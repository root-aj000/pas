"""
One worker per GPU. Trains a named subset of the ensemble's members.

Not run by hand and not part of `run_pipeline.py`: `train_ensemble` spawns this
with `CUDA_VISIBLE_DEVICES` set to a single GPU, one process per device, and each
worker writes the same `oof_<member>.npy` / `test_<member>.npy` files the
single-process path would. The parent then reads those back and stacks them, so
nothing downstream knows or cares that the work was split.

    CUDA_VISIBLE_DEVICES=1 python -m src.pipeline.gpu_shard \
        --members cat_obl,lgb_deep --artifacts artifacts/ensemble

Why a process and not a thread: a CUDA context belongs to a process. Two threads
inside one process share device 0 and the second one queues behind the first, so
threading would look like it worked and deliver one GPU's throughput.

Why the frames are re-read here rather than passed in: 700k rows times 160
columns is gigabytes, and pickling that through a pipe costs more than the minute
it takes to read the CSVs. Every worker builds its frames with
`load_ensemble_frames`, the same function the parent uses, so a worker cannot
train on different rows from the stack.
"""

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

from src.components.ensemble import (
    fit_member_to_artifacts,
    load_ensemble_frames,
    resolve_device,
)
from src.config.configuration import PipelineConfigReader
from src.utils.common import get_logger, log_step, setup_logging


def main(argv: list[str] | None = None) -> int:
    """Train the requested members and write their predictions.

    Args:
        argv: Command line. Defaults to `sys.argv[1:]`.

    Returns:
        0 on success, 1 if a member could not be trained.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--members",
        required=True,
        help="Comma-separated member names, exactly as config.yaml spells them.",
    )
    parser.add_argument(
        "--artifacts",
        required=True,
        type=Path,
        help="Where oof_<member>.npy and test_<member>.npy are written.",
    )
    args = parser.parse_args(argv)

    setup_logging()
    logger = get_logger()
    wanted = [name.strip() for name in args.members.split(",") if name.strip()]

    reader = PipelineConfigReader()
    config = reader.create_ensemble_config()
    pipeline = reader.create_pipeline_config()
    features = list(pipeline.training.features)
    categorical = [
        column
        for column in reader.create_cleaning_config().categorical_columns
        if column in features
    ]
    target_column = str(pipeline.training.target_column)

    known = {member.name: member for member in config.members}
    unknown = [name for name in wanted if name not in known]
    if unknown:
        logger.error("no such member in config.yaml: %s", unknown)
        return 1

    device = resolve_device(config.device)
    # Same reason as the parent: `train_member` reads `config.device`, so the
    # resolved value has to go back into the config, not just into the log line.
    config = replace(config, device=device)
    log_step(
        "gpu_shard_start",
        members=len(wanted),
        device=device,
        visible=os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
    )

    train_frame, competition_frame = load_ensemble_frames(
        pipeline.artifacts_root / "data_cleaning_encoding", features, categorical
    )

    for name in wanted:
        spec = known[name]
        oof, test, auc = fit_member_to_artifacts(
            spec,
            train_frame,
            competition_frame,
            features,
            categorical,
            config,
            target_column,
            args.artifacts,
        )
        log_step("member", member=name, oof_auc=round(auc, 6), device=device)
        del oof, test

    log_step("gpu_shard_done", members=len(wanted))
    return 0


if __name__ == "__main__":
    sys.exit(main())
