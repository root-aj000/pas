"""
Train one ensemble member per process, on CPU, so peak memory stays bounded.

Run with: python -m src.pipeline.cpu_member --members cat_obl --artifacts artifacts/ensemble

WHY THIS EXISTS
The normal path keeps every member's folds in one process and, on the GPU, splits
members across workers with `run_gpu_shards`. On a CPU machine neither applies:
there is one device, so `plan_gpu_shards` returns a single group and all members
train sequentially inside one interpreter.

That is fine for correctness and bad for memory. At 699,635 rows by ~178 columns
the parent holds roughly 2 GB of frames, and every fold adds a copy of the
training slice plus the in-fold target-encoding block - 31 encodings cross-fitted
over 5 inner folds, which is itself a few hundred MB of intermediate groupbys. A
14 GB laptop with a browser open does not have that headroom, and the failure mode
is an OOM kill part-way through a run that has already spent hours.

So this module does what the GPU path does for devices, applied to processes on one
device: one member per subprocess, so the previous member's memory is returned to
the OS before the next one allocates. `run_cpu_members` is the driver; run this
file directly to train a single member and debug it.

WHAT IS AND IS NOT THE SAME
Identical numbers, deliberately. This calls the same `fit_member_to_artifacts` with
the same config, so the seed is still `config.seed + fold` and never depends on
which process ran it. A CPU run through this module and an inline run produce the
same `.npy` files. The only difference is that the parent process does not hold
the frames.

The config comes from a payload file, not from config.yaml. `run_cpu_members`
writes it before starting any worker, so a worker trains the member its caller
asked for. Reading config.yaml here instead made the workers a second source of
truth: a caller passing its own two-member config got the file's nine members, or
an error naming members it had never mentioned, and nothing outside this module
could tell which had happened.

Note the memory cost that is NOT solved here: reading the frames still needs
~2 GB inside each worker, so peak is one worker's frames rather than the parent's
plus a worker's. On a 14 GB machine that is the difference between fitting and
not.
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from src.components.ensemble import (
    EnsembleConfig,
    config_from_payload,
    config_payload,
    fit_member_to_artifacts,
    load_ensemble_frames,
    resolve_device,
)
from src.config.configuration import PipelineConfigReader
from src.utils.common import get_logger, log_step, setup_logging


def train_one(
    wanted: list[str],
    artifacts_dir: Path,
    config: EnsembleConfig,
    data_dir: Path,
    features: list[str],
    categorical: list[str],
    target_column: str,
) -> int:
    """Train the named members in THIS process, one after another.

    Args:
        wanted: Member names, exactly as the caller spelled them.
        artifacts_dir: Where `oof_<name>.npy` and `test_<name>.npy` go.
        config: The ensemble settings, handed over by the caller rather than
            re-read from config.yaml.
        data_dir: The `data_cleaning_encoding` folder to read frames from.
        features: The feature columns config.yaml asks for.
        categorical: Categorical column names within them.
        target_column: Label name.

    Returns:
        0 on success.
    """
    logger = get_logger()

    known = {member.name: member for member in config.members}
    unknown = [name for name in wanted if name not in known]
    if unknown:
        logger.error("no such member in the handed-over config: %s", unknown)
        return 1

    device = resolve_device(config.device)
    # Same reason as the GPU shard: `train_member` reads `config.device` when it
    # calls `apply_device`, so the resolved value has to go back into the config
    # rather than staying a local.
    config = replace(config, device=device)
    log_step(
        "cpu_member_start",
        members=len(wanted),
        device=device,
        names=",".join(wanted),
    )

    train_frame, competition_frame = load_ensemble_frames(data_dir, features, categorical)
    log_step(
        "cpu_member_frames",
        train_rows=len(train_frame),
        competition_rows=len(competition_frame),
        features=len(features),
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
            artifacts_dir,
        )
        log_step("member", member=name, oof_auc=round(auc, 6), device=device)
        # Drop the arrays before the next member: on a CPU run these are the
        # largest allocations still alive, and they are pure output.
        del oof, test

    log_step("cpu_member_done", members=len(wanted))
    return 0


def run_cpu_members(
    names: list[str],
    config: EnsembleConfig,
    artifacts_dir: Path,
    data_dir: Path,
    features: list[str],
    categorical: list[str],
    target_column: str,
    log_dir: Path | None = None,
) -> None:
    """Train each named member in its own subprocess, sequentially.

    Args:
        names: Member names, exactly as the caller spelled them.
        config: The ensemble settings. Serialised into the payload file each
            worker reads, so a worker trains what the caller asked for.
        artifacts_dir: Where the workers write their `.npy` predictions.
        data_dir: The `data_cleaning_encoding` folder the workers read frames
            from.
        features: The feature columns config.yaml asks for.
        categorical: Categorical column names within them.
        target_column: Label name.
        log_dir: One log per member. Defaults to artifacts_dir / "cpu_logs".

    Raises:
        RuntimeError: If any member's process exits non-zero. Its log is named in
            the message, because a member that dies in a subprocess otherwise
            fails silently and the stack is built from whichever members survived.

    Note:
    The payload exists because a subprocess cannot inherit the caller's arguments.
    Reading config.yaml in the worker instead - which is what this did - meant the
    worker trained the file's members rather than the ones requested, so a caller
    passing its own config got either the wrong models or an error naming models
    it had never asked for. `train_ensemble` is supposed to be a function of its
    arguments; this is what makes it one.
    """
    logger = get_logger()
    if log_dir is None:
        log_dir = artifacts_dir / "cpu_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Written before the first worker starts, so a crash mid-loop leaves a
    # payload on disk rather than a half-written one.
    payload_path = artifacts_dir / "cpu_members_payload.json"
    payload = config_payload(config)
    payload["data_dir"] = str(data_dir)
    payload["features"] = list(features)
    payload["categorical"] = list(categorical)
    payload["target_column"] = str(target_column)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(payload, default=repr))

    failures: list[str] = []
    for name in names:
        log_path = log_dir / f"{name}.log"
        environment = dict(os.environ)
        # Keep BLAS from oversubscribing: the members already set n_jobs, and a
        # second layer of threading on top is what turns "slow" into "OOM".
        environment.setdefault("OMP_NUM_THREADS", "8")

        command = [
            sys.executable,
            "-m",
            "src.pipeline.cpu_member",
            "--members",
            name,
            "--artifacts",
            str(artifacts_dir),
            "--payload",
            str(payload_path),
        ]
        log_step("cpu_member_launch", member=name, log=str(log_path))

        with log_path.open("w", encoding="utf-8") as handle:
            # `subprocess.run` blocking is correct here and is the whole point:
            # one member at a time is what keeps peak memory bounded. The GPU
            # path deliberately does NOT block, because it has one process per
            # device to overlap.
            return_code = subprocess.run(
                command,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode

        if return_code != 0:
            failures.append(f"{name} exited {return_code}; log {log_path}")
            logger.error("member %s failed; log %s", name, log_path)
        else:
            logger.info("member %s finished", name)

    if failures:
        raise RuntimeError(
            "CPU members failed: "
            + "; ".join(failures)
            + ". The surviving .npy files are on disk, but they must not be "
            "stacked against a member that is missing."
        )


def main(argv: list[str] | None = None) -> int:
    """Entry point for `python -m src.pipeline.cpu_member`.

    Args:
        argv: Command line. Defaults to `sys.argv[1:]`.

    Returns:
        0 on success, 1 on a bad invocation or a member that could not be trained.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--members",
        required=True,
        help="Comma-separated member names, exactly as the caller spells them.",
    )
    parser.add_argument(
        "--artifacts",
        required=True,
        type=Path,
        help="Where oof_<member>.npy and test_<member>.npy are written.",
    )
    parser.add_argument(
        "--payload",
        type=Path,
        default=None,
        help="JSON file written by `run_cpu_members`, holding the caller's config, "
        "feature list and data directory. Required when `train_ensemble` is the "
        "caller; when omitted the members are resolved from config.yaml, which is "
        "only correct for running this file by hand.",
    )
    args = parser.parse_args(argv)

    setup_logging()
    logger = get_logger()
    wanted = [name.strip() for name in args.members.split(",") if name.strip()]
    if not wanted:
        logger.error("no members requested")
        return 1

    if args.payload is not None:
        if not args.payload.exists():
            logger.error("payload not found: %s", args.payload)
            return 1
        payload = json.loads(args.payload.read_text())
        config = config_from_payload(payload)
        data_dir = Path(str(payload["data_dir"]))
        features = [str(c) for c in payload.get("features", [])]
        categorical = [str(c) for c in payload.get("categorical", [])]
        target_column = str(payload.get("target_column", ""))
        if not features or not target_column:
            logger.error("payload is missing features or target_column: %s", args.payload)
            return 1
        return train_one(
            wanted, args.artifacts, config, data_dir, features, categorical,
            target_column,
        )

    # Hand-run only. `train_ensemble` always writes a payload, so reaching here
    # means someone typed the command themselves.
    reader = PipelineConfigReader()
    pipeline = reader.create_pipeline_config()
    config = reader.create_ensemble_config()
    features = list(pipeline.training.features)
    categorical = [
        column
        for column in reader.create_cleaning_config().categorical_columns
        if column in features
    ]
    return train_one(
        wanted,
        args.artifacts,
        config,
        pipeline.artifacts_root / "data_cleaning_encoding",
        features,
        categorical,
        str(pipeline.training.target_column),
    )


if __name__ == "__main__":
    sys.exit(main())
