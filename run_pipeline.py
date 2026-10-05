"""
The one and only entry point. Runs every stage in order.

    python run_pipeline.py

No arguments. Ever. See .dev/DEBUGGING.md section 8.5.

Everything that varies lives in config.yaml, so changing a setting is a
committed edit and the git history shows which settings produced which result.
If this command could take flags, somebody would eventually run it with flags
nobody wrote down.

Every run writes its own log to logs/, carrying the git commit, a hash of
config.yaml, the seed, the package versions and the exit code. Nothing is ever
deleted.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

from src.config.configuration import PipelineConfigReader
from src.pipeline.stage_01_data_ingestion import run_pipeline as run_stage_01
from src.pipeline.stage_02_data_cleaning_encoding import run_pipeline as run_stage_02
from src.pipeline.stage_03_model_training import run_pipeline as run_stage_03
from src.pipeline.stage_04_model_evaluation import run_pipeline as run_stage_04
from src.utils.common import (
    build_run_log_name,
    ensure_project_root,
    log_run_footer,
    log_run_header,
    log_step,
    next_model_version,
    setup_logging,
)

COMMAND = "python run_pipeline.py"
# Resolved against the project root at startup, so this file runs from any
# directory on any machine. A hardcoded absolute path lived here briefly and
# broke every run that was not on that one machine.


class RunLogWriter:
    """Sends logging output to both the console and a per-run log file.

    The console copy is for whoever is watching. The file copy is the evidence:
    append-only, one file per run, never edited or deleted.

    Args:
        log_path: The logs folder.
        prefix: What the run was, used in the file name.
        started_at: When the run started.
    """

    def __init__(self, log_path: Path, prefix: str, started_at: datetime) -> None:
        import logging

        log_path.mkdir(parents=True, exist_ok=True)
        self.file_path = log_path / build_run_log_name(prefix, started_at)
        self.handler = logging.FileHandler(self.file_path, encoding="utf-8")
        self.handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S"
            )
        )
        logging.getLogger().addHandler(self.handler)

    def detach(self) -> Path:
        """Stop writing to the file and return its path.

        Returns:
            The log file that was written.
        """
        import logging

        logging.getLogger().removeHandler(self.handler)
        self.handler.close()
        return self.file_path


def main() -> int:
    """Run all four stages in order, logging everything.

    Returns:
        0 if every stage succeeded, 1 if any stage raised.

    Note:
        A failure is logged with its exit code and the log file is kept. A failed
        run is evidence - .dev/RULES.md rule 11 and .lead/08-TRAPS.md Trap 11 both
        turn on having it.
    """
    started_at = datetime.now().astimezone()
    ensure_project_root()
    setup_logging()
    reader = PipelineConfigReader()
    seed = int(reader.config["random_seed"])

    from src.config.configuration import resolve_project_path

    run_log = RunLogWriter(
        resolve_project_path(reader.config["log_path"]), "full_run", started_at
    )
    log_run_header(COMMAND, reader.config_path, seed)

    exit_code = 0
    status = "SUCCESS"
    facts: dict[str, object] = {}
    started = time.monotonic()

    try:
        log_step("run", stage="01_data_ingestion")
        run_stage_01()

        log_step("run", stage="02_data_cleaning_encoding")
        run_stage_02()

        model_version = next_model_version(reader.create_pipeline_config().models_root)
        log_step("run", stage="03_model_training", model_version=model_version)
        model_version = run_stage_03(model_version=model_version)
        facts["model_version"] = f"model_{model_version}"

        log_step("run", stage="04_model_evaluation")
        run_stage_04(model_version=model_version)

        submission_dir = (
            resolve_project_path(reader.config["report_path"]) / "submissions"
        )
        written = sorted(submission_dir.glob("submission_model_*.csv"))
        if written:
            facts["submission"] = written[-1].name
    # Catching everything is deliberate, not lazy. This is the entry point: a
    # failure here must be written into the run log with its exit code, not die
    # with a traceback and no record. The error is logged in full below, so it is
    # recorded rather than swallowed.
    except Exception as error:  # noqa: BLE001
        exit_code = 1
        status = "FAILURE"
        facts["error"] = f"{type(error).__name__}: {error}"
        setup_logging().exception("run failed")

    log_run_footer(
        status,
        exit_code,
        time.monotonic() - started,
        **facts,
    )
    log_file = run_log.detach()
    print(f"\nrun log: {log_file}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
