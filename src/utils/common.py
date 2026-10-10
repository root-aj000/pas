"""
Shared helpers used by more than one component.

This is the only module for helpers that genuinely appear everywhere: logging
setup, the one-line step log, the column check, seeding, the forensic run header,
and save/load. A helper used by exactly one component stays in that component -
see .dev/RULES.md rule 7, "no utils dumping ground".
"""

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER_NAME = "pipeline"

# Where Kaggle mounts attached datasets. Read-only. If a data file is missing
# from the project tree, these directories are scanned for a matching filename
# before giving up. Local files always win - this never overrides data/ when it
# exists.
KAGGLE_INPUT_DIRS = (Path("/kaggle/input"),)


def find_project_root(marker: str = "config.yaml") -> Path:
    """Return the project root: the directory holding config.yaml.

    Args:
        marker: The file that identifies the root.

    Returns:
        The project root as an absolute path.

    Raises:
        FileNotFoundError: If no directory from here up to the filesystem root
            holds the marker. Lists everywhere it looked.
    """
    searched: list[Path] = []
    candidates = [
        Path.cwd(),
        *Path.cwd().parents,
        Path(__file__).resolve().parent,
        *Path(__file__).resolve().parents,
    ]
    for directory in dict.fromkeys(candidates):
        searched.append(directory)
        if (directory / marker).exists():
            return directory.resolve()
    raise FileNotFoundError(
        f"Could not find {marker} in the current directory or any parent, "
        f"nor near {__file__}. Looked in: {[str(d) for d in searched]}. "
        "Run from inside the project, or clone it first."
    )


def ensure_project_root(marker: str = "config.yaml") -> Path:
    """Change into the project root so relative paths work from anywhere.

    Args:
        marker: The file that identifies the root.

    Returns:
        The project root.

    Note:
    Every entry point - run_pipeline.py and each research script - calls this
    first. That is the single reason `Path("data/train.csv")` works whether you
    run `python run_pipeline.py` from the root, `python research/foo.py` from
    anywhere, or `python /kaggle/working/pas/run_pipeline.py` from a notebook.
    """
    root = find_project_root(marker)
    os.chdir(root)
    return root


def describe_kaggle_input(limit: int = 30) -> str:
    """Return a listing of what is actually mounted under /kaggle/input.

    Args:
        limit: Maximum entries to list. Kaggle mounts are small, so this is a
            safety cap rather than a real constraint.

    Returns:
        One path per line, or "(nothing mounted)" when the directory is absent.
        Used in the error message so a missing dataset diagnoses itself instead
        of just failing.
    """
    lines: list[str] = []
    for mount in KAGGLE_INPUT_DIRS:
        if not mount.is_dir():
            lines.append(f"{mount}/  (not present - no dataset attached?)")
            continue
        entries = sorted(p for p in mount.rglob("*") if p.is_file())[:limit]
        if not entries:
            lines.append(f"{mount}/  (present but empty)")
        lines.extend(f"  {path}" for path in entries)
    return "\n".join(lines)


def find_kaggle_file(
    filename: str, preferred_root: str | Path | None = None
) -> Path | None:
    """Look for a data file in the Kaggle input mounts, at any depth.

    Args:
        filename: Just the file name, for example "train.csv".
        preferred_root: A directory to search first. When the file exists there
            that match wins over anything else mounted.

    Returns:
        The match, or None if there is none.

    Raises:
        FileNotFoundError: If the file is mounted in more than one place and no
            preferred_root was given, because there is no way to tell which one
            was meant and guessing wrong trains on the wrong data silently.

    Note:
    Recursive on purpose. A competition dataset may mount as
    `/kaggle/input/<slug>/train.csv`, as
    `/kaggle/input/<slug>/competitions/playground-series-s6e10/train.csv`, or
    with the files one level deeper still. A one-level glob misses the last two.

    The ambiguity guard exists because two mounted datasets can both hold a
    train.csv. Sorted order used to decide, which meant the answer changed if a
    path changed - no error, just a different dataset and a different score.
    """
    if preferred_root is not None:
        direct = Path(preferred_root) / filename
        if direct.is_file():
            return direct

    matches: list[Path] = []
    for mount in KAGGLE_INPUT_DIRS:
        if not mount.is_dir():
            continue
        matches.extend(
            candidate
            for candidate in sorted(mount.rglob(filename))
            if candidate.is_file()
        )
    if not matches:
        return None
    if preferred_root is not None and len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        listed = "\n  ".join(str(m) for m in matches)
        raise FileNotFoundError(
            f"{filename} is mounted in {len(matches)} places and config.yaml does "
            f"not say which to use:\n  {listed}\n"
            "Set kaggle_dataset_path in config.yaml to the directory that holds "
            "the files you want."
        )
    return matches[0]


def setup_logging(name: str = LOGGER_NAME) -> logging.Logger:
    """Return a logger that prints to the console with a timestamp.

    Args:
        name: The logger's name. One name for the whole pipeline, so every line
            lands in the same run log and reads as one story.

    Returns:
        A configured logger.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    # Lightning advertises its cloud logger on every Trainer it builds, once per
    # member per fold, via `rank_zero_info`. The message reaches the console
    # through `pytorch_lightning.utilities.rank_zero`'s logger - not the
    # `lightning_utilities` one, and not the "lightning"/"pytorch_lightning"
    # names either, because that module sets its own level when it is imported.
    # Silencing the parent is undone by that. Nobody here logs to the cloud, so
    # the tip is noise in the middle of a run log.
    for noisy in (
        "pytorch_lightning.utilities.rank_zero",
        "lightning_utilities.core.rank_zero",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return the shared logger.

    Args:
        name: The logger's name.

    Returns:
        The shared logger. setup_logging must have been called first.
    """
    return logging.getLogger(name)


def log_step(
    step: str,
    rows_in: int | None = None,
    rows_out: Any = None,
    **details: Any,
) -> None:
    """Log one pipeline step: what it took in, what it did, what it produced.

    Args:
        step: Short name of the step, for example "clean".
        rows_in: Row count before the step, or None if not applicable.
        rows_out: Row count after the step, or None if not applicable. Any, not
            int, because callers also pass numpy integer counts.
        **details: Any other facts worth logging, for example dropped=412.

    Example:
        log_step("clean", rows_in=15231, rows_out=14819, dropped_duplicates=412)
    """
    parts = [f"[{step}]"]
    if rows_in is not None:
        parts.append(f"rows_in={rows_in}")
    if rows_out is not None:
        parts.append(f"rows_out={rows_out}")
    parts.extend(f"{key}={value}" for key, value in details.items())
    get_logger().info(" ".join(parts))


def check_columns(name: str, actual: set[str], expected: set[str]) -> None:
    """Stop the program if the expected columns are not all present.

    Args:
        name: Name of the step being checked, used in the error message.
        actual: Column names that exist.
        expected: Column names the code is about to use.

    Raises:
        ValueError: If any expected column is missing, listing exactly which.
    """
    missing = expected - actual
    if missing:
        raise ValueError(
            f"{name}: missing columns {sorted(missing)}. "
            f"Available columns: {sorted(actual)}"
        )


def seed_everything(seed: int) -> None:
    """Fix every source of randomness, so a run can be repeated exactly.

    Args:
        seed: The seed value. The same seed means the same result.

    Note:
        Setting PYTHONHASHSEED only affects child processes, not this one. It is
        still worth setting for anything launched later.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    get_logger().info("Random seed set to %d", seed)


def seed_torch(seed: int) -> None:
    """Fix PyTorch's random number generators, so GPU runs are repeatable.

    Args:
        seed: The seed value, the same one passed to seed_everything().

    Note:
        Imported lazily so PyTorch stays an optional dependency. This project has
        no need for it - the EDA chose gradient boosting over a neural network -
        but the owner trains on a Kaggle GPU, so the helper is here rather than
        rediscovered later.
    """
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    get_logger().info("PyTorch seed set to %d", seed)


def hash_file(file_path: Path) -> str:
    """Return a short hash of a file's bytes.

    Args:
        file_path: The file to hash.

    Returns:
        The first 8 hex characters of the SHA-256 of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot hash a file that does not exist: {file_path}")
    return hashlib.sha256(file_path.read_bytes()).hexdigest()[:8]


def get_git_commit() -> str:
    """Return the current git commit, or "unknown" outside a repository.

    Returns:
        The short commit hash, or the string "unknown".

    Note:
        The commit and the config hash are the two facts that make a run
        provable. Without them a log is a story with no evidence.
        .dev/DEBUGGING.md section 8.2.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def get_installed_packages() -> dict[str, str]:
    """Return the versions of the packages that affect a result.

    Returns:
        Library name to version, for the libraries this project depends on.
    """
    packages = ["pandas", "numpy", "scikit-learn", "xgboost", "scipy"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            module = __import__(package)
        except ImportError:
            continue
        versions[package] = getattr(module, "__version__", "unknown")
    return versions


def log_run_header(command: str, config_path: Path, seed: int) -> None:
    """Log the facts that identify this run forever, before any work starts.

    Args:
        command: The exact command that was run.
        config_path: Path to config.yaml, which is hashed to prove the settings.
        seed: The random seed used.

    Note:
        git commit and config hash are the two facts that make a run provable.
        Without them a log cannot be checked against the code that produced it.
    """
    logger = get_logger()
    logger.info("=" * 70)
    logger.info("RUN STARTED")
    logger.info("  command:      %s", command)
    logger.info(
        "  started:      %s", datetime.now().astimezone().isoformat(timespec="seconds")
    )
    logger.info("  host:         %s", platform.node())
    logger.info("  python:       %s", sys.version.split()[0])
    logger.info("  git_commit:   %s", get_git_commit())
    logger.info("  config_hash:  %s", hash_file(config_path))
    logger.info("  random_seed:  %s", seed)
    logger.info(
        "  packages:     %s",
        ", ".join(
            f"{name} {version}" for name, version in get_installed_packages().items()
        ),
    )
    logger.info("=" * 70)


def log_run_footer(
    status: str, exit_code: int, duration_seconds: float, **facts: Any
) -> None:
    """Log how the run ended.

    Args:
        status: "SUCCESS" or "FAILURE".
        exit_code: The process exit code.
        duration_seconds: How long the run took.
        **facts: Any other headline numbers worth recording, e.g. roc_auc=0.957.
    """
    logger = get_logger()
    logger.info("=" * 70)
    logger.info("RUN FINISHED")
    logger.info("  status:       %s", status)
    logger.info("  exit_code:    %d", exit_code)
    logger.info("  duration:     %d seconds", round(duration_seconds))
    for key, value in facts.items():
        logger.info("  %-13s %s", f"{key}:", value)
    logger.info("=" * 70)


def build_run_log_name(prefix: str, started_at: datetime) -> str:
    """Return the file name for a run log, as date_time_what.

    Args:
        prefix: What the run was, e.g. "full_run".
        started_at: When the run started.

    Returns:
        A file name like "2026-10-04_2215_full_run.log".

    Note:
        Sorted by name, a list of these reads as a history of the project's work.
        .dev/DEBUGGING.md section 8.1.
    """
    return f"{started_at.strftime('%Y-%m-%d_%H%M')}_{prefix}.log"


def save_dataframe(frame: pd.DataFrame, output_path: Path) -> Path:
    """Save a dataframe to CSV, creating the parent folder if needed.

    Args:
        frame: The rows to save.
        output_path: Where to write, including the file name.

    Returns:
        The path written to.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def save_json(payload: dict[str, Any], output_path: Path) -> Path:
    """Save a dictionary as JSON, creating the parent folder if needed.

    Args:
        payload: The values to save.
        output_path: Where to write, including the file name.

    Returns:
        The path written to.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(payload, file, indent=2, default=_json_safe)
    return output_path


def _json_safe(value: Any) -> Any:
    """Return `value` in a form json can serialise, without losing it.

    Args:
        value: Whatever `json.dump` could not handle.

    Returns:
        A plain list for a numpy array, a plain float for a numpy scalar, and
        `str(value)` for anything else.

    Note:
    `default=str` was the previous answer to "json.dump cannot serialise this",
    and it turns a numpy array into its *repr*, which is not valid JSON - the
    file parses back as one enormous string rather than a list. `save_json` is
    what writes `ensemble_summary.json`, whose `competition_probabilities` is a
    numpy array, so the written summary held `"0.0123, 0.9876, ..."` as a string.
    The run itself was unaffected - the in-memory array went to the submission -
    but every consumer that read the summary back, which is the entire reason it
    is written, got a string. `np.ndarray.tolist()` is the fix, and it is the
    difference between a summary anyone can plot and a file that only appears to
    have parsed.
    """
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def load_json(input_path: Path) -> dict[str, Any]:
    """Read a JSON file into a dictionary.

    Args:
        input_path: The file to read.

    Returns:
        The parsed contents.

    Raises:
        FileNotFoundError: If the file does not exist. A missing model bundle
            means the pipeline was run out of order, which must stop the run.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Expected a JSON file at {input_path}")
    with input_path.open() as file:
        return json.load(file)


def apply_categorical_encoding(
    frame: pd.DataFrame,
    categorical_columns: list[str],
    categorical_encoding: str,
    step_name: str,
    category_maps: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Re-apply the categorical dtype after reading a prepared CSV.

    Args:
        frame: Rows just read from artifacts/.
        categorical_columns: The columns that were categorical in stage 2.
        categorical_encoding: "native" or "one_hot", as stage 2 recorded it.
        step_name: Which step is reading, used in the error message.
        category_maps: Column name to the level list stage 2 pinned, as recorded
            in features.json. Restores the exact categories; without it the levels
            are inferred from this frame's own rows.

    Returns:
        A new frame with the categorical columns as pandas `category` dtype, or
        unchanged when the encoding was one_hot.

    Note:
    CSV does not carry pandas dtypes. A `category` column written to CSV and read
    back comes in as `str`, which xgboost then rejects with an error about
    `enable_categorical` - pointing at the model rather than at the file format.
    So the dtype is re-applied from what stage 2 recorded, rather than being
    guessed at the point of use.

    The levels come from `category_maps`, not from `astype("category")`. Inferring
    them here made the encoding depend on which rows happened to be in this frame:
    a level absent from the validation split's rows changed that split's codes and
    its category count, so the same column meant different things to stage 3 and
    stage 4 - the drift stage 2 pins its categories specifically to prevent, two
    stages away from where it was introduced. It also dropped the `_cat_` twins
    entirely, because they were never named in the contract: all 62 came back from
    the CSV as int64, and an embedding per distinct value - the entire reason the
    twins exist - was never created.

    A frame read from an older features.json, or one whose contract predates
    category maps, falls back to inference. That is a downgrade rather than a
    failure, because the alternative is refusing to read artifacts this project
    already has on disk; `log` it so the downgrade is visible.

    Raises:
        ValueError: If the encoding is not one we support, or a column recorded as
            categorical is missing. Either means the artifact was built by a
            different config than the one being used now.
    """
    if categorical_encoding not in ("native", "one_hot"):
        raise ValueError(
            f"{step_name}: unknown categorical_encoding '{categorical_encoding}'. "
            "Supported: 'native', 'one_hot'."
        )
    if categorical_encoding == "one_hot":
        return frame

    missing = set(categorical_columns) - set(frame.columns)
    if missing:
        raise ValueError(
            f"{step_name}: artifacts were written with native categorical columns "
            f"{sorted(categorical_columns)}, but these are missing: {sorted(missing)}. "
            "Re-run stage 2 so the artifacts match config.yaml."
        )

    maps = category_maps or {}
    if not maps:
        log_step(
            "category_maps_missing",
            step_name=step_name,
            columns=len(categorical_columns),
            reason="features.json predates category_maps; levels inferred per frame",
        )
    typed = frame.copy()
    for column in categorical_columns:
        levels = maps.get(column)
        if levels:
            # `astype(str)` first, and it is load-bearing.
            #
            # A `_cat_` twin holds the string "3", which pandas writes to CSV as a
            # bare `3` and reads back as int64. Casting int64 3 against the level
            # list ["0","1","2",...] matches nothing, so every row became NaN and
            # all twenty twins were dead columns - while `cat.codes` was -1
            # everywhere, so a check comparing codes across splits saw -1 == -1 and
            # called them consistent. Levels are stored as strings in the contract
            # for exactly this reason, and the frame has to be put in the same type
            # before the comparison can mean anything.
            typed[column] = pd.Categorical(
                typed[column].astype(str), categories=levels
            )
        else:
            typed[column] = typed[column].astype("category")
    return typed


def next_model_version(models_path: Path) -> int:
    """Return the next unused model version number.

    Args:
        models_path: The folder holding model_1, model_2, and so on.

    Returns:
        One more than the highest version already present, or 1 if none are.

    Note:
        A trained model is never overwritten. .dev/PROJECT-STRUCTURE.md: "Never
        overwrite a trained model."
    """
    if not models_path.exists():
        return 1
    versions = [
        int(folder.name.split("_")[-1])
        for folder in models_path.iterdir()
        if folder.is_dir()
        and folder.name.startswith("model_")
        and folder.name.split("_")[-1].isdigit()
    ]
    return max(versions, default=0) + 1
