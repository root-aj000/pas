"""Measure the baselines required by .lead/00-PROBLEM-AND-BASELINE.md Step 0.4.

Step 0.4 asks for at least two baselines before any model is built: "do
nothing", and "one simple rule". They are the numbers every later model has to
beat. This script measures them on data/train.csv and prints them as a table.

Why this measures on the WHOLE training file, with no train/test split:

A baseline here is a rule with no fitted parameters. "Always answer False" has
nothing to learn from data, so measuring it on a subset would only add noise.
The same is true of the simple rule: the threshold is written down in this file
before the script is run, not chosen from the results. There is nothing here
that can be overfitted, so there is nothing to hold back.

Run with:  python research/02_baselines.py
Writes:    nothing. The printed table is copied into
           docs/problem_statement.md by hand, and this run is kept in
           logs/dev/ as the evidence for those numbers.

Uses only the Python standard library. The project has no pandas installed yet,
and 700,000 rows does not need one.
"""

import csv
import logging
from pathlib import Path

# The rule we are registering as "one simple rule", fixed before the run.
#
# Why this feature and this cut: a passenger's overall satisfaction in this
# survey is dominated by whether the things they remember were good, and
# in-flight entertainment is the service the survey consistently shows as the
# strongest single driver. The scale is 0-5, so "rated it 5" means "the best
# possible answer". One feature, one threshold, no tuning.
SIMPLE_RULE_COLUMN = "Inflight entertainment"
SIMPLE_RULE_SATISFIED_IF_RATING_IS = 5

# The 16 columns that are the passenger's own 0-5 ratings of the flight.
# Scored individually below so that no single-feature result can be quietly
# cherry-picked: every one of them is printed, not just the best.
SERVICE_RATING_COLUMNS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
]

REQUIRED_COLUMNS = set(SERVICE_RATING_COLUMNS) | {
    "satisfaction",
    "Flight Distance",
    "Age",
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
}

# Resolved from this file's location, not the working directory, so the script
# runs from anywhere. Kept stdlib-only by design - see the module docstring.
TRAIN_PATH = Path(__file__).resolve().parent.parent / "data" / "train.csv"

logger = logging.getLogger("baselines")


class RuleScores:
    """How a rule did: how many rows it got right, and in which direction.

    Args:
        name: What the rule is called, for printing.
        true_positive: Predicted satisfied, actually satisfied.
        false_positive: Predicted satisfied, actually not satisfied.
        true_negative: Predicted not satisfied, actually not satisfied.
        false_negative: Predicted not satisfied, actually satisfied.
    """

    def __init__(
        self,
        name: str,
        true_positive: int,
        false_positive: int,
        true_negative: int,
        false_negative: int,
    ) -> None:
        self.name = name
        self.true_positive = true_positive
        self.false_positive = false_positive
        self.true_negative = true_negative
        self.false_negative = false_negative

    @property
    def total(self) -> int:
        """How many rows this rule was scored on."""
        return (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
        )

    @property
    def accuracy(self) -> float:
        """Share of rows the rule got right, from 0 to 1."""
        if self.total == 0:
            return 0.0
        return (self.true_positive + self.true_negative) / self.total

    @property
    def precision(self) -> float:
        """Of the rows it called satisfied, how many really were.

        Returns 1.0 when the rule called nobody satisfied, because there were
        no false alarms to count. An empty prediction is not a clever rule.
        """
        predicted_positive = self.true_positive + self.false_positive
        if predicted_positive == 0:
            return 1.0
        return self.true_positive / predicted_positive

    @property
    def recall(self) -> float:
        """Of the rows that really were satisfied, how many it caught."""
        actually_positive = self.true_positive + self.false_negative
        if actually_positive == 0:
            return 0.0
        return self.true_positive / actually_positive


def check_required_columns(name: str, actual: list[str], expected: set[str]) -> None:
    """Stop the script if a column we need is not in the file.

    Args:
        name: Which step is being checked, used in the error message.
        actual: Column names read from the file header.
        expected: Column names the rest of this script is going to use.

    Raises:
        ValueError: If any expected column is missing, listing which.
    """
    missing = expected - set(actual)
    if missing:
        raise ValueError(
            f"{name}: missing columns {sorted(missing)}. Columns found: {actual}"
        )


def add_score(
    scores: RuleScores, predicted_satisfied: bool, actually_satisfied: bool
) -> None:
    """Record one row's outcome in the running tally for a rule.

    Args:
        scores: The tally to update, modified in place.
        predicted_satisfied: What the rule said.
        actually_satisfied: What the answer in the file says.
    """
    if predicted_satisfied and actually_satisfied:
        scores.true_positive += 1
    elif predicted_satisfied and not actually_satisfied:
        scores.false_positive += 1
    elif not predicted_satisfied and not actually_satisfied:
        scores.true_negative += 1
    else:
        scores.false_negative += 1


def read_train_file(train_path: Path) -> tuple[list[str], list[list[str]]]:
    """Read the training CSV and return its header and all rows.

    Args:
        train_path: Path to the training CSV.

    Returns:
        The column names, and the rows as lists of strings.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file has no rows. An empty file means the download
            failed, so there is nothing to measure.
    """
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    with train_path.open(newline="") as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Training data has no header row: {train_path}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Training data has a header but no rows: {train_path}")

    return header, rows


def score_every_rating_column(
    header: list[str], rows: list[list[str]], rating_columns: list[str]
) -> list[RuleScores]:
    """Score "rated it 5 means satisfied" for each rating column.

    Args:
        header: Column names, used to find each column's position.
        rows: The data rows.
        rating_columns: The columns to score, each holding a 0-5 rating.

    Returns:
        One RuleScores per column, in the order given.
    """
    positions = {name: header.index(name) for name in rating_columns}
    satisfied_position = header.index("satisfaction")

    tallies = {name: RuleScores(name, 0, 0, 0, 0) for name in rating_columns}

    for row in rows:
        actually_satisfied = row[satisfied_position] == "True"
        for name in rating_columns:
            rating = row[positions[name]]
            predicted_satisfied = rating == str(SIMPLE_RULE_SATISFIED_IF_RATING_IS)
            add_score(tallies[name], predicted_satisfied, actually_satisfied)

    return [tallies[name] for name in rating_columns]


def measure_base_rate(rows: list[list[str]], satisfied_position: int) -> RuleScores:
    """Score the "do nothing" baseline: predict nobody is satisfied.

    Args:
        rows: The data rows.
        satisfied_position: Index of the `satisfaction` column.

    Returns:
        The tally for the do-nothing rule.
    """
    scores = RuleScores("do nothing (always predict False)", 0, 0, 0, 0)
    for row in rows:
        actually_satisfied = row[satisfied_position] == "True"
        add_score(
            scores, predicted_satisfied=False, actually_satisfied=actually_satisfied
        )
    return scores


def print_table(title: str, scores_list: list[RuleScores]) -> None:
    """Print one table of results, best accuracy first.

    Args:
        title: Heading for the table.
        scores_list: The rules to print.
    """
    logger.info("")
    logger.info(title)
    logger.info(
        "%-42s %10s %10s %10s %12s",
        "rule",
        "accuracy",
        "precision",
        "recall",
        "calls satisfied",
    )
    for scores in sorted(scores_list, key=lambda item: item.accuracy, reverse=True):
        logger.info(
            "%-42s %10.4f %10.4f %10.4f %12d",
            scores.name,
            scores.accuracy,
            scores.precision,
            scores.recall,
            scores.true_positive + scores.false_positive,
        )


def self_check() -> None:
    """Prove the scoring arithmetic on a small case with a known answer.

    Four rows, two of them satisfied. The rule calls three rows satisfied and is
    wrong about one of those three. So it should score accuracy 0.75, precision
    2/3 (two right out of three calls) and recall 1.0 (it caught both).
    """
    scores = RuleScores("self check", 0, 0, 0, 0)
    add_score(scores, predicted_satisfied=True, actually_satisfied=True)
    add_score(scores, predicted_satisfied=True, actually_satisfied=False)
    add_score(scores, predicted_satisfied=True, actually_satisfied=True)
    add_score(scores, predicted_satisfied=False, actually_satisfied=False)

    assert scores.total == 4, f"expected 4 rows scored, got {scores.total}"
    assert scores.accuracy == 0.75, f"expected 0.75, got {scores.accuracy}"
    assert scores.precision == 2 / 3, f"expected 2/3, got {scores.precision}"
    assert scores.recall == 1.0, f"expected 1.0, got {scores.recall}"


def main() -> None:
    """Measure and print every baseline required by Step 0.4."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    self_check()

    header, rows = read_train_file(TRAIN_PATH)
    check_required_columns("baselines", header, REQUIRED_COLUMNS)
    logger.info("read %s: rows=%d cols=%d", TRAIN_PATH, len(rows), len(header))

    satisfied_position = header.index("satisfaction")
    satisfied_rows = sum(1 for row in rows if row[satisfied_position] == "True")
    logger.info(
        "base rate: satisfied=%d of %d = %.4f",
        satisfied_rows,
        len(rows),
        satisfied_rows / len(rows),
    )

    do_nothing = measure_base_rate(rows, satisfied_position)
    rating_rules = score_every_rating_column(header, rows, SERVICE_RATING_COLUMNS)

    chosen = next(
        scores for scores in rating_rules if scores.name == SIMPLE_RULE_COLUMN
    )

    print_table(
        "Baseline 1 — do nothing, and the registered simple rule", [do_nothing, chosen]
    )
    print_table(
        "Every 0-5 rating scored as 'rated 5 means satisfied' — "
        "printed in full so no single-feature result is hidden",
        rating_rules,
    )

    logger.info("")
    logger.info(
        "REGISTERED simple rule: %s == %d -> predict satisfied",
        SIMPLE_RULE_COLUMN,
        SIMPLE_RULE_SATISFIED_IF_RATING_IS,
    )
    logger.info(
        "Any model must beat accuracy %.4f to be worth building.",
        chosen.accuracy,
    )
    logger.info(
        "Accuracy is the wrong headline number here. With a %.2f%% base rate, "
        "always answering False already gets %.2f%%.",
        100 * satisfied_rows / len(rows),
        100 * do_nothing.accuracy,
    )


if __name__ == "__main__":
    main()
