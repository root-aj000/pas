"""
Values that never change.

Column names, the label, and the fixed category lists. Anything here is a fact
about the dataset, not a setting we tune - which is the test
.dev/RULES.md rule 8 uses to decide between this file and config.yaml.

If a value could plausibly change between experiments, it belongs in config.yaml
instead.
"""

# The column holding the answer. Given to us by the competition.
TARGET_COLUMN = "satisfaction"

# A row number in the organisers' shuffled list, never a passenger. Every
# training id is lower than every test id, so a model that saw it would learn
# their 70/30 cut. See docs/eda_findings.md section 9.
ID_COLUMN = "id"

# The passenger's own 0-5 ratings of the flight. Bounded by definition, so an
# outlier is impossible in these columns.
SERVICE_RATING_COLUMNS: list[str] = [
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

# Short categorical columns, one-hot encoded in stage 2.
CATEGORICAL_COLUMNS: list[str] = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]

# Numeric columns that are not service ratings.
CONTINUOUS_COLUMNS: list[str] = [
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]

# Every column a model is allowed to consider, before encoding.
CANDIDATE_FEATURE_COLUMNS: list[str] = (
    SERVICE_RATING_COLUMNS + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
)

# The lowest and highest value a service rating can hold. Twelve of the thirteen
# rating columns contain a 0; Baggage handling never does, and whether that 0
# means "worst possible" or "not applicable" is open question 3.
# docs/column_dictionary.md Note 3.
MIN_SERVICE_RATING = 0
MAX_SERVICE_RATING = 5

# The label values. Read as booleans by pandas, spelled out here so the mapping
# is written down rather than assumed.
SATISFIED_LABEL = "True"
NOT_SATISFIED_LABEL = "False"

# The three outcomes for a flight's arrival delay. `unknown` exists because 204
# rows in the train split have no recorded arrival delay and we do not know what
# a blank means. They are kept as their own group rather than filled in.
# docs/column_dictionary.md Note 1.
ARRIVAL_DELAY_ON_TIME = "on_time"
ARRIVAL_DELAY_DELAYED = "delayed"
ARRIVAL_DELAY_UNKNOWN = "unknown"
ARRIVAL_DELAY_STATUS_COLUMN = "arrival_delay_status"

# The rules that must beat this. Measured in research/02_eda.ipynb.
# See docs/eda_findings.md section 4.
BASELINE_RULE_COLUMN = "Class"
BASELINE_RULE_VALUE = "Business"
BASELINE_ACCURACY = 0.7770
BASELINE_ROC_AUC = 0.7786

# An untuned default HistGradientBoosting reaches this on the validation split.
# The real bar is this, not the one-column rule above.
UNTUNED_MODEL_ACCURACY = 0.924218
UNTUNED_MODEL_ROC_AUC = 0.957312
