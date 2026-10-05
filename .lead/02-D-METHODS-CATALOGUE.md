# 02-D — Method Catalogue

**What this file is:** the lookup table. *Task → library → exact function → when
to use it → what goes wrong.*

It exists so that nobody has to guess a function name, and so everyone picks the
same approach for the same problem.

**How to use it:**

1. Find your task in the contents below.
2. Read the row for that task.
3. Start with the **recommended first choice** — the boring, well-supported one.
4. Check the "watch out for" column before you commit.
5. Not in the table? Use the search procedure in
   [`02-C-CHOOSING-THE-METHOD.md`](02-C-CHOOSING-THE-METHOD.md) Step 6.

> **Names change between versions.** Everything here is scikit-learn 1.5-era
> naming. If a name does not exist in your installed version, search the docs for
> it — do not guess. `pip show scikit-learn` tells you the version.

## Contents

1. [Classification — binary and multiclass](#1-classification--binary-and-multiclass)
2. [Regression](#2-regression)
3. [Imbalanced data](#3-imbalanced-data)
4. [Time series forecasting](#4-time-series-forecasting)
5. [Anomaly detection](#5-anomaly-detection)
6. [Clustering](#6-clustering)
7. [Dimensionality reduction](#7-dimensionality-reduction)
8. [Feature selection](#8-feature-selection)
9. [Preprocessing and encoding](#9-preprocessing-and-encoding)
10. [Pipelines](#10-pipelines)
11. [Text](#11-text)
12. [Images](#12-images)
13. [Hyperparameter search](#13-hyperparameter-search)
14. [Interpretability](#14-interpretability)
15. [Calibration and threshold choice](#15-calibration-and-threshold-choice)
16. [Metrics](#16-metrics)
17. [Cross-validation](#17-cross-validation)
18. [Saving and loading](#18-saving-and-loading)
19. [Where to look things up](#19-where-to-look-things-up)

---

## 1. Classification — binary and multiclass

**First choice for most tabular problems: a gradient boosted tree.** Start with
`HistGradientBoostingClassifier` from scikit-learn — fast, handles missing
values, no tuning needed to work. Move to XGBoost or LightGBM when the data is
large or you need more control.

| Method | Library + exact call | Use it when | Watch out for |
|---|---|---|---|
| **Gradient boosting** | `sklearn.ensemble.HistGradientBoostingClassifier` | **Default first choice on tabular data.** Handles missing values natively | Slower to tune; `max_iter` is the main knob |
| **Random forest** | `sklearn.ensemble.RandomForestClassifier` | Many mixed features, you want few settings, parallelisable | Large memory use; cannot extrapolate |
| **Logistic regression** | `sklearn.linear_model.LogisticRegression` | You need a baseline, or a model you can explain in one sentence | Needs scaling; only linear relationships |
| **Single decision tree** | `sklearn.tree.DecisionTreeClassifier` | You want to print it and show a non-technical person exactly why | Overfits almost always alone |
| **Extra trees** | `sklearn.ensemble.ExtraTreesClassifier` | Randomised splits, slightly more varied than a forest | Same memory cost as a forest |
| **Gradient boosting (classic)** | `sklearn.ensemble.GradientBoostingClassifier` | You need `loss="quantile"` or similar options | Much slower than `HistGradientBoosting` |
| **k-NN** | `sklearn.neighbors.KNeighborsClassifier` | Very small data, or many mixed-type features | Needs scaling; slow at prediction time; cannot extrapolate |
| **SVM** | `sklearn.svm.SVC` | Small-to-medium clean data, high-dimensional features (text) | Needs scaling; slow on large data |
| **Linear SVM** | `sklearn.svm.LinearSVC` | Large data where `SVC` is too slow | Same scaling requirement |
| **SGD** | `sklearn.linear_model.SGDClassifier` | Very large data, online learning | Sensitive to settings; results vary |
| **Naive Bayes** | `sklearn.naive_bayes.GaussianNB`, `MultinomialNB`, `BernoulliNB` | Text data, or a fast sanity baseline | Assumes features are independent |
| **Neural net (MLP)** | `sklearn.neural_network.MLPClassifier` | You are deliberately choosing a neural network | Needs scaling; easy to overfit; not reproducible without care |
| **XGBoost** | `xgboost.XGBClassifier` | Tabular, competitive leaderboards, you want strong regularisation controls | Extra dependency; verbose defaults |
| **LightGBM** | `lightgbm.LGBMClassifier` | Large tabular data, trains fastest | Extra dependency; verbose defaults |
| **CatBoost** | `catboost.CatBoostClassifier` | **Many categorical features.** Handles them natively, saves you encoding work | Extra dependency; slower than LightGBM |

**Multiclass:** all of the above take `target` as strings — no encoding needed.
Same functions, same rules. Sklearn figures it out automatically.

**A note on defaults.** Every one of these works with `random_state=42` and no
other arguments. Run that first. Then change one setting at a time.

---

## 2. Regression

Predicting a number. The same family of methods applies.

| Method | Library + exact call | Use it when | Watch out for |
|---|---|---|---|
| **Gradient boosting** | `sklearn.ensemble.HistGradientBoostingRegressor` | **Default first choice on tabular data** | Same as classifier |
| **Random forest** | `sklearn.ensemble.RandomForestRegressor` | Mixed features, few settings needed | Same as classifier |
| **Linear regression** | `sklearn.linear_model.LinearRegression` | You want to see a coefficient per feature | Cannot capture curves; affected by outliers |
| **Ridge** | `sklearn.linear_model.Ridge` | Many correlated features; stabilises the fit | One hyperparameter, `alpha` |
| **Lasso** | `sklearn.linear_model.Lasso` | You want it to **drop** unhelpful features automatically | One hyperparameter; drops features at random when correlated |
| **ElasticNet** | `sklearn.linear_model.ElasticNet` | You want Lasso's sparsity without the instability | Two hyperparameters |
| **XGBoost** | `xgboost.XGBRegressor` | Large tabular data | Extra dependency |
| **LightGBM** | `lightgbm.LGBMRegressor` | Large tabular data, fast | Extra dependency |
| **CatBoost** | `catboost.CatBoostRegressor` | Many categorical features | Extra dependency |

**Prediction intervals, not just point predictions** — often what a business
actually needs:

```python
# Predict a range: the 10th and 90th percentile of the outcome.
model = HistGradientBoostingRegressor(loss="quantile", quantile=0.1)
```

Fit it twice, at `quantile=0.1` and `quantile=0.9`, and you have an honest range.
Ask whether the decision needs a number or a range. Often it needs a range.

---

## 3. Imbalanced data

For problems where the outcome is rare (churn, fraud, failure) — the common case
in real projects.

| Technique | Library + exact call | Use it when | Watch out for |
|---|---|---|---|
| **Class weight** | `Classifier(class_weight="balanced")` | **Always try this first.** One argument, no new data | Shifts the decision boundary; may hurt precision badly |
| **Balanced class weight** | `xgboost.XGBClassifier(scale_pos_weight=N)` | XGBoost, where it is the standard lever | Tune it, do not guess it |
| **Random undersampling** | `imblearn.RandomUnderSampler` | You have plenty of data | Throws away real examples |
| **Random oversampling** | `imblearn.RandomOverSampler` | Data is scarce | Duplicates; risks memorising duplicates |
| **SMOTE** | `imblearn.SMOTE` | Classic tabular imbalance | Invented points can be nonsense; only for numeric features |
| **SMOTE with categories** | `imblearn.SMOTENC` | You have categorical columns | More settings to get right |
| **Borderline SMOTE** | `imblearn.BorderlineSMOTE` | Plain SMOTE is not helping | Same caveats |
| **Ensemble resampling** | `imblearn.ensemble.BalancedRandomForestClassifier` | You want imbalance handled internally | Larger and slower |

**The order to try them, and this matters:**

1. **`class_weight="balanced"`.** One argument. Try it first, always.
2. **Then just move the threshold.** Cheapest and most effective of all — see
   [`02-C-CHOOSING-THE-METHOD.md`](02-C-CHOOSING-THE-METHOD.md) Step 10 in
   [`04-EVALUATION-AND-GO-NO-GO.md`](04-EVALUATION-AND-GO-NO-GO.md) Step 4.4.
3. **Then resampling**, inside a pipeline so it only touches training folds.
4. **Never resample before splitting.** Resampling before the split leaks
   synthetic neighbours of test rows into training.

```python
# Correct: the pipeline guarantees SMOTE only sees training data in each fold.
from imblearn.pipeline import Pipeline as ImbPipeline

pipeline = ImbPipeline(
    [
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(random_state=42)),
    ]
)
```

---

## 4. Time series forecasting

Two families: statistical, and machine learning. ML wins when you have many
related series or many extra features.

| Method | Library + exact call | Use it when | Watch out for |
|---|---|---|---|
| **Seasonal naive** | Manual: use the value from one season ago | **Always the baseline.** No library needed | Very simple — and surprisingly hard to beat |
| **Exponential smoothing** | `statsmodels.tsa.holtwinters.ExponentialSmoothing` | One series, clear trend and seasonality | One series at a time |
| **Simple smoothing** | `statsmodels.tsa.holtwinters.SimpleExpSmoothing` | One series, no seasonality | Same |
| **ARIMA / SARIMA** | `statsmodels.tsa.statespace.sarimax.SARIMAX` | One series, autocorrelation, you understand the diagnostics | Fitting is fiddly; needs stationarity |
| **AutoReg** | `statsmodels.tsa.ar_model.AutoReg` | Simple autocorrelation, fast | Limited flexibility |
| **ML on lag features** | Lag features + `HistGradientBoostingRegressor` | **Many related series, or extra features** (price, weather, holidays) | You must build the lag features yourself |
| **Decomposition check** | `statsmodels.tsa.seasonal.STL`, `seasonal_decompose` | You want to see trend vs season vs noise first | Diagnostic, not a model |

**Forecasting with machine learning, properly:**

```python
"""Forecast demand with lag features and gradient boosting.

Why: demand depends on the same day last week and last month, plus known future
factors like holidays and price. A model that can see those lags handles it
better than a univariate statistical model.
"""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FORECAST_HORIZON = 7  # days


def add_lag_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add 'value as it was N days ago' columns.

    Only rows with a full history survive, because early rows have no lags.
    """
    enriched = data.copy()
    for lag_days in [1, 7, 14, 28]:
        enriched[f"value_lag_{lag_days}"] = enriched["value"].shift(lag_days)
    return enriched.dropna()


def add_calendar_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-week and month, which are known for future dates."""
    enriched = data.copy()
    enriched["day_of_week"] = enriched["date"].dt.dayofweek
    enriched["month"] = enriched["date"].dt.month
    return enriched
```

**The three rules for time series, which are the whole difference from tabular:**

1. **Split by time.** Never randomly. See
   [`01-DATA.md`](01-DATA.md) Step 1.7.
2. **Never use a feature that is only known after the moment you predict** —
   including "actual sales today" when predicting today's sales.
3. **Evaluate per horizon.** A 1-day forecast and a 30-day forecast are different
   problems. Score them separately.

---

## 5. Anomaly detection

| Method | Library + exact call | Use it when | Watch out for |
|---|---|---|---|
| **Isolation Forest** | `sklearn.ensemble.IsolationForest` | **Start here.** Fast, works on mixed data, no assumptions about the shape | `contamination` must be set — it sets the false-alarm rate |
| **Local Outlier Factor** | `sklearn.neighbors.LocalOutlierFactor` | Anomalies are rare *and* in dense groups | Cannot predict on new data — `novelty=True` required |
| **Elliptic Envelope** | `sklearn.covariance.EllipticEnvelope` | Data is roughly elliptical and clean | Assumes normality; fails on real messy data |
| **One-class SVM** | `sklearn.svm.OneClassSVM` | Small, high-dimensional data | Needs scaling; slow |
| **Rolling z-score** | Manual: rolling mean and standard deviation | A single time series; you want something explainable | Only catches simple spikes |
| **Seasonal residual** | STL decomposition, then threshold the residual | Seasonal data; you want to explain the anomaly | Needs enough history |
| **Model residuals** | Train a normal model, alert on large errors | You already have a model | Only as good as the model |

**Always set `contamination`.** Unsupervised means "tell me what is odd", and
nobody has ever been happy with the answer to "how odd is odd". The number is a
business decision, like the threshold in
[`04-EVALUATION-AND-GO-NO-GO.md`](04-EVALUATION-AND-GO-NO-GO.md).

**And always look at what it flagged.** The first 50 alerts will be mostly
false positives — a known data entry, a known holiday. Anomaly detection that
nobody has reviewed is not finished.

---

## 6. Clustering

Grouping records with no labels. Useful for exploration and segmentation.

| Method | Library + exact call | Use it when | Watch out for |
|---|---|---|---|
| **K-Means** | `sklearn.cluster.KMeans` | Clear, round-ish groups; you know roughly how many | Needs scaling; needs `n_clusters`; assumes round clusters |
| **Mini-batch K-Means** | `sklearn.cluster.MiniBatchKMeans` | Very large data | Less precise |
| **DBSCAN** | `sklearn.cluster.DBSCAN` | **Groups of any shape, and you want "unknown" as an answer** | Sensitive to `eps`; struggles with varying density |
| **Hierarchical** | `sklearn.cluster.AgglomerativeClustering` | You want to see the tree of merges | Slow on big data |
| **Gaussian Mixture** | `sklearn.mixture.GaussianMixture` | Soft cluster membership; overlapping groups | More settings, more ways to get it wrong |

**Measuring a clustering** — and this is the part people skip:

| Measure | Function | Note |
|---|---|---|
| How well separated | `sklearn.metrics.silhouette_score` | 0 to 1. Higher is better. Also `silhouette_samples` per row |
| Better than random | `sklearn.metrics.adjusted_rand_score` | Compare against random labels as a floor |
| Same information, two views | `homogeneity_score`, `completeness_score`, `v_measure_score` | Compare two clusterings |
| Shape based | `calinski_harabasz_score` | Higher is better |

**The real test is not a score.** It is: can you describe each cluster in one
plain sentence, and does that description match something the business already
recognises? If not, the clustering is decorative.

---

## 7. Dimensionality reduction

| Method | Library + exact call | Use it when | Watch out for |
|---|---|---|---|
| **PCA** | `sklearn.decomposition.PCA` | Compressing many correlated numeric features | Linear only; needs scaling; components are unlabelled |
| **Truncated PCA** | `sklearn.decomposition.TruncatedSCA` | Same, faster on wide data | Same |
| **t-SNE** | `sklearn.manifold.TSNE` | **Visualising** high-dimensional data in 2D | **Never use as pipeline input.** Distances are not preserved |
| **UMAP** | `umap.UMAP` | Visualising large high-dimensional data | Same warning as t-SNE |
| **Non-linear** | `KernelPCA`, `Isomap`, `LocallyLinearEmbedding`, `MDS` | A specific non-linear structure is suspected | Slow; hard to maintain |

**The rule that saves people a week:** t-SNE and UMAP are **drawing tools**. They
are for looking at data, and their axes mean nothing. You cannot say "this
customer is 0.3 away from that one" — that is not true after t-SNE. If you need
distances to mean something, use PCA.

---

## 8. Feature selection

Fewer features means less noise, faster training, and a model someone can read.

| Method | Library + exact call | Use it when | Watch out for |
|---|---|---|---|
| **Drop low variance** | `sklearn.feature_selection.VarianceThreshold` | Columns that never change | A constant column has no signal by definition |
| **Statistical test** | `SelectKBest(score_func=f_classif)` | Quick, works on numeric data | Misses interactions; biased by high-cardinality features |
| **Mutual information** | `mutual_info_classif`, `mutual_info_regression` | Non-linear relationships | Slower; needs more samples |
| **Chi-squared** | `SelectKBest(score_func=chi2)` | Non-negative numeric data | Same caveats |
| **Lasso's own selection** | `sklearn.linear_model.Lasso` | You want to also see the surviving features' effects | Drops correlated features at random |
| **Model importance** | `SelectFromModel` | You already have a model that ranks features | Circular if the model is bad |
| **Exhaustive search** | `SequentialFeatureSelector` | Small feature counts only | Very slow above ~20 features |
| **Recursive elimination** | `sklearn.feature_selection.RFE` | Small-to-medium feature counts | Slow |
| **Permutation importance** | `sklearn.inspection.permutation_importance` | **Best general answer.** Tells you what the model actually relies on | Slower; needs a trained model |

**The order that works:** start with the features that survived the target
comparison in [`02-B-FINDING-SIGNALS.md`](02-B-FINDING-SIGNALS.md), then check
permutation importance, then try removing the useless ones and see whether
validation improves. Stop when it stops improving.

---

## 9. Preprocessing and encoding

| Job | Library + exact call | Notes |
|---|---|---|
| Missing numbers | `sklearn.impute.SimpleImputer(strategy="median")` | Median, not mean — robust to outliers |
| Missing categories | `SimpleImputer(strategy="most_frequent")` | Or add a "was missing" flag column, which often predicts more |
| Advanced missing | `KNNImputer`, `IterativeImputer` | Only if the median leaves real signal behind |
| Scale numbers | `StandardScaler` | To mean 0, std 1. Needed for k-NN, SVM, neural nets, linear models with different units |
| Scale to range | `MinMaxScaler`, `MaxAbsScaler` | When you genuinely need 0–1 |
| Scale robustly | `RobustScaler` | Heavy-tailed data with outliers |
| Make normal-ish | `PowerTransformer`, `QuantileTransformer` | Skewed features before a linear model |
| One-hot categories | `OneHotEncoder(handle_unknown="ignore")` | **Always** set `handle_unknown="ignore"` or unseen categories crash the model in production |
| Ordered categories | `OrdinalEncoder` | Only when the order is real (small/medium/large) |
| High-cardinality categories | `TargetEncoder` (sklearn ≥ 1.3) | Customer IDs, product IDs. Use inside a pipeline to avoid leakage |
| Bin a continuous feature | `KBinsDiscretizer` | When the relationship is a step, not a line |
| Crossed categories | `sklearn.preprocessing.PolynomialFeatures` | Interactions; explodes the count quickly |
| Dates | `FunctionTransformer` | Write your own small function — clearer than `dt` chains |

**A warning on `TargetEncoder`:** it uses the target to encode, so it must be
inside a `Pipeline` with cross-validation. Fitted on the whole dataset it leaks
the answer into the features.

---

## 10. Pipelines

**Use a `Pipeline` for anything with more than one preprocessing step.** It is the
single biggest correctness win in this catalogue.

```python
"""Preprocessing and model in one object, so it cannot be fitted wrongly.

Why this matters: a Pipeline remembers what it learned while fitting. That is
what stops a scaler or an encoder being fitted on the test data by accident.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_COLUMNS = ["tenure_days", "avg_spend_per_visit", "support_tickets"]
CATEGORY_COLUMNS = ["plan", "region"]


def build_preprocessor() -> ColumnTransformer:
    """Return the cleaning and encoding steps, applied to the right columns."""
    numeric_steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    category_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore" means a category unseen in training does not
        # crash the model in production. Always set it.
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ]
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC_COLUMNS),
            ("categories", Pipeline(category_steps), CATEGORY_COLUMNS),
        ]
    )


def build_full_pipeline(model_name: str = "hist_gradient_boosting") -> Pipeline:
    """Return preprocessing plus the named model, as one fitted-together object."""
    from src.components.model_training import build_model

    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", build_model(model_name, {}, seed=42)),
        ]
    )
```

| Function | When |
|---|---|
| `Pipeline` | Chained transforms ending in an estimator |
| `make_pipeline` | Same, names steps automatically. Readability suffers — name steps yourself when you will debug it |
| `ColumnTransformer` | Apply different steps to different columns |
| `FeatureUnion` | Run two transform paths and combine — used rarely |
| `TransformedTargetRegressor` | Transform the target (e.g. `log1p`) and invert it automatically |

**What a pipeline buys you:** preprocessing cannot be fitted on the wrong data; the
whole thing saves and loads as one file; and cross-validation works correctly
without any special care. This is why the production architecture puts
preprocessing in the training component rather than in a notebook.

---

## 11. Text

| Job | Library + exact call | Notes |
|---|---|---|
| Turn text into numbers | `sklearn.feature_extraction.text.TfidfVectorizer` | **Start here.** Word and character variants both available |
| Count words | `CountVectorizer` | Sometimes better than TF-IDF for short texts |
| Combine text and numbers | `scipy.sparse.hstack`, or `ColumnTransformer` | Sparse and dense arrays must be combined, not concatenated |
| Classify text | `TfidfVectorizer` + `LogisticRegression` or `LinearSVC` | The standard strong baseline for text classification |
| Meaningful embeddings | `sentence_transformers.SentenceTransformer` | Use when TF-IDF is not good enough. Slower, larger |
| Pretrained language model | `transformers.AutoTokenizer` + `AutoModel` | Only when you need real language understanding |
| Sentiment | `transformers` pipeline | Off-the-shelf, no training |

**TF-IDF plus a linear model is remarkably hard to beat** for ordinary
classification tasks, and it trains in seconds and explains itself ("these words
drove the decision"). Do not reach for a language model before you have tried it.

---

## 12. Images

| Job | Library + exact call | Notes |
|---|---|---|
| Load and transform images | `torchvision.transforms` | Resize, normalise, augment |
| Pretrained CNN | `torchvision.models.resnet50(weights="IMAGENET50_Weights.IMAGENET1K_V2")` | **Always start from pretrained weights** |
| Modern pretrained models | `timm.create_model("efficientnet_b0", pretrained=True)` | Often better per unit of compute |
| Fine-tune | Replace the last layer, then train a few epochs at low LR | **Freeze everything else** at first |
| Classification head | `torch.nn.Linear(in_features, n_classes)` | The only layer you change initially |
| Augmentation | `RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter` | What makes small image datasets work |
| Transfer learning | Freeze early layers, train later ones | The standard approach when you have < 10k images |

**Order of work:** pretrained model + logistic regression on the features →
fine-tune the last layer → fine-tune more layers. Stop when validation stops
improving. Training a CNN from scratch needs tens of thousands of images and a
GPU.

---

## 13. Hyperparameter search

| Tool | Call | Notes |
|---|---|---|
| Exhaustive grid | `GridSearchCV` | Small grids only — cost grows multiplicatively |
| Random search | `RandomizedSearchCV` | **Better default.** Same budget explores more combinations |
| Successive halving | `HalvingRandomSearchCV`, `HalvingGridSearchCV` | Cheap early trials, full trials only for the good ones. Best value |
| Bayesian optimisation | `optuna.create_study(direction="maximize")` | Best for larger spaces. Use in [`03-TRAIN-AND-TUNE.md`](03-TRAIN-AND-TUNE.md) |
| Cross-validate in one call | `sklearn.model_selection.cross_validate` | Fits and scores k folds, returns a dict per metric |

Always pass `random_state` and `n_jobs=-1` where supported. Always give a search
a **time budget**, not a fixed number of tries.

---

## 14. Interpretability

Required before almost any real deployment.

| Question | Library + exact call | Notes |
|---|---|---|
| Which features matter? | `sklearn.inspection.permutation_importance` | **Most trustworthy.** Shuffles one feature and measures the damage |
| How does one feature affect predictions? | `sklearn.inspection.partial_dependence` | A chart, not a number. Good for stakeholders |
| Explain one prediction | `shap.Explainer(model).shap_values(x)` | Per-prediction reasons. The best tool for "why did it say that" |
| Tree explanation | `shap.TreeExplainer(model)` | Fastest for tree models |
| Linear explanation | Look at `.coef_` | Direct, and the whole reason to use a linear model |
| What is the model learning? | Partial dependence, or a single decision tree | **A small decision tree printed in full is the best explainer ever written** |

---

## 15. Calibration and threshold choice

A probability of 0.8 should mean "80% of the time this happens". Models are often
wrong about this, and it matters when someone acts on the probability.

| Job | Library + exact call | Notes |
|---|---|---|
| Check calibration | `sklearn.calibration.CalibrationDisplay.from_predictions` | A chart. Reliability diagram |
| Fix calibration | `CalibratedClassifierCV(estimator=model, method="isotonic")` | Use when probabilities are used in a decision |
| Check how well calibrated | `sklearn.metrics.brier_score_loss` | Lower is better. 0 to 1 |
| Find the trade-off | `sklearn.metrics.precision_recall_curve` | **Gives you every possible threshold with its result** |
| Find best F1 | `precision_recall_curve` → pick the point that maximises F1 | A starting point only |

**The threshold is a business decision, not a default.** See
[`04-EVALUATION-AND-GO-NO-GO.md`](04-EVALUATION-AND-GO-NO-GO.md) Step 4.4 for how
to make it with a person and write the reason next to the number.

```python
"""Choose the threshold from the trade-off curve, then check the result.

Why not just use 0.5? Because on an imbalanced problem the useful threshold is
usually far from 0.5. The curve shows every possible threshold and what it costs.
"""

import numpy as np
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_true, probabilities)

# How many calls does the team have to make each week? Put the limit here.
MAX_CALLS_PER_WEEK = 500

chosen = 0.5
for index in range(len(thresholds) - 1, -1, -1):
    if np.sum(probabilities >= thresholds[index]) <= MAX_CALLS_PER_WEEK:
        chosen = float(thresholds[index])
        break

print(f"threshold={chosen:.3f} recall={recall[index]:.3f}")
```

---

## 16. Metrics

**Classification:**

| Function | What it gives |
|---|---|
| `sklearn.metrics.classification_report` | Precision, recall, F1 per class. **Start here** |
| `confusion_matrix`, `ConfusionMatrixDisplay` | The raw counts. Read it in plain language |
| `precision_score`, `recall_score`, `f1_score` | One at a time. Use `pos_label=` for binary |
| `roc_auc_score`, `roc_curve` | Ranking quality. Can be misleading on rare positives |
| `average_precision_score` | **Better than AUC when positives are rare** |
| `precision_recall_curve` | Every threshold with its precision and recall |
| `log_loss` | How wrong the probabilities are. Lower is better |
| `brier_score_loss` | Probability quality. Lower is better |
| `matthews_corrcoef` | A single balanced number. 0 to 1 |
| `balanced_accuracy_score` | Accuracy that ignores class sizes |

**Regression:**

| Function | What it gives |
|---|---|
| `mean_absolute_error` | Average error in the units you care about. **Start here** |
| `root_mean_squared_error` | Penalises big errors more. `mean_squared_error ** 0.5` on older sklearn |
| `mean_squared_error` | Penalises big errors, less readable units |
| `mean_absolute_percentage_error` | Error as a percentage |
| `median_absolute_error` | Robust to outliers |
| `r2_score` | Fraction of variance explained. **Can be negative — do not trust it alone** |
| `max_error` | The worst single error. Always worth looking at |

**Clustering:** `silhouette_score`, `adjusted_rand_score`, `homogeneity_score`,
`completeness_score`, `calinski_harabasz_score` — see section 6.

**The only rule that matters:** report the number the business decision uses, and
report the confusion matrix next to it. If they disagree, they are both true and
you must explain both.

---

## 17. Cross-validation

| Function | When |
|---|---|
| `KFold(n_splits=5)` | Regression, or balanced classification |
| `StratifiedKFold(n_splits=5)` | **Classification.** Keeps the class ratio in every fold |
| `TimeSeriesSplit(n_splits=5)` | **Time-ordered data.** Never use random folds here |
| `GroupKFold` | When the same customer/session appears in several rows |
| `StratifiedGroupKFold` | Grouped data that is also imbalanced |
| `cross_validate(estimator, X, y, cv=5, scoring={"recall": "recall", ...})` | Fits and scores in one call. **Returns a dict of arrays** — always ask for `return_train_score=True` to see overfitting |
| `cross_val_score` | One metric only. Simpler |
| `GridSearchCV(cv=...)` | Search and cross-validation in one step |

**When to cross-validate, and when not to:**

| Situation | Do |
|---|---|
| Small or medium data (< 100k rows) | Cross-validate. 5 folds is fine |
| Large data | A single fixed holdout is enough, and faster |
| Time series | `TimeSeriesSplit`, never random folds |
| Grouped data | `GroupKFold`, so groups do not leak across folds |

**Always pass `return_train_score=True`** when comparing methods. A huge gap
between train and validation score is the clearest signal of overfitting.

---

## 18. Saving and loading

| Job | Call | Notes |
|---|---|---|
| Save / load a model | `joblib.dump(model, "model.pkl")`, `joblib.load("model.pkl")` | **Use `joblib`, not `pickle`.** Handles large numpy arrays properly |
| Save the feature list | `json.dump(features, open("features.json", "w"))` | Save separately, and check it at prediction time |
| Save config with the model | `joblib.dump({"model": model, "features": features, "config": config})` | Then a mismatch is impossible |
| Portable format | `skl2onnx` | Only if another language must read the model |
| Never | `pickle.load` on a file you did not create | **Arbitrary code execution.** Only load files you trust |

Save the feature list **in order**. A model given its columns in the wrong order
still returns numbers — wrong numbers, silently. See
[`.dev/DEBUGGING.md`](../.dev/DEBUGGING.md) section 4.

---

## 19. Where to look things up

| What | Where |
|---|---|
| Which algorithm for my task | scikit-learn [algorithm cheat sheet](https://scikit-learn.org/stable/algorithm_selection.html) |
| Anything scikit-learn | The library's own docs — a task-based "User Guide" plus an API reference |
| Class reference | `help(sklearn.ensemble.HistGradientBoostingClassifier)` in Python |
| What is installed | `pip list`, `pip show scikit-learn` |
| Realistic full examples | scikit-learn's **examples** section, which are usually more useful than the API docs |
| Anything else | The library's documentation site. Every serious library has one |

**A habit worth forming:** when you use a function for the first time, open its
docstring in the Python REPL. Thirty seconds, and you learn the parameters and
see an example:

```python
import inspect
from sklearn.ensemble import HistGradientBoostingClassifier

print(inspect.signature(HistGradientBoostingClassifier.__init__))
```

---

## The short version

If you remember nothing else from this file:

| Situation | Use |
|---|---|
| Tabular, classification | `HistGradientBoostingClassifier` |
| Tabular, regression | `HistGradientBoostingRegressor` |
| Need to explain every decision | `LogisticRegression`, or one small `DecisionTreeClassifier` |
| Rare outcome | `class_weight="balanced"`, then move the threshold |
| Many categorical columns | `catboost.CatBoostClassifier` |
| Text | `TfidfVectorizer` + `LogisticRegression` |
| Images | Pretrained `resnet50`, fine-tune the last layer |
| Unlabelled groups | `KMeans`, or `DBSCAN` if shapes vary |
| Something unusual in the data | `IsolationForest` |
| Predicting a number over time | Lag features + `HistGradientBoostingRegressor` |

Start simple, measure honestly, and only add complexity when a number demands
it.