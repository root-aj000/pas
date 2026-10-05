# Kaggle GPU Runbook

**For:** running the heavy experiments on a Kaggle GPU (P100 or 2× T4).
**Read this on Kaggle, in order, top to bottom.**

`config.yaml` sets `device: "cpu"`. Change it to `"cuda"` for every command below.
Nothing else changes.

---

## 1. Attach two datasets to a new GPU notebook

| Dataset | What it holds | Why two |
|---|---|---|
| **This repo** (uploaded as a Kaggle Dataset) | Everything except `data/` and `.venv`, which are gitignored | Code, config, docs |
| **The competition files** (`train.csv`, `test.csv`, `sample_submission.csv`) | The data | Already on Kaggle — search Datasets for the competition name |

Do NOT use the Kaggle API or any access token. Upload through the website and
attach through the notebook's **Add data** button. Standing project rule — see
`docs/open_questions.md`.

Do NOT edit `config.yaml` paths. They are relative (`data/train.csv`) and resolve
against the project root on any machine. When `data/` is absent, the pipeline scans
`/kaggle/input/` for the same filenames automatically and logs which file it found.
An absolute Kaggle path in `config.yaml` is honoured untouched — and breaks every
machine that is not Kaggle. That happened once already; the comment in `config.yaml`
says so.

## 2. Set up (first cell)

```python
# Mounted paths. Replace <repo> and <data> with the names you see under /kaggle/input/.
!cp -r /kaggle/input/<repo>/* /kaggle/working/PAS/
!mkdir -p /kaggle/working/PAS/data
!cp /kaggle/input/<data>/*.csv /kaggle/working/PAS/data/
!pip install -q -r /kaggle/working/PAS/requirements-kaggle.txt
```

```python
import os
os.chdir("/kaggle/working/PAS")
!python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The last line must print `True`. If it prints `False`, stop — the notebook is not
on a GPU runtime. Settings → Accelerator → GPU.

Do NOT run `pip install -r requirements.txt`. It replaces Kaggle's GPU-enabled
xgboost with a CPU build.

## 3. Verify the pipeline reproduces the CPU numbers (second cell)

```python
!python run_pipeline.py
```

Expected, from `config.yaml` as committed:

| Metric | Validation | Test |
|---|---|---|
| ROC-AUC | 0.958844 | **0.957894** |
| Accuracy | 0.925924 | 0.915585 |

**If test ROC-AUC differs by more than 0.0005, stop.** A GPU/CPU divergence means
something is wrong — usually a library version, not the model. Compare
`!pip show xgboost pandas scikit-learn` against `requirements.txt` and record the
difference in the dev log before continuing.

## 4. Run the experiments, cheapest first

Each command appends to `reports/` and prints one number that matters.

```python
# RealMLP on our 22 features. Their recipe: 8 members, 3 epochs.
!python research/realmlp_compare.py --quick     # smoke test, ~1 min
!python research/realmlp_compare.py             # full, ~10-20 min on GPU
```

```python
# Optuna, all nine parameters. Set DEVICE = "cuda" at the top of the file first.
!python research/optuna_search.py               # 15-min budget, set in the file
```

```python
# Three boosters, out-of-fold blending. ~10 min on GPU.
!python research/boosting_comparison.py --quick  # smoke test
!python research/boosting_comparison.py          # full
```

## 5. What to bring back

Not the models — `models/*.pkl` is gitignored and too large to move. Bring back:

| File | Why |
|---|---|
| `reports/*.csv` | Every number, in tables |
| The console output | The run log |
| `config.yaml`, if you changed it | Only if a setting beat the committed one on validation |

Write the numbers into `docs/experiment_log.md` as new rows. Old rows are never
edited.

## 6. GPU-specific notes

| | |
|---|---|
| **T4 ×2 vs P100** | Take T4 ×2 if offered. Tensor cores for FP16, 32GB total. See `docs/method_plan.md` §4 |
| **XGBoost on GPU** | Needs `tree_method: "hist"` (already set) plus `device: "cuda"`. No other change |
| **Determinism** | GPU training is not bit-reproducible. Same seed, same code, slightly different 5th decimal between runs. Compare at 0.0005 tolerance, not exactly |
| **Session limit** | ~9–12 hours. `research/optuna_search.py` has its own 15-minute budget; raise `SEARCH_SECONDS` if the session allows |
| **Memory** | ~30GB system RAM. Do not load `train.csv` twice in one process, and do not run 5-fold CV with `n_jobs=-1` on both levels |

## What has changed

- **2026-10-05:** first version. Covers pipeline verification, RealMLP, Optuna
  and the three-booster comparison. Written because GPU work moved to Kaggle.
