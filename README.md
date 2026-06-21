# Multi-SIM Prediction — Reproducible MLOps Project

![CI Build](https://github.com/nurlan-abdullazada/My-MLOps-2/actions/workflows/ci-build.yaml/badge.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-green.svg)](https://shields.io/)

An end-to-end, reproducible machine-learning project that predicts whether a telecom customer is a **multi-SIM user** (a "multisimmer") from demographic and account data. Built on a clean **Cookiecutter Data Science** structure, dependency-managed with **uv**, and linted in CI with **Ruff**.

---

## Problem

Telecom operators benefit from identifying customers who use multiple SIM cards, since multi-SIM behavior is a strong signal for churn, cross-sell, and tariff-targeting decisions. This project frames the task as a **binary classification** problem and trains a tuned **XGBoost** model to estimate, for each customer, the probability of being a multisimmer.

The dataset (`multisim_dataset.parquet`) contains:

| Type        | Features |
|-------------|----------|
| Numerical   | `age`, `tenure`, `age_dev` |
| Categorical | `trf` (tariff plan), `gndr` (gender), `dev_man` (device manufacturer), `is_dualsim`, `region` |
| Target      | `target` (1 = multisimmer, 0 = single-SIM) |

---

## Approach

The full training flow is implemented as a single **scikit-learn `Pipeline`**, so preprocessing and the model are fitted and serialized together (no train/serve skew):

1. **Preprocessing** via `ColumnTransformer`
   - Numerical: median imputation → `StandardScaler`
   - Categorical: most-frequent imputation → `OneHotEncoder(handle_unknown="ignore")`
2. **Model**: `XGBClassifier` with hyperparameters tuned through an **Optuna** study (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, regularization terms, etc.).
3. **Evaluation**: stratified train/test split, reporting **Accuracy**, **ROC AUC**, and a full classification report.
4. **Serialization**: the fitted pipeline is saved to disk with `joblib` for reuse in prediction.

---

## Results

Exploratory analysis and final model diagnostics are stored in `reports/figures/`:

- **Distribution of the target variable** — class balance of multisimmers vs. single-SIM users.
- **Distribution of tariff plans** — categorical breakdown across the customer base.
- **Top 15 feature importances** from the final XGBoost model — the strongest predictors of multi-SIM behavior.

![Top 15 Feature Importances](reports/figures/Top%2015%20Feature%20Importances%20from%20Final%20XGBoost%20Model.png)

---

## Project Structure

```
.
├── data
│   ├── external          <- Data from third-party sources
│   ├── interim           <- Intermediate, transformed data
│   └── processed         <- Final, canonical datasets for modeling
├── models                <- Trained / serialized models and predictions
├── notebooks             <- Jupyter notebooks (EDA, experimentation)
├── reports
│   └── figures           <- Generated graphics and figures
├── src
│   ├── data
│   │   └── make_dataset.py       <- Data acquisition / generation
│   ├── features
│   │   └── build_features.py     <- Feature engineering
│   ├── models
│   │   ├── train_model.py        <- Train & evaluate the XGBoost pipeline
│   │   └── predict_model.py      <- Run predictions with the saved model
│   └── visualization
│       └── visualize.py          <- Exploratory & results visualizations
├── pyproject.toml        <- Project metadata & dependencies (uv-installable)
├── uv.lock               <- Locked environment for full reproducibility
└── .github/workflows
    └── ci-build.yaml     <- CI: Ruff lint via uvx
```

---

## Tech Stack

| Area                | Tools |
|---------------------|-------|
| Modeling            | XGBoost, scikit-learn |
| Hyperparameter tuning | Optuna |
| Data                | pandas, Parquet |
| Serialization       | joblib |
| Environment / deps  | uv (`uv.lock` committed) |
| Code quality        | Ruff, Black, isort |
| CI                  | GitHub Actions |
| Python              | 3.12+ |

---

## Getting Started

This project uses **[uv](https://docs.astral.sh/uv/)** for fast, reproducible environment management.

```bash
# Create the virtual environment and sync dependencies from the lockfile
uv sync

# Add a runtime dependency
uv add numpy

# Train the model
uv run python -m src.models.train_model

# Run predictions on new data
uv run python -m src.models.predict_model
```

### Code quality (via uvx — no dev deps added to the project)

```bash
uvx ruff check .          # Lint
uvx isort .               # Sort imports
uvx black .               # Format
uvx ruff check --fix .    # Apply safe fixes
```

---

## Reproducibility & CI

- **Locked environment** — `uv.lock` pins exact dependency versions so the environment can be recreated identically on any machine.
- **Continuous Integration** — every push and pull request to `main` runs a Ruff lint check through GitHub Actions, keeping the codebase consistent.
- **Pipeline-based modeling** — preprocessing and the estimator are bundled in one serialized artifact, ensuring identical transformations at train and inference time.

---

## License

Distributed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for details.
