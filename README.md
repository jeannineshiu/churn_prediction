# Churn Prediction — End-to-End ML System

A production-ready machine learning pipeline for predicting customer churn using the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). The system is structured as a modular Python project following ML engineering best practices: config-driven, pipeline-based, and fully reproducible.

---

## Problem Statement

Customer churn is a critical business metric in the telecom industry. This project builds a binary classifier to predict whether a customer will churn (`Yes`/`No`) based on their account details, services subscribed, and billing information.

**Target variable:** `Churn` (1 = churned, 0 = retained)

---

## Business Value

### What business questions does this solve?

| Business Question | How This Model Answers It |
|---|---|
| Which customers are likely to leave next month? | Outputs a churn probability score (0–100%) for every customer |
| Who should the retention team call first? | Rank customers by risk score → prioritise top-N for outreach |
| What is driving customers to leave? | Feature importance reveals top churn signals (e.g. contract type, tenure, monthly charges) |
| Is our retention campaign reaching the right people? | Compare predicted vs actual churn on a held-out set to validate targeting |

---

### How much can this save?

The telecom industry rule of thumb: **acquiring a new customer costs 5–25× more than retaining an existing one.**

Using the IBM Telco dataset as a reference point (7,043 customers, ~27% churn rate):

| Scenario | Without Model | With Model | Savings |
|---|---|---|---|
| **Retention campaign cost** | Blast all 7,043 customers @ ~$30/contact = **$211K** | Target top 700 high-risk customers @ ~$30/contact = **$21K** | ~**$190K saved** per campaign cycle |
| **Revenue at risk** | ~1,900 churners × $65/mo avg spend = **$1.5M/yr** | Prevent 15–20% churn with proactive retention = **$225K–$300K recovered** | Up to **$300K/yr in retained revenue** |
| **Analyst time** | Manually reviewing accounts: ~2–3 hrs/analyst/day | Automated daily risk scoring pipeline: **< 5 min** | Frees ~**10+ analyst-hours/week** |

> These are illustrative estimates based on industry benchmarks and dataset statistics. Actual impact scales with your customer base size and retention campaign conversion rate.

---

### Real-world workflow enabled

```
Daily batch scoring
        │
        ▼
 Risk score per customer
        │
        ├── Score > 0.7  →  Flag for urgent retention call
        ├── Score 0.4–0.7 →  Add to nurture email campaign
        └── Score < 0.4  →  No action needed (save cost)
```

This means retention teams stop guessing and start acting on data — **spending budget only where it matters most.**

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 0.758 |
| Weighted F1 | 0.762 |
| Macro F1 | 0.699 |

Model: **CatBoostClassifier** tuned with **Optuna** (50 trials, 5-fold StratifiedKFold CV)

---

## Project Structure

```
churn_prediction/
├── config/
│   └── config.yaml                  # All pipeline parameters
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   └── prod/
│       └── latest_model.cbm         # Trained CatBoost model (generated after training)
├── app-ml/
│   ├── src/
│   │   └── pipelines/
│   │       ├── preprocessing.py     # Data cleaning & target encoding
│   │       ├── feature_engineering.py  # Categorical label encoding
│   │       ├── training.py          # Optuna hyperparameter tuning + CatBoost
│   │       ├── inference.py         # Model loading & prediction
│   │       ├── postprocessing.py    # Evaluation & model saving
│   │       └── pipeline_runner.py   # Orchestrates all pipeline stages
│   └── entrypoint/
│       ├── train.py                 # Training entrypoint
│       └── inference.py             # Inference entrypoint
└── common/
    ├── data_manager.py              # Data I/O, train/test split, upsampling
    └── utils.py                     # Config reader, model save/load, plotting
```

---

## Pipeline Overview

```
Raw CSV
   │
   ▼
[Preprocessing]
  - Fix TotalCharges (empty string → NaN → float)
  - Drop customers with tenure == 0
  - Drop customerID column
  - Encode target: Yes → 1, No → 0
   │
   ▼
[Feature Engineering]
  - LabelEncoder on all categorical (object) columns
  - Tree-based models handle label-encoded features well; avoids one-hot sparsity
   │
   ▼
[Train/Test Split]
  - Stratified split (80/20) to preserve class balance
  - Optional minority-class upsampling on training set
   │
   ▼
[Training]
  - Optuna study (50 trials, TPE sampler)
  - 5-fold StratifiedKFold CV per trial, maximising macro F1
  - Final model trained on full training set with best hyperparameters
   │
   ▼
[Inference]
  - Load saved .cbm model
  - Predict on held-out test set
   │
   ▼
[Postprocessing]
  - Classification report (precision, recall, F1 per class)
  - Confusion matrix plot saved to inference_results.png
```

---

## Demo UI

An interactive Dash app for demoing the system — input any customer profile and get an instant churn prediction with probability.

### Churn Risk Prediction
![Churn Risk](images/prediction_risk.png)

### Safe Prediction (Likely to Stay)
![Safe Prediction](images/prediction_safe.png)

### Model Overview
![Model Overview](images/model_overview.png)

### Launch the app

```bash
pip install -r app-ui/requirements.txt
python app-ui/app.py
```

Open `http://localhost:8050` in your browser.

| Feature | Description |
|---|---|
| 🎲 Random Customer | Load a real customer from the test set with their true churn label |
| 🔮 Predict | Run inference and display churn probability + color-coded result |
| Model Overview tab | Confusion matrix, ROC curve, feature importance |

---

## Docker Architecture

### Why Docker?

In production ML systems, a common failure mode is the **"works on my machine" problem** — a model trained in one environment behaves differently when deployed in another due to differences in Python versions, package versions, or OS-level libraries. Docker solves this by packaging the entire runtime environment (code, dependencies, OS libraries) into a portable image that runs identically everywhere.

Beyond reproducibility, Docker enables **separation of concerns**: the training job and the serving application have different dependency footprints and different lifecycles. By running them as independent containers, we can:

- Scale or redeploy the UI without retraining
- Retrain the model without touching the serving layer
- Run the training job on a different machine (e.g., a GPU server) and ship only the model artifact to the serving layer

### Service Design

This project uses two containers orchestrated with Docker Compose:

```
┌─────────────────────────────────────────────────────────────┐
│                      docker-compose up                       │
│                                                             │
│  ┌─────────────────┐        named volume        ┌────────────────────┐  │
│  │  app-ml-train   │  ──── models_volume ────▶  │     app-ui         │  │
│  │                 │      (latest_model.cbm)     │                    │  │
│  │  - Preprocess   │                             │  - Loads model     │  │
│  │  - Optuna tune  │                             │  - Serves Dash app │  │
│  │  - Train model  │                             │  - Port 8050       │  │
│  │  → exits        │                             │                    │  │
│  └─────────────────┘                             └────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

| Design Decision | Rationale |
|---|---|
| Two separate images | Training needs `optuna`/`catboost`; UI needs `dash`/`plotly`. Separate images keep each lean and independently deployable. |
| Named Docker volume for models | Model artifacts are written by the training container and read by the UI. A named volume lives inside Docker's Linux VM filesystem, avoiding macOS/Windows host filesystem I/O issues. |
| Static files (data, config) baked into image | Config and training data are versioned with the code. Baking them into the image makes each image fully self-contained and traceable to a specific git commit. |
| UI waits for model file | `app-ui` polls for `latest_model.cbm` before starting gunicorn, creating a lightweight dependency without requiring a separate health-check service. |

---

## Quickstart

### Option A — Docker (recommended)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/).

```bash
git clone https://github.com/jeannineshiu/churn_prediction.git
cd churn_prediction

docker-compose up --build
```

Then open `http://localhost:8050` in your browser.

Other useful commands:

```bash
# Train only (no UI)
docker-compose run app-ml-train

# Start UI only (model already trained)
docker-compose up app-ui

# Stop and remove containers
docker-compose down

# Stop and remove containers + delete trained model volume (force retrain)
docker-compose down -v
```

---

### Option B — Local (conda)

### 1. Clone the repository

```bash
git clone https://github.com/jeannineshiu/churn_prediction.git
cd churn_prediction
```

### 2. Create and activate the conda environment

```bash
conda create -n churn-prediction python=3.10 -y
conda activate churn-prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -r app-ui/requirements.txt
```

### 4. Train the model

```bash
python app-ml/entrypoint/train.py
```

This will:
- Run Optuna hyperparameter search (50 trials)
- Train the final model on the full training set
- Save the model to `models/prod/latest_model.cbm`

### 5. Run inference (optional)

```bash
python app-ml/entrypoint/inference.py
```

Prints the classification report and saves a confusion matrix to `inference_results.png`.

### 6. Launch the UI

```bash
python app-ui/app.py
```

Open `http://localhost:8050` in your browser.

---

## Configuration

All pipeline behaviour is controlled via [`config/config.yaml`](config/config.yaml). Key parameters:

```yaml
pipeline_runner:
  test_size: 0.20        # Fraction held out for evaluation
  upsample: true         # Minority-class upsampling on training set

training:
  iterations: 1000
  early_stopping_rounds: 100
  n_cv_splits: 5

  optuna:
    n_trials: 50         # Increase for better tuning at the cost of time
    search_space:
      learning_rate: [0.01, 0.2]
      depth: [3, 8]
      l2_leaf_reg: [0.5, 5.0]
```

---

## Dataset

**IBM Telco Customer Churn** — 7,043 customers, 20 features + 1 target.

Key feature groups:
- **Demographics**: gender, SeniorCitizen, Partner, Dependents
- **Account**: tenure, Contract, PaperlessBilling, PaymentMethod
- **Services**: PhoneService, InternetService, OnlineSecurity, TechSupport, etc.
- **Billing**: MonthlyCharges, TotalCharges

**Class distribution**: ~73% No Churn / ~27% Churn → imbalanced; addressed via stratified splitting and upsampling.

---

## Key Design Decisions

- **LabelEncoder over One-Hot Encoding** — tree-based models (CatBoost, Random Forest) handle ordinal-encoded categoricals natively; one-hot encoding would add ~30 sparse columns with no benefit.
- **Upsampling over class weights** — minority upsampling on the training set improves recall for the churn class without requiring model-level adjustments.
- **StratifiedKFold in Optuna** — preserves class distribution in each fold, giving a reliable F1 estimate on an imbalanced dataset.
- **Config-driven** — all hyperparameter ranges, paths, and split ratios live in `config.yaml` so experiments are reproducible without touching source code.
