# Bank Transaction Fraud Detector — Autoencoder + XGBoost

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbhiramPaku/BankTransactionFraudDetectorAE/blob/main/VIMiniProject.ipynb)

A hybrid fraud-detection pipeline that combines an **Autoencoder** (for unsupervised feature extraction) with **XGBoost** (for classification) to detect fraudulent bank transactions on the PaySim1 dataset.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
  - [1. Data Loading & Preprocessing](#1-data-loading--preprocessing)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Autoencoder (Feature Extraction)](#3-autoencoder-feature-extraction)
  - [4. Hybrid Feature Merge](#4-hybrid-feature-merge)
  - [5. XGBoost Classifier](#5-xgboost-classifier)
  - [6. Threshold Tuning & Evaluation](#6-threshold-tuning--evaluation)
- [Results](#results)
- [Requirements](#requirements)
- [How to Run](#how-to-run)

---

## Overview

Financial fraud detection is a classic class-imbalance problem: fraudulent transactions are rare (< 0.2 % of all records) yet extremely costly. This project addresses the problem with a two-stage approach:

1. **Autoencoder** — trained exclusively on normal (non-fraud) transactions to learn a compressed latent representation of "normal" behaviour.
2. **XGBoost** — trained on the concatenation of the original scaled features *and* the latent encoding produced by the autoencoder, giving the classifier richer signal about how anomalous each transaction is.

---

## Dataset

| Property | Value |
|---|---|
| Source | [PaySim1 — Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) |
| License | CC-BY-SA-4.0 |
| Rows | 6,362,620 |
| Columns | 11 |
| Fraud rate | ~0.13 % |

**Columns**

| Column | Description |
|---|---|
| `step` | Unit of time (1 step = 1 hour, max 743) |
| `type` | Transaction type: CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER |
| `amount` | Transaction amount |
| `nameOrig` | Originating account ID |
| `oldbalanceOrg` | Originating account balance before transaction |
| `newbalanceOrig` | Originating account balance after transaction |
| `nameDest` | Destination account ID |
| `oldbalanceDest` | Destination account balance before transaction |
| `newbalanceDest` | Destination account balance after transaction |
| `isFraud` | Target label — 1 if fraud, 0 otherwise |
| `isFlaggedFraud` | Existing rule-based flag (not used as a feature) |

---

## Project Structure

```
BankTransactionFraudDetectorAE/
├── VIMiniProject.ipynb      # Main Jupyter / Colab notebook
├── autoencoder_model.h5     # Saved Keras autoencoder model
├── encoder_model.h5         # Saved Keras encoder (latent extractor)
└── README.md
```

---

## Pipeline

### 1. Data Loading & Preprocessing

- Dataset downloaded from Kaggle using the `kaggle` CLI.
- Transaction type (`type`) is label-encoded with `sklearn.preprocessing.LabelEncoder`.
- Identifier columns (`nameOrig`, `nameDest`) and the rule-based flag (`isFlaggedFraud`) are dropped.
- Features are scaled with `StandardScaler` after a stratified 80/20 train-test split.

### 2. Feature Engineering

Two balance-discrepancy features are engineered to capture inconsistencies that are strongly correlated with fraud:

```python
df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
```

Feature importance was assessed with:
- **ANOVA F-test** for continuous numerical features.
- **Chi-squared test** for encoded categorical features (`type`, `step`).

### 3. Autoencoder (Feature Extraction)

A symmetric autoencoder is trained **only on normal (non-fraud) transactions** so that it learns to reconstruct normal behaviour accurately. Fraudulent transactions are expected to produce higher reconstruction errors.

**Architecture**

```
Input (d) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(8, ReLU)  ← latent space
                                                ↓
Output (d) ← Dense(32, Linear) ← Dense(16, ReLU) ← Dense(8, ReLU)
```

| Hyper-parameter | Value |
|---|---|
| Encoding dimension | 8 |
| Optimizer | Adam |
| Loss | MSE |
| Epochs | 15 |
| Batch size | 512 |

The encoder sub-model is saved as `encoder_model.h5` and is used to extract 8-dimensional latent features for every sample.

### 4. Hybrid Feature Merge

The original scaled features and the latent encoding are concatenated:

```python
X_train_final = np.hstack((X_train_scaled, X_train_encoded))
X_test_final  = np.hstack((X_test_scaled,  X_test_encoded))
```

This gives the downstream classifier access to both the raw transaction signal and the autoencoder's anomaly signal.

### 5. XGBoost Classifier

| Hyper-parameter | Value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 8 |
| `learning_rate` | 0.1 |
| `scale_pos_weight` | 99 (handles class imbalance) |
| `eval_metric` | logloss |

`scale_pos_weight ≈ (# negatives) / (# positives)` is used to compensate for the severe class imbalance.

### 6. Threshold Tuning & Evaluation

The default decision threshold of 0.5 is replaced with a custom threshold of **0.05** to maximise recall (catching as many real frauds as possible) at the cost of slightly lower precision.

```python
custom_threshold = 0.05
y_pred = (model.predict_proba(X_test_final)[:, 1] >= custom_threshold).astype(int)
```

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 99.96 % |
| Precision | 0.7773 |
| Recall | 0.9963 |
| F1-Score | 0.8733 |
| PR-AUC | 0.9921 |

**Confusion Matrix**

|  | Predicted Normal | Predicted Fraud |
|---|---|---|
| **Actual Normal** | 1,270,412 | 469 |
| **Actual Fraud** | 6 | 1,637 |

Only **6 fraud transactions** out of 1,643 were missed (false negatives).

---

## Requirements

```
python >= 3.8
tensorflow / keras
xgboost
scikit-learn
pandas
numpy
matplotlib
seaborn
kaggle (CLI)
```

Install with:

```bash
pip install tensorflow xgboost scikit-learn pandas numpy matplotlib seaborn kaggle
```

---

## How to Run

1. **Open in Google Colab** (recommended) — click the badge at the top of this README.

2. **Run locally**:

   ```bash
   # 1. Clone the repo
   git clone https://github.com/AbhiramPaku/BankTransactionFraudDetectorAE.git
   cd BankTransactionFraudDetectorAE

   # 2. Install dependencies
   pip install tensorflow xgboost scikit-learn pandas numpy matplotlib seaborn kaggle

   # 3. Set up Kaggle credentials (kaggle.json in ~/.kaggle/)
   kaggle datasets download -d ealaxi/paysim1

   # 4. Launch the notebook
   jupyter notebook VIMiniProject.ipynb
   ```

   Pre-trained model weights (`autoencoder_model.h5`, `encoder_model.h5`) are included so you can skip the training step and go straight to inference if needed.