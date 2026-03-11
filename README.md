# ML-fundamentals-2026

Individual Assignment 1 Machine Learning - Nicolas Grass

---

Note that this README was partially written by AI

## Assignment 1 — Data Preparation & Feature Engineering

**Notebook:** `assignment_1_nicolas_grass.ipynb`  
**Dataset:** [UCI Bank Marketing Dataset](https://www.kaggle.com/) — `bank-additional.csv`  
**Deadline:** 11 March 2026

### Problem

Given client and campaign information available at the time of contact, predict whether a client subscribes to a term deposit following a telemarketing campaign conducted by a Portuguese banking institution.

### Pipeline Overview

The tasks are executed in the following order to ensure a leak-free preprocessing pipeline:

| #   | Task                                 | Key Decision                                               |
| --- | ------------------------------------ | ---------------------------------------------------------- |
| 1   | Identifying the Prediction Target    | `y` (term deposit subscription)                            |
| 2   | Data Loading and Exploration         | EDA on 4,119 observations, 20 features                     |
| 3   | Task Ordering                        | Justified pipeline sequence                                |
| 4   | Data Splitting                       | 70 / 15 / 15 stratified split                              |
| 5   | Managing Missing Values              | Mode imputation + `unknown` as category                    |
| 6   | Encoding Categorical Variables       | Ordinal (education) + OHE (9 nominal cols)                 |
| 7   | Feature Selection                    | Variance threshold + correlation pruning + leakage removal |
| 8   | Feature Scaling                      | StandardScaler on continuous features only                 |
| 9   | Addressing Class Imbalance           | SMOTE on training set only                                 |
| 10  | Training a Logistic Regression Model | Evaluated on untouched validation set                      |

### Key Results

| Metric             | Score |
| ------------------ | ----- |
| Accuracy           | 78.3% |
| Zero Rule Baseline | 89.2% |
| Precision          | 28.7% |
| Recall             | 67.2% |
| F1 Score           | 40.2% |
| ROC AUC            | 0.786 |

Accuracy intentionally falls below the Zero Rule baseline — this is expected after SMOTE resampling, which shifts the decision boundary toward the minority class to improve recall for true subscribers.

### Notable Design Decisions

- **`duration` removed** — call duration is only known after the call ends (temporal leakage).
- **`default` and `education` unknowns retained** — their unknown groups show meaningfully different subscription rates, so missingness carries predictive signal.
- **`pdays` engineered** into two features (`previously_contacted`, `pdays_clean`) — 96.5% of values were the sentinel `999`, making the raw column unsuitable as a continuous feature.
- **`euribor3m` and `nr.employed` removed** — Pearson correlation > 0.85 with `emp.var.rate`, causing multicollinearity.
- **Cyclical encoding rejected** for `month` and `day_of_week` — campaign behavior does not follow a periodic pattern that would justify treating months or days as points on a circle.

### Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

### Running the Notebook

1. Download `bank-additional.csv` from Kaggle and place it in the same directory as the notebook.
2. Open the notebook and run **Restart Kernel and Run All**.
3. All cells must execute without errors from top to bottom.

### Repository Structure

```
ML-fundamentals-2026/
│
├── README.md
├── .gitignore
├── assignment_1_nicolas_grass.ipynb
└── bank-additional.csv          # not tracked by git (add to .gitignore)
```

### AI Disclaimer

AI assistance was used for generating markdown tables, styling visualizations, and verifying expected outputs. All analytical decisions, justifications, and pipeline design are original work.

---
