# Credit Risk Scorer

A machine learning pipeline that predicts credit default risk and serves predictions through an interactive web dashboard.

## Features

- **3 models compared**: Logistic Regression, Decision Tree, Random Forest
- **24 features**: 14 raw financial inputs + 10 engineered features
- **Live web app**: Streamlit dashboard with real-time credit scoring
- **Full evaluation**: Precision, Recall, F1, ROC-AUC, threshold analysis

## Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.630 | 0.727 | 0.675 | 0.815 |
| Decision Tree | 0.650 | 0.641 | 0.646 | 0.787 |
| **Random Forest** | **0.670** | **0.665** | **0.667** | **0.820** |

Optimal deployment threshold: **0.45**

## Project Structure

```
credit_scoring/
├── app.py                  # Streamlit web dashboard
├── data_generator.py       # Synthetic dataset generator
├── feature_engineering.py  # Feature engineering (24 features)
├── train_evaluate.py       # Full training + evaluation pipeline
├── save_model.py           # Serialize trained model to disk
├── data/                   # Generated CSV dataset
├── model/                  # Saved model artifacts (.pkl)
└── outputs/                # Charts and evaluation reports
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. Generate data + train all models + produce charts:**
```bash
python credit_scoring/train_evaluate.py
```

**2. Save the Random Forest model for the web app:**
```bash
python credit_scoring/save_model.py
```

**3. Launch the live scoring dashboard:**
```bash
streamlit run credit_scoring/app.py
```

Then open [http://localhost:8501](http://localhost:8501).

## Engineered Features

| Feature | Formula | Purpose |
|---|---|---|
| `debt_to_income` | total_debt / income | Core DTI ratio |
| `payment_reliability` | on_time / total payments | Recent payment behaviour |
| `savings_to_income` | savings / income | Financial cushion |
| `employment_stability` | type_weight x log1p(months) | Job type + tenure |
| `high_utilization` | utilization > 75% | Severe credit stress flag |
| `derogatory_intensity` | log1p(derogatory marks) | Compressed derogatory count |
| `inquiry_burden` | log1p(inquiries 12m) | Credit-seeking behaviour |
| `credit_maturity` | log1p(history months) | Credit history length |
| `debt_per_account` | debt / accounts | Average account burden |
| `age_credit_experience` | age x credit_maturity / 100 | Sustained history interaction |

## Tech Stack

- Python 3.12
- scikit-learn, pandas, numpy, matplotlib, seaborn
- Streamlit, joblib
