import numpy as np
import pandas as pd

EMPLOYMENT_LABELS = {0: "Unemployed", 1: "Part-time", 2: "Full-time", 3: "Self-employed"}


def build_features(df):
    data = df.copy()

    data["debt_to_income"] = data["total_debt"] / (data["annual_income"] + 1e-6)

    total_payments = data["on_time_payments_6m"] + data["late_payments_6m"]
    data["payment_reliability"] = data["on_time_payments_6m"] / (total_payments + 1e-6)

    data["savings_to_income"] = data["savings_balance"] / (data["annual_income"] + 1e-6)

    type_weights = {0: 0.0, 1: 0.5, 2: 1.0, 3: 0.85}
    emp_weight = data["employment_type"].map(type_weights)
    data["employment_stability"] = emp_weight * np.log1p(data["months_employed"])

    data["high_utilization"] = (data["credit_utilization"] > 0.75).astype(int)

    data["derogatory_intensity"] = np.log1p(data["num_derogatory_marks"])

    data["inquiry_burden"] = np.log1p(data["num_inquiries_12m"])

    data["credit_maturity"] = np.log1p(data["credit_history_months"])

    data["debt_per_account"] = data["total_debt"] / (data["num_accounts"] + 1)

    data["age_credit_experience"] = data["age"] * data["credit_maturity"] / 100.0

    raw_features = [
        "age",
        "annual_income",
        "employment_type",
        "months_employed",
        "total_debt",
        "credit_utilization",
        "num_accounts",
        "on_time_payments_6m",
        "late_payments_6m",
        "credit_history_months",
        "num_derogatory_marks",
        "num_inquiries_12m",
        "num_open_accounts",
        "savings_balance",
    ]

    engineered_features = [
        "debt_to_income",
        "payment_reliability",
        "savings_to_income",
        "employment_stability",
        "high_utilization",
        "derogatory_intensity",
        "inquiry_burden",
        "credit_maturity",
        "debt_per_account",
        "age_credit_experience",
    ]

    all_features = raw_features + engineered_features
    X = data[all_features]
    y = data["default"]

    return X, y


def feature_summary():
    records = [
        ("debt_to_income",        "total_debt / annual_income",              "Core DTI ratio; primary repayment-capacity signal"),
        ("payment_reliability",   "on_time / (on_time + late)",              "6-month on-time payment fraction; recent behaviour"),
        ("savings_to_income",     "savings_balance / annual_income",         "Financial cushion / shock absorption capacity"),
        ("employment_stability",  "type_weight x log1p(months_employed)",    "Combines employment type and tenure into one signal"),
        ("high_utilization",      "credit_utilization > 0.75",               "Binary flag: severe credit utilisation stress"),
        ("derogatory_intensity",  "log1p(num_derogatory_marks)",             "Log-scaled derogatory record count; compressed outliers"),
        ("inquiry_burden",        "log1p(num_inquiries_12m)",                "Credit-seeking behaviour in the past 12 months"),
        ("credit_maturity",       "log1p(credit_history_months)",            "Log-scaled credit history length"),
        ("debt_per_account",      "total_debt / (num_accounts + 1)",         "Average debt burden per open account"),
        ("age_credit_experience", "age x credit_maturity / 100",             "Interaction: age + sustained credit history"),
    ]
    return pd.DataFrame(records, columns=["Feature", "Formula", "Rationale"])
