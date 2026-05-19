import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_SAMPLES = 10_000


def generate_credit_dataset(n=N_SAMPLES, seed=SEED):
    rng = np.random.default_rng(seed)

    risk_score = rng.beta(a=2, b=5, size=n)

    age = rng.normal(loc=38, scale=12, size=n).clip(18, 75).astype(int)

    employment_type = rng.choice([0, 1, 2, 3], p=[0.06, 0.12, 0.62, 0.20], size=n)

    months_employed = np.where(
        employment_type == 0,
        0,
        (rng.exponential(scale=48, size=n) * (1 - 0.6 * risk_score)).clip(0, 360).astype(int)
    )

    base_income = rng.lognormal(mean=10.8, sigma=0.55, size=n)
    income_multiplier = np.select(
        [employment_type == 0, employment_type == 1],
        [0.15, 0.55],
        default=1.0
    )
    annual_income = (base_income * income_multiplier * (1 - 0.3 * risk_score)).clip(5_000, 500_000)

    total_debt = (annual_income * rng.uniform(0.1, 3.5, size=n) * (1 + 1.5 * risk_score)).clip(0)

    credit_utilization = (rng.beta(a=2, b=5, size=n) + 0.5 * risk_score).clip(0, 1)

    num_accounts = rng.integers(1, 20, size=n)
    late_payments_6m = rng.poisson(lam=(0.3 + 3.0 * risk_score), size=n).clip(0, 12)
    on_time_payments_6m = (num_accounts * 6 - late_payments_6m).clip(0)

    credit_history_months = (age * rng.uniform(3, 12, size=n) - 100 * risk_score).clip(0, 400).astype(int)

    num_derogatory_marks = rng.poisson(lam=(0.1 + 4.0 * risk_score), size=n).clip(0, 10)

    num_inquiries_12m = rng.poisson(lam=(0.5 + 2.5 * risk_score), size=n).clip(0)

    num_open_accounts = rng.integers(0, 15, size=n)

    savings_balance = (annual_income * rng.beta(a=1.5, b=4, size=n) * (1 - 0.7 * risk_score)).clip(0)

    log_odds = (
        -3.5
        + 5.0 * risk_score
        + 1.2 * (total_debt / (annual_income + 1e-6) > 2).astype(float)
        + 0.8 * (credit_utilization > 0.75).astype(float)
        + 0.6 * (late_payments_6m > 2).astype(float)
        + 1.5 * (num_derogatory_marks > 1).astype(float)
        - 0.5 * (months_employed > 24).astype(float)
        + 0.4 * (employment_type == 0).astype(float)
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    default = rng.binomial(n=1, p=prob_default).astype(int)

    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income.round(2),
        "employment_type": employment_type,
        "months_employed": months_employed,
        "total_debt": total_debt.round(2),
        "credit_utilization": credit_utilization.round(4),
        "num_accounts": num_accounts,
        "on_time_payments_6m": on_time_payments_6m,
        "late_payments_6m": late_payments_6m,
        "credit_history_months": credit_history_months,
        "num_derogatory_marks": num_derogatory_marks,
        "num_inquiries_12m": num_inquiries_12m,
        "num_open_accounts": num_open_accounts,
        "savings_balance": savings_balance.round(2),
        "default": default,
    })

    return df


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)
    df = generate_credit_dataset()
    out_path = out_dir / "credit_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Dataset saved: {out_path}")
    print(f"Shape: {df.shape}")
    print(f"Default rate: {df['default'].mean():.1%}")
    print(df.describe().T.to_string())
