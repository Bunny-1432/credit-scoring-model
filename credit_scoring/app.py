import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from feature_engineering import build_features

BASE_DIR  = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"

st.set_page_config(
    page_title="Credit Risk Scorer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main { background: #0f1117; }

.stApp {
    background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 100%);
}

.score-card {
    background: linear-gradient(135deg, #1e2235 0%, #252840 100%);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

.score-number {
    font-size: 72px;
    font-weight: 700;
    line-height: 1;
    margin: 8px 0;
}

.score-label {
    font-size: 14px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #8892a4;
    margin-bottom: 8px;
}

.verdict-approve {
    background: linear-gradient(135deg, #0d3320, #1a5c38);
    border: 1px solid #2ecc71;
    border-radius: 12px;
    padding: 16px 24px;
    color: #2ecc71;
    font-size: 18px;
    font-weight: 600;
    text-align: center;
    margin-top: 16px;
}

.verdict-deny {
    background: linear-gradient(135deg, #3a0d0d, #6b1a1a);
    border: 1px solid #e74c3c;
    border-radius: 12px;
    padding: 16px 24px;
    color: #e74c3c;
    font-size: 18px;
    font-weight: 600;
    text-align: center;
    margin-top: 16px;
}

.verdict-review {
    background: linear-gradient(135deg, #332d00, #5c4d00);
    border: 1px solid #f39c12;
    border-radius: 12px;
    padding: 16px 24px;
    color: #f39c12;
    font-size: 18px;
    font-weight: 600;
    text-align: center;
    margin-top: 16px;
}

.metric-card {
    background: #1e2235;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.06);
    text-align: center;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #5c9de0;
}

.metric-label {
    font-size: 12px;
    color: #8892a4;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

.section-header {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #5c9de0;
    padding: 8px 0;
    border-bottom: 1px solid rgba(92,157,224,0.2);
    margin-bottom: 16px;
}

.stSlider > div > div { accent-color: #5c9de0; }

div[data-testid="stSidebar"] {
    background: #141724;
    border-right: 1px solid rgba(255,255,255,0.06);
}

.stSelectbox > div, .stNumberInput > div {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_path = MODEL_DIR / "rf_model.pkl"
    feat_path  = MODEL_DIR / "feature_names.pkl"
    if not model_path.exists():
        st.error("Model not found. Please run `python save_model.py` first.")
        st.stop()
    model    = joblib.load(model_path)
    features = joblib.load(feat_path)
    return model, features


def get_risk_color(prob):
    if prob < 0.35:
        return "#2ecc71"
    elif prob < 0.60:
        return "#f39c12"
    else:
        return "#e74c3c"


def credit_score_from_prob(prob):
    return int(850 - prob * (850 - 300))


def make_gauge(prob):
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#1e2235")
    ax.set_facecolor("#1e2235")

    theta_start = np.pi
    theta_end   = 0.0
    n_pts       = 200
    thetas      = np.linspace(theta_start, theta_end, n_pts)

    colors_bg = ["#2ecc71", "#f39c12", "#e74c3c"]
    splits    = [0, n_pts // 3, 2 * n_pts // 3, n_pts]
    for i, c in enumerate(colors_bg):
        seg = thetas[splits[i]:splits[i+1]]
        ax.plot(seg, [0.85] * len(seg), lw=14, color=c, alpha=0.25, solid_capstyle="butt")

    needle_theta = theta_start + prob * (theta_end - theta_start)
    ax.annotate("", xy=(needle_theta, 0.75), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="white", lw=2.5))

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)

    color = get_risk_color(prob)
    ax.text(np.pi / 2, 0.35, f"{prob:.1%}", ha="center", va="center",
            fontsize=18, fontweight="bold", color=color,
            transform=ax.transData)

    fig.tight_layout(pad=0.2)
    return fig


def make_feature_bar(model, feature_names, input_df):
    clf = model.named_steps["clf"]
    imp = clf.feature_importances_
    imp_series = pd.Series(imp, index=feature_names).sort_values().tail(10)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor("#1e2235")
    ax.set_facecolor("#1e2235")

    colors = ["#5c9de0" if v < imp_series.max() * 0.6 else "#a278d4"
              for v in imp_series.values]
    bars = ax.barh(imp_series.index, imp_series.values, color=colors,
                   edgecolor="none", height=0.6)

    ax.set_xlabel("Importance", color="#8892a4", fontsize=9)
    ax.tick_params(colors="#8892a4", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, color="#2a2f45", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.5)
    return fig


def build_input_row(inputs):
    raw = pd.DataFrame([{
        "age":                   inputs["age"],
        "annual_income":         inputs["annual_income"],
        "employment_type":       inputs["employment_type"],
        "months_employed":       inputs["months_employed"],
        "total_debt":            inputs["total_debt"],
        "credit_utilization":    inputs["credit_utilization"],
        "num_accounts":          inputs["num_accounts"],
        "on_time_payments_6m":   inputs["on_time_payments_6m"],
        "late_payments_6m":      inputs["late_payments_6m"],
        "credit_history_months": inputs["credit_history_months"],
        "num_derogatory_marks":  inputs["num_derogatory_marks"],
        "num_inquiries_12m":     inputs["num_inquiries_12m"],
        "num_open_accounts":     inputs["num_open_accounts"],
        "savings_balance":       inputs["savings_balance"],
        "default":               0,
    }])
    X, _ = build_features(raw)
    return X


model, feature_names = load_model()

st.markdown("## 💳 Credit Risk Scorer")
st.markdown("<p style='color:#8892a4;margin-top:-12px;'>Real-time applicant credit risk assessment powered by Random Forest</p>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.markdown("### 👤 Applicant Profile")
    st.markdown('<p class="section-header">Personal</p>', unsafe_allow_html=True)

    age = st.slider("Age", 18, 75, 35)
    employment_type = st.selectbox(
        "Employment Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: ["Unemployed", "Part-time", "Full-time", "Self-employed"][x],
        index=2,
    )
    months_employed = st.slider("Months at Current Employer", 0, 360, 36)

    st.markdown('<p class="section-header">Income & Debt</p>', unsafe_allow_html=True)
    annual_income  = st.number_input("Annual Income ($)", 5_000, 500_000, 55_000, step=1_000)
    total_debt     = st.number_input("Total Debt ($)", 0, 1_000_000, 15_000, step=500)
    savings_balance = st.number_input("Savings Balance ($)", 0, 500_000, 8_000, step=500)

    st.markdown('<p class="section-header">Credit Profile</p>', unsafe_allow_html=True)
    credit_utilization    = st.slider("Credit Utilization", 0.0, 1.0, 0.30, 0.01, format="%.2f")
    credit_history_months = st.slider("Credit History (months)", 0, 400, 72)
    num_accounts          = st.slider("Number of Accounts", 1, 20, 4)
    num_open_accounts     = st.slider("Open Accounts", 0, 15, 3)
    num_derogatory_marks  = st.slider("Derogatory Marks", 0, 10, 0)
    num_inquiries_12m     = st.slider("Credit Inquiries (12m)", 0, 20, 1)

    st.markdown('<p class="section-header">Payment History</p>', unsafe_allow_html=True)
    on_time_payments_6m = st.slider("On-time Payments (6m)", 0, 120, 22)
    late_payments_6m    = st.slider("Late Payments (6m)", 0, 12, 0)

    assess_btn = st.button("Run Credit Assessment", type="primary", use_container_width=True)

inputs = {
    "age": age, "annual_income": annual_income, "employment_type": employment_type,
    "months_employed": months_employed, "total_debt": total_debt,
    "credit_utilization": credit_utilization, "num_accounts": num_accounts,
    "on_time_payments_6m": on_time_payments_6m, "late_payments_6m": late_payments_6m,
    "credit_history_months": credit_history_months, "num_derogatory_marks": num_derogatory_marks,
    "num_inquiries_12m": num_inquiries_12m, "num_open_accounts": num_open_accounts,
    "savings_balance": savings_balance,
}

X_input  = build_input_row(inputs)
prob_def = float(model.predict_proba(X_input)[0, 1])
cs       = credit_score_from_prob(prob_def)
dti      = total_debt / (annual_income + 1e-6)
pay_rel  = on_time_payments_6m / max(on_time_payments_6m + late_payments_6m, 1)

col_score, col_gauge, col_feat = st.columns([1, 1.1, 1.4])

with col_score:
    color = get_risk_color(prob_def)
    st.markdown(f"""
    <div class="score-card">
        <div class="score-label">Credit Score</div>
        <div class="score-number" style="color:{color}">{cs}</div>
        <div style="color:#8892a4;font-size:12px;">300 – 850 scale</div>
    </div>
    """, unsafe_allow_html=True)

    if prob_def < 0.35:
        st.markdown('<div class="verdict-approve">✅ Recommend: APPROVE</div>', unsafe_allow_html=True)
    elif prob_def < 0.60:
        st.markdown('<div class="verdict-review">⚠️ Recommend: MANUAL REVIEW</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="verdict-deny">❌ Recommend: DECLINE</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
        <div class="metric-card">
            <div class="metric-value">{prob_def:.1%}</div>
            <div class="metric-label">Default Risk</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{dti:.2f}</div>
            <div class="metric-label">DTI Ratio</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{pay_rel:.0%}</div>
            <div class="metric-label">Pay Reliability</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{credit_utilization:.0%}</div>
            <div class="metric-label">Utilization</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_gauge:
    st.markdown("**Default Probability Gauge**")
    fig_gauge = make_gauge(prob_def)
    st.pyplot(fig_gauge, use_container_width=True)
    plt.close(fig_gauge)

    threshold = 0.45
    st.markdown(f"""
    <div style="background:#1e2235;border-radius:10px;padding:14px;border:1px solid rgba(255,255,255,0.06);margin-top:8px;">
        <div style="font-size:12px;color:#8892a4;margin-bottom:6px;">DEPLOYMENT THRESHOLD</div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="color:#f0f0f0;font-size:14px;">Decision cutoff</span>
            <span style="color:#5c9de0;font-weight:700;">{threshold}</span>
        </div>
        <div style="background:#2a2f45;border-radius:6px;height:6px;margin-top:8px;">
            <div style="background:{'#e74c3c' if prob_def >= threshold else '#2ecc71'};
                        width:{min(prob_def/threshold*50, 100):.0f}%;height:6px;border-radius:6px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_feat:
    st.markdown("**Top 10 Feature Importances**")
    fig_feat = make_feature_bar(model, feature_names, X_input)
    st.pyplot(fig_feat, use_container_width=True)
    plt.close(fig_feat)

st.markdown("---")
st.markdown("### 📋 Risk Factor Breakdown")

factors = [
    ("Debt-to-Income Ratio",     dti,                  0.43,  False, "DTI > 0.43 is a lender red flag"),
    ("Credit Utilization",       credit_utilization,   0.75,  False, "> 75% signals maxed credit lines"),
    ("Payment Reliability",      pay_rel,              0.90,  True,  "< 90% on-time payments is concerning"),
    ("Derogatory Marks",         num_derogatory_marks, 1,     False, "Any derogatory mark significantly raises risk"),
    ("Credit Inquiries (12m)",   num_inquiries_12m,    3,     False, "> 3 inquiries suggests credit-seeking stress"),
]

cols = st.columns(len(factors))
for col, (label, val, threshold_v, higher_is_better, note) in zip(cols, factors):
    with col:
        if higher_is_better:
            status = "🟢" if val >= threshold_v else "🔴"
        else:
            status = "🟢" if val < threshold_v else "🔴"
        st.markdown(f"""
        <div class="metric-card" style="padding:16px;">
            <div style="font-size:20px;margin-bottom:4px;">{status}</div>
            <div style="font-size:15px;font-weight:600;color:#f0f0f0;">{val:.2f}</div>
            <div style="font-size:11px;color:#8892a4;margin-top:4px;">{label}</div>
            <div style="font-size:10px;color:#555e70;margin-top:6px;">{note}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("🔬 View Raw Feature Vector"):
    st.dataframe(X_input.T.rename(columns={0: "Value"}).style.format("{:.4f}"), use_container_width=True)

st.markdown("""
<div style="text-align:center;color:#3d4455;font-size:12px;margin-top:24px;">
    Credit Risk Scorer &nbsp;|&nbsp; Random Forest (AUC = 0.82) &nbsp;|&nbsp; Optimal threshold = 0.45
</div>
""", unsafe_allow_html=True)
