import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

def train_and_save():
    df = pd.read_csv(DATA_DIR / "credit_data.csv")

    from feature_engineering import build_features
    X, y = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw_dict = dict(zip(classes, weights))

    scale_pos_weight = cw_dict[1] / cw_dict[0] if 0 in cw_dict and 1 in cw_dict else 1.0

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, MODEL_DIR / "xgb_model.pkl")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_names.pkl")

    print(f"Model saved to {MODEL_DIR / 'xgb_model.pkl'}")
    print(f"Features: {list(X.columns)}")

if __name__ == "__main__":
    train_and_save()
