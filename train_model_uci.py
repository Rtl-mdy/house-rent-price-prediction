from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "apartments_for_rent_classified_10K.csv"
CLEAN_PATH = ROOT / "data" / "cleaned_apartment_rent.csv"
MODEL_PATH = ROOT / "models" / "rent_model_uci.joblib"
RESULTS_PATH = ROOT / "outputs" / "model_comparison.csv"
FEATURES = ["bedrooms", "bathrooms", "square_feet", "cityname", "state"]
TARGET = "price"

def load_clean_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\\nDownload 'apartments_for_rent_classified_10K.csv' "
            "from the UCI Apartment for Rent Classified (10K) dataset page and place it in data/."
        )
    df = pd.read_csv(DATA_PATH, sep=";", encoding="cp1252")
    for col in ["price", "square_feet", "bedrooms", "bathrooms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["currency"] == "USD") &
            (df["price_type"].astype(str).str.lower() == "monthly")]
    df = df[df["price"].between(100, 20000) &
            df["square_feet"].between(100, 10000)]
    df_clean = df.dropna(subset=["price", "cityname", "state"]).copy()
    return df_clean

def make_pipeline(estimator):
    numeric = ["bedrooms", "bathrooms", "square_feet"]
    categorical = ["cityname", "state"]
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])
    return Pipeline([("preprocessor", preprocessor), ("regressor", estimator)])

def main():
    for p in [CLEAN_PATH.parent, MODEL_PATH.parent, RESULTS_PATH.parent]:
        p.mkdir(parents=True, exist_ok=True)
    df = load_clean_data()
    df[FEATURES + [TARGET]].to_csv(CLEAN_PATH, index=False)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # The reference report selects the depth-10 tree using 5-fold CV.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    depth_scores = {}
    for depth in [1, 2, 3, 5, 8, 10, 15, 20, None]:
        tree = make_pipeline(DecisionTreeRegressor(max_depth=depth, random_state=42))
        depth_scores[depth] = cross_val_score(
            tree, X_train, y_train, cv=cv, scoring="r2"
        ).mean()
    best_depth = max(depth_scores, key=depth_scores.get)

    candidates = {
        f"Linear Regression": LinearRegression(),
        f"Decision Tree (depth {best_depth})": DecisionTreeRegressor(
            max_depth=best_depth, random_state=42
        ),
        "Decision Tree (abs. error, depth 8)": DecisionTreeRegressor(
            criterion="absolute_error", max_depth=8, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=50, random_state=42, min_samples_leaf=2, n_jobs=-1
        ),
    }

    records = []
    fitted = {}
    for name, estimator in candidates.items():
        pipe = make_pipeline(estimator)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted[name] = pipe
        records.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": mean_squared_error(y_test, pred) ** 0.5,
            "R2": r2_score(y_test, pred),
            "Train R2": r2_score(y_train, pipe.predict(X_train)),
        })

    results = pd.DataFrame(records).sort_values("RMSE")
    results.to_csv(RESULTS_PATH, index=False)
    best_name = "Random Forest"
    joblib.dump(fitted[best_name], MODEL_PATH)

    print(f"Original rows: 10,000 | Usable rows: {len(df):,}")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved final model: {MODEL_PATH}")
    print(f"Saved cleaned data: {CLEAN_PATH}")
    print(f"Saved comparison: {RESULTS_PATH}")

    ax = results.plot(x="Model", y="MAE", kind="bar", legend=False, rot=20)
    ax.set_ylabel("MAE (USD)")
    ax.set_title("Model Comparison — lower MAE is better")
    plt.tight_layout()
    plt.savefig(ROOT / "graphs" / "model_mae_comparison.png", dpi=160)
    plt.close()

if __name__ == "__main__":
    main()
