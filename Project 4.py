import argparse
import os
import warnings
from typing import Tuple, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------
# CONFIG - change to match your dataset
# ----------------------------
CONFIG = {
    "target": "price_usd",          # target column name in CSV
    "id_col": None,                 # optional ID column to drop, e.g. 'model_id'
    "datetime_cols": [],            # e.g. ['release_date'] (not used by default)
    # Known features expected in dataset (will try to use if present)
    "expected_numeric": [
        "memory_gb", "ram_gb", "battery_mah",
        "rear_cam_mp", "front_cam_mp", "height_mm"
    ],
    "expected_categorical": [
        "model", "color", "processor", "ai_lens"
    ],
    # For high-cardinality categorical columns (like model), we'll frequency-encode if > `ohe_threshold`
    "onehot_threshold": 25,
    # Output folder
    "out_dir": "outputs"
}

# ----------------------------
# Utilities
# ----------------------------
def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

# ----------------------------
# Feature engineering / preprocessing builder
# ----------------------------
def build_preprocessor(df: pd.DataFrame, cfg: dict) -> Tuple[ColumnTransformer, List[str]]:
    """
    Build a ColumnTransformer that:
    - imputes numeric columns (median) + scales (StandardScaler)
    - imputes categorical (most frequent) + OneHotEncode (or frequency encode for high-cardinality)
    Returns (preprocessor, output_feature_names) where feature names are approximate.
    """
    num_cols = [c for c in cfg["expected_numeric"] if c in df.columns]
    cat_cols = [c for c in cfg["expected_categorical"] if c in df.columns]

    # If dataset contains other categorical columns, add them
    other_cat = [c for c in df.columns if df[c].dtype == "object" and c not in cat_cols]
    # Avoid including the target or ID
    other_cat = [c for c in other_cat if c != cfg["target"] and c != cfg["id_col"]]
    cat_cols += other_cat

    # Decide which categorical cols get one-hot vs frequency encoding
    onehot_cols, freq_cols = [], []
    for c in cat_cols:
        n_unique = df[c].nunique(dropna=True)
        if n_unique <= cfg["onehot_threshold"]:
            onehot_cols.append(c)
        else:
            freq_cols.append(c)

    transformers = []
    feature_names = []

    # Numeric pipeline
    if num_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, num_cols))
        feature_names += num_cols

    # One-hot categorical pipeline
    if onehot_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ])
        transformers.append(("cat_ohe", cat_pipeline, onehot_cols))
        # We'll build names later once fitted.

    # Frequency (count) encoding -> implemented as a transformer using pandas in fit_transform wrapper
    # We'll implement a simple custom transformer for freq-encoding below
    if freq_cols:
        # For frequency encoding we'll just encode into new numeric columns before ColumnTransformer
        pass

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)

    return preprocessor, num_cols, onehot_cols, freq_cols

# Simple helper to frequency-encode specified columns (inplace copy)
def add_frequency_encoding(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        freq = df[c].value_counts(dropna=False).to_dict()
        df[f"{c}__freq"] = df[c].map(freq).astype(float)
    return df

# ----------------------------
# Modeling & evaluation helpers
# ----------------------------
def evaluate_model(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    return {"mae": mae, "rmse": rmse, "r2": r2, "preds": preds}

# ----------------------------
# Main pipeline
# ----------------------------
def run_pipeline(csv_path: str, cfg: dict):
    out_dir = cfg["out_dir"]
    ensure_out_dir(out_dir)

    print(f"Loading data from: {csv_path}")
    df = load_data(csv_path)
    print("Initial shape:", df.shape)

    # Basic cleaning: drop ID, drop rows missing target
    if cfg["id_col"] and cfg["id_col"] in df.columns:
        df = df.drop(columns=[cfg["id_col"]])
    if cfg["target"] not in df.columns:
        raise ValueError(f"Target column '{cfg['target']}' not found in dataset.")

    df = df.dropna(subset=[cfg["target"]])
    print("After dropping rows without target:", df.shape)

    # Derived features (if original columns exist)
    if {"rear_cam_mp", "front_cam_mp"}.issubset(df.columns):
        df["camera_total_mp"] = df["rear_cam_mp"].fillna(0) + df["front_cam_mp"].fillna(0)
    if {"memory_gb", "ram_gb"}.issubset(df.columns):
        df["mem_to_ram_ratio"] = df["memory_gb"] / (df["ram_gb"].replace(0, np.nan))

    # Frequency-encode high-cardinality categorical features
    preprocessor, num_cols, onehot_cols, freq_cols = build_preprocessor(df, cfg)
    if freq_cols:
        df = add_frequency_encoding(df, freq_cols)
        # After encoding, treat these new numeric columns as numeric features
        freq_cols_encoded = [f"{c}__freq" for c in freq_cols]
        num_cols += freq_cols_encoded

    # Final check for numeric columns present
    num_cols = [c for c in num_cols if c in df.columns]
    onehot_cols = [c for c in onehot_cols if c in df.columns]

    # Compose full preprocessor now (rebuild to include new numerics)
    transformers = []
    if num_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, num_cols))
    if onehot_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ])
        transformers.append(("cat_ohe", cat_pipeline, onehot_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)

    # Prepare X and y
    y = df[cfg["target"]].astype(float)
    X = df.drop(columns=[cfg["target"]])

    # Train/test split
    X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print("Train/test sizes:", X_train_df.shape, X_test_df.shape)

    # Fit preprocessor
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    # Build feature names list (approximate)
    feature_names = []
    feature_names += num_cols
    if onehot_cols:
        ohe = preprocessor.named_transformers_["cat_ohe"].named_steps["onehot"]
        ohe_names = list(ohe.get_feature_names_out(onehot_cols))
        feature_names += ohe_names

    # ----------------------------
    # Feature analysis: correlation (numeric) and mutual information
    # ----------------------------
    numeric_for_corr = [c for c in num_cols if c in df.columns]
    corr_series = None
    if numeric_for_corr:
        corr_series = df[numeric_for_corr + [cfg["target"]]].corr()[cfg["target"]].sort_values(ascending=False)
        corr_series.to_csv(os.path.join(out_dir, "correlation_with_target.csv"))
        print("\nTop correlations (numeric):\n", corr_series.head(10))

    # Mutual information (non-linear) on preprocessed features
    mi = mutual_info_regression(X_train, y_train, random_state=42)
    mi_series = pd.Series(mi, index=feature_names).sort_values(ascending=False)
    mi_series.head(15).to_csv(os.path.join(out_dir, "mutual_info_top15.csv"))
    print("\nTop mutual information features:\n", mi_series.head(10))

    # PCA (to understand explained variance)
    try:
        pca = PCA(n_components=min(10, X_train.shape[1]), random_state=42)
        pca.fit(X_train)
        explained_var = np.cumsum(pca.explained_variance_ratio_)
        # Save explained variance to CSV
        pd.Series(explained_var, index=[f"PC{i+1}" for i in range(len(explained_var))]).to_csv(
            os.path.join(out_dir, "pca_explained_variance.csv")
        )
    except Exception:
        explained_var = None

    # ----------------------------
    # Models: baseline training
    # ----------------------------
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
    }

    eval_results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        eval_results[name] = {"mae": metrics["mae"], "rmse": metrics["rmse"], "r2": metrics["r2"]}
        trained_models[name] = model
        print(f"{name} -> MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.3f}")

    # Save evaluation summary
    eval_df = pd.DataFrame(eval_results).T.sort_values("mae")
    eval_df.to_csv(os.path.join(out_dir, "model_evaluation.csv"))

    # ----------------------------
    # Feature importances from RandomForest + permutation importance
    # ----------------------------
    rf = trained_models.get("RandomForest")
    if rf is not None:
        try:
            importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
            importances.head(30).to_csv(os.path.join(out_dir, "rf_feature_importances_top30.csv"))
            # Permutation importance
            perm = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=42, n_jobs=-1)
            perm_imp = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
            perm_imp.head(30).to_csv(os.path.join(out_dir, "perm_importances_top30.csv"))
            print("\nTop RandomForest importances:\n", importances.head(10))
            print("\nTop permutation importances:\n", perm_imp.head(10))
        except Exception as e:
            print("Could not compute feature importances:", e)

    # ----------------------------
    # Save artifacts: preprocessor + best model (by MAE)
    # ----------------------------
    best_model_name = eval_df.index[0]
    best_model = trained_models[best_model_name]

    artifact_preprocessor_path = os.path.join(out_dir, "preprocessor.joblib")
    artifact_model_path = os.path.join(out_dir, f"best_model_{best_model_name}.joblib")
    joblib.dump(preprocessor, artifact_preprocessor_path)
    joblib.dump(best_model, artifact_model_path)
    print(f"\nSaved preprocessor -> {artifact_preprocessor_path}")
    print(f"Saved best model ({best_model_name}) -> {artifact_model_path}")

    # ----------------------------
    # Plots
    # ----------------------------
    # 1) Correlation bar (numeric)
    if corr_series is not None:
        plt.figure(figsize=(8, 4))
        corr_series.sort_values(ascending=False).plot(kind="bar")
        plt.title("Correlation with target (numeric features)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "correlation_with_target.png"))
        plt.close()

    # 2) Mutual info top features
    plt.figure(figsize=(8, 4))
    mi_series.head(15).plot(kind="bar")
    plt.title("Top 15 features by mutual information")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mutual_info_top15.png"))
    plt.close()

    # 3) PCA explained variance
    if explained_var is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, len(explained_var) + 1), explained_var, marker="o")
        plt.xlabel("Number of PCA components")
        plt.ylabel("Cumulative explained variance")
        plt.title("PCA cumulative explained variance")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "pca_explained_variance.png"))
        plt.close()

    # 4) RF feature importances (top 20)
    try:
        if rf is not None:
            plt.figure(figsize=(8, 6))
            importances.head(20).sort_values().plot(kind="barh")
            plt.title("RandomForest top 20 feature importances")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "rf_top20_importances.png"))
            plt.close()
    except Exception:
        pass

    # Final summary saved
    print("\nPipeline completed. Outputs saved in:", out_dir)
    print("Key files:")
    for fname in sorted(os.listdir(out_dir)):
        print(" -", fname)


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Mobile Phone Price Prediction Pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file containing the data.")
    parser.add_argument("--target", type=str, default=None, help="Name of the target column (overrides config).")
    parser.add_argument("--out", type=str, default=None, help="Output folder (overrides config).")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = CONFIG.copy()
    if args.target:
        cfg["target"] = args.target
    if args.out:
        cfg["out_dir"] = args.out
    run_pipeline(args.data, cfg)

if __name__ == "__main__":
    main()
