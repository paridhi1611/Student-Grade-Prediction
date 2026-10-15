"""
Student Grade Prediction using Linear Regression
Dataset: UCI Student Performance (ID 320), CC BY 4.0
This version excludes G1 and G2 to avoid using earlier grades to predict the final grade.
Run: python student_grade_prediction.py
"""
from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    print("Downloading the official UCI Student Performance dataset...")
    dataset = fetch_ucirepo(id=320)
    X = dataset.data.features.copy()
    y_all = dataset.data.targets.copy()

    # Predict Mathematics final grade G3; do not use G1/G2 to avoid grade leakage.
    if "G3" not in y_all.columns:
        raise ValueError("Expected target column G3 was not found in the UCI dataset.")
    y = pd.to_numeric(y_all["G3"], errors="coerce")

    # Keep only non-grade predictors; G1/G2 are not in X on the current UCI release,
    # but this explicit guard prevents accidental leakage if dataset shape changes.
    X = X.drop(columns=[c for c in ["G1", "G2", "G3"] if c in X.columns], errors="ignore")
    valid_rows = y.notna()
    X, y = X.loc[valid_rows].copy(), y.loc[valid_rows].copy()

    # Use one fixed seed so results are reproducible.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", LinearRegression()),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "dataset": "UCI Student Performance (ID 320)",
        "target": "G3 (final mathematics grade, scale 0-20)",
        "features_used": X.shape[1],
        "records_used": int(len(X)),
        "test_records": int(len(X_test)),
        "model": "Linear Regression",
        "predictors_excluded_to_avoid_leakage": ["G1", "G2"],
        "MAE": round(float(mean_absolute_error(y_test, predictions)), 4),
        "RMSE": round(float(mean_squared_error(y_test, predictions) ** 0.5), 4),
        "R2": round(float(r2_score(y_test, predictions)), 4),
    }
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    results = pd.DataFrame({
        "actual_final_grade": y_test.to_numpy(),
        "predicted_final_grade": predictions,
    })
    results.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    # Actual vs predicted chart
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=results, x="actual_final_grade", y="predicted_final_grade", alpha=0.75)
    plt.plot([0, 20], [0, 20], linestyle="--")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.title("Actual vs Predicted Final Grades")
    plt.xlabel("Actual final grade (G3)")
    plt.ylabel("Predicted final grade")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close()

    # Residual distribution
    results["residual"] = results["actual_final_grade"] - results["predicted_final_grade"]
    plt.figure(figsize=(7, 5))
    sns.histplot(results["residual"], kde=True)
    plt.title("Distribution of Prediction Errors")
    plt.xlabel("Residual (actual - predicted)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residual_distribution.png", dpi=160)
    plt.close()

    # Save the exact split for audit/reproducibility
    print("\n=== STUDENT GRADE PREDICTION RESULTS ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"\nOutputs saved in: {OUTPUT_DIR}")
    print("Note: this is a small dataset from two Portuguese schools; results may not generalize elsewhere.")

if __name__ == "__main__":
    main()
