
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    ConfusionMatrixDisplay,
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"
CHARTS_DIR = BASE_DIR / "charts"
REPORTS_DIR = BASE_DIR / "reports"


def main():
    if not DATA_PATH.exists():
        print("Dataset not found.")
        return

    MODEL_DIR.mkdir(exist_ok=True)
    CHARTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Remove exact duplicate transactions before splitting.
    original_rows = len(df)
    df = df.drop_duplicates().copy()

    print(f"Original rows: {original_rows:,}")
    print(f"Rows after duplicate removal: {len(df):,}")

    X = df.drop(columns=["Class", "Time"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"\nTraining records: {len(X_train):,}")
    print(f"Testing records: {len(X_test):,}")
    print(f"Fraud cases in training: {int(y_train.sum()):,}")
    print(f"Fraud cases in testing: {int(y_test.sum()):,}")

    # SMOTE is applied only to the training data.
    models = {
        "Logistic Regression": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=42
            )),
        ]),
        "Decision Tree": ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("classifier", DecisionTreeClassifier(
                max_depth=12,
                min_samples_leaf=5,
                random_state=42
            )),
        ]),
        "Random Forest": ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42
            )),
        ]),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print("\n" + "=" * 60)
        print(f"Training: {name}")
        print("=" * 60)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )
        pr_auc = average_precision_score(
            y_test, probabilities
        )

        results.append({
            "Model": name,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
            "PR_AUC": pr_auc,
        })

        trained_models[name] = model

        print(classification_report(
            y_test,
            predictions,
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        ))

        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"PR-AUC:    {pr_auc:.4f}")

        # Save confusion matrix for each model.
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, predictions),
            display_labels=["Legitimate", "Fraud"],
        ).plot(cmap="Blues", values_format="d")

        plt.title(f"{name} - Confusion Matrix")
        plt.tight_layout()
        filename = name.lower().replace(" ", "_")
        plt.savefig(
            CHARTS_DIR / f"{filename}_confusion_matrix.png",
            dpi=150,
        )
        plt.close()

    # Compare all models.
    results_df = pd.DataFrame(results).sort_values(
        "PR_AUC", ascending=False
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))

    results_df.to_csv(
        REPORTS_DIR / "model_comparison.csv",
        index=False,
    )

    # Plot model comparison.
    comparison = results_df.set_index("Model")[
        ["Precision", "Recall", "F1_Score", "PR_AUC"]
    ]

    comparison.plot(
        kind="bar",
        figsize=(11, 6),
        ylim=(0, 1.05),
    )

    plt.title("Fraud Detection Model Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(
        CHARTS_DIR / "model_comparison.png",
        dpi=150,
    )
    plt.close()

    # Save the best model based on PR-AUC.
    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]

    joblib.dump(
        best_model,
        MODEL_DIR / "best_fraud_detection_model.joblib",
    )

    print(f"\nBest model by PR-AUC: {best_name}")
    print("Trained model saved successfully.")
    print("\nMachine learning training completed!")


if __name__ == "__main__":
    main()
