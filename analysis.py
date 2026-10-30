
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
CHARTS_DIR = BASE_DIR / "charts"
REPORTS_DIR = BASE_DIR / "reports"


def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return

    CHARTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    print("Loading credit card transactions...")
    df = pd.read_csv(DATA_PATH)

    print("\n" + "=" * 60)
    print("          CREDIT CARD FRAUD DATA ANALYSIS")
    print("=" * 60)

    print(f"\nDataset shape: {df.shape}")
    print(f"Total transactions: {len(df):,}")
    print(f"Number of columns: {len(df.columns)}")

    print("\nFirst five records:")
    print(df.head().to_string())

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates:,}")

    # Transaction distribution
    class_counts = df["Class"].value_counts().sort_index()
    fraud_count = int(class_counts.get(1, 0))
    legitimate_count = int(class_counts.get(0, 0))
    total = len(df)

    fraud_percentage = fraud_count / total * 100 if total else 0

    print("\nTransaction distribution:")
    print(f"Legitimate transactions: {legitimate_count:,}")
    print(f"Fraudulent transactions: {fraud_count:,}")
    print(f"Fraud percentage: {fraud_percentage:.4f}%")

    # Transaction amount statistics
    print("\nTransaction amount statistics:")
    print(df["Amount"].describe().round(2).to_string())

    print("\nAverage transaction amount by class:")
    print(df.groupby("Class")["Amount"].mean().round(2).to_string())

    # Fraud distribution chart
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Class")
    plt.title("Legitimate vs Fraudulent Transactions")
    plt.xlabel("Transaction Class (0 = Legitimate, 1 = Fraud)")
    plt.ylabel("Number of Transactions")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "transaction_distribution.png", dpi=150)
    plt.close()

    # Transaction amount distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=df[df["Amount"] <= df["Amount"].quantile(0.99)],
        x="Amount",
        hue="Class",
        bins=50,
        element="step",
        stat="density",
        common_norm=False
    )
    plt.title("Transaction Amount Distribution (Below 99th Percentile)")
    plt.xlabel("Transaction Amount")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "transaction_amount_distribution.png", dpi=150)
    plt.close()

    # Correlation with fraud class
    correlations = (
        df.corr(numeric_only=True)["Class"]
        .drop("Class")
        .sort_values()
    )

    print("\nFeatures most negatively correlated with fraud:")
    print(correlations.head(5).round(4).to_string())

    print("\nFeatures most positively correlated with fraud:")
    print(correlations.tail(5).sort_values(ascending=False).round(4).to_string())

    plt.figure(figsize=(10, 7))
    correlations.plot(kind="barh")
    plt.title("Feature Correlation with Fraud Class")
    plt.xlabel("Correlation")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "feature_correlation.png", dpi=150)
    plt.close()

    # Export summary
    summary = pd.DataFrame({
        "Metric": [
            "Total Transactions",
            "Legitimate Transactions",
            "Fraudulent Transactions",
            "Fraud Percentage",
            "Duplicate Rows",
            "Missing Values"
        ],
        "Value": [
            total,
            legitimate_count,
            fraud_count,
            round(fraud_percentage, 4),
            int(duplicates),
            int(df.isnull().sum().sum())
        ]
    })

    summary.to_csv(REPORTS_DIR / "dataset_summary.csv", index=False)
    correlations.to_csv(REPORTS_DIR / "feature_correlations.csv")

    print(f"\nCharts saved to: {CHARTS_DIR}")
    print(f"Reports saved to: {REPORTS_DIR}")
    print("\nData analysis completed successfully!")


if __name__ == "__main__":
    main()
