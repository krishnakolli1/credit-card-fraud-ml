# Credit Card Fraud Detection Using Machine Learning

## Project Overview
This project uses machine learning to identify potentially fraudulent credit card transactions. It explores transaction patterns, handles imbalanced data using SMOTE, trains multiple classification models, and compares their performance.

## Dataset
- Source: Kaggle – Credit Card Fraud Detection (ULB Machine Learning Group)
- Original records: 284,807
- Records after duplicate removal: 283,726
- Features: 30 input features
- Target: `Class` (0 = Legitimate, 1 = Fraud)
- Fraud cases in cleaned dataset: 473

**Note:** The dataset is not included in this repository. Download `creditcard.csv` from Kaggle and place it inside the `data/` folder before running the scripts.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Joblib

## Project Workflow
1. Load and explore the transaction dataset.
2. Check missing values, duplicates, and class distribution.
3. Analyze transaction amounts and feature correlations.
4. Remove duplicate records.
5. Split the data into training and testing sets using stratification.
6. Apply SMOTE to the training data to address class imbalance.
7. Train Logistic Regression, Decision Tree, and Random Forest models.
8. Evaluate models using Precision, Recall, F1-score, and PR-AUC.
9. Save the best-performing model and evaluation charts.

## Model Performance

| Model | Precision | Recall | F1-score | PR-AUC |
|---|---:|---:|---:|---:|
| Random Forest | 78.57% | 81.05% | 79.79% | 81.03% |
| Logistic Regression | 5.27% | 87.37% | 9.93% | 67.63% |
| Decision Tree | 16.56% | 78.95% | 27.37% | 41.30% |

Random Forest achieved the highest PR-AUC and F1-score among the three models.

## Key Findings
- Fraudulent transactions represented approximately 0.17% of the original dataset.
- Fraud detection is challenging because the dataset is highly imbalanced.
- SMOTE was applied only to the training data.
- Random Forest provided the best overall balance between precision and recall in this experiment.

## Project Structure
```text
credit-card-fraud-ml/
├── data/
├── models/
├── charts/
├── reports/
├── analysis.py
├── train_model.py
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run

1. Clone this repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the Kaggle dataset and place `creditcard.csv` in the `data/` folder.
4. Run the analysis:

```bash
python analysis.py
```

5. Train and evaluate the models:

```bash
python train_model.py
```

## Future Improvements
- Tune model hyperparameters.
- Experiment with different classification thresholds.
- Explore additional imbalance-handling techniques.
- Build an interactive fraud detection dashboard.
- Develop a simple web application for transaction predictions.

## Disclaimer
This project is for educational purposes and is not intended for real-world financial fraud decisions.