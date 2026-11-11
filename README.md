# FraudShield - Credit Card Fraud Detection System

## 1. Project Overview

FraudShield is a machine learning-based credit card fraud detection system developed to identify potentially fraudulent transactions.

The project uses multiple classification algorithms and compares their performance to select a suitable model. It also provides an interactive Streamlit dashboard for transaction prediction, model evaluation, and explainable AI-based insights.

The selected model is a Random Forest Classifier

## 2. Key Features

- Machine learning-based credit card fraud detection
- Interactive Streamlit dashboard
- Transaction-level fraud probability estimation
- Logistic Regression, Decision Tree, and Random Forest comparison
- Automated decision threshold selection
- Confusion matrix and classification performance analysis
- SHAP-based global feature importance
- Individual SHAP Waterfall Plot for prediction explanations
- Transaction and fraud distribution visualizations

## 3. Dataset Description

The project uses a dataset containing 10,000 credit card transactions and 10 columns.

**Target variable:** `is_fraud`

**Features used by the final model:**
- `amount`
- `transaction_hour`
- `merchant_category`
- `device_trust_score`
- `velocity_last_24h`
- `cardholder_age`

The transaction identifier is not used for model training. Other available dataset columns, including foreign transaction and location mismatch indicators, are also excluded from the final model.

### Dataset Distribution

| Transaction Type | Records |
|---|---:|
| Genuine | 9,849 |
| Fraudulent | 151 |
| Total | 10,000 |

The dataset is imbalanced, with fraudulent transactions representing approximately 1.51% of all records.

## 4. Machine Learning Algorithms

Three classification algorithms are evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

### Training Methodology

- Training set: 60%
- Validation set: 20%
- Testing set: 20%
- Random state: 42
- Stratified splitting to maintain class distribution
- Numerical feature preprocessing using median imputation and standardization
- Categorical feature preprocessing using one-hot encoding
- Class weighting to address class imbalance
- Decision threshold selection using validation data

The final model is selected based on validation F1-score and evaluated on the separate test set.

## 5. Model Performance

The following results were obtained from the held-out test set of 2,000 transactions.

| Metric | Logistic Regression | Decision Tree | Random Forest |
|---|---:|---:|---:|
| Accuracy | 96.95% | 97.35% | 97.80% |
| Precision | 18.37% | 24.44% | 29.41% |
| Recall | 30.00% | 36.67% | 33.33% |
| F1-Score | 22.78% | 29.33% | 31.25% |
| ROC-AUC | 0.9109 | 0.7127 | 0.9248 |
| PR-AUC | 0.1947 | 0.2333 | 0.2760 |

**Selected Model: Random Forest Classifier**

Random Forest achieved the highest validation F1-score of approximately 0.5373 and the highest test F1-score of 0.3125 among the evaluated models.

### Random Forest Confusion Matrix

| | Predicted Genuine | Predicted Fraud |
|---|---:|---:|
| Actual Genuine | 1946 | 24 |
| Actual Fraud | 20 | 10 |

The model correctly identified 10 out of 30 fraudulent transactions in the test set.

### Decision Threshold

The selected model uses a decision threshold of approximately **59.34%**, optimized using validation data.

A transaction is classified as potentially fraudulent when its predicted fraud probability meets or exceeds this threshold.

## 6. Explainable Artificial Intelligence (XAI)

FraudShield integrates SHAP (SHapley Additive exPlanations) to improve model interpretability.

Two types of SHAP visualizations are included:

- **SHAP Summary Plot:** Displays the overall contribution of features across multiple transactions.
- **SHAP Waterfall Plot:** Explains how individual transaction features contribute to a specific prediction.

These visualizations help understand the factors influencing the model's predictions.

## 7. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Matplotlib
- SHAP
- Joblib

## 8. Project Structure

```text
FraudShield/
|-- app.py
|-- train_model.py
|-- diagnose_model.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- credit_card_fraud_10k.csv
|-- models/
|   |-- fraud_model.pkl
|-- outputs/
|   |-- model_comparison.csv
|   |-- model_metadata.json
|   |-- test_predictions.csv
|-- .gitignore
```

## 9. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/ramandeeprimpy29-rgb/FraudShield.git
```

### Step 2: Navigate to the Project Directory

```bash
cd FraudShield
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
python -m streamlit run app.py
```

The Streamlit application will open in the browser.

## 10. Model Training

To train and compare the machine learning algorithms again, run:

```bash
python train_model.py
```

The training process generates the trained model, model comparison results, metadata, and test predictions.

## 11. Model Evaluation and Limitations

Although the model achieves 97.80% test accuracy, its fraud recall is 33.33%.

This means that it identifies 10 of the 30 fraudulent transactions in the test set, while 20 fraudulent transactions remain undetected.

The imbalanced dataset affects fraud detection performance. Accuracy alone is therefore insufficient to evaluate this classification problem.

The system is an academic prototype and should not be treated as a production-ready financial fraud detection solution.

## 12. Future Scope

- Improve fraud detection recall.
- Explore advanced class imbalance handling techniques.
- Evaluate additional machine learning algorithms.
- Perform further hyperparameter optimization.
- Test on larger and more diverse transaction datasets.
- Explore model monitoring and deployment improvements.

## 13. Disclaimer

FraudShield is developed for educational and academic purposes. Its predictions are model-generated estimates and should not be treated as definitive financial decisions.

Model performance depends on dataset quality, feature representation, and the characteristics of unseen transactions.

## Author

Developed as a Machine Learning Project.
