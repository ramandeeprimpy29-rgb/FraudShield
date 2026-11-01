# FraudShield – Credit Card Fraud Detection System

## Project Overview
FraudShield is a machine learning-based credit card fraud detection system that identifies potentially fraudulent transactions using transaction-related features. It uses Logistic Regression and Random Forest algorithms to classify transactions as genuine or fraudulent.

## Key Features
- Machine learning-based fraud prediction
- Interactive Streamlit dashboard
- Transaction fraud probability estimation
- Genuine vs. fraudulent transaction visualization
- Confusion matrix analysis
- Model performance comparison
- Merchant category analysis

## Dataset
The project uses a dataset containing 10,000 credit card transactions with 10 features.

**Target variable:** `is_fraud`

**Features include:**
- Transaction amount
- Transaction hour
- Merchant category
- Foreign transaction indicator
- Location mismatch
- Device trust score
- Transaction velocity
- Cardholder age

Dataset distribution:
- Genuine transactions: 9,849
- Fraudulent transactions: 151

## Machine Learning Models
1. Logistic Regression
2. Random Forest Classifier

## Model Performance

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 95.80% | 99.75% |
| Precision | 25.89% | 100% |
| Recall | 96.67% | 83.33% |
| F1-Score | 40.85% | 90.91% |
| ROC-AUC | 0.9933 | 1.0000 |

Random Forest was selected as the best model based on its F1-score.

**Note:** These results are from a held-out test set of 2,000 transactions containing 30 fraud cases. Performance on other datasets may differ.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib

## Project Structure
```text
FraudShield/
├── app.py
├── train_model.py
├── requirements.txt
├── data/
│   └── credit_card_fraud_10k.csv
├── models/
│   └── fraud_model.pkl
├── outputs/
└── .gitignore
```

## Installation and Setup

1. Clone the repository:
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Navigate to the project folder:
```bash
cd FraudShield
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:
```bash
python -m streamlit run app.py
```

## Model Training
To train the models again, run:
```bash
python train_model.py
```

## Disclaimer
This project is developed for educational and academic purposes. Predictions should not be treated as definitive financial decisions. Model performance depends on the quality and representativeness of the dataset.

## Author
Developed as a Machine Learning Project.
