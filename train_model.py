
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    precision_recall_curve
)

# Load dataset
df = pd.read_csv("data/credit_card_fraud_10k.csv")

FEATURES = [
    "amount",
    "transaction_hour",
    "merchant_category",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age"
]

TARGET = "is_fraud"

X = df[FEATURES].copy()
y = df[TARGET].copy()

print("Dataset loaded:", df.shape)
print("Selected features:", FEATURES)
print("Fraud distribution:\n", y.value_counts())

# Split: 60% training, 20% validation, 20% testing
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=42, stratify=y_temp
)

print("\nTraining:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# Preprocessing
numeric_features = [
    "amount", "transaction_hour", "device_trust_score",
    "velocity_last_24h", "cardholder_age"
]

categorical_features = ["merchant_category"]


def create_preprocessor():
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])


# Models
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=8,
        min_samples_leaf=5,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
}

results = []
best_model = None
best_name = ""
best_validation_f1 = -1
best_threshold = 0.50

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Train and evaluate
for name, classifier in models.items():

    print("\nTraining:", name)

    model = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    # Select threshold using validation data
    val_probabilities = model.predict_proba(X_val)[:, 1]

    precision_values, recall_values, thresholds = (
        precision_recall_curve(y_val, val_probabilities)
    )

    if len(thresholds) > 0:
        f1_values = (
            2 * precision_values[:-1] * recall_values[:-1]
            / (
                precision_values[:-1]
                + recall_values[:-1]
                + 1e-12
            )
        )

        best_index = int(f1_values.argmax())
        threshold = float(thresholds[best_index])
        validation_f1 = float(f1_values[best_index])
    else:
        threshold = 0.50
        validation_f1 = 0.0

    # Evaluate on untouched test data
    test_probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (test_probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, test_probabilities)
    pr_auc = average_precision_score(y_test, test_probabilities)

    print("Validation F1:", round(validation_f1, 4))
    print("Threshold:", round(threshold, 4))
    print("Test Accuracy:", round(accuracy, 4))
    print("Test Precision:", round(precision, 4))
    print("Test Recall:", round(recall, 4))
    print("Test F1:", round(f1, 4))
    print("Test ROC-AUC:", round(roc_auc, 4))
    print("Test PR-AUC:", round(pr_auc, 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    results.append({
        "Model": name,
        "Validation_F1": validation_f1,
        "Threshold": threshold,
        "Test_Accuracy": accuracy,
        "Test_Precision": precision,
        "Test_Recall": recall,
        "Test_F1": f1,
        "Test_ROC_AUC": roc_auc,
        "Test_PR_AUC": pr_auc
    })

    if validation_f1 > best_validation_f1:
        best_validation_f1 = validation_f1
        best_model = model
        best_name = name
        best_threshold = threshold

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/model_comparison.csv", index=False)

joblib.dump(best_model, "models/fraud_model.pkl")

metadata = {
    "best_model": best_name,
    "decision_threshold": best_threshold,
    "validation_f1": best_validation_f1,
    "features": FEATURES,
    "target": TARGET,
    "random_state": 42
}

with open("outputs/model_metadata.json", "w") as file:
    json.dump(metadata, file, indent=4)

# Save test predictions
test_output = pd.DataFrame({
    "actual": y_test.to_numpy(),
    "predicted": (
        best_model.predict_proba(X_test)[:, 1]
        >= best_threshold
    ).astype(int),
    "fraud_probability": best_model.predict_proba(X_test)[:, 1]
})

test_output.to_csv("outputs/test_predictions.csv", index=False)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))
print("\nSelected Model:", best_name)
print("Selected Threshold:", round(best_threshold, 4))
print("Training completed successfully!")
