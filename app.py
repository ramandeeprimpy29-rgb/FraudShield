
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st


st.set_page_config(
    page_title="FraudShield | Credit Card Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "models" / "fraud_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "data" / "credit_card_fraud_10k.csv")


@st.cache_data
def load_metadata():
    with open(BASE_DIR / "outputs" / "model_metadata.json", "r") as file:
        return json.load(file)


model = load_model()
df = load_data()
metadata = load_metadata()

DECISION_THRESHOLD = metadata["decision_threshold"]
FEATURES = metadata["features"]


@st.cache_resource
def load_shap_explainer():
    return shap.TreeExplainer(model.named_steps["classifier"])


def prepare_shap_values(input_frame):
    preprocessor = model.named_steps["preprocessor"]

    transformed = preprocessor.transform(input_frame)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    transformed_df = pd.DataFrame(
        transformed,
        columns=feature_names
    )

    explainer = load_shap_explainer()
    shap_values = explainer.shap_values(transformed)

    if isinstance(shap_values, list):
        fraud_values = shap_values[1]
    else:
        fraud_values = shap_values[:, :, 1]

    base_value = float(explainer.expected_value[1])

    return transformed_df, fraud_values, base_value


model_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [96.95, 97.35, 97.80],
    "Precision": [18.37, 24.44, 29.41],
    "Recall": [30.00, 36.67, 33.33],
    "F1 Score": [22.78, 29.33, 31.25],
    "ROC-AUC": [91.09, 71.27, 92.48],
    "PR-AUC": [19.47, 23.33, 27.60]
})

confusion_matrix = [[1946, 24], [20, 10]]

st.markdown("""
<style>
.stApp {
    background-color: #080F1E;
    color: #FFFFFF;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background-color: #111D32;
}

h1, h2, h3, h4, p, label, span {
    color: #FFFFFF;
}

.main-title {
    color: #FFFFFF;
    font-size: 34px;
    font-weight: 800;
}

.subtitle {
    color: #C5D2E5;
    font-size: 15px;
}

.metric-card {
    background-color: #14243B;
    border: 1px solid #29415F;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
}

.metric-label {
    color: #C5D2E5;
    font-size: 14px;
}

.metric-value {
    color: #FFFFFF;
    font-size: 27px;
    font-weight: bold;
}

div.stButton > button {
    background-color: #087EBC;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: bold;
    padding: 10px 20px;
}

div.stButton > button:hover {
    background-color: #09669A;
    color: white;
}

.stTextInput input, .stNumberInput input {
    background-color: #14243B;
    color: white;
}

div[data-baseweb="select"] > div {
    background-color: #14243B;
    color: white;
}

hr {
    border-color: #29415F;
}
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🛡️ FraudShield")
    st.caption("Credit Card Fraud Detection")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "🔍 Live Prediction",
            "📘 About Project"
        ]
    )

    st.divider()
    st.markdown("### Project Information")
    st.write("Machine Learning Based")
    st.write("Credit Card Security System")

st.markdown(
    '<div class="main-title">FraudShield</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Intelligent Credit Card Fraud Detection System</div>',
    unsafe_allow_html=True
)

st.write("")

# Dashboard
if page == "📊 Dashboard":

    st.subheader("📊 Project Dashboard")
    st.write(
        "Overview of the transaction dataset and machine learning model performance."
    )

    total_transactions = len(df)
    fraud_count = int(df["is_fraud"].sum())
    genuine_count = total_transactions - fraud_count
    fraud_percentage = (fraud_count / total_transactions) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Fraud Transactions", f"{fraud_count:,}")
    col3.metric("Genuine Transactions", f"{genuine_count:,}")
    col4.metric("Fraud Percentage", f"{fraud_percentage:.2f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Transaction Distribution")

        distribution = pd.DataFrame({
            "Transaction Type": ["Genuine", "Fraud"],
            "Count": [genuine_count, fraud_count]
        })

        fig = px.pie(
            distribution,
            names="Transaction Type",
            values="Count",
            hole=0.55,
            title="Genuine vs Fraud Transactions"
        )

        fig.update_layout(
            paper_bgcolor="#080F1E",
            plot_bgcolor="#080F1E",
            font_color="white"
        )

        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Fraud Distribution")

        fig = px.bar(
            distribution,
            x="Transaction Type",
            y="Count",
            color="Transaction Type",
            text="Count",
            title="Number of Transactions"
        )

        fig.update_layout(
            paper_bgcolor="#080F1E",
            plot_bgcolor="#080F1E",
            font_color="white",
            showlegend=False
        )

        st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader("Model Performance Comparison")

    st.caption(
        "All metrics are calculated on the same held-out test dataset."
    )

    st.dataframe(
        model_results.style.format({
            "Accuracy": "{:.2f}%",
            "Precision": "{:.2f}%",
            "Recall": "{:.2f}%",
            "F1 Score": "{:.2f}%",
            "ROC-AUC": "{:.2f}%",
            "PR-AUC": "{:.2f}%"
        }),
        width="stretch",
        hide_index=True
    )

    st.subheader("Model Metric Comparison")

    metric_choice = st.selectbox(
        "Select performance metric",
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"]
    )

    fig = px.bar(
        model_results,
        x="Model",
        y=metric_choice,
        color="Model",
        text=metric_choice,
        title=f"{metric_choice} Comparison"
    )

    fig.update_layout(
        paper_bgcolor="#080F1E",
        plot_bgcolor="#080F1E",
        font_color="white",
        showlegend=False
    )

    st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader("Random Forest Confusion Matrix")

    cm = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix,
            x=["Predicted Genuine", "Predicted Fraud"],
            y=["Actual Genuine", "Actual Fraud"],
            text=confusion_matrix,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True
        )
    )

    cm.update_layout(
        paper_bgcolor="#080F1E",
        plot_bgcolor="#080F1E",
        font_color="white",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class"
    )

    st.plotly_chart(cm, width="stretch")

    st.caption(
        "The confusion matrix represents the Random Forest model's "
        "held-out test results."
    )

    st.divider()

    st.subheader("SHAP Explainability")
    st.write(
        "This plot shows how different transaction features influence "
        "the Random Forest fraud predictions."
    )

    try:
        shap_sample = df[FEATURES].sample(
            n=min(200, len(df)),
            random_state=42
        )

        shap_features, shap_values, base_value = prepare_shap_values(
            shap_sample
        )

        shap.summary_plot(
            shap_values,
            shap_features,
            max_display=8,
            plot_size=(8, 4),
            show=False
        )

        fig = plt.gcf()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    except Exception as error:
        st.warning(f"SHAP summary plot could not be generated: {error}")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        width="stretch",
        hide_index=True
    )

# Live Prediction
elif page == "🔍 Live Prediction":

    st.subheader("🔍 Transaction Analysis")

    st.write(
        "Enter transaction details to estimate the possibility of credit card fraud."
    )

    merchant_categories = sorted(
        df["merchant_category"].dropna().astype(str).unique().tolist()
    )

    amount_max = max(
        5000.0,
        float(df["amount"].max())
    )

    velocity_max = max(
        50,
        int(df["velocity_last_24h"].max())
    )

    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            max_value=amount_max,
            value=500.0,
            step=50.0
        )

        transaction_hour = st.slider(
            "Transaction Hour",
            min_value=0,
            max_value=23,
            value=12
        )

        merchant_category = st.selectbox(
            "Merchant Category",
            merchant_categories
        )

    with col2:
        device_trust_score = st.slider(
            "Device Trust Score",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05
        )

        velocity_last_24h = st.number_input(
            "Transactions in Last 24 Hours",
            min_value=0,
            max_value=velocity_max,
            value=min(2, velocity_max),
            step=1
        )

        cardholder_age = st.number_input(
            "Cardholder Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1
        )

    st.divider()

    analyze_clicked = st.button(
        "🔎 Analyze Transaction",
        width="stretch"
    )

    if analyze_clicked:

        input_values = {
            "amount": amount,
            "transaction_hour": transaction_hour,
            "merchant_category": merchant_category,
            "device_trust_score": device_trust_score,
            "velocity_last_24h": velocity_last_24h,
            "cardholder_age": cardholder_age
        }

        input_data = pd.DataFrame(
            [[input_values[feature] for feature in FEATURES]],
            columns=FEATURES
        )

        try:
            probabilities = model.predict_proba(input_data)[0]

            fraud_class_index = list(model.classes_).index(1)

            fraud_probability = float(
                probabilities[fraud_class_index]
            )

            prediction = int(
                fraud_probability >= DECISION_THRESHOLD
            )

            st.divider()
            st.subheader("Prediction Result")

            result1, result2 = st.columns(2)

            with result1:
                if prediction == 1:
                    st.error("⚠️ Potential Fraud Transaction")
                else:
                    st.success("✅ Likely Genuine Transaction")

                st.metric(
                    "Fraud Probability",
                    f"{fraud_probability * 100:.2f}%"
                )

                st.metric(
                    "Decision Threshold",
                    f"{DECISION_THRESHOLD * 100:.2f}%"
                )

            with result2:
                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=fraud_probability * 100,
                        title={"text": "Fraud Probability (%)"},
                        number={"suffix": "%"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#19B5FE"},
                            "threshold": {
                                "line": {"color": "#FFFFFF", "width": 3},
                                "thickness": 0.8,
                                "value": DECISION_THRESHOLD * 100
                            },
                            "steps": [
                                {"range": [0, 30], "color": "#087F65"},
                                {"range": [30, 60], "color": "#A66A12"},
                                {"range": [60, 100], "color": "#A83242"}
                            ]
                        }
                    )
                )

                gauge.update_layout(
                    paper_bgcolor="#14243B",
                    font_color="white"
                )

                st.plotly_chart(gauge, width="stretch")

            st.divider()
            st.subheader("Individual Transaction Explanation")

            try:
                shap_features, shap_values, base_value = prepare_shap_values(
                    input_data
                )

                explanation = shap.Explanation(
                    values=shap_values[0],
                    base_values=base_value,
                    data=shap_features.iloc[0].values,
                    feature_names=shap_features.columns.tolist()
                )

                shap.plots.waterfall(
                    explanation,
                    max_display=8,
                    show=False
                )

                fig = plt.gcf()
                fig.set_size_inches(8, 4)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)

                st.caption(
                    "The waterfall plot explains how individual features "
                    "influence the model's fraud prediction."
                )

            except Exception as error:
                st.warning(
                    f"Individual SHAP explanation unavailable: {error}"
                )

            st.subheader("Transaction Details")

            st.dataframe(
                input_data,
                width="stretch",
                hide_index=True
            )

            st.caption(
                "This result is a machine learning estimate, not a confirmed "
                "fraud determination. The prediction uses the selected model "
                "and its validation-based decision threshold."
            )

        except Exception as error:
            st.error(f"Prediction could not be completed: {error}")

    else:
        st.info(
            "Enter the transaction details and click Analyze Transaction."
        )

# About Project
elif page == "📘 About Project":

    st.subheader("📘 About FraudShield")

    st.write(
        "FraudShield is a machine learning based credit card fraud detection "
        "system. It analyzes transaction characteristics and estimates "
        "whether a transaction may be fraudulent or genuine."
    )

    st.divider()

    st.markdown("### Project Objectives")

    st.write("- Identify potentially fraudulent transactions.")
    st.write("- Analyze transaction-related patterns.")
    st.write("- Apply machine learning classification algorithms.")
    st.write("- Compare models using multiple evaluation metrics.")
    st.write("- Provide an interactive transaction prediction interface.")
    st.write("- Explain model predictions using SHAP.")

    st.markdown("### Dataset Information")

    st.write(f"- Total records: {len(df):,}")
    st.write(f"- Total columns: {len(df.columns)}")
    st.write(f"- Fraud transactions: {int(df['is_fraud'].sum()):,}")
    st.write(f"- Genuine transactions: {int((df['is_fraud'] == 0).sum()):,}")
    st.write("- Target column: is_fraud")
    st.write("- Fraud class: 1")
    st.write("- Genuine class: 0")

    st.markdown("### Selected Input Features")

    for feature in FEATURES:
        st.write(f"- {feature}")

    st.markdown("### Machine Learning Algorithms")

    st.write("1. Logistic Regression")
    st.write("2. Decision Tree Classifier")
    st.write("3. Random Forest Classifier")

    st.success(
        f"Selected Model: {metadata['best_model']}"
    )

    st.write(
        f"Decision Threshold: {DECISION_THRESHOLD:.4f}"
    )

    st.markdown("### Technologies Used")

    st.write("- Python")
    st.write("- Pandas and NumPy")
    st.write("- Scikit-learn")
    st.write("- Streamlit")
    st.write("- Plotly")
    st.write("- Matplotlib")
    st.write("- SHAP")
    st.write("- Joblib")

    st.divider()

    st.markdown("### Developed By")
    st.markdown("**Ramandeep Rimpy**")
    st.write("M.Tech Data Science and Analytics")

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#C5D2E5; padding:10px;">
        FraudShield | Machine Learning Based Credit Card Fraud Detection
    </div>
    """,
    unsafe_allow_html=True
)
