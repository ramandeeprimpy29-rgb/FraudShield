
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

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


model = load_model()
df = load_data()

st.markdown("""
<style>
.stApp {
    background-color: #080F1E;
    color: #FFFFFF;
}

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
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

if page == "📊 Dashboard":

    st.subheader("📊 Project Dashboard")
    st.write("Overview of the transaction dataset and model performance.")

    total_transactions = len(df)
    fraud_transactions = int(df["is_fraud"].sum())
    genuine_transactions = total_transactions - fraud_transactions

    fraud_percentage = (
        fraud_transactions / total_transactions
    ) * 100

    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        ("Total Transactions", f"{total_transactions:,}"),
        ("Genuine Transactions", f"{genuine_transactions:,}"),
        ("Fraud Transactions", f"{fraud_transactions:,}"),
        ("Fraud Percentage", f"{fraud_percentage:.2f}%")
    ]

    for col, (label, value) in zip(
        [col1, col2, col3, col4],
        metrics
    ):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")
    st.divider()

    st.subheader("Transaction Insights")

    avg_amount = df["amount"].mean()
    max_amount = df["amount"].max()

    fraud_rate = (
        df.groupby("merchant_category")["is_fraud"].mean() * 100
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Average Transaction Amount", f"₹{avg_amount:,.2f}")

    with col2:
        st.metric("Highest Transaction Amount", f"₹{max_amount:,.2f}")

    st.markdown("#### Fraud Rate by Merchant Category")

    fraud_rate_df = fraud_rate.reset_index()
    fraud_rate_df.columns = ["Merchant Category", "Fraud Rate"]

    fig = px.bar(
        fraud_rate_df,
        x="Merchant Category",
        y="Fraud Rate",
        color="Fraud Rate",
        color_continuous_scale="Reds"
    )

    fig.update_layout(
        paper_bgcolor="#14243B",
        plot_bgcolor="#14243B",
        font_color="white",
        yaxis_title="Fraud Rate (%)"
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Transaction Analysis")

    chart1, chart2 = st.columns(2)

    with chart1:
        st.markdown("#### Transaction Distribution")

        distribution = pd.DataFrame({
            "Transaction Type": ["Genuine", "Fraud"],
            "Count": [genuine_transactions, fraud_transactions]
        })

        fig = px.pie(
            distribution,
            names="Transaction Type",
            values="Count",
            hole=0.55,
            color="Transaction Type",
            color_discrete_map={
                "Genuine": "#08BFA5",
                "Fraud": "#F04452"
            }
        )

        fig.update_layout(
            paper_bgcolor="#14243B",
            plot_bgcolor="#14243B",
            font_color="white",
            legend_title_text="Transaction Type"
        )

        st.plotly_chart(fig, width="stretch")

    with chart2:
        st.markdown("#### Transactions by Merchant Category")

        merchant_counts = (
            df["merchant_category"]
            .value_counts()
            .reset_index()
        )

        merchant_counts.columns = ["Merchant Category", "Count"]

        fig = px.bar(
            merchant_counts,
            x="Merchant Category",
            y="Count",
            color="Count",
            color_continuous_scale="Blues"
        )

        fig.update_layout(
            paper_bgcolor="#14243B",
            plot_bgcolor="#14243B",
            font_color="white",
            xaxis_title="Merchant Category",
            yaxis_title="Number of Transactions"
        )

        st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader("Model Performance")

    st.caption(
        "Decision Tree performance on the held-out test dataset."
    )

    m1, m2, m3, m4 = st.columns(4)

    performance = [
        ("Accuracy", "99.95%"),
        ("Precision", "96.77%"),
        ("Recall", "100%"),
        ("F1 Score", "98.36%")
    ]

    for col, (label, value) in zip(
        [m1, m2, m3, m4],
        performance
    ):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.success("Best Performing Model: Decision Tree Classifier")

    st.write("")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Decision Tree Confusion Matrix")

        confusion = [
            [1969, 1],
            [0, 30]
        ]

        fig = px.imshow(
            confusion,
            text_auto=True,
            labels={
                "x": "Predicted",
                "y": "Actual",
                "color": "Transactions"
            },
            x=["Genuine", "Fraud"],
            y=["Genuine", "Fraud"],
            color_continuous_scale="Blues"
        )

        fig.update_layout(
            paper_bgcolor="#14243B",
            plot_bgcolor="#14243B",
            font_color="white"
        )

        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("#### Algorithm Comparison")

        comparison = pd.DataFrame({
            "Algorithm": [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest"
            ],
            "Accuracy": [95.80, 99.95, 99.75],
            "Precision": [25.89, 96.77, 100.00],
            "Recall": [96.67, 100.00, 83.33],
            "F1 Score": [40.85, 98.36, 90.91]
        })

        fig = px.bar(
            comparison,
            x="Algorithm",
            y=[
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ],
            barmode="group",
            color_discrete_sequence=[
                "#19B5FE",
                "#9B59B6",
                "#08BFA5",
                "#F39C12"
            ]
        )

        fig.update_layout(
            paper_bgcolor="#14243B",
            plot_bgcolor="#14243B",
            font_color="white",
            yaxis_title="Score (%)",
            legend_title="Metrics"
        )

        st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader("Model Comparison Summary")

    summary = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ],
        "Accuracy": ["95.80%", "99.95%", "99.75%"],
        "Precision": ["25.89%", "96.77%", "100.00%"],
        "Recall": ["96.67%", "100.00%", "83.33%"],
        "F1 Score": ["40.85%", "98.36%", "90.91%"],
        "ROC-AUC": ["99.33%", "99.97%", "100.00%"]
    })

    st.dataframe(
        summary,
        width="stretch",
        hide_index=True
    )

elif page == "🔍 Live Prediction":

    st.subheader("🔍 Transaction Analysis")

    st.write(
        "Enter transaction details to check whether "
        "the transaction is fraudulent."
    )

    st.markdown("### Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input(
            "Transaction Amount",
            min_value=0.01,
            value=1000.0,
            step=100.0,
            help="Transaction amount must be greater than zero."
        )

        transaction_hour = st.slider(
            "Transaction Hour",
            min_value=0,
            max_value=23,
            value=12
        )

        merchant_category = st.selectbox(
            "Merchant Category",
            options=sorted(
                df["merchant_category"].dropna().unique()
            )
        )

        foreign_transaction = st.selectbox(
            "Foreign Transaction",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with col2:
        st.markdown("#### 📍 Location Verification")

        usual_location = st.text_input(
            "Cardholder's Usual Location",
            value="Ludhiana",
            placeholder="Enter usual city"
        )

        transaction_location = st.text_input(
            "Current Transaction Location",
            value="Ludhiana",
            placeholder="Enter transaction city"
        )

        usual_location_clean = usual_location.strip().casefold()
        transaction_location_clean = transaction_location.strip().casefold()

        if usual_location_clean and transaction_location_clean:

            location_mismatch = int(
                usual_location_clean != transaction_location_clean
            )

            if location_mismatch == 1:
                st.warning(
                    "⚠️ Location Mismatch Detected: "
                    "The transaction location differs from the usual location."
                )
            else:
                st.success(
                    "✅ Location Match: Both locations are the same."
                )

        else:
            location_mismatch = 0
            st.info("Enter both locations to compare them.")

        st.caption(
            "Location is compared using manually entered city names. "
            "This feature does not use GPS tracking."
        )

        device_trust_score = st.slider(
            "Device Trust Score",
            min_value=0.0,
            max_value=1.0,
            value=0.8
        )

        velocity_last_24h = st.number_input(
            "Transactions in Last 24 Hours",
            min_value=0,
            value=2,
            step=1
        )

        cardholder_age = st.number_input(
            "Cardholder Age",
            min_value=18,
            max_value=100,
            value=30
        )

    st.divider()

    live_mode = st.checkbox(
        "Enable Live Prediction",
        value=False,
        help="When enabled, prediction updates automatically when inputs change."
    )

    analyze_clicked = st.button(
        "🔎 Analyze Transaction",
        width="stretch"
    )

    if live_mode or analyze_clicked:

        if not usual_location_clean or not transaction_location_clean:
            st.error(
                "Please enter both locations before analyzing the transaction."
            )

        else:
            input_data = pd.DataFrame([{
                "amount": amount,
                "transaction_hour": transaction_hour,
                "merchant_category": merchant_category,
                "foreign_transaction": foreign_transaction,
                "location_mismatch": location_mismatch,
                "device_trust_score": device_trust_score,
                "velocity_last_24h": velocity_last_24h,
                "cardholder_age": cardholder_age
            }])

            try:
                prediction = model.predict(input_data)[0]

                probabilities = model.predict_proba(input_data)[0]

                fraud_class_index = list(model.classes_).index(1)
                fraud_probability = (
                    float(probabilities[fraud_class_index]) * 100
                )

                st.divider()
                st.subheader("Prediction Result")

                result1, result2 = st.columns(2)

                with result1:
                    if prediction == 1:
                        st.error("⚠️ Potential Fraud Transaction")
                    else:
                        st.success("✅ Genuine Transaction")

                    st.metric(
                        "Fraud Probability",
                        f"{fraud_probability:.2f}%"
                    )

                    if fraud_probability < 30:
                        st.success("Risk Level: LOW")
                    elif fraud_probability < 70:
                        st.warning("Risk Level: MEDIUM")
                    else:
                        st.error("Risk Level: HIGH")

                with result2:
                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=fraud_probability,
                            title={"text": "Fraud Probability (%)"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "#19B5FE"},
                                "steps": [
                                    {"range": [0, 30], "color": "#087F65"},
                                    {"range": [30, 70], "color": "#A66A12"},
                                    {"range": [70, 100], "color": "#A83242"}
                                ]
                            }
                        )
                    )

                    fig.update_layout(
                        paper_bgcolor="#14243B",
                        font_color="white"
                    )

                    st.plotly_chart(fig, width="stretch")

                st.caption(
                    "Predictions are machine learning estimates "
                    "and should not be treated as a confirmed fraud determination."
                )

            except Exception as e:
                st.error(f"Prediction could not be completed: {e}")

    else:
        st.info(
            "Enter the transaction details and click Analyze Transaction, "
            "or enable Live Prediction."
        )

elif page == "📘 About Project":

    st.subheader("📘 About FraudShield")

    st.write(
        "FraudShield is a machine learning based credit card "
        "fraud detection project. It analyzes transaction details "
        "and predicts whether a transaction may be fraudulent or genuine."
    )

    st.divider()

    st.markdown("### Project Objectives")

    st.write("- Detect potentially fraudulent transactions.")
    st.write("- Analyze transaction-related information.")
    st.write("- Apply machine learning classification algorithms.")
    st.write("- Compare model performance using classification metrics.")
    st.write("- Display prediction results through an interactive dashboard.")

    st.markdown("### Dataset Information")

    st.write(f"- Total records: {len(df):,}")
    st.write(f"- Total columns: {len(df.columns)}")
    st.write("- Target column: is_fraud")
    st.write("- Fraud class: 1")
    st.write("- Genuine class: 0")

    st.markdown("### Machine Learning Algorithms")

    st.write("1. Logistic Regression")
    st.write("2. Decision Tree Classifier")
    st.write("3. Random Forest Classifier")

    st.success("Best Performing Model: Decision Tree Classifier")

    st.markdown("### Technologies Used")

    st.write("- Python")
    st.write("- Pandas")
    st.write("- Scikit-learn")
    st.write("- Streamlit")
    st.write("- Plotly")

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
