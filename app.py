
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="FraudShield | Credit Card Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# LOAD MODEL AND DATASET
# ==================================================
model = joblib.load("models/fraud_model.pkl")
data = pd.read_csv("data/credit_card_fraud_10k.csv")

total = len(data)
fraud = int(data["is_fraud"].sum())
genuine = total - fraud
fraud_rate = (fraud / total) * 100

# ==================================================
# CUSTOM CSS
# ==================================================
st.markdown("""
<style>
.stApp {
    background: #08111f;
    color: #e8f0fc;
}

[data-testid="stSidebar"] {
    background: #0e1b2d;
    border-right: 1px solid #263b56;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #f1f6ff !important;
}

p, label {
    color: #b4c5d9;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #152943, #102035);
    border: 1px solid #29415e;
    border-radius: 15px;
    padding: 20px;
}

div[data-testid="stMetricLabel"] {
    color: #a9bdd5;
}

div[data-testid="stMetricValue"] {
    color: #40c8ee;
}

div[data-testid="stForm"] {
    background: #11233a;
    border: 1px solid #29415e;
    border-radius: 16px;
    padding: 22px;
}

.stButton button, .stFormSubmitButton button {
    background: linear-gradient(90deg, #087eae, #315eea);
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 45px;
    font-weight: 600;
}

hr {
    border-color: #263b56;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("# 🛡️ FraudShield")
    st.caption("SMART TRANSACTION SECURITY")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Fraud Detection", "About Project"]
    )

    st.divider()

    st.caption("MACHINE LEARNING POWERED")
    st.caption("Credit Card Security System")
    st.caption("Version 1.0")

# ==================================================
# MAIN HEADER
# ==================================================
st.markdown("# 🛡️ FraudShield")
st.markdown("### Credit Card Fraud Intelligence Platform")

st.caption(
    "Monitor transactions. Detect suspicious patterns. "
    "Protect financial activity."
)

st.divider()

# ==================================================
# DASHBOARD PAGE
# ==================================================
if page == "Dashboard":

    st.subheader("📊 System Overview")

    # KPI CARDS
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Fraud Detected", f"{fraud:,}")
    c3.metric("Genuine Transactions", f"{genuine:,}")
    c4.metric("Fraud Rate", f"{fraud_rate:.2f}%")

    st.write("")
    st.divider()

    # TRANSACTION ANALYTICS
    st.subheader("Transaction Analytics")

    left, right = st.columns(2, gap="large")

    # DONUT CHART
    with left:

        st.markdown("#### Transaction Distribution")

        donut = go.Figure(
            data=[
                go.Pie(
                    labels=["Genuine", "Fraud"],
                    values=[genuine, fraud],
                    hole=0.72,
                    marker=dict(
                        colors=["#22c7a9", "#f05b78"],
                        line=dict(color="#08111f", width=3)
                    ),
                    textinfo="percent",
                    hovertemplate=(
                        "%{label}: %{value:,} transactions"
                        "<extra></extra>"
                    )
                )
            ]
        )

        donut.update_layout(
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8f0fc"),
            margin=dict(t=25, b=35, l=20, r=20),
            legend=dict(orientation="h", y=-0.08),
            showlegend=True
        )

        st.plotly_chart(
            donut,
            use_container_width=True,
            config={"displayModeBar": False},
            key="transaction_donut"
        )

    # MERCHANT CATEGORY CHART
    with right:

        st.markdown("#### Merchant Category Analysis")

        merchant_counts = (
            data["merchant_category"]
            .value_counts()
            .sort_values()
        )

        merchant_chart = go.Figure(
            go.Bar(
                x=merchant_counts.values,
                y=merchant_counts.index,
                orientation="h",
                marker=dict(
                    color=merchant_counts.values,
                    colorscale=[
                        [0, "#315eea"],
                        [1, "#31c6db"]
                    ]
                ),
                text=merchant_counts.values,
                textposition="outside",
                hovertemplate=(
                    "%{y}: %{x:,} transactions"
                    "<extra></extra>"
                )
            )
        )

        merchant_chart.update_layout(
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8f0fc"),
            margin=dict(t=25, b=35, l=20, r=45),
            xaxis=dict(
                title="Number of Transactions",
                showgrid=True,
                gridcolor="#263b56",
                zeroline=False
            ),
            yaxis=dict(
                title="",
                showgrid=False
            ),
            showlegend=False
        )

        st.plotly_chart(
            merchant_chart,
            use_container_width=True,
            config={"displayModeBar": False},
            key="merchant_chart"
        )

    # MODEL PERFORMANCE
    st.divider()
    st.subheader("🤖 Model Performance")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Accuracy", "99.75%")
    m2.metric("Precision", "100%")
    m3.metric("Recall", "83.33%")
    m4.metric("F1 Score", "90.91%")

    st.caption(
        "Random Forest | Held-out test set | 2,000 transactions"
    )

    # CONFUSION MATRIX AND MODEL COMPARISON
    st.divider()
    st.subheader("📈 Model Evaluation")

    left, right = st.columns(2, gap="large")

    # CONFUSION MATRIX
    with left:

        st.markdown("#### Random Forest Confusion Matrix")

        matrix = go.Figure(
            data=go.Heatmap(
                z=[
                    [1970, 0],
                    [5, 25]
                ],
                x=[
                    "Predicted Genuine",
                    "Predicted Fraud"
                ],
                y=[
                    "Actual Genuine",
                    "Actual Fraud"
                ],
                colorscale=[
                    [0, "#172b43"],
                    [1, "#17a589"]
                ],
                text=[
                    [1970, 0],
                    [5, 25]
                ],
                texttemplate="%{text}",
                showscale=False
            )
        )

        matrix.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8f0fc"),
            margin=dict(t=25, b=65, l=20, r=20)
        )

        st.plotly_chart(
            matrix,
            use_container_width=True,
            config={"displayModeBar": False},
            key="confusion_matrix"
        )

        st.caption(
            "TP: 25 | TN: 1970 | FP: 0 | FN: 5"
        )

    # MODEL COMPARISON
    with right:

        st.markdown("#### Algorithm Comparison")

        comparison = go.Figure()

        comparison.add_trace(
            go.Bar(
                name="Logistic Regression",
                x=["Accuracy", "Precision", "Recall", "F1 Score"],
                y=[95.80, 25.89, 96.67, 40.85],
                marker_color="#818cf8"
            )
        )

        comparison.add_trace(
            go.Bar(
                name="Random Forest",
                x=["Accuracy", "Precision", "Recall", "F1 Score"],
                y=[99.75, 100, 83.33, 90.91],
                marker_color="#22c7a9"
            )
        )

        comparison.update_layout(
            barmode="group",
            height=350,
            yaxis=dict(
                range=[0, 110],
                title="Score (%)",
                gridcolor="#263b56"
            ),
            xaxis=dict(
                showgrid=False
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8f0fc"),
            legend=dict(
                orientation="h",
                y=-0.28
            ),
            margin=dict(t=25, b=80, l=20, r=20)
        )

        st.plotly_chart(
            comparison,
            use_container_width=True,
            config={"displayModeBar": False},
            key="model_comparison"
        )

    st.caption(
        "Comparison metrics are from the recorded held-out test results."
    )

# ==================================================
# FRAUD DETECTION PAGE
# ==================================================
elif page == "Fraud Detection":

    st.subheader("🔍 Transaction Risk Analysis")

    st.caption(
        "Enter transaction information to generate a machine learning prediction."
    )

    with st.form("fraud_form"):

        col1, col2 = st.columns(2, gap="large")

        with col1:

            amount = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=500.0
            )

            hour = st.slider(
                "Transaction Hour",
                0, 23, 12
            )

            merchant = st.selectbox(
                "Merchant Category",
                [
                    "Food",
                    "Clothing",
                    "Travel",
                    "Grocery",
                    "Electronics"
                ]
            )

            foreign = st.selectbox(
                "Foreign Transaction",
                ["No", "Yes"]
            )

        with col2:

            mismatch = st.selectbox(
                "Location Mismatch",
                ["No", "Yes"]
            )

            trust = st.slider(
                "Device Trust Score",
                25, 99, 75
            )

            velocity = st.slider(
                "Transactions in Last 24 Hours",
                0, 9, 2
            )

            age = st.number_input(
                "Cardholder Age",
                min_value=18,
                max_value=69,
                value=25
            )

        submitted = st.form_submit_button(
            "🔎 Analyze Transaction"
        )

    if submitted:

        input_data = pd.DataFrame([{
            "amount": amount,
            "transaction_hour": hour,
            "merchant_category": merchant,
            "foreign_transaction": int(foreign == "Yes"),
            "location_mismatch": int(mismatch == "Yes"),
            "device_trust_score": trust,
            "velocity_last_24h": velocity,
            "cardholder_age": age
        }])

        prediction = model.predict(input_data)[0]

        probability = float(
            model.predict_proba(input_data)[0][1]
        )

        st.divider()
        st.subheader("Prediction Result")

        result_col, gauge_col = st.columns(2)

        with result_col:

            if prediction == 1:

                st.error("🚨 Potential Fraud Detected")

                st.write(
                    "The model classified this transaction as potentially fraudulent."
                )

            else:

                st.success("✅ Genuine Transaction")

                st.write(
                    "The model classified this transaction as genuine."
                )

            st.metric(
                "Fraud Probability",
                f"{probability * 100:.2f}%"
            )

        # FRAUD PROBABILITY GAUGE
        with gauge_col:

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%"},
                    title={"text": "Fraud Risk Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#31c6db"},
                        "steps": [
                            {
                                "range": [0, 50],
                                "color": "#173a43"
                            },
                            {
                                "range": [50, 100],
                                "color": "#493047"
                            }
                        ]
                    }
                )
            )

            gauge.update_layout(
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8f0fc"),
                margin=dict(t=45, b=15, l=20, r=20)
            )

            st.plotly_chart(
                gauge,
                use_container_width=True,
                config={"displayModeBar": False}
            )

        st.caption(
            "This model prediction is an analytical aid, "
            "not a final financial decision."
        )

# ==================================================
# ABOUT PROJECT PAGE
# ==================================================
elif page == "About Project":

    st.subheader("📌 About FraudShield")

    st.write("""
    FraudShield is a machine learning-based credit card fraud detection
    application designed to identify potentially suspicious transactions.

    It analyzes transaction attributes such as transaction amount,
    merchant category, device trust score, foreign transaction status,
    location mismatch and transaction frequency.
    """)

    st.divider()

    st.markdown("### Technology Stack")

    tech1, tech2, tech3 = st.columns(3)

    tech1.info("Python\n\nPandas\n\nNumPy")
    tech2.info("Scikit-learn\n\nRandom Forest\n\nLogistic Regression")
    tech3.info("Streamlit\n\nPlotly\n\nJoblib")

    st.divider()

    st.markdown("### Dataset Summary")

    d1, d2, d3 = st.columns(3)

    d1.metric("Total Records", f"{total:,}")
    d2.metric("Fraud Records", f"{fraud:,}")
    d3.metric("Genuine Records", f"{genuine:,}")

    st.markdown("### Project Objective")

    st.write(
        "To demonstrate how machine learning techniques can help "
        "identify potentially fraudulent credit card transactions."
    )

# ==================================================
# FOOTER
# ==================================================
st.divider()

st.caption(
    "🛡️ FraudShield | Credit Card Fraud Detection | Academic ML Project"
)
