import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── 1. Page Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 2. Load Model & Threshold ────────────────────────────────────────────
MODEL_PATH = os.path.join("models", "churn_pipeline.joblib")
THRESHOLD_PATH = os.path.join("models", "churn_threshold.joblib")
METADATA_PATH = os.path.join("models", "model_metadata.joblib")

@st.cache_resource
def load_model_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(THRESHOLD_PATH):
        return None, 0.38, None
    try:
        pipeline = joblib.load(MODEL_PATH)
        threshold = float(joblib.load(THRESHOLD_PATH))
        metadata = joblib.load(METADATA_PATH) if os.path.exists(METADATA_PATH) else None
        return pipeline, threshold, metadata
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, 0.38, None

pipeline, threshold, metadata = load_model_artifacts()

# ── 3. Application Header ────────────────────────────────────────────────
st.title("📡 Telco Customer Churn Risk Predictor")
st.markdown(
    "Predict the probability that an individual customer will cancel their subscription and "
    "identify the key operational drivers influencing their retention risk."
)

if pipeline is None:
    st.warning(
        "⚠️ Model artifacts not found in `models/`. Please run `notebooks/churn.ipynb` "
        "or execute the training script to serialize the tuned model pipeline."
    )

st.markdown("---")

# ── 4. Input Form (Sliders & Dropdowns) ──────────────────────────────────
st.subheader("📋 Enter Customer Account & Service Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📄 Contract & Billing")
    contract = st.selectbox(
        "Contract Type",
        options=["Month-to-month", "One year", "Two year"],
        index=0,
        help="Month-to-month contracts have historically exhibited the highest churn rates."
    )
    tenure = st.slider(
        "Tenure (Months with Company)",
        min_value=0,
        max_value=72,
        value=12,
        help="Number of continuous months the customer has been with the provider."
    )
    monthly_charges = st.slider(
        "Monthly Charges ($)",
        min_value=18.0,
        max_value=120.0,
        value=70.0,
        step=0.5,
        help="The amount charged to the customer each billing cycle."
    )
    # Default total charges calculated as tenure * monthly_charges
    auto_total = round(float(tenure * monthly_charges), 2)
    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=auto_total,
        step=10.0,
        help="Cumulative historical charges billed to the customer so far."
    )
    payment_method = st.selectbox(
        "Payment Method",
        options=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ],
        index=0,
        help="Electronic check users experience significantly higher churn frequency."
    )
    paperless_billing = st.selectbox(
        "Paperless Billing",
        options=["Yes", "No"],
        index=0
    )

with col2:
    st.markdown("#### 🌐 Internet & Core Services")
    internet_service = st.selectbox(
        "Internet Service Provider",
        options=["Fiber optic", "DSL", "No"],
        index=0,
        help="Fiber optic subscribers exhibit elevated churn (~42%) vs DSL (~19%)."
    )
    phone_service = st.selectbox(
        "Phone Service",
        options=["Yes", "No"],
        index=0
    )
    multiple_lines_options = ["No", "Yes"] if phone_service == "Yes" else ["No phone service"]
    multiple_lines = st.selectbox(
        "Multiple Lines",
        options=multiple_lines_options,
        index=0
    )
    online_security_options = ["No", "Yes"] if internet_service != "No" else ["No internet service"]
    online_security = st.selectbox(
        "Online Security",
        options=online_security_options,
        index=0
    )
    online_backup_options = ["No", "Yes"] if internet_service != "No" else ["No internet service"]
    online_backup = st.selectbox(
        "Online Backup",
        options=online_backup_options,
        index=0
    )
    device_protection_options = ["No", "Yes"] if internet_service != "No" else ["No internet service"]
    device_protection = st.selectbox(
        "Device Protection",
        options=device_protection_options,
        index=0
    )

with col3:
    st.markdown("#### 📺 Entertainment & Demographics")
    tech_support_options = ["No", "Yes"] if internet_service != "No" else ["No internet service"]
    tech_support = st.selectbox(
        "Tech Support",
        options=tech_support_options,
        index=0
    )
    streaming_tv_options = ["No", "Yes"] if internet_service != "No" else ["No internet service"]
    streaming_tv = st.selectbox(
        "Streaming TV",
        options=streaming_tv_options,
        index=0
    )
    streaming_movies_options = ["No", "Yes"] if internet_service != "No" else ["No internet service"]
    streaming_movies = st.selectbox(
        "Streaming Movies",
        options=streaming_movies_options,
        index=0
    )
    gender = st.selectbox("Gender", options=["Female", "Male"], index=0)
    senior_citizen_label = st.selectbox("Senior Citizen", options=["No (0)", "Yes (1)"], index=0)
    senior_citizen = 1 if "Yes" in senior_citizen_label else 0
    partner = st.selectbox("Partner", options=["No", "Yes"], index=0)
    dependents = st.selectbox("Dependents", options=["No", "Yes"], index=0)

# ── 5. Assemble Input Data exactly matching training features ───────────
input_data = {
    "gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [int(tenure)],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [float(monthly_charges)],
    "TotalCharges": [float(total_charges)],
}

input_df = pd.DataFrame(input_data)

st.markdown("---")

# ── 6. Prediction & Factor Attribution ──────────────────────────────────
st.subheader("🎯 Prediction & Risk Assessment")

if st.button("🚀 Analyze Churn Risk", type="primary", use_container_width=True):
    if pipeline is not None:
        try:
            # Predict probability
            prob = float(pipeline.predict_proba(input_df)[0, 1])
            is_high_risk = prob >= threshold

            # KPI Display
            res_col1, res_col2, res_col3 = st.columns(3)

            with res_col1:
                st.metric(
                    label="Predicted Churn Probability",
                    value=f"{prob:.1%}",
                    delta=f"{prob - threshold:+.1%} vs Threshold",
                    delta_color="inverse"
                )

            with res_col2:
                if is_high_risk:
                    st.error("### 🔴 HIGH CHURN RISK")
                    st.caption(f"Exceeds optimal decision threshold ({threshold:.1%})")
                else:
                    st.success("### 🟢 LOW CHURN RISK")
                    st.caption(f"Below optimal decision threshold ({threshold:.1%})")

            with res_col3:
                st.metric(
                    label="Decision Threshold (F2-Tuned)",
                    value=f"{threshold:.1%}",
                    help="Calibrated to optimize recall for churners while maintaining acceptable precision."
                )

            st.progress(min(prob, 1.0))

            # ── Factor Attribution ──
            st.markdown("### 🔍 Key Factors Driving This Prediction")

            risk_drivers = []
            protective_factors = []

            # 1. Contract
            if contract == "Month-to-month":
                risk_drivers.append("**Month-to-month Contract**: Lacks long-term commitment; major churn driver.")
            else:
                protective_factors.append(f"**{contract} Contract**: Long-term contractual lock-in strongly protects retention.")

            # 2. Tenure
            if tenure <= 6:
                risk_drivers.append(f"**Early Tenure ({tenure} months)**: New accounts in months 1–6 face peak customer defection hazard.")
            elif tenure >= 24:
                protective_factors.append(f"**Mature Tenure ({tenure} months)**: Account is well-established past the 2-year mark.")

            # 3. Internet Service
            if internet_service == "Fiber optic":
                risk_drivers.append("**Fiber Optic Internet**: Premium service cohort with elevated churn rate (~42%).")
            elif internet_service == "DSL":
                protective_factors.append("**DSL Internet**: More price-stable with lower historical churn (~19%).")

            # 4. Payment Method
            if payment_method == "Electronic check":
                risk_drivers.append("**Electronic Check**: High-friction manual payment method with 45.3% churn rate.")
            elif "automatic" in payment_method:
                protective_factors.append(f"**Auto-Pay ({payment_method.split(' ')[0]})**: Seamless automated billing reduces friction.")

            # 5. Monthly Charges
            if monthly_charges >= 75.0:
                risk_drivers.append(f"**High Monthly Charges (${monthly_charges:.2f})**: Above-average billing heightens price sensitivity.")
            elif monthly_charges <= 45.0:
                protective_factors.append(f"**Low Monthly Charges (${monthly_charges:.2f})**: Budget-friendly plan minimizes bill shock.")

            # 6. Tech Support / Online Security
            if internet_service != "No" and tech_support == "No":
                risk_drivers.append("**No Tech Support**: Absence of dedicated support increases defection during technical friction.")
            if internet_service != "No" and online_security == "No":
                risk_drivers.append("**No Online Security**: Missing bundled security reduces perceived value.")

            fact_col1, fact_col2 = st.columns(2)

            with fact_col1:
                st.markdown("#### ⚠️ Risk-Elevating Factors")
                if risk_drivers:
                    for d in risk_drivers:
                        st.markdown(f"- {d}")
                else:
                    st.markdown("_No major high-risk indicators detected for this profile._")

            with fact_col2:
                st.markdown("#### 🛡️ Retention-Protecting Factors")
                if protective_factors:
                    for p in protective_factors:
                        st.markdown(f"- {p}")
                else:
                    st.markdown("_No significant protective buffers detected for this profile._")

            # ── Actionable Retention Prescription ──
            st.markdown("#### 💡 Recommended Next Action")
            if is_high_risk:
                if contract == "Month-to-month":
                    st.info(
                        "**Immediate Action**: Offer a 10%–15% annual contract discount or a complimentary service add-on "
                        "(e.g., free speed boost or Tech Support) in exchange for switching to a 1-Year agreement."
                    )
                elif payment_method == "Electronic check":
                    st.info(
                        "**Immediate Action**: Prompt the customer to switch to automated Credit Card or Bank Transfer billing "
                        "with a one-time $10 bill credit."
                    )
                else:
                    st.info(
                        "**Immediate Action**: Schedule a proactive check-in call with customer success to review service quality."
                    )
            else:
                st.success(
                    "Customer appears stable and content. Continue monitoring normal usage patterns and maintain regular loyalty engagement."
                )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.error("Model pipeline is not loaded. Please ensure models are trained and saved in `models/`.")

# ── Sidebar Info ────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/telecommunication.png", width=70)
    st.markdown("### Model Information")
    st.markdown("- **Algorithm**: Tuned XGBoost Classifier")
    st.markdown("- **Imbalance Handling**: `scale_pos_weight`")
    st.markdown(f"- **Operating Threshold**: `{threshold:.1%}`")
    st.markdown("- **Optimization Target**: Recall-prioritized ($F_2$-score)")
    st.markdown("---")
    st.markdown("### Features Used (19 total)")
    st.caption("3 Numerical: tenure, MonthlyCharges, TotalCharges")
    st.caption("16 Categorical: Contract, InternetService, PaymentMethod, Services, Demographics")
