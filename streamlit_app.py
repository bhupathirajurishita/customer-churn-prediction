
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(
    page_title="ChurnGuard",
    page_icon="📊",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: #0b0d0f;
    color: #f5f5f5;
}
.block-container {
    max-width: 900px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
}
.subtitle {
    text-align: center;
    color: #9aa3ad;
    font-size: 16px;
    margin-bottom: 30px;
}
.section {
    color: #21c78a;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.5px;
    margin-top: 25px;
    margin-bottom: 10px;
}
.result-card {
    background: #12161a;
    border: 1px solid #293139;
    border-radius: 16px;
    padding: 25px;
    margin-top: 25px;
    text-align: center;
}
.result-title {
    font-size: 28px;
    font-weight: 800;
}
.result-subtitle {
    color: #9aa3ad;
    margin-top: 8px;
}
.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 10px;
    font-weight: 700;
    border: 1px solid #293139;
}
.stButton > button:hover {
    border-color: #21c78a;
}
.footer {
    text-align: center;
    color: #68727c;
    font-size: 12px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# IBM Telco Customer Churn sample dataset.
# The local CSV is used when available; the public IBM copy is the fallback
# so the deployed app does not require the large CSV in GitHub.
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)

LOCAL_DATA = Path("churn_preprocessed(20260924-104145).csv")


@st.cache_data
def load_data():
    if LOCAL_DATA.exists():
        df = pd.read_csv(LOCAL_DATA)
    else:
        df = pd.read_csv(DATA_URL)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


@st.cache_resource
def train_model(df):
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]

    categorical = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical
            ),
            (
                "numerical",
                "passthrough",
                numerical
            )
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Small grid keeps Streamlit startup practical while still using
    # systematic hyperparameter tuning with 5-fold cross-validation.
    params = {
        "model__n_estimators": [200],
        "model__max_depth": [12],
        "model__min_samples_split": [5]
    }

    search = GridSearchCV(
        pipeline,
        params,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    return search.best_estimator_


with st.spinner("📊 Preparing the churn prediction model..."):
    df = load_data()
    model = train_model(df)

st.markdown(
    '<div class="title">📊 CHURNGUARD</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Enter customer details and estimate churn risk.</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section">CUSTOMER PROFILE</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

with col2:
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )
    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

st.markdown(
    '<div class="section">SERVICES</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )
    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

with col2:
    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

with col3:
    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )
    monthly = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=0.01
    )
    total = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=840.0,
        step=0.01
    )

st.write("")

if st.button("✨ Predict Churn Risk", use_container_width=True):

    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    probability = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        title = "⚠️ Higher Churn Risk"
        message = "The model predicts this customer is likely to churn."
    else:
        title = "✅ Lower Churn Risk"
        message = "The model predicts this customer is likely to stay."

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-title">{title}</div>
            <div class="result-subtitle">{message}</div>
            <div style="
                font-size:26px;
                font-weight:800;
                margin-top:18px;
                color:#21c78a;
            ">
                Churn probability: {probability * 100:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div class="footer">
        Built with Python · Pandas · Scikit-learn · Random Forest · Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
