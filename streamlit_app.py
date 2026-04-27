import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans

st.set_page_config(page_title="Health Prediction Dashboard", layout="wide")

# =========================================================
# helper functions
# =========================================================
def get_bmi(weight, height_cm):
    h = height_cm / 100
    if h <= 0:
        return 0
    return round(weight / (h * h), 2)

def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    return "Obese"

def ideal_weight_range(height_cm):
    h = height_cm / 100
    low = 18.5 * (h * h)
    high = 24.9 * (h * h)
    return round(low, 1), round(high, 1)

def cluster_meaning(cluster_id):
    mapping = {
        0: "Lower-risk profile group",
        1: "Moderate profile group",
        2: "Higher-risk / higher-cost profile group"
    }
    return mapping.get(cluster_id, f"Cluster {cluster_id}")

def build_recommendations(bmi_text, diabetes_pred, heart_pred, smoker):
    recs = []

    if bmi_text == "Underweight":
        recs.append("Balanced diet aur protein-rich meals increase karo.")
    elif bmi_text == "Normal":
        recs.append("Current lifestyle maintain rakho aur regular activity continue karo.")
    elif bmi_text == "Overweight":
        recs.append("Daily walk, sugar control aur portion control useful rahega.")
    else:
        recs.append("Weight management aur routine health checkup par focus karo.")

    if diabetes_pred == 1:
        recs.append("Glucose monitoring, low-sugar diet aur physical activity improve karo.")
    else:
        recs.append("Diabetes risk low hai, fir bhi healthy food routine maintain rakho.")

    if heart_pred == 1:
        recs.append("Salt kam lo, stress manage karo, aur heart checkups consider karo.")
    else:
        recs.append("Heart risk low dikhta hai, healthy habits continue rakho.")

    if smoker == "yes":
        recs.append("Smoking health risk aur insurance cost dono badhati hai.")
    else:
        recs.append("Non-smoker hona positive factor hai for health and insurance profile.")

    return recs

# =========================================================
# load and train models
# =========================================================

# ---------------- diabetes ----------------
diabetes_df = pd.read_csv("diabetes.csv")
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in zero_cols:
    diabetes_df[col] = diabetes_df[col].replace(0, np.nan)
    diabetes_df[col] = diabetes_df[col].fillna(diabetes_df[col].median())

X_diabetes = diabetes_df.drop("Outcome", axis=1)
y_diabetes = diabetes_df["Outcome"]

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_diabetes, y_diabetes, test_size=0.2, random_state=42, stratify=y_diabetes
)

diabetes_scaler = StandardScaler()
X_train_d_scaled = diabetes_scaler.fit_transform(X_train_d)
X_test_d_scaled = diabetes_scaler.transform(X_test_d)

diabetes_model = LogisticRegression(max_iter=5000)
diabetes_model.fit(X_train_d_scaled, y_train_d)
diabetes_accuracy = round(accuracy_score(y_test_d, diabetes_model.predict(X_test_d_scaled)) * 100, 2)

# ---------------- heart ----------------
heart_df = pd.read_csv("heart_disease.csv")
heart_df = heart_df.replace("?", np.nan)

for col in heart_df.columns:
    heart_df[col] = pd.to_numeric(heart_df[col], errors="coerce")

heart_df["target"] = heart_df["num"].apply(lambda x: 0 if x == 0 else 1)

drop_cols = [col for col in ["id", "num", "dataset"] if col in heart_df.columns]
heart_df = heart_df.drop(columns=drop_cols)

X_heart = heart_df.drop("target", axis=1)
y_heart = heart_df["target"]

for col in X_heart.columns:
    X_heart[col] = pd.to_numeric(X_heart[col], errors="coerce")

heart_num_cols = X_heart.columns.tolist()

heart_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), heart_num_cols)
])

heart_model = Pipeline(steps=[
    ("preprocessor", heart_preprocessor),
    ("classifier", LogisticRegression(max_iter=5000))
])

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_heart, y_heart, test_size=0.2, random_state=42, stratify=y_heart
)

heart_model.fit(X_train_h, y_train_h)
heart_accuracy = round(accuracy_score(y_test_h, heart_model.predict(X_test_h)) * 100, 2)

# ---------------- insurance ----------------
insurance_df = pd.read_csv("height_weight.csv")

X_insurance = insurance_df.drop("charges", axis=1)
y_insurance = insurance_df["charges"]

insurance_cat_cols = X_insurance.select_dtypes(include=["object"]).columns.tolist()
insurance_num_cols = X_insurance.select_dtypes(include=["int64", "float64"]).columns.tolist()

insurance_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), insurance_num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ]), insurance_cat_cols)
])

insurance_model = Pipeline(steps=[
    ("preprocessor", insurance_preprocessor),
    ("regressor", LinearRegression())
])

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_insurance, y_insurance, test_size=0.2, random_state=42
)

insurance_model.fit(X_train_i, y_train_i)
insurance_test_pred = insurance_model.predict(X_test_i)
insurance_r2 = round(r2_score(y_test_i, insurance_test_pred), 4)
insurance_mae = round(mean_absolute_error(y_test_i, insurance_test_pred), 2)
insurance_rmse = round(np.sqrt(mean_squared_error(y_test_i, insurance_test_pred)), 2)

# ---------------- clustering ----------------
cluster_df = insurance_df[["age", "sex", "bmi", "children", "smoker", "region"]].copy()

cluster_cat_cols = ["sex", "smoker", "region"]
cluster_num_cols = ["age", "bmi", "children"]

cluster_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), cluster_num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ]), cluster_cat_cols)
])

X_cluster_processed = cluster_preprocessor.fit_transform(cluster_df)
kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_model.fit(X_cluster_processed)

# =========================================================
# session state defaults
# =========================================================
defaults = {
    "age": 22,
    "sex": "male",
    "height": 170.0,
    "weight": 70.0,
    "children": 0,
    "smoker": "no",
    "region": "northwest",
    "pregnancies": 0,
    "glucose": 120.0,
    "bloodpressure": 80.0,
    "skinthickness": 20.0,
    "insulin": 85.0,
    "dpf": 0.5,
    "cp": 1.0,
    "trestbps": 120.0,
    "chol": 200.0,
    "fbs": 0.0,
    "restecg": 1.0,
    "thalch": 150.0,
    "exang": 0.0,
    "oldpeak": 1.2,
    "slope": 2.0,
    "ca": 0.0,
    "thal": 0.0
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

def fill_demo():
    for key, value in defaults.items():
        st.session_state[key] = value

# =========================================================
# UI styling
# =========================================================
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
[data-testid="stMetric"] {
    background: #f7faff;
    border: 1px solid #dbe6f5;
    padding: 12px;
    border-radius: 14px;
}
.demo-box {
    background: #f8fbff;
    border: 1px solid #d9e5f4;
    padding: 14px;
    border-radius: 14px;
    margin-bottom: 14px;
}
.title-box {
    background: linear-gradient(135deg, #eef4ff, #f8fbff);
    border: 1px solid #dbe6f5;
    padding: 18px;
    border-radius: 18px;
    margin-bottom: 18px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-box">
    <h1 style="margin-bottom:8px;">Smart Health Prediction Dashboard</h1>
    <p style="margin:0; color:#55606f;">
        BMI, Diabetes, Heart Disease, Insurance aur Clustering output ek hi jagah.
    </p>
</div>
""", unsafe_allow_html=True)

with st.container():
    col_a, col_b = st.columns([4, 1])
    with col_a:
        st.markdown("""
        <div class="demo-box">
            <b>Demo values:</b> Age 22, Male, Height 170, Weight 70, Glucose 120, BP 80, Chol 200, Thalch 150, Smoker no, Region northwest
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.button("Fill Demo Inputs", on_click=fill_demo, use_container_width=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Basic Details")
    age = st.number_input("Age", step=1, key="age")
    sex = st.selectbox("Gender", ["male", "female"], key="sex")
    height = st.number_input("Height (cm)", key="height")
    weight = st.number_input("Weight (kg)", key="weight")
    children = st.number_input("Children", step=1, key="children")
    smoker = st.selectbox("Smoker", ["no", "yes"], key="smoker")
    region = st.selectbox("Region", ["northwest", "northeast", "southwest", "southeast"], key="region")

with c2:
    st.subheader("Diabetes Inputs")
    pregnancies = st.number_input("Pregnancies", step=1, key="pregnancies")
    glucose = st.number_input("Glucose", key="glucose")
    bloodpressure = st.number_input("Blood Pressure", key="bloodpressure")
    skinthickness = st.number_input("Skin Thickness", key="skinthickness")
    insulin = st.number_input("Insulin", key="insulin")
    dpf = st.number_input("DPF", key="dpf")

with c3:
    st.subheader("Heart Inputs")
    cp = st.number_input("CP", key="cp")
    trestbps = st.number_input("Trestbps", key="trestbps")
    chol = st.number_input("Chol", key="chol")
    fbs = st.number_input("FBS", key="fbs")
    restecg = st.number_input("Restecg", key="restecg")
    thalch = st.number_input("Thalch", key="thalch")
    exang = st.number_input("Exang", key="exang")
    oldpeak = st.number_input("Oldpeak", key="oldpeak")
    slope = st.number_input("Slope", key="slope")
    ca = st.number_input("CA", key="ca")
    thal = st.number_input("Thal", key="thal")

if st.button("Generate Full Output", use_container_width=True):
    try:
        bmi = get_bmi(weight, height)
        bmi_text = bmi_category(bmi)
        ideal_low, ideal_high = ideal_weight_range(height)

        diabetes_input = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bloodpressure,
            "SkinThickness": skinthickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }])

        diabetes_scaled = diabetes_scaler.transform(diabetes_input)
        diabetes_pred = int(diabetes_model.predict(diabetes_scaled)[0])
        diabetes_prob = float(diabetes_model.predict_proba(diabetes_scaled)[0][1])
        diabetes_result = "Diabetes Risk" if diabetes_pred == 1 else "No Diabetes Risk"

        heart_input = pd.DataFrame([{
            "age": age,
            "sex": 1 if sex == "male" else 0,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalch": thalch,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }])

        for col in heart_input.columns:
            heart_input[col] = pd.to_numeric(heart_input[col], errors="coerce")

        heart_pred = int(heart_model.predict(heart_input)[0])
        heart_prob = float(heart_model.predict_proba(heart_input)[0][1])
        heart_result = "Heart Disease Risk" if heart_pred == 1 else "No Heart Disease Risk"

        insurance_input = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }])

        insurance_pred = round(float(insurance_model.predict(insurance_input)[0]), 2)

        cluster_input = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }])

        cluster_processed = cluster_preprocessor.transform(cluster_input)
        cluster_id = int(kmeans_model.predict(cluster_processed)[0])
        cluster_text = cluster_meaning(cluster_id)

        recommendations = build_recommendations(bmi_text, diabetes_pred, heart_pred, smoker)

        summary = (
            f"Your BMI is {bmi}, which falls in the {bmi_text} category. "
            f"Diabetes result is {diabetes_result}. "
            f"Heart disease result is {heart_result}. "
            f"Estimated insurance charges are ₹ {insurance_pred}. "
            f"Your profile belongs to {cluster_text}."
        )

        st.success("Output generated successfully")

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("BMI", bmi, bmi_text)
        r2.metric("Diabetes", diabetes_result, f"{round(diabetes_prob * 100, 2)}%")
        r3.metric("Heart", heart_result, f"{round(heart_prob * 100, 2)}%")
        r4.metric("Insurance", f"₹ {insurance_pred}")
        r5.metric("Cluster", cluster_id, cluster_text)

        st.subheader("Ideal Weight Range")
        st.write(f"{ideal_low} kg - {ideal_high} kg")

        st.subheader("Recommendations")
        for item in recommendations:
            st.write("- " + item)

        st.subheader("Summary")
        st.write(summary)

        report_text = f"""
HEALTH PREDICTION REPORT
========================================
BMI: {bmi}
BMI Category: {bmi_text}
Ideal Weight Range: {ideal_low} kg - {ideal_high} kg

Diabetes Prediction: {diabetes_result}
Diabetes Confidence: {round(diabetes_prob * 100, 2)}%

Heart Prediction: {heart_result}
Heart Confidence: {round(heart_prob * 100, 2)}%

Insurance Charges: ₹ {insurance_pred}

Cluster: {cluster_id} - {cluster_text}

RECOMMENDATIONS:
"""
        for i, item in enumerate(recommendations, start=1):
            report_text += f"{i}. {item}\n"

        report_text += f"\nSUMMARY:\n{summary}\n"

        st.download_button(
            label="Download Report",
            data=report_text,
            file_name="health_report.txt",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"Error while generating output: {e}")

st.subheader("Model Scores")
s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Diabetes Accuracy", f"{diabetes_accuracy}%")
s2.metric("Heart Accuracy", f"{heart_accuracy}%")
s3.metric("Insurance R²", insurance_r2)
s4.metric("Insurance MAE", insurance_mae)
s5.metric("Insurance RMSE", insurance_rmse)