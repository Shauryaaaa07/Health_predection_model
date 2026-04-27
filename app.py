from flask import Flask, request, render_template_string, make_response
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans

app = Flask(__name__)

# =========================================================
# helper functions
# =========================================================
def safe_float(value, default=0.0):
    try:
        return float(value)
    except:
        return default

def safe_int(value, default=0):
    try:
        return int(float(value))
    except:
        return default

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
    heart_df[col] = pd.to_numeric(heart_df[col], errors="ignore")

heart_df["target"] = heart_df["num"].apply(lambda x: 0 if x == 0 else 1)
heart_df = heart_df.drop(columns=["id", "num"])

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
# html
# =========================================================
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Health Prediction Dashboard</title>
    <style>
        *{box-sizing:border-box;margin:0;padding:0}
        body{
            font-family:Arial,sans-serif;
            background:linear-gradient(135deg,#eef3fb,#f8fbff);
            color:#1f2937;
            padding:20px;
        }
        .container{
            max-width:1200px;
            margin:auto;
        }
        .hero{
            background:#fff;
            border-radius:24px;
            padding:26px;
            box-shadow:0 12px 28px rgba(0,0,0,.06);
            margin-bottom:20px;
        }
        .badge{
            display:inline-block;
            background:#e7efff;
            color:#17468a;
            padding:8px 14px;
            border-radius:999px;
            font-size:13px;
            font-weight:700;
            margin-bottom:14px;
        }
        .hero h1{
            color:#184989;
            font-size:34px;
            margin-bottom:10px;
        }
        .hero p{
            color:#55606f;
            line-height:1.7;
        }
        .box{
            background:#fff;
            border-radius:20px;
            padding:22px;
            box-shadow:0 10px 24px rgba(0,0,0,.05);
            margin-bottom:18px;
        }
        .demo-inline{
            display:flex;
            flex-wrap:wrap;
            gap:10px;
            margin-top:10px;
        }
        .demo-pill{
            background:#f4f8ff;
            border:1px solid #d9e4f6;
            padding:8px 12px;
            border-radius:999px;
            font-size:13px;
            color:#244b85;
        }
        .grid{
            display:grid;
            grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
            gap:18px;
        }
        h2{
            color:#1b4b88;
            margin-bottom:14px;
            font-size:21px;
        }
        label{
            display:block;
            font-weight:700;
            margin-bottom:6px;
            font-size:14px;
            color:#334155;
        }
        input,select{
            width:100%;
            padding:11px 12px;
            border:1px solid #d4ddeb;
            border-radius:12px;
            margin-bottom:12px;
            background:#fbfdff;
        }
        .btn-row{
            display:flex;
            gap:12px;
            flex-wrap:wrap;
        }
        button{
            flex:1;
            min-width:220px;
            padding:14px;
            border:none;
            border-radius:14px;
            background:#1f63d3;
            color:#fff;
            font-size:16px;
            font-weight:700;
            cursor:pointer;
        }
        button:hover{background:#174faa}
        .ghost-btn{
            display:inline-block;
            text-decoration:none;
            text-align:center;
            padding:14px 18px;
            border-radius:14px;
            background:#0f766e;
            color:#fff;
            font-weight:700;
            min-width:220px;
        }
        .result-grid{
            display:grid;
            grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
            gap:16px;
        }
        .card{
            border-radius:18px;
            padding:18px;
            color:#fff;
        }
        .blue{background:linear-gradient(135deg,#1f64d7,#4f8ef7)}
        .green{background:linear-gradient(135deg,#13975a,#3fc178)}
        .red{background:linear-gradient(135deg,#cb3939,#f16868)}
        .purple{background:linear-gradient(135deg,#6d34c3,#9e67ee)}
        .orange{background:linear-gradient(135deg,#de8b15,#ffb444)}
        .card h3{margin-bottom:10px;font-size:18px}
        .card p{line-height:1.7}
        .score-grid{
            display:grid;
            grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
            gap:14px;
        }
        .score-box{
            background:#f7faff;
            border:1px solid #dce7f4;
            border-radius:16px;
            padding:16px;
        }
        .score-box h4{
            color:#1b4c88;
            margin-bottom:8px;
            font-size:15px;
        }
        .score-box p{
            font-weight:700;
            font-size:17px;
        }
        .summary{
            background:#f9fbff;
            border:1px solid #dce7f4;
            border-radius:18px;
            padding:18px;
            line-height:1.8;
            color:#374151;
        }
        .reco-list{
            padding-left:18px;
            line-height:1.9;
            color:#374151;
        }
        @media(max-width:768px){
            .hero h1{font-size:28px}
        }
    </style>
    <script>
        function fillDemo(){
            document.getElementsByName('age')[0].value = 22;
            document.getElementsByName('sex')[0].value = 'male';
            document.getElementsByName('height')[0].value = 170;
            document.getElementsByName('weight')[0].value = 70;
            document.getElementsByName('children')[0].value = 0;
            document.getElementsByName('smoker')[0].value = 'no';
            document.getElementsByName('region')[0].value = 'northwest';
            document.getElementsByName('pregnancies')[0].value = 0;
            document.getElementsByName('glucose')[0].value = 120;
            document.getElementsByName('bloodpressure')[0].value = 80;
            document.getElementsByName('skinthickness')[0].value = 20;
            document.getElementsByName('insulin')[0].value = 85;
            document.getElementsByName('dpf')[0].value = 0.5;
            document.getElementsByName('cp')[0].value = 1;
            document.getElementsByName('trestbps')[0].value = 120;
            document.getElementsByName('chol')[0].value = 200;
            document.getElementsByName('fbs')[0].value = 0;
            document.getElementsByName('restecg')[0].value = 1;
            document.getElementsByName('thalch')[0].value = 150;
            document.getElementsByName('exang')[0].value = 0;
            document.getElementsByName('oldpeak')[0].value = 1.2;
            document.getElementsByName('slope')[0].value = 2;
            document.getElementsByName('ca')[0].value = 0;
            document.getElementsByName('thal')[0].value = 0;
        }
    </script>
</head>
<body>
<div class="container">

    <div class="hero">
        <div class="badge">Major Project Dashboard</div>
        <h1>Smart Health Prediction Dashboard</h1>
        <p>
            Is dashboard me user input ke basis par BMI, Diabetes Risk, Heart Disease Risk,
            Insurance Charges aur final clustering output generate hota hai.
        </p>
    </div>

    <div class="box">
        <h2>Demo Values</h2>
        <div class="demo-inline">
            <span class="demo-pill">Age: 22</span>
            <span class="demo-pill">Gender: male</span>
            <span class="demo-pill">Height: 170</span>
            <span class="demo-pill">Weight: 70</span>
            <span class="demo-pill">Glucose: 120</span>
            <span class="demo-pill">BP: 80</span>
            <span class="demo-pill">Chol: 200</span>
            <span class="demo-pill">Thalch: 150</span>
            <span class="demo-pill">Smoker: no</span>
            <span class="demo-pill">Region: northwest</span>
        </div>
        <div style="margin-top:14px;">
            <button type="button" onclick="fillDemo()">Fill Demo Inputs</button>
        </div>
    </div>

    <form method="POST" class="box">
        <div class="grid">
            <div>
                <h2>Basic Details</h2>
                <label>Age</label>
                <input type="number" name="age" required value="{{ values.get('age','') }}">

                <label>Gender</label>
                <select name="sex" required>
                    <option value="male" {% if values.get('sex') == 'male' %}selected{% endif %}>Male</option>
                    <option value="female" {% if values.get('sex') == 'female' %}selected{% endif %}>Female</option>
                </select>

                <label>Height (cm)</label>
                <input type="number" step="any" name="height" required value="{{ values.get('height','') }}">

                <label>Weight (kg)</label>
                <input type="number" step="any" name="weight" required value="{{ values.get('weight','') }}">

                <label>Children</label>
                <input type="number" name="children" required value="{{ values.get('children','0') }}">

                <label>Smoker</label>
                <select name="smoker" required>
                    <option value="no" {% if values.get('smoker') == 'no' %}selected{% endif %}>No</option>
                    <option value="yes" {% if values.get('smoker') == 'yes' %}selected{% endif %}>Yes</option>
                </select>

                <label>Region</label>
                <select name="region" required>
                    <option value="northwest" {% if values.get('region') == 'northwest' %}selected{% endif %}>Northwest</option>
                    <option value="northeast" {% if values.get('region') == 'northeast' %}selected{% endif %}>Northeast</option>
                    <option value="southwest" {% if values.get('region') == 'southwest' %}selected{% endif %}>Southwest</option>
                    <option value="southeast" {% if values.get('region') == 'southeast' %}selected{% endif %}>Southeast</option>
                </select>
            </div>

            <div>
                <h2>Diabetes Inputs</h2>
                <label>Pregnancies</label>
                <input type="number" name="pregnancies" required value="{{ values.get('pregnancies','0') }}">

                <label>Glucose</label>
                <input type="number" step="any" name="glucose" required value="{{ values.get('glucose','') }}">

                <label>Blood Pressure</label>
                <input type="number" step="any" name="bloodpressure" required value="{{ values.get('bloodpressure','') }}">

                <label>Skin Thickness</label>
                <input type="number" step="any" name="skinthickness" required value="{{ values.get('skinthickness','') }}">

                <label>Insulin</label>
                <input type="number" step="any" name="insulin" required value="{{ values.get('insulin','') }}">

                <label>DPF</label>
                <input type="number" step="any" name="dpf" required value="{{ values.get('dpf','') }}">
            </div>

            <div>
                <h2>Heart Inputs</h2>
                <label>CP</label>
                <input type="number" step="any" name="cp" required value="{{ values.get('cp','') }}">

                <label>Trestbps</label>
                <input type="number" step="any" name="trestbps" required value="{{ values.get('trestbps','') }}">

                <label>Chol</label>
                <input type="number" step="any" name="chol" required value="{{ values.get('chol','') }}">

                <label>FBS</label>
                <input type="number" step="any" name="fbs" required value="{{ values.get('fbs','') }}">

                <label>Restecg</label>
                <input type="number" step="any" name="restecg" required value="{{ values.get('restecg','') }}">

                <label>Thalch</label>
                <input type="number" step="any" name="thalch" required value="{{ values.get('thalch','') }}">

                <label>Exang</label>
                <input type="number" step="any" name="exang" required value="{{ values.get('exang','') }}">

                <label>Oldpeak</label>
                <input type="number" step="any" name="oldpeak" required value="{{ values.get('oldpeak','') }}">

                <label>Slope</label>
                <input type="number" step="any" name="slope" required value="{{ values.get('slope','') }}">

                <label>CA</label>
                <input type="number" step="any" name="ca" required value="{{ values.get('ca','') }}">

                <label>Thal</label>
                <input type="number" step="any" name="thal" required value="{{ values.get('thal','') }}">
            </div>
        </div>

        <div class="btn-row" style="margin-top:8px;">
            <button type="submit">Generate Full Output</button>
            {% if result %}
            <a class="ghost-btn" href="/download-report?age={{ values.get('age','') }}&sex={{ values.get('sex','') }}&height={{ values.get('height','') }}&weight={{ values.get('weight','') }}&children={{ values.get('children','0') }}&smoker={{ values.get('smoker','no') }}&region={{ values.get('region','northwest') }}&pregnancies={{ values.get('pregnancies','0') }}&glucose={{ values.get('glucose','') }}&bloodpressure={{ values.get('bloodpressure','') }}&skinthickness={{ values.get('skinthickness','') }}&insulin={{ values.get('insulin','') }}&dpf={{ values.get('dpf','') }}&cp={{ values.get('cp','') }}&trestbps={{ values.get('trestbps','') }}&chol={{ values.get('chol','') }}&fbs={{ values.get('fbs','') }}&restecg={{ values.get('restecg','') }}&thalch={{ values.get('thalch','') }}&exang={{ values.get('exang','') }}&oldpeak={{ values.get('oldpeak','') }}&slope={{ values.get('slope','') }}&ca={{ values.get('ca','') }}&thal={{ values.get('thal','') }}">Download Report</a>
            {% endif %}
        </div>
    </form>

    {% if result %}
    <div class="box">
        <h2>Prediction Output</h2>
        <div class="result-grid">
            <div class="card blue">
                <h3>BMI</h3>
                <p><strong>{{ result.bmi }}</strong></p>
                <p>{{ result.bmi_category }}</p>
                <p>Ideal Range: {{ result.ideal_low }} kg - {{ result.ideal_high }} kg</p>
            </div>

            <div class="card {% if result.diabetes_result == 'Diabetes Risk' %}red{% else %}green{% endif %}">
                <h3>Diabetes</h3>
                <p>{{ result.diabetes_result }}</p>
                <p>Confidence: {{ result.diabetes_confidence }}%</p>
            </div>

            <div class="card {% if result.heart_result == 'Heart Disease Risk' %}red{% else %}green{% endif %}">
                <h3>Heart</h3>
                <p>{{ result.heart_result }}</p>
                <p>Confidence: {{ result.heart_confidence }}%</p>
            </div>

            <div class="card purple">
                <h3>Insurance</h3>
                <p>₹ {{ result.insurance_charges }}</p>
                <p>Estimated yearly profile cost</p>
            </div>

            <div class="card orange">
                <h3>Clustering</h3>
                <p>Cluster ID: {{ result.cluster_id }}</p>
                <p>{{ result.cluster_text }}</p>
            </div>
        </div>
    </div>

    <div class="box">
        <h2>Recommendations</h2>
        <ul class="reco-list">
            {% for item in result.recommendations %}
                <li>{{ item }}</li>
            {% endfor %}
        </ul>
    </div>

    <div class="box">
        <h2>Summary</h2>
        <div class="summary">{{ result.summary }}</div>
    </div>
    {% endif %}

    <div class="box">
        <h2>Model Scores</h2>
        <div class="score-grid">
            <div class="score-box">
                <h4>Diabetes Accuracy</h4>
                <p>{{ diabetes_accuracy }}%</p>
            </div>
            <div class="score-box">
                <h4>Heart Accuracy</h4>
                <p>{{ heart_accuracy }}%</p>
            </div>
            <div class="score-box">
                <h4>Insurance R² Score</h4>
                <p>{{ insurance_r2 }}</p>
            </div>
            <div class="score-box">
                <h4>Insurance MAE</h4>
                <p>{{ insurance_mae }}</p>
            </div>
            <div class="score-box">
                <h4>Insurance RMSE</h4>
                <p>{{ insurance_rmse }}</p>
            </div>
        </div>
    </div>

</div>
</body>
</html>
"""

# =========================================================
# core prediction function
# =========================================================
def generate_result(source):
    age = safe_int(source.get("age"))
    sex = source.get("sex")
    height = safe_float(source.get("height"))
    weight = safe_float(source.get("weight"))
    children = safe_int(source.get("children"))
    smoker = source.get("smoker")
    region = source.get("region")

    bmi = get_bmi(weight, height)
    bmi_text = bmi_category(bmi)
    ideal_low, ideal_high = ideal_weight_range(height)

    diabetes_input = pd.DataFrame([{
        "Pregnancies": safe_int(source.get("pregnancies")),
        "Glucose": safe_float(source.get("glucose")),
        "BloodPressure": safe_float(source.get("bloodpressure")),
        "SkinThickness": safe_float(source.get("skinthickness")),
        "Insulin": safe_float(source.get("insulin")),
        "BMI": bmi,
        "DiabetesPedigreeFunction": safe_float(source.get("dpf")),
        "Age": age
    }])

    diabetes_scaled = diabetes_scaler.transform(diabetes_input)
    diabetes_pred = int(diabetes_model.predict(diabetes_scaled)[0])
    diabetes_prob = float(diabetes_model.predict_proba(diabetes_scaled)[0][1])
    diabetes_result = "Diabetes Risk" if diabetes_pred == 1 else "No Diabetes Risk"

    heart_input = pd.DataFrame([{
        "age": age,
        "sex": 1 if sex == "male" else 0,
        "cp": safe_float(source.get("cp")),
        "trestbps": safe_float(source.get("trestbps")),
        "chol": safe_float(source.get("chol")),
        "fbs": safe_float(source.get("fbs")),
        "restecg": safe_float(source.get("restecg")),
        "thalch": safe_float(source.get("thalch")),
        "exang": safe_float(source.get("exang")),
        "oldpeak": safe_float(source.get("oldpeak")),
        "slope": safe_float(source.get("slope")),
        "ca": safe_float(source.get("ca")),
        "thal": safe_float(source.get("thal"))
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

    return {
        "bmi": bmi,
        "bmi_category": bmi_text,
        "ideal_low": ideal_low,
        "ideal_high": ideal_high,
        "diabetes_result": diabetes_result,
        "diabetes_confidence": round(diabetes_prob * 100, 2),
        "heart_result": heart_result,
        "heart_confidence": round(heart_prob * 100, 2),
        "insurance_charges": insurance_pred,
        "cluster_id": cluster_id,
        "cluster_text": cluster_text,
        "recommendations": recommendations,
        "summary": summary
    }

# =========================================================
# routes
# =========================================================
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    values = {}

    if request.method == "POST":
        values = request.form.to_dict()
        result = generate_result(request.form)

    return render_template_string(
        html,
        result=result,
        values=values,
        diabetes_accuracy=diabetes_accuracy,
        heart_accuracy=heart_accuracy,
        insurance_r2=insurance_r2,
        insurance_mae=insurance_mae,
        insurance_rmse=insurance_rmse
    )

@app.route("/download-report")
def download_report():
    result = generate_result(request.args)

    lines = [
        "HEALTH PREDICTION REPORT",
        "=" * 40,
        f"BMI: {result['bmi']}",
        f"BMI Category: {result['bmi_category']}",
        f"Ideal Weight Range: {result['ideal_low']} kg - {result['ideal_high']} kg",
        f"Diabetes Prediction: {result['diabetes_result']}",
        f"Diabetes Confidence: {result['diabetes_confidence']}%",
        f"Heart Prediction: {result['heart_result']}",
        f"Heart Confidence: {result['heart_confidence']}%",
        f"Insurance Charges: ₹ {result['insurance_charges']}",
        f"Cluster: {result['cluster_id']} - {result['cluster_text']}",
        "",
        "RECOMMENDATIONS:"
    ]

    for i, item in enumerate(result["recommendations"], start=1):
        lines.append(f"{i}. {item}")

    lines.append("")
    lines.append("SUMMARY:")
    lines.append(result["summary"])

    response = make_response("\\n".join(lines))
    response.headers["Content-Type"] = "text/plain"
    response.headers["Content-Disposition"] = "attachment; filename=health_report.txt"
    return response

if __name__ == "__main__":
    app.run(debug=True)