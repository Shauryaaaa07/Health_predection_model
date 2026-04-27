from flask import Flask, request, render_template_string
import warnings
warnings.filterwarnings("ignore")

import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, r2_score
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

def cluster_meaning(cluster_id):
    mapping = {
        0: "Cluster 0: comparatively lower-risk profile group",
        1: "Cluster 1: moderate health-risk profile group",
        2: "Cluster 2: higher health-risk / higher-cost profile group"
    }
    return mapping.get(cluster_id, f"Cluster {cluster_id}")

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img

# =========================================================
# diabetes model
# =========================================================
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

# =========================================================
# heart model
# =========================================================
heart_df = pd.read_csv("heart_disease.csv")
heart_df["target"] = heart_df["num"].apply(lambda x: 0 if x == 0 else 1)
heart_df = heart_df.drop(columns=["id", "num"])

X_heart = heart_df.drop("target", axis=1)
y_heart = heart_df["target"]

heart_cat_cols = X_heart.select_dtypes(include=["object"]).columns.tolist()
heart_num_cols = X_heart.select_dtypes(include=["int64", "float64"]).columns.tolist()

heart_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), heart_num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ]), heart_cat_cols)
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

# =========================================================
# insurance model
# =========================================================
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
insurance_r2 = round(r2_score(y_test_i, insurance_model.predict(X_test_i)), 4)

# =========================================================
# clustering model
# =========================================================
cluster_df = insurance_df.copy()

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

X_cluster = cluster_df[["age", "sex", "bmi", "children", "smoker", "region"]]
X_cluster_processed = cluster_preprocessor.fit_transform(X_cluster)

kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_model.fit(X_cluster_processed)

# =========================================================
# UI
# =========================================================
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Health Dashboard</title>
    <style>
        body{
            font-family: Arial, sans-serif;
            background:#f4f7fb;
            padding:20px;
            color:#222;
        }
        .container{
            max-width:1150px;
            margin:auto;
        }
        .box{
            background:white;
            padding:22px;
            border-radius:16px;
            box-shadow:0 8px 20px rgba(0,0,0,0.07);
            margin-bottom:20px;
        }
        h1,h2{
            color:#1a4b8f;
        }
        .grid{
            display:grid;
            grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
            gap:18px;
        }
        label{
            display:block;
            margin-bottom:6px;
            font-weight:bold;
            font-size:14px;
        }
        input,select{
            width:100%;
            padding:10px 12px;
            margin-bottom:12px;
            border:1px solid #d0d8e5;
            border-radius:10px;
        }
        button{
            width:100%;
            padding:14px;
            border:none;
            border-radius:12px;
            background:#1e63d0;
            color:white;
            font-weight:bold;
            font-size:16px;
            cursor:pointer;
        }
        .result-grid{
            display:grid;
            grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
            gap:16px;
        }
        .card{
            padding:16px;
            border-radius:14px;
            color:white;
        }
        .blue{background:#2d6cdf;}
        .green{background:#20a35a;}
        .red{background:#d9534f;}
        .purple{background:#7a45b8;}
        .orange{background:#f39c12;}
        .muted{
            color:#555;
            line-height:1.7;
        }
        .demo{
            background:#f7faff;
            border:1px solid #d8e5f7;
            padding:14px;
            border-radius:12px;
            line-height:1.8;
        }
        .graph-box{
            text-align:center;
        }
        .graph-box img{
            width:100%;
            max-width:760px;
            border-radius:12px;
            border:1px solid #d0d8e5;
        }
    </style>
</head>
<body>
<div class="container">

    <div class="box">
        <h1>Health Prediction Dashboard</h1>
        <p class="muted">
            BMI, Diabetes, Heart Disease, Insurance aur Clustering ke liye simple dashboard.
        </p>
    </div>

    <div class="box">
        <h2>Demo Values</h2>
        <div class="demo">
            Age = 22<br>
            Gender = male<br>
            Height = 170<br>
            Weight = 70<br>
            Children = 0<br>
            Smoker = no<br>
            Region = northwest<br>
            Pregnancies = 0<br>
            Glucose = 120<br>
            Blood Pressure = 80<br>
            Skin Thickness = 20<br>
            Insulin = 85<br>
            DPF = 0.5<br>
            CP = 1<br>
            Trestbps = 120<br>
            Chol = 200<br>
            FBS = 0<br>
            Restecg = 1<br>
            Thalch = 150<br>
            Exang = 0<br>
            Oldpeak = 1.2<br>
            Slope = 2<br>
            CA = 0<br>
            Thal = 0
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

        <br>
        <button type="submit">Generate Result</button>
    </form>

    {% if result %}
    <div class="box">
        <h2>Prediction Results</h2>
        <div class="result-grid">
            <div class="card blue">
                <h3>BMI</h3>
                <p>{{ result.bmi }}</p>
                <p>{{ result.bmi_category }}</p>
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
            </div>

            <div class="card orange">
                <h3>Clustering</h3>
                <p>Cluster ID: {{ result.cluster_id }}</p>
                <p>{{ result.cluster_text }}</p>
            </div>
        </div>
    </div>

    <div class="box graph-box">
        <h2>Cluster Graph</h2>
        <img src="data:image/png;base64,{{ result.cluster_graph }}">
    </div>

    <div class="box">
        <h2>Summary</h2>
        <p class="muted">{{ result.summary }}</p>
    </div>
    {% endif %}

    <div class="box">
        <h2>Model Scores</h2>
        <p class="muted">
            Diabetes Accuracy: {{ diabetes_accuracy }}%<br>
            Heart Accuracy: {{ heart_accuracy }}%<br>
            Insurance R² Score: {{ insurance_r2 }}
        </p>
    </div>

</div>
</body>
</html>
"""

# =========================================================
# route
# =========================================================
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    values = {}

    if request.method == "POST":
        values = request.form.to_dict()

        age = safe_int(request.form.get("age"))
        sex = request.form.get("sex")
        height = safe_float(request.form.get("height"))
        weight = safe_float(request.form.get("weight"))
        children = safe_int(request.form.get("children"))
        smoker = request.form.get("smoker")
        region = request.form.get("region")

        bmi = get_bmi(weight, height)
        bmi_text = bmi_category(bmi)

        # diabetes
        diabetes_input = pd.DataFrame([{
            "Pregnancies": safe_int(request.form.get("pregnancies")),
            "Glucose": safe_float(request.form.get("glucose")),
            "BloodPressure": safe_float(request.form.get("bloodpressure")),
            "SkinThickness": safe_float(request.form.get("skinthickness")),
            "Insulin": safe_float(request.form.get("insulin")),
            "BMI": bmi,
            "DiabetesPedigreeFunction": safe_float(request.form.get("dpf")),
            "Age": age
        }])
        diabetes_scaled = diabetes_scaler.transform(diabetes_input)
        diabetes_pred = int(diabetes_model.predict(diabetes_scaled)[0])
        diabetes_prob = float(diabetes_model.predict_proba(diabetes_scaled)[0][1])
        diabetes_result = "Diabetes Risk" if diabetes_pred == 1 else "No Diabetes Risk"

        # heart
        heart_input = pd.DataFrame([{
            "age": age,
            "sex": 1 if sex == "male" else 0,
            "cp": safe_float(request.form.get("cp")),
            "trestbps": safe_float(request.form.get("trestbps")),
            "chol": safe_float(request.form.get("chol")),
            "fbs": safe_float(request.form.get("fbs")),
            "restecg": safe_float(request.form.get("restecg")),
            "thalch": safe_float(request.form.get("thalch")),
            "exang": safe_float(request.form.get("exang")),
            "oldpeak": safe_float(request.form.get("oldpeak")),
            "slope": safe_float(request.form.get("slope")),
            "ca": safe_float(request.form.get("ca")),
            "thal": safe_float(request.form.get("thal"))
        }])
        heart_pred = int(heart_model.predict(heart_input)[0])
        heart_prob = float(heart_model.predict_proba(heart_input)[0][1])
        heart_result = "Heart Disease Risk" if heart_pred == 1 else "No Heart Disease Risk"

        # insurance
        insurance_input = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }])
        insurance_pred = round(float(insurance_model.predict(insurance_input)[0]), 2)

        # clustering
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

        # cluster graph
        fig, ax = plt.subplots(figsize=(7, 5))
        train_plot_df = cluster_df.copy()
        train_plot_df["cluster"] = kmeans_model.labels_

        for c in sorted(train_plot_df["cluster"].unique()):
            temp = train_plot_df[train_plot_df["cluster"] == c]
            ax.scatter(temp["age"], temp["bmi"], label=f"Cluster {c}", alpha=0.6)

        ax.scatter(age, bmi, color="black", s=140, marker="X", label="User Input")
        ax.set_title("Cluster Visualization (Age vs BMI)")
        ax.set_xlabel("Age")
        ax.set_ylabel("BMI")
        ax.legend()

        cluster_graph = fig_to_base64(fig)

        summary = (
            f"Your BMI is {bmi}, which is in the {bmi_text} category. "
            f"Diabetes result: {diabetes_result}. "
            f"Heart result: {heart_result}. "
            f"Estimated insurance charges are ₹ {insurance_pred}. "
            f"Your profile falls in {cluster_text}."
        )

        result = {
            "bmi": bmi,
            "bmi_category": bmi_text,
            "diabetes_result": diabetes_result,
            "diabetes_confidence": round(diabetes_prob * 100, 2),
            "heart_result": heart_result,
            "heart_confidence": round(heart_prob * 100, 2),
            "insurance_charges": insurance_pred,
            "cluster_id": cluster_id,
            "cluster_text": cluster_text,
            "cluster_graph": cluster_graph,
            "summary": summary
        }

    return render_template_string(
        html,
        result=result,
        values=values,
        diabetes_accuracy=diabetes_accuracy,
        heart_accuracy=heart_accuracy,
        insurance_r2=insurance_r2
    )

if __name__ == "__main__":
    app.run(debug=True)