from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import json
import os
import requests
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# Global state
model = None
scaler = None
df_global = None
model_accuracy = None
feature_importance = None

FEATURE_COLS = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
TARGET_COL = "target"


def train_model(df):
    global model, scaler, model_accuracy, feature_importance
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    model_accuracy = round(accuracy_score(y_test, preds) * 100, 1)
    feature_importance = dict(zip(FEATURE_COLS, [round(float(v)*100, 2) for v in model.feature_importances_]))


def compute_stats(df):
    total   = len(df)
    disease = int((df[TARGET_COL] == 1).sum())
    no_dis  = total - disease

    # age groups
    bins   = [0, 40, 50, 60, 120]
    labels = ["<40","40-49","50-59","60+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    age_disease = df.groupby("age_group", observed=True)[TARGET_COL].agg(["sum","count"]).reset_index()
    age_chart = {
        "labels": list(age_disease["age_group"].astype(str)),
        "disease": list(age_disease["sum"].astype(int)),
        "no_disease": list((age_disease["count"] - age_disease["sum"]).astype(int))
    }

    # sex
    sex_stats = df.groupby("sex")[TARGET_COL].agg(["sum","count"])
    male_rate   = round(float(sex_stats.loc[1,"sum"]/sex_stats.loc[1,"count"]*100), 1) if 1 in sex_stats.index else 0
    female_rate = round(float(sex_stats.loc[0,"sum"]/sex_stats.loc[0,"count"]*100), 1) if 0 in sex_stats.index else 0

    # chest pain
    cp_counts = df["cp"].value_counts().sort_index()
    cp_chart  = {"labels": [f"Type {i}" for i in cp_counts.index.tolist()],
                 "counts": cp_counts.values.tolist()}

    # slope
    slope_counts = df["slope"].value_counts().sort_index()
    slope_chart  = {"labels": [f"Slope {i}" for i in slope_counts.index.tolist()],
                    "counts": slope_counts.values.tolist()}

    # oldpeak
    op_disease = round(float(df[df[TARGET_COL]==1]["oldpeak"].mean()), 2)
    op_no      = round(float(df[df[TARGET_COL]==0]["oldpeak"].mean()), 2)

    return {
        "total": total,
        "disease": disease,
        "no_disease": no_dis,
        "disease_pct": round(disease/total*100, 1),
        "avg_age": round(float(df["age"].mean()), 1),
        "avg_chol": round(float(df["chol"].mean()), 1),
        "avg_thalach": round(float(df["thalach"].mean()), 1),
        "avg_trestbps": round(float(df["trestbps"].mean()), 1),
        "male_disease_rate": male_rate,
        "female_disease_rate": female_rate,
        "male_total": int(sex_stats.loc[1,"count"]) if 1 in sex_stats.index else 0,
        "female_total": int(sex_stats.loc[0,"count"]) if 0 in sex_stats.index else 0,
        "fbs_positive": int((df["fbs"]==1).sum()),
        "op_disease": op_disease,
        "op_no_disease": op_no,
        "age_chart": age_chart,
        "cp_chart": cp_chart,
        "slope_chart": slope_chart,
        "model_accuracy": model_accuracy,
        "feature_importance": feature_importance
    }


@app.route("/api/upload", methods=["POST"])
def upload():
    global df_global
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files accepted"}), 400
    try:
        content = file.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))
        # Validate columns
        missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400
        df = df[FEATURE_COLS + [TARGET_COL]].dropna()
        df_global = df
        train_model(df)
        stats = compute_stats(df.copy())
        return jsonify({"success": True, "stats": stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def stats():
    if df_global is None:
        return jsonify({"error": "No dataset loaded"}), 400
    return jsonify(compute_stats(df_global.copy()))


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "No model trained. Upload a dataset first."}), 400
    try:
        data = request.get_json()
        values = [float(data[f]) for f in FEATURE_COLS]
        arr = np.array(values).reshape(1, -1)
        arr_s = scaler.transform(arr)
        pred = int(model.predict(arr_s)[0])
        proba = model.predict_proba(arr_s)[0]
        risk_pct = round(float(proba[1]) * 100, 1)
        risk_level = "High" if risk_pct >= 65 else ("Medium" if risk_pct >= 40 else "Low")

        # top contributing features
        importances = model.feature_importances_
        top = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])[:5]
        top_features = [{"name": k, "importance": round(float(v)*100,1)} for k,v in top]

        return jsonify({
            "prediction": pred,
            "risk_pct": risk_pct,
            "risk_level": risk_level,
            "top_features": top_features,
            "no_disease_pct": round(float(proba[0])*100,1)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "dataset_loaded": df_global is not None, "model_trained": model is not None})


def get_available_groq_models(key):
    """Fetch list of available models from Groq API."""
    try:
        headers = {'Authorization': f'Bearer {key}'}
        resp = requests.get('https://api.groq.com/openai/v1/models', headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if 'data' in data:
            return [m['id'] for m in data['data']]
        return []
    except:
        return []


def pick_best_model(available_models):
    """Pick the best available model from list."""
    # Preferred models in order (most capable to least)
    preferred = [
        'llama-3.3-70b-versatile',
        'mixtral-8x7b-32768',
        'llama-3.2-90b-vision-preview',
        'llama-3.1-70b-versatile',
        'llama-3.2-11b-vision-preview',
        'gemma-7b-it',
        'gemma2-9b-it'
    ]
    for model in preferred:
        if model in available_models:
            return model
    # Fallback: return first available
    return available_models[0] if available_models else None


@app.route('/api/ai-report', methods=['POST'])
def ai_report():
    """Proxy endpoint to generate AI clinical report using Groq API.
    Expects JSON: { prompt: "..." }
    Loads .env file or uses system env var GROQ_API_KEY.
    Automatically detects and uses the best available model.
    """
    load_dotenv()
    data = request.get_json() or {}
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    key = os.environ.get('GROQ_API_KEY')
    if not key:
        return jsonify({'error': 'Groq API key not configured on server (set GROQ_API_KEY).'}), 500

    try:
        # Auto-detect available models
        available = get_available_groq_models(key)
        if not available:
            return jsonify({'error': 'Could not fetch available models from Groq API.'}), 500
        
        model = pick_best_model(available)
        if not model:
            return jsonify({'error': 'No suitable models available on Groq API.'}), 500

        # Call Groq REST API with the best available model
        payload = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 1000,
            'temperature': 0.7
        }
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {key}'
        }
        resp = requests.post('https://api.groq.com/openai/v1/chat/completions', headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        # Groq uses OpenAI-compatible format with choices[0].message.content
        if isinstance(j, dict) and 'choices' in j and len(j['choices']) > 0:
            text = j['choices'][0]['message']['content']
        else:
            text = str(j)
        return jsonify({'text': text})
    except requests.exceptions.HTTPError as he:
        try:
            error_detail = he.response.json()
        except:
            error_detail = he.response.text if he.response else str(he)
        return jsonify({'error': f'Groq API error: {error_detail}'}), 500
    except Exception as e:
        return jsonify({'error': f'AI request failed: {str(e)}'}), 500


if __name__ == "__main__":
    # Local development entry point.
    # In production (Render), gunicorn imports `app` directly and this block is not used.
    # The host reads the port from the PORT environment variable; we fall back to 5001 locally.
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
