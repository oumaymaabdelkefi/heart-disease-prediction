# Heart Disease Prediction Using Machine Learning

A classification study on the UCI Heart Disease dataset — a full-stack web dashboard that puts the analysis in front of a user: upload a dataset, explore it visually, and get a risk prediction for an individual patient.

**Authors:** Noura Haddaoui, Oumayma Abdelkefi
**Course:** Data Analytics — South Mediterranean University
**Instructor:** Hamza Moharrar · April 2026

> **Academic project.** This is not a medical device. Nothing it outputs should inform diagnosis or treatment.

---

## What's here

The project has two halves that answer different questions.

The **notebook** (`data/data.ipynb`) is the study: it audits the data, explores it, and fits an interpretable Logistic Regression to find out *which clinical factors matter and in which direction*. This is the work documented in the written report.

The **dashboard** (`heart_dashboard/`) is the application: a Flask API that trains a Random Forest on whatever CSV you upload, and a browser frontend that visualises the dataset and returns per-patient risk scores. It optimises for predictive accuracy rather than interpretability.

```
heart-disease-prediction/
├── index.html                      # Root launcher
├── data/
│   ├── heart.csv                   # Raw dataset (1,025 records)
│   ├── heart_cleaned.csv           # Deduplicated (302 unique records)
│   └── data.ipynb                  # EDA + Logistic Regression study
└── heart_dashboard/
    ├── backend/
    │   ├── app.py                  # Flask API
    │   ├── requirements.txt
    │   └── .env.example
    └── frontend/
        ├── index.html
        ├── partials/               # home · dashboard · predict · upload · articles · doctors · sidebar
        └── assets/
```

---

## Results

Logistic Regression, 80/20 split (241 train / 61 test), `random_state=42`, StandardScaler normalisation.

| Metric | No Disease (n=32) | Disease (n=29) | Weighted avg |
|---|---|---|---|
| Precision | 0.88 | 0.70 | 0.79 |
| Recall | 0.66 | **0.90** | 0.77 |
| F1-score | 0.75 | 0.79 | 0.77 |

**Overall accuracy: 77.05%**

Confusion matrix:

|  | Predicted: No Disease | Predicted: Disease |
|---|---|---|
| **Actual: No Disease** | 21 (TN) | 11 (FP) |
| **Actual: Disease** | 3 (FN) | 26 (TP) |

The headline number is not the accuracy — it's the **90% recall on positive cases**. The model caught 26 of 29 real cases and missed 3. In a screening context that trade-off is the right one: an unnecessary follow-up test costs far less than an undetected cardiac condition.

### Strongest predictors

Ranked by coefficient magnitude:

| Feature | Coefficient | Reading |
|---|---|---|
| `cp` — chest pain type | +1.004 | Atypical/asymptomatic pain sharply raises predicted risk |
| `sex` | −0.856 | Dataset-specific artefact — see caveat below |
| `oldpeak` — ST depression | −0.796 | Encoding-specific direction |
| `ca` — vessels by fluoroscopy | −0.589 | More coloured vessels → lower predicted probability |
| `trestbps` — resting BP | −0.557 | Inverse in this cohort |
| `restecg` — resting ECG | +0.496 | Abnormal readings raise risk |
| `thalach` — max heart rate | +0.481 | Higher exercise HR correlates with disease here |

Chest pain type and the ST-segment features align with established cardiology. The negative `sex` coefficient does **not** mean men are at lower risk — it reflects the demographic composition and diagnostic patterns of the 1980s Cleveland cohort. Don't carry that coefficient to another population.

---

## The duplicate problem

`heart.csv` ships with 1,025 rows, but **723 of them are duplicates — 70.5% of the file**. After `drop_duplicates()`, 302 unique records remain.

This matters more than it sounds. Training on the raw file lets identical rows land in both the training and test sets, so the model is scored on records it has already memorised. Accuracy comes out inflated and meaningless. Every result above uses `heart_cleaned.csv`, and the dashboard should be fed the same file.

No missing values were found in any of the 14 columns.

---

## Running it

### Backend

```bash
cd heart_dashboard/backend

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
python app.py                     # → http://localhost:5001
```

### Frontend

Serve the repo root over HTTP. Opening the file directly via `file://` will fail — the page fetches its partials.

```bash
# from the repo root, second terminal
python -m http.server 8080
```

Open <http://localhost:8080>, go to the **Upload** tab, and upload `data/heart_cleaned.csv`. The backend trains a Random Forest on it immediately and the dashboard and prediction tabs come alive.

### Notebook

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook data/data.ipynb
```

---

## API

Base URL `http://localhost:5001`

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/upload` | Upload CSV → trains model, returns accuracy |
| GET | `/api/stats` | Aggregate statistics for the dashboard |
| POST | `/api/predict` | Risk prediction for one patient |
| GET | `/api/health` | Server status, whether a dataset is loaded |
| POST | `/api/ai-report` | Groq-backed narrative report (optional) |

**`POST /api/predict`**

```json
{
  "age": 55, "sex": 1, "cp": 0, "trestbps": 130, "chol": 250,
  "fbs": 0, "restecg": 1, "thalach": 148, "exang": 0,
  "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 2
}
```

```json
{
  "prediction": 1,
  "risk_pct": 73.5,
  "risk_level": "High",
  "no_disease_pct": 26.5,
  "top_features": [
    { "name": "ca", "importance": 14.2 },
    { "name": "thalach", "importance": 12.8 }
  ]
}
```

### Optional: AI reports

`/api/ai-report` proxies to Groq. Without a key every other endpoint still works — only this one is disabled.

```bash
cd heart_dashboard/backend
cp .env.example .env      # then paste your key from console.groq.com/keys
```

`.env` is gitignored. Keep it that way.

---

## Dataset reference

Cleveland Heart Disease dataset (UCI / Kaggle), originally compiled by Detrano et al. (1989).

| Feature | Description | Values |
|---|---|---|
| `age` | Age in years | continuous |
| `sex` | Sex | 0 = female, 1 = male |
| `cp` | Chest pain type | 0 = asymptomatic, 1 = atypical angina, 2 = non-anginal, 3 = typical angina |
| `trestbps` | Resting blood pressure at admission | mm Hg |
| `chol` | Serum cholesterol | mg/dl |
| `fbs` | Fasting blood sugar > 120 mg/dl | 0 or 1 |
| `restecg` | Resting ECG | 0 = normal, 1 = ST-T abnormality, 2 = LVH |
| `thalach` | Max heart rate during stress test | bpm |
| `exang` | Exercise-induced angina | 0 or 1 |
| `oldpeak` | ST depression induced by exercise | continuous |
| `slope` | Slope of peak exercise ST segment | 0 = upsloping, 1 = flat, 2 = downsloping |
| `ca` | Major vessels coloured by fluoroscopy | 0–3 |
| `thal` | Thalassemia | 0 = normal, 1 = fixed defect, 2 = reversible defect |
| `target` | Heart disease present | 0 = no, 1 = yes |

---

## Methodology

CRISP-DM, applied in sequence: business understanding → data understanding → preparation → modelling → evaluation.

Preprocessing removed 723 duplicates, confirmed zero missing values, separated 13 features from the binary target, split 80/20 with a fixed seed, and applied StandardScaler — **fitted on the training set only**, then used to transform both, so no test-set information leaks into preprocessing.

The notebook model uses scikit-learn's `LogisticRegression` defaults (`lbfgs`, L2 penalty, `C=1.0`, `max_iter=100`). No hyperparameter search was run; that's a documented gap rather than an oversight. The dashboard API instead uses a Random Forest — 150 trees, `max_depth=8` — because it doesn't need coefficient interpretability.

---

## Limitations

- **302 unique records** is a small effective sample, which limits statistical power and generalisability.
- **One algorithm.** No comparison against ensemble or non-linear classifiers. Mohan et al. (2019) report 88.7% on comparable data using hybrid Random Forest, so there is clear headroom.
- **No hyperparameter tuning.** No cross-validated grid search was performed.
- **Dataset age.** Collected at the Cleveland Clinic in the late 1980s; demographics, diagnostic criteria, and measurement standards have moved on.
- **Internal validation only.** Performance was measured on a split of the same dataset — no external cohort.
- **Linear decision boundary.** Logistic Regression cannot capture non-linear interactions among clinical variables.

## Future work

Benchmark Random Forest, XGBoost, LightGBM and SVM on the same preprocessed data; run cross-validated hyperparameter optimisation; validate against an external cohort; explore SHAP values for per-prediction explanations.

---

## Tech stack

**Analysis** — Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · Jupyter
**Backend** — Flask · flask-cors · scikit-learn · pandas · python-dotenv
**Frontend** — vanilla JS · Chart.js

## References

Chapman, P. et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide.*
Detrano, R. et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology.*
Mohan, S., Thirumalai, C., & Srivastava, G. (2019). Effective heart disease prediction using hybrid machine learning techniques. *IEEE Access.*
Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR.*
Rajpurkar, P. et al. (2022). AI in health and medicine. *Nature Medicine.*
Steyerberg, E. W. (2019). *Clinical Prediction Models.* Springer.
World Health Organization (2021). *Cardiovascular diseases (CVDs) fact sheet.*
