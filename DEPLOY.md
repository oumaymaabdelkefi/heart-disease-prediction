# Deploying the Heart Disease Dashboard (free, on Render)

This app has **two parts** that both go online separately:

1. **Backend** — the Flask ML API (folder `heart_dashboard/backend`)
2. **Frontend** — the static dashboard (folder `heart_dashboard/frontend`)

We deploy the backend first (to get its URL), paste that URL into the
frontend, then deploy the frontend. Both are free on Render, no credit card.

> **What's already been changed for you**
> - `app.py` now reads the port from the `PORT` environment variable and runs
>   with `debug=False` (production-safe). Locally it still uses port 5001.
> - `requirements.txt` now includes `gunicorn` (the production server).
> - `index.html` now auto-switches: on your own computer it uses
>   `localhost:5001`; when deployed it uses the `BACKEND_URL` you set (Step 3).
> - The Windows `venv/` folder was removed. Never commit it.

---

## Before you start

- A free **GitHub** account — https://github.com
- A free **Render** account — https://render.com (sign up with GitHub, easiest)
- (Optional) A free **Groq** API key for the AI-report feature —
  https://console.groq.com/keys — skip if you don't need that one feature.

---

## Step 1 — Put the project on GitHub

1. Create a new **empty** repository on GitHub (e.g. `heart-disease-prediction`).
   Do **not** add a README/gitignore there — this folder already has them.
2. In a terminal, from **this folder** (the one containing `DEPLOY.md`):

   ```bash
   git init
   git add .
   git commit -m "Deploy-ready heart disease dashboard"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/heart-disease-prediction.git
   git push -u origin main
   ```

   Replace the URL with the one GitHub shows you.

---

## Step 2 — Deploy the BACKEND (Flask API)

On Render: **New +** → **Web Service** → connect your GitHub repo.

Fill in exactly:

| Field | Value |
|---|---|
| **Root Directory** | `heart_dashboard/backend` |
| **Runtime / Language** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app` |
| **Instance Type** | Free |

If you want the AI-report feature, under **Environment** add a variable:
- Key: `GROQ_API_KEY`  Value: *your Groq key*

Click **Create Web Service** and wait for the build to finish (a few minutes).
When it's live, copy the URL at the top — it looks like:

```
https://heart-disease-prediction-xxxx.onrender.com
```

Quick check: open that URL with `/api/health` on the end. You should see
`{"status":"ok", ...}`.

---

## Step 3 — Point the frontend at your backend

1. Open `heart_dashboard/frontend/index.html`.
2. Near the top of the script find this line:

   ```js
   const BACKEND_URL = 'https://REPLACE-WITH-YOUR-BACKEND.onrender.com';
   ```

3. Replace it with your real backend URL from Step 2.
   **No trailing slash, no `/api`** — just the base:

   ```js
   const BACKEND_URL = 'https://heart-disease-prediction-xxxx.onrender.com';
   ```

4. Commit and push:

   ```bash
   git add heart_dashboard/frontend/index.html
   git commit -m "Point frontend at deployed backend"
   git push
   ```

---

## Step 4 — Deploy the FRONTEND (static site)

On Render: **New +** → **Static Site** → same GitHub repo.

| Field | Value |
|---|---|
| **Root Directory** | *(leave blank)* |
| **Build Command** | *(leave blank)* |
| **Publish Directory** | `heart_dashboard/frontend` |

Click **Create Static Site**. When it's live you get a public link like
`https://heart-disease-dashboard.onrender.com` — **this is the link you share.**

---

## Step 5 — Use it

1. Open your frontend link.
2. Go to **Upload Dataset** and upload `data/heart_cleaned.csv`.
   (Use the *cleaned* file — the raw one is 70% duplicates.)
3. The Dashboard and Predict tabs come alive.

---

## Important things to know about the free tier

- **The backend sleeps after ~15 min of inactivity.** The first request after
  that takes about a minute to wake up — the page may seem frozen briefly.
  This is normal on the free plan.
- **When the backend sleeps/restarts, the trained model is wiped**, because it
  lives in memory. After a wake-up you must **re-upload the CSV** before
  Dashboard/Predict work again. For a live class demo: upload the CSV right
  before you present. (Fixing this properly means saving the model to disk —
  ask if you want that.)
- The **Find Doctors map tiles** are a separate issue unrelated to deployment
  (they need a CARTO/OpenStreetMap tile source). The hospital search itself
  works regardless.

---

## Local development still works

Nothing here breaks running it on your own machine. As before:

```bash
# terminal 1
cd heart_dashboard/backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py                   # http://localhost:5001

# terminal 2, from the repo root
python -m http.server 8080      # http://localhost:8080
```
