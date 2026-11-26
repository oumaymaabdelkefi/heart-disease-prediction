"""Data-drift monitoring for the heart disease model.

The reference distribution is the training split the model was fitted on.
Each incoming batch is compared against it feature by feature:

* continuous features  -> two-sample Kolmogorov-Smirnov test
* categorical features -> chi-square test of homogeneity on the category counts
* every feature        -> Population Stability Index (PSI), an effect-size measure

Running 13 tests per batch inflates false alarms (at alpha=0.05 there is a
~49% chance at least one clean feature "fails" by luck), so p-values are
compared against a Bonferroni-corrected threshold alpha / n_features.

A feature is flagged as drifted only when the test is significant AND the
PSI shows a non-trivial shift (PSI >= 0.1). Requiring both stops large batches
from flagging tiny, irrelevant differences that happen to be significant.

Prediction drift is checked too: the KS test on the model's predicted risk
for the reference set vs the batch.
"""

import numpy as np
import pandas as pd
from scipy import stats

CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

ALPHA = 0.05
PSI_MODERATE = 0.1   # PSI < 0.1: stable, 0.1-0.25: moderate shift, >= 0.25: major shift
PSI_MAJOR = 0.25
N_BINS = 10
EPS = 1e-4
PRED_ALPHA = 0.01      # prediction drift: KS on predicted risk ...
PRED_MIN_SHIFT = 5.0   # ... and mean risk moved by at least 5 percentage points

SCENARIOS = {
    "none":        "No shift — batches resampled from the training population",
    "aging":       "Older population — age rises, max heart rate falls",
    "cholesterol": "Lab assay change — cholesterol readings inflated",
    "bp_sensor":   "Miscalibrated BP cuff — resting BP offset upward",
    "female":      "Demographic shift — far more female patients",
    "case_mix":    "Case-mix shift — clinic referrals skew toward sick patients",
}


# ─── statistics ──────────────────────────────────────────────────────────────

def _bin_edges(ref):
    """Quantile bin edges from the reference, so each bin holds ~10% of it."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, N_BINS + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _psi(ref_frac, cur_frac):
    ref_frac = np.clip(ref_frac, EPS, None)
    cur_frac = np.clip(cur_frac, EPS, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def _test_feature(name, ref, cur):
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)

    if name in CONTINUOUS:
        statistic, p = stats.ks_2samp(ref, cur)
        edges = _bin_edges(ref)
        ref_counts = np.histogram(ref, edges)[0]
        cur_counts = np.histogram(cur, edges)[0]
        # open-ended outer bins, so values outside the reference range still count
        inner = edges[1:-1]
        labels = ([f"< {inner[0]:g}"]
                  + [f"{lo:g}–{hi:g}" for lo, hi in zip(inner[:-1], inner[1:])]
                  + [f"≥ {inner[-1]:g}"])
        test = "KS"
    else:
        cats = sorted(set(ref) | set(cur))
        ref_counts = np.array([(ref == c).sum() for c in cats])
        cur_counts = np.array([(cur == c).sum() for c in cats])
        table = np.vstack([ref_counts, cur_counts])
        table = table[:, table.sum(axis=0) > 0]
        if table.shape[1] < 2:
            statistic, p = 0.0, 1.0
        else:
            statistic, p, _, _ = stats.chi2_contingency(table)
        labels = [f"{int(c)}" for c in cats]
        test = "Chi²"

    ref_frac = ref_counts / max(ref_counts.sum(), 1)
    cur_frac = cur_counts / max(cur_counts.sum(), 1)
    psi = _psi(ref_frac, cur_frac)

    return {
        "feature": name,
        "test": test,
        "statistic": round(float(statistic), 4),
        "p_value": float(p),
        "psi": round(psi, 4),
        "ref_mean": round(float(ref.mean()), 2),
        "cur_mean": round(float(cur.mean()), 2),
        "hist": {
            "labels": labels,
            "reference": [round(float(v) * 100, 1) for v in ref_frac],
            "current": [round(float(v) * 100, 1) for v in cur_frac],
        },
    }


def _severity(psi, significant):
    if not significant or psi < PSI_MODERATE:
        return "stable"
    return "major" if psi >= PSI_MAJOR else "moderate"


def detect_drift(reference, batch, feature_cols, predict_proba=None):
    """Compare one batch against the reference. Returns a JSON-able report."""
    threshold = ALPHA / len(feature_cols)
    features = []
    for f in feature_cols:
        r = _test_feature(f, reference[f], batch[f])
        r["significant"] = r["p_value"] < threshold
        r["status"] = _severity(r["psi"], r["significant"])
        r["p_value"] = float(f"{r['p_value']:.3g}")
        features.append(r)

    drifted = [r for r in features if r["status"] != "stable"]

    prediction = None
    if predict_proba is not None:
        ref_p = predict_proba(reference[feature_cols])
        cur_p = predict_proba(batch[feature_cols])
        ks, p = stats.ks_2samp(ref_p, cur_p)
        shift_pts = float(cur_p.mean() - ref_p.mean()) * 100
        prediction = {
            "ref_mean_risk": round(float(ref_p.mean()) * 100, 1),
            "cur_mean_risk": round(float(cur_p.mean()) * 100, 1),
            "shift_pts": round(shift_pts, 1),
            "ks": round(float(ks), 4),
            "p_value": float(f"{p:.3g}"),
            # same idea as for features: significant AND a material effect size
            "drifted": bool(p < PRED_ALPHA and abs(shift_pts) >= PRED_MIN_SHIFT),
        }

    share = len(drifted) / len(feature_cols)
    if any(r["status"] == "major" for r in drifted) or share >= 0.3:
        overall = "major"
    elif drifted:
        overall = "moderate"
    elif prediction and prediction["drifted"]:
        # Shift spread thinly over many features can slip past every per-feature
        # test yet still move the model's output — the case that matters most.
        overall = "moderate"
    else:
        overall = "stable"

    return {
        "n_rows": int(len(batch)),
        "n_reference": int(len(reference)),
        "alpha": ALPHA,
        "corrected_alpha": round(threshold, 5),
        "n_drifted": len(drifted),
        "drifted_features": [r["feature"] for r in drifted],
        "max_psi": round(max(r["psi"] for r in features), 4),
        "overall": overall,
        "features": sorted(features, key=lambda r: -r["psi"]),
        "prediction": prediction,
    }


# ─── simulation ──────────────────────────────────────────────────────────────

def _apply_shift(df, scenario, k, rng, pool):
    """Return a shifted copy of df. k in [0, 1] is how strong the shift is."""
    df = df.copy()
    n = len(df)
    if k <= 0 or scenario == "none":
        return df

    if scenario == "aging":
        years = 10 * k
        df["age"] = df["age"] + years + rng.normal(0, 2, n)
        # max heart rate declines ~0.7 bpm/year of age (Tanaka formula)
        df["thalach"] = df["thalach"] - 0.7 * years + rng.normal(0, 3, n)
    elif scenario == "cholesterol":
        df["chol"] = df["chol"] * (1 + 0.25 * k) + rng.normal(0, 5, n)
    elif scenario == "bp_sensor":
        df["trestbps"] = df["trestbps"] + 18 * k + rng.normal(0, 3, n)
    elif scenario in ("female", "case_mix"):
        # covariate / prior shift: resample from the pool with skewed weights
        if scenario == "female":
            w = np.where(pool["sex"] == 0, 1 + 8 * k, 1.0)
        else:
            w = np.where(pool["target"] == 1, 1 + 8 * k, 1.0)
        w = w / w.sum()
        idx = rng.choice(len(pool), size=n, replace=True, p=w)
        df = pool.iloc[idx].reset_index(drop=True).copy()

    # keep values physically plausible
    df["age"] = df["age"].clip(18, 100).round()
    df["thalach"] = df["thalach"].clip(60, 210).round()
    df["trestbps"] = df["trestbps"].clip(80, 220).round()
    df["chol"] = df["chol"].clip(100, 700).round()
    return df


def simulate_stream(pool, scenario, intensity, batch_size, n_batches, onset, seed=42):
    """Yield (index, drift_strength, batch_df).

    `pool` should be the reference (training) population: a clean batch is then
    a bootstrap sample of the same population, so any drift the tests report is
    exactly the shift injected here. (Bootstrapping a small held-out split would
    instead amplify its chance differences from the training set into false alarms.)

    Batches before `onset` are clean resamples of `pool`. From `onset` on,
    the shift ramps linearly up to `intensity` by the last batch, which
    mimics drift that creeps in gradually rather than all at once.
    """
    rng = np.random.default_rng(seed)
    ramp_len = max(n_batches - onset, 1)
    for i in range(n_batches):
        if i < onset:
            k = 0.0
        else:
            k = intensity * min((i - onset + 1) / ramp_len, 1.0)
        idx = rng.choice(len(pool), size=batch_size, replace=True)
        base = pool.iloc[idx].reset_index(drop=True)
        yield i, round(k, 3), _apply_shift(base, scenario, k, rng, pool)
