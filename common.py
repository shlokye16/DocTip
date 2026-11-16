"""Shared pieces: synthetic triage data, tiny MLP weights, counterfactual metrics.

Everything here is SYNTHETIC. Not for clinical use.
"""
import json
import numpy as np

FEATURES = [
    "age_norm",            # (age - 55) / 15
    "chest_pain_severity", # 0..1
    "ecg_abnormal",        # 0/1
    "troponin_elevated",   # 0/1
    "diabetes",            # 0/1
    "smoker",              # 0/1
    "female",              # 0/1  <- protected
    "medicaid",            # 0/1  <- protected (insurance proxy)
]
PROTECTED = ["female", "medicaid"]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_data(n, seed):
    """Returns X, y_hist (biased historical labels), p_true (unbiased clinical need)."""
    rng = np.random.default_rng(seed)
    age = rng.integers(30, 86, n)
    X = np.column_stack([
        (age - 55) / 15,
        rng.uniform(0, 1, n),
        rng.binomial(1, 0.25, n),
        rng.binomial(1, 0.20, n),
        rng.binomial(1, 0.25, n),
        rng.binomial(1, 0.30, n),
        rng.binomial(1, 0.50, n),
        rng.binomial(1, 0.30, n),
    ]).astype(np.float64)

    clinical = (-3.0 + 0.8 * X[:, 0] + 2.0 * X[:, 1] + 1.8 * X[:, 2]
                + 3.0 * X[:, 3] + 0.6 * X[:, 4] + 0.5 * X[:, 5])
    p_true = sigmoid(clinical)

    # PLANTED BIAS: historical labels under-call urgency for women and Medicaid patients
    biased_logit = clinical - 1.6 * X[:, 6] - 0.7 * X[:, 7]
    y_hist = rng.binomial(1, sigmoid(biased_logit))
    return X, y_hist, p_true


# ---------- weights as plain JSON (this is what the LLM reads/writes) ----------
def export_weights(clf):
    return {
        "architecture": "8 inputs -> 8 ReLU hidden units -> 1 sigmoid output (urgency of cardiac workup)",
        "features": FEATURES,
        "W1": np.round(clf.coefs_[0], 3).tolist(),       # shape (8 inputs, 8 hidden)
        "b1": np.round(clf.intercepts_[0], 3).tolist(),  # shape (8,)
        "W2": np.round(clf.coefs_[1], 3).tolist(),       # shape (8, 1)
        "b2": np.round(clf.intercepts_[1], 3).tolist(),  # shape (1,)
    }


def save_weights(path, w):
    with open(path, "w") as f:
        json.dump(w, f, indent=1)


def load_weights(path):
    with open(path) as f:
        return json.load(f)


def forward(w, X):
    X = np.asarray(X, dtype=np.float64)
    h = np.maximum(0, X @ np.array(w["W1"]) + np.array(w["b1"]))
    z = h @ np.array(w["W2"]) + np.array(w["b2"])
    return sigmoid(z).ravel()


# ---------- counterfactual testing ----------
def flip(X, feature):
    X2 = np.array(X, dtype=np.float64, copy=True)
    i = FEATURES.index(feature)
    X2[..., i] = 1 - X2[..., i]
    return X2


def disparity(w, X):
    """Mean |p(x) - p(x with one protected attribute flipped)| over a population."""
    p = forward(w, X)
    out = {f: float(np.mean(np.abs(p - forward(w, flip(X, f))))) for f in PROTECTED}
    out["max"] = max(out.values())
    return out


def evaluate(w):
    X, _, p_true = make_data(3000, seed=99)
    p = forward(w, X)
    return {
        "disparity": disparity(w, X),
        "accuracy_vs_unbiased_need": float(np.mean((p > 0.5) == (p_true > 0.5))),
    }


def encounter_to_x(e):
    """e: dict with age, chest_pain_severity, ecg_abnormal, troponin_elevated,
    diabetes, smoker, female, medicaid"""
    return np.array([
        (float(e["age"]) - 55) / 15,
        float(e["chest_pain_severity"]),
        float(bool(e["ecg_abnormal"])),
        float(bool(e["troponin_elevated"])),
        float(bool(e["diabetes"])),
        float(bool(e["smoker"])),
        float(bool(e["female"])),
        float(bool(e["medicaid"])),
    ])
