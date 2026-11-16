"""Step 2: Weight-Space Programming.

A general-purpose LLM reads the raw weights (as JSON text) and writes edited weights.
No gradient descent, no training loop, no loss function.

Usage:
  python wsp_edit.py              # ask a local Ollama model (ollama serve must be running)
  python wsp_edit.py --fallback   # hand-written baseline edit (NOT WSP) for demo backup
"""
import json
import os
import sys
import numpy as np
from common import (FEATURES, PROTECTED, load_weights, save_weights, evaluate, make_data, forward, disparity)
from llm import chat_json, DEFAULT_MODEL

MODEL = os.getenv("WSP_MODEL", DEFAULT_MODEL)  # bigger local model = better edits
KEYS = ["W1", "b1", "W2", "b2"]

PROMPT = """You are performing Weight-Space Programming: directly reading and editing a
neural network's weights. Do NOT write training code or simulate gradient descent. Reason
about what the weights compute and change the numbers.

NETWORK
{architecture}
Input order (index -> name): {features}
W1 has shape (8 inputs, 8 hidden): row i holds the weights from input feature i to each
hidden unit. b1 shape (8,). W2 shape (8, 1). b2 shape (1,).
Output = sigmoid(relu(x @ W1 + b1) @ W2 + b2) = urgency of a cardiac workup.

PROBLEM
This model was trained on historically biased labels. Its output changes when the protected
attributes {protected} are flipped while all clinical features stay identical. A fair model
should not change its output when only those attributes change, but must keep using the
clinical features (age, chest pain severity, ECG, troponin, diabetes, smoker).

TASK
1. Work out how the protected attributes influence the output through the weights.
2. Edit the weights so that influence is removed, with the smallest edit that works.
3. Keep the model's use of clinical features and its overall calibration as intact as possible.

CURRENT WEIGHTS (JSON)
{weights}

Return ONLY a JSON object with keys: "reasoning" (short string), "W1", "b1", "W2", "b2",
with exactly the same shapes as above."""


def shapes(w):
    return {k: np.array(w[k]).shape for k in KEYS}


def validate(new, old):
    for k in KEYS:
        if k not in new:
            raise ValueError(f"missing key {k}")
    if shapes(new) != shapes(old):
        raise ValueError(f"shape mismatch: {shapes(new)} vs {shapes(old)}")
    for k in KEYS:
        if not np.all(np.isfinite(np.array(new[k], dtype=float))):
            raise ValueError(f"non-finite values in {k}")


def ask_llm(w, feedback=""):
    payload = {k: w[k] for k in KEYS}
    prompt = PROMPT.format(
        architecture=w["architecture"],
        features=", ".join(f"{i}={n}" for i, n in enumerate(FEATURES)),
        protected=", ".join(PROTECTED),
        weights=json.dumps(payload),
    ) + (f"\n\nYour previous answer was rejected: {feedback}. Try again." if feedback else "")
    return chat_json("You edit neural network weights directly and reply with JSON only.",
                     prompt, model=MODEL)


def fallback_edit(w):
    """Baseline, NOT weight-space programming: zero the input rows of protected features."""
    new = json.loads(json.dumps(w))
    for f in PROTECTED:
        new["W1"][FEATURES.index(f)] = [0.0] * len(new["W1"][0])
    new["reasoning"] = "Manual baseline: zeroed the W1 rows for protected attributes."
    return new


def frac_changed(old, new):
    ch = tot = 0
    for k in KEYS:
        a, b = np.array(old[k], float), np.array(new[k], float)
        ch += int(np.sum(np.abs(a - b) > 1e-6))
        tot += a.size
    return ch / tot


def robust_evaluate(w, seeds=(99, 2026, 7301, 48117)):
    """Evaluate on several fixed synthetic holdout populations, not the training sample."""
    rows = []
    for seed in seeds:
        X, _, p_true = make_data(5000, seed=seed)
        p = forward(w, X)
        rows.append({
            "seed": seed,
            "disparity": disparity(w, X),
            "accuracy_vs_unbiased_need": float(np.mean((p > 0.5) == (p_true > 0.5))),
            "brier_vs_unbiased_need": float(np.mean((p - p_true) ** 2)),
        })
    return {
        "seeds": list(seeds),
        "mean_max_disparity": float(np.mean([r["disparity"]["max"] for r in rows])),
        "mean_sex_disparity": float(np.mean([r["disparity"]["female"] for r in rows])),
        "mean_insurance_disparity": float(np.mean([r["disparity"]["medicaid"] for r in rows])),
        "mean_accuracy": float(np.mean([r["accuracy_vs_unbiased_need"] for r in rows])),
        "mean_brier": float(np.mean([r["brier_vs_unbiased_need"] for r in rows])),
        "per_seed": rows,
    }


def main():
    old = load_weights("weights_biased.json")
    source = "baseline"
    if "--fallback" in sys.argv:
        new = fallback_edit(old)
    else:
        source, feedback, new = "llm-wsp", "", None
        for attempt in range(3):
            try:
                cand = ask_llm(old, feedback)
                validate(cand, old)
                new = cand
                break
            except Exception as e:  # bad JSON, wrong shape, API error
                feedback = str(e)
                print(f"attempt {attempt + 1} failed: {feedback}")
        if new is None:
            print("LLM edit failed 3 times -> use --fallback or report the negative result.")
            sys.exit(1)

    patched = {"architecture": old["architecture"], "features": old["features"],
               **{k: new[k] for k in KEYS}}

    # Reject numerically extreme edits before evaluating them.
    max_abs_weight = max(float(np.max(np.abs(np.asarray(patched[k], dtype=float)))) for k in KEYS)
    before, after = robust_evaluate(old), robust_evaluate(patched)
    disparity_drop = before["mean_max_disparity"] - after["mean_max_disparity"]
    accuracy_delta = after["mean_accuracy"] - before["mean_accuracy"]
    brier_delta = after["mean_brier"] - before["mean_brier"]
    accepted = (
        max_abs_weight <= 25.0
        and disparity_drop >= max(0.01, before["mean_max_disparity"] * 0.50)
        and accuracy_delta >= -0.02
        and brier_delta <= 0.02
    )
    report = {
        "source": source,
        "model": MODEL if source == "llm-wsp" else None,
        "reasoning": new.get("reasoning", ""),
        "fraction_of_weights_changed": frac_changed(old, patched),
        "max_abs_weight": max_abs_weight,
        "accepted": accepted,
        "acceptance_criteria": {
            "max_abs_weight_lte": 25.0,
            "disparity_reduction_at_least": "50% and 0.01 absolute",
            "accuracy_drop_at_most": 0.02,
            "brier_increase_at_most": 0.02,
        },
        "before": before,
        "after": after,
    }
    with open("wsp_report.json", "w") as f:
        json.dump(report, f, indent=1)

    print(f"\nSource: {source}")
    print("Reasoning:", report["reasoning"])
    print(f"Weights changed: {report['fraction_of_weights_changed']:.1%}")
    print(f"Mean max disparity: {before['mean_max_disparity']:.3f} -> {after['mean_max_disparity']:.3f}")
    print(f"Mean accuracy:      {before['mean_accuracy']:.3f} -> {after['mean_accuracy']:.3f}")
    print(f"Mean Brier score:   {before['mean_brier']:.3f} -> {after['mean_brier']:.3f}")
    print("Edit accepted:" , accepted)
    if accepted:
        save_weights("weights_patched.json", patched)
    else:
        print("Edit rejected; weights_patched.json was not overwritten. Inspect wsp_report.json.")
        sys.exit(2)


if __name__ == "__main__":
    main()
