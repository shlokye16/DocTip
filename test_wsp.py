"""Offline quality gates for the synthetic weight-space experiment."""
import copy
import numpy as np
from common import load_weights
from wsp_edit import validate, fallback_edit, robust_evaluate


def test_weight_validation_rejects_wrong_shape():
    original = load_weights("weights_biased.json")
    candidate = copy.deepcopy(original)
    candidate["W1"] = [[0.0]]
    try:
        validate(candidate, original)
    except ValueError:
        return
    raise AssertionError("wrong-shaped weight matrix was accepted")


def test_weight_validation_rejects_non_finite():
    original = load_weights("weights_biased.json")
    candidate = copy.deepcopy(original)
    candidate["b2"] = [float("nan")]
    try:
        validate(candidate, original)
    except ValueError:
        return
    raise AssertionError("non-finite weight was accepted")


def test_manual_baseline_reduces_planted_disparity():
    original = load_weights("weights_biased.json")
    baseline = fallback_edit(original)
    before = robust_evaluate(original)
    after = robust_evaluate(baseline)
    assert after["mean_max_disparity"] < before["mean_max_disparity"] * 0.5
    assert after["mean_accuracy"] >= before["mean_accuracy"] - 0.02
    assert after["mean_brier"] <= before["mean_brier"] + 0.02


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("ALL WSP QUALITY GATES PASSED")
