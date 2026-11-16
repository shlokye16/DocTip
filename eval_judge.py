"""Evaluate Astra's answer-quality judge on a small labelled synthetic set.

Run with: uv run python eval_judge.py
This requires OPENAI_API_KEY and a valid LLM_MODEL. Results assess this prompt's
classification behavior only; they are not clinical validation.
"""
import json
from engine import load_profiles, llm_judge

PROFILES = load_profiles()
CASES = [
    ("sickle_cell", 0, "The patient has a pain score of 9/10, this is different from baseline, temperature is 38.4 C, and oxygen saturation is 91%.", True),
    ("sickle_cell", 0, "They come here all the time, so it is probably the same thing as usual.", False),
    ("sickle_cell", 1, "The individualized plan documents the patient's prior response, and I will document dose timing and reassess pain and respiratory status.", True),
    ("sickle_cell", 1, "I think this is enough because the patient looks comfortable.", False),
    ("acs_chest_pain", 0, "The ECG has no acute ischemic changes, serial high-sensitivity troponins are below the assay threshold, and the HEART score is 2.", True),
    ("acs_chest_pain", 0, "She is young and seems anxious, so I do not think it is cardiac.", False),
    ("acs_chest_pain", 1, "We have two non-rising troponins, a nonischemic ECG, documented risk score, and explicit return precautions.", True),
    ("acs_chest_pain", 1, "I usually discharge people like this.", False),
    ("preeclampsia", 0, "Blood pressures were 164/112 and 162/110 fifteen minutes apart, with headache and elevated liver enzymes.", True),
    ("preeclampsia", 0, "She is probably just stressed and pregnancy causes headaches.", False),
    ("preeclampsia", 1, "Repeat blood pressures, urine protein, platelets, creatinine, liver enzymes, and neurologic symptoms are documented.", True),
    ("preeclampsia", 1, "She says she feels okay now, so I am not concerned.", False),
]


def main():
    tp = fp = tn = fn = 0
    rows = []
    for profile_id, question_index, answer, expected in CASES:
        profile = PROFILES[profile_id]
        question = profile["questions"][question_index]
        result = llm_judge(profile, question, answer)
        predicted = bool(result["sufficient"])
        tp += predicted and expected
        fp += predicted and not expected
        tn += (not predicted) and (not expected)
        fn += (not predicted) and expected
        rows.append({"profile": profile_id, "expected": expected, "predicted": predicted, "answer": answer})
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    report = {"n": len(CASES), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
              "accuracy": (tp + tn) / len(CASES), "precision": precision,
              "recall": recall, "f1": f1, "cases": rows,
              "note": "Synthetic prompt-evaluation set; not clinical validation."}
    with open("judge_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
    print("Case report saved to judge_eval_report.json")


if __name__ == "__main__":
    main()
