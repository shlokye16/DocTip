"""Offline tests (no LLM, no network): uv run python test_engine.py"""
import sys

from engine import (Engine, Session, load_profiles, match_profile, extract_context,
                    match_fhir_conditions)

PROFILES = load_profiles()
ENGINE = Engine(PROFILES, judge=None)  # heuristic judge only

VAGUE = "I just think it is probably nothing."
SPECIFIC = ("His pain score is nine out of ten, hemoglobin is near baseline at seven, there is no fever, "
            "and the chest x-ray is clear so acute chest syndrome is unlikely.")
PLAN = "The plan is IV opioid at his individualized dose within thirty minutes with reassessment every hour."
CF = "Yes the same, because his history and findings point to the same diagnosis whoever he is."


def new():
    return Session("t")


def run(steps, s=None):
    """Visit-phase turns."""
    s = s or new()
    return s, [ENGINE.turn(s, t) for t in steps]


def debrief(first, steps=()):
    """Flag with `first` during the visit, end the visit, then answer the debrief questions.
    Returns (session, visit_reply, [end_visit_reply, *answer_replies])."""
    s = new()
    v = ENGINE.turn(s, first)
    out = [ENGINE.end_visit(s)] + [ENGINE.turn(s, t) for t in steps]
    return s, v, out


# ------------------------------------------------------------------ general
def test_quiet_when_not_flagged():
    _, rs = run(["Sixty-year-old with a sprained ankle, wrap and rest.", "Also mild seasonal allergies."])
    assert all(r["reply"] is None and not r["flagged"] for r in rs)


def test_context_extraction_and_no_false_positives():
    c = extract_context("58-year-old Black woman on Medicaid")
    assert c == {"age": 58, "sex": "woman", "race": "Black", "insurance": "medicaid"}, c
    assert "race" not in extract_context("white blood cell count is high, black stool noted")
    assert extract_context("a female patient")["sex"] == "woman"


# ------------------------------------------------------------------ visit phase (tips only)
def test_visit_gives_one_short_tip_and_no_question():
    s, (r,) = run(["58-year-old Black woman with sickle cell disease in a pain crisis. "
                   "I'm thinking ibuprofen and discharge."])
    assert r["flagged"] and r["just_flagged"] and r["phase"] == "visit"
    assert r["reply"].startswith("Tip:") and "individualized" in r["reply"] and "?" not in r["reply"]
    assert s.queue == [] and s.qa == []


def test_visit_profile_tip_when_no_concern_voiced():
    _, (r,) = run(["The patient has sickle cell disease."])
    assert r["reply"].startswith("Sickle cell tip:") and r["just_flagged"]


def test_visit_is_silent_after_tip_and_patient_speech_is_not_judged():
    s, rs = run(["Sickle cell disease, pain crisis.", "It hurts a lot, my back and legs.", "How long will this take?"])
    assert rs[0]["reply"] and rs[1]["reply"] is None and rs[2]["reply"] is None
    assert s.qa == [] and s.phase == "visit"


def test_new_concern_during_visit_gets_its_own_tip_once():
    s, rs = run(["Sickle cell crisis, admit for observation.",
                 "Honestly he is probably drug seeking anyway.",
                 "Yeah, drug seeking again."])
    assert "drug_seeking_label" in s.concerns and "drug-seeking" in rs[1]["reply"]
    assert not rs[1]["just_flagged"] and rs[2]["reply"] is None


def test_tips_are_capped_per_visit():
    s, rs = run(["Sickle cell crisis.", "He is drug seeking.", "Just ibuprofen.", "Normal vitals, looks comfortable."])
    assert sum(1 for r in rs if r["reply"]) == 3
    assert len(s.concerns) == 3  # the 4th concern is still recorded for the debrief


def test_end_visit_without_flag_closes_quietly():
    s, _ = run(["Sprained ankle."])
    r = ENGINE.end_visit(s)
    assert r["done"] and "Nothing was flagged" in r["reply"] and r["note"] is None


# ------------------------------------------------------------------ sickle cell debrief
def test_sickle_debrief_opens_after_end_visit():
    s, v, (d,) = debrief("58-year-old Black woman with sickle cell disease in a pain crisis. "
                         "I'm thinking ibuprofen and discharge.")
    assert d["phase"] == "debrief" and "visit is over" in d["reply"] and "individualized plan" in d["reply"]
    assert d["total_questions"] == 3 and s.queue[-1]["kind"] == "counterfactual"


def test_sickle_abbreviation_and_label():
    s, v, _ = debrief("SCD patient, probably drug seeking, wants more morphine.")
    assert v["flagged"] and "drug-seeking" in v["reply"]
    assert "drug_seeking_label" in s.concerns


def test_sickle_full_dialogue_with_followup():
    s, v, rs = debrief("42-year-old Black man with sickle cell disease, vaso-occlusive crisis.",
                       [VAGUE,      # insufficient -> follow-up
                        SPECIFIC,   # accepted -> Q2
                        PLAN,       # accepted -> Q3 counterfactual
                        CF])        # accepted -> close
    assert "acute chest syndrome" in rs[0]["reply"]
    assert "specific finding" in rs[1]["reply"] and not rs[1]["done"]
    assert rs[2]["reply"].startswith("Thanks.") and "individualized plan" in rs[2]["reply"]
    assert "consistency check" in rs[3]["reply"]
    assert rs[4]["done"] and "logged" in rs[4]["reply"]
    assert rs[4]["note"] and "sickle" in rs[4]["note"].lower() and "42-year-old Black man" in rs[4]["note"]
    assert len(s.qa) == 3 and s.qa[0]["specific"] is True  # follow-up combined with the first answer


def test_sickle_never_more_than_three_questions():
    s, v, _ = debrief("SCD patient, drug seeking, just ibuprofen, normal vitals, looks comfortable.")
    assert len(s.queue) == 3 and s.queue[-1]["kind"] == "counterfactual"
    assert [q["kind"] for q in s.queue] == ["concern", "concern", "counterfactual"]


def test_concern_voiced_mid_debrief_answer_is_inserted():
    s, v, _ = debrief("Sickle cell crisis, admit for observation.", ["Honestly he is probably drug seeking anyway."])
    assert "drug_seeking_label" in s.concerns
    assert any(q["id"] == "drug_seeking_label" for q in s.queue) and len(s.queue) <= 3


def test_addendum_after_close():
    s, v, rs = debrief("Sickle cell crisis.", [SPECIFIC, PLAN, CF])
    assert rs[-1]["done"]
    r = ENGINE.turn(s, "Actually I will increase the opioid dose and reassess in 15 minutes.")
    assert r["reply"] == "Added to the note." and len(s.addenda) == 1


def test_fhir_conditions_match():
    bundle = {"conditions": {"entry": [
        {"resource": {"code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "D57.00"}]}}}]}}
    assert [d["id"] for d in match_fhir_conditions(PROFILES, bundle)] == ["sickle_cell"]
    other = {"conditions": {"entry": [{"resource": {"code": {"text": "Seasonal allergic rhinitis"}}}]}}
    assert match_fhir_conditions(PROFILES, other) == []


# ------------------------------------------------------------------ chest pain / ACS
ACS_SPECIFIC = ("The ECG shows no ST changes, serial high-sensitivity troponins are flat, and her HEART score "
                "is low, so ACS, PE and dissection are unlikely.")
ACS_PLAN = "A HEART score of two supports observation and outpatient follow-up, and she returns if pain recurs within hours."


def test_acs_detection_and_two_concerns():
    s, v, _ = debrief("62-year-old woman with chest pain. I think it's anxiety, so I'll discharge her.")
    assert v["disease"]["id"] == "acs_chest_pain" and "objective findings" in v["reply"]
    assert [q["kind"] for q in s.queue] == ["concern", "concern", "counterfactual"]


def test_acs_abbreviation_mi_and_dialogue():
    s, v, rs = debrief("Rule out MI in a 55-year-old man.", [ACS_SPECIFIC, ACS_PLAN, CF])
    assert v["disease"]["id"] == "acs_chest_pain" and "ECG" in v["reply"]
    assert "ECG" in rs[0]["reply"]
    assert rs[2]["reply"].startswith("Thanks.") and "consistency check" in rs[2]["reply"]
    assert rs[3]["done"] and "Chest pain" in rs[3]["note"]


def test_named_disease_beats_symptom_trigger():
    _, (r,) = run(["Black woman with chest pain and sickle cell disease."])
    assert r["disease"]["id"] == "sickle_cell"


# ------------------------------------------------------------------ preeclampsia
PE_SPECIFIC = ("Her blood pressure was 128 over 80 on two checks four hours apart, urine protein is negative, "
               "platelets and liver enzymes are normal, and she has no headache or vision changes.")
PE_PLAN = "We recheck her blood pressure in forty-eight hours and she returns for severe headache, vision changes, or upper abdominal pain."


def test_preeclampsia_detection_with_benign_framing():
    s, v, _ = debrief("Postpartum Black woman with headache and swelling, probably just anxiety, preeclampsia unlikely.")
    assert v["disease"]["id"] == "preeclampsia" and "before calling it benign" in v["reply"]
    assert s.context["race"] == "Black" and s.context["sex"] == "woman"


def test_preeclampsia_hellp_and_full_dialogue():
    s, v, rs = debrief("Suspected HELLP in a 29-year-old pregnant woman.", [PE_SPECIFIC, PE_PLAN, CF])
    assert v["disease"]["id"] == "preeclampsia"
    assert rs[3]["done"] and len(s.qa) == 3 and all(a["specific"] for a in s.qa)


def test_word_boundary_blocks_fake_evidence():
    # "although" / "last" must not count as the lab abbreviations "alt" / "ast"
    s, v, rs = debrief("Preeclampsia is on the list.",
                       ["Although it is probably fine, last time was fine also and she seemed okay to me."])
    assert not rs[1]["done"] and "specific blood pressures" in rs[1]["reply"]


def test_pregnancy_alone_does_not_trigger():
    _, (r,) = run(["Pregnant patient with morning sickness, advise small frequent meals."])
    assert r["reply"] is None


# ------------------------------------------------------------------ cross-profile
def test_all_profiles_well_formed():
    import re
    assert len(PROFILES) == 3
    for d in PROFILES.values():
        qs = d["questions"]
        assert sum(q["kind"] == "counterfactual" for q in qs) == 1, d["id"]
        assert len({q["id"] for q in qs + d["concern_patterns"]}) == len(qs) + len(d["concern_patterns"]), d["id"]
        for c in d["concern_patterns"]:
            re.compile(c["pattern"])
            assert c["question"] and c["followup"]
        assert d["intro"] and d["guideline_note"].startswith("DRAFT") and d["evidence_terms"]


def test_fhir_other_profiles():
    def b(code):
        return {"conditions": {"entry": [{"resource": {"code": {"coding": [{"code": code}]}}}]}}
    assert [d["id"] for d in match_fhir_conditions(PROFILES, b("O14.10"))] == ["preeclampsia"]
    assert [d["id"] for d in match_fhir_conditions(PROFILES, b("I21.4"))] == ["acs_chest_pain"]
    assert match_fhir_conditions(PROFILES, b("J06.9")) == []


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa
                failed += 1
                print("FAIL", name, "->", repr(e))
    print(f"\n{'ALL PASSED' if not failed else str(failed) + ' FAILED'}")
    sys.exit(1 if failed else 0)
