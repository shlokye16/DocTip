"""Clinician reasoning-check API and CDS Hooks demo service.

Synthetic/demo use only. The CDS Hooks endpoint returns a static informational card
when an encounter contains one of the configured conditions. It does not diagnose,
recommend treatment, or write to an EHR.
"""
import json
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from engine import Engine, Session, load_profiles, llm_judge, build_note, match_fhir_conditions

APP_VERSION = "0.2.0"
PROFILES = load_profiles()
ENGINE = Engine(PROFILES, judge=llm_judge)
SESSIONS: dict[str, Session] = {}

app = FastAPI(
    title="Clinician Reasoning Check",
    version=APP_VERSION,
    description="Synthetic hackathon prototype; not for clinical use.",
)

WEB_ROOT = Path(__file__).resolve().parent


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def homepage():
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/evals.html", include_in_schema=False)
def evidence_page():
    return FileResponse(WEB_ROOT / "evals.html")


@app.get("/blob.png", include_in_schema=False)
def orb_image():
    return FileResponse(WEB_ROOT / "blob.png")


class TurnRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


def _cds_card(profile: dict[str, Any], hook_instance: str) -> dict[str, Any]:
    # Keep card content static and sourced from the locally reviewed profile.
    # No patient identifiers or encounter details are sent to an LLM.
    return {
        "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{hook_instance}:{profile['id']}")),
        "summary": f"Reasoning check available: {profile['name']}",
        "indicator": "info",
        "detail": (
            f"{profile['intro']} This optional prompt helps document objective reasoning. "
            "It does not diagnose or recommend treatment.\n\n"
            f"Reference note: {profile.get('guideline_note', 'Draft reference; clinician review required.')}"
        ),
        "source": {
            "label": "Clinician Reasoning Check (SaHacks prototype)",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION, "profiles_loaded": len(PROFILES)}


def _read_json(name: str):
    path = Path(__file__).parent / name
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


@app.get("/api/evals")
def evals():
    """Latest WSP and judge reports for the evidence page. Missing files return null."""
    return {
        "wsp": _read_json("wsp_report.json"),
        "judge": _read_json("judge_eval_report.json"),
        "weights_patched": _read_json("weights_patched.json"),
    }


@app.get("/cds-services")
def cds_discovery():
    """CDS Hooks discovery endpoint used by an EHR to find available services."""
    return {
        "services": [
            {
                "hook": "patient-view",
                "id": "clinician-reasoning-check",
                "title": "Clinician Reasoning Check",
                "description": "Offers a brief reasoning prompt for selected conditions.",
                "prefetch": {
                    "conditions": "Condition?patient={{context.patientId}}&_count=100"
                },
            }
        ]
    }


@app.post("/cds-services/clinician-reasoning-check")
def cds_hook(payload: dict[str, Any]):
    hook = payload.get("hook")
    if hook != "patient-view":
        raise HTTPException(status_code=400, detail="Unsupported hook; expected patient-view.")

    context = payload.get("context") or {}
    patient_id = context.get("patientId")
    if not patient_id:
        raise HTTPException(status_code=400, detail="context.patientId is required.")

    hook_instance = str(payload.get("hookInstance") or uuid.uuid4())
    prefetch = payload.get("prefetch") or {}
    matched = match_fhir_conditions(PROFILES, {"conditions": prefetch.get("conditions")})
    cards = [_cds_card(profile, hook_instance) for profile in matched]
    return {"cards": cards}


@app.post("/api/sessions")
def create_session():
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = Session(session_id)
    return {"session_id": session_id}


@app.post("/api/sessions/{session_id}/turn")
def session_turn(session_id: str, request: TurnRequest):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    result = ENGINE.turn(session, request.text)
    return result


@app.post("/api/sessions/{session_id}/end")
def session_end(session_id: str):
    """End the visit and start the longer debrief conversation."""
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    return ENGINE.end_visit(session)


@app.get("/api/sessions/{session_id}/note")
def session_note(session_id: str):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    if not session.closed or not session.profile_id:
        return {"ready": False, "note": None}
    return {"ready": True, **build_note(session, PROFILES[session.profile_id])}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    if SESSIONS.pop(session_id, None) is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    return {"deleted": True}
