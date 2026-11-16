# Clinician Reasoning Check — SacHacks prototype

An ambient-listening, clinician-facing reasoning prompt for seven configured condition profiles:
sickle cell disease, chest pain / ischemic heart disease / acute coronary syndrome,
preeclampsia, endometriosis, HIV/AIDS, depression, and tuberculosis.

The additional profiles address pain dismissal, identity-based assumptions, interpretation
of distress, and barriers to care. Each has explicit disease triggers, short visit tips,
condition-specific debrief questions, a consistency check, and ICD-10 matching for the
CDS Hooks demo. Source links are included in the profiles. Depression coverage does
not extend to every mental illness; cardiac coverage does not cover every heart disorder.
Demographic characteristics alone do not trigger these profiles. One condition is
selected per encounter (the first detected profile); start a new encounter to change it.
Mention matching does not determine whether a condition is present, ruled out, or historical.
These prompts do not provide emergency triage or replace clinical assessment.

**Demo only. Synthetic/demo use only. Not a diagnostic tool, not clinically validated, and not approved for production or EHR deployment.** The profile text is draft material and needs clinician review.

## What is in this build

- A quiet assistant orb with two phases. **During the visit**, with explicit consent, the browser continuously transcribes nearby conversation and sends final transcript segments to the local demo API. When a configured condition (or a configured concern) is mentioned, the assistant shows one short tip for the clinician and plays a soft, low beep (max 3 tips per visit). It never asks questions while the patient is in the room. **After the visit**, the clinician presses **End visit** and the assistant runs the longer Socratic conversation (answer by voice or typing) and produces the reasoning note.
- A deterministic, profile-based engine: short tips during the visit, then up to three Socratic questions after it, with at most one follow-up per question. Tip and question wording both live in the reviewed JSON profiles.
- GPT-6 Astra via the OpenAI API to judge whether answers contain specific, objective reasoning. If the API is unavailable or times out, a local heuristic judge is used.
- A CDS Hooks discovery endpoint and a `patient-view` service that returns an informational card when the FHIR Conditions prefetch contains one of the configured conditions.
- In-memory demo sessions. No patient data should be entered; no EHR writes occur.

## Requirements and setup

Install [uv](https://docs.astral.sh/uv/), Node.js, and npm. Configure your OpenAI API key:

```bash
export OPENAI_API_KEY="..."
export LLM_MODEL="gpt-6-astra"       # verify the exact model ID enabled for your API account
export LLM_TIMEOUT_SECONDS=8         # bounded LLM request timeout
npm run setup
npm run dev
```

Open http://localhost:5173 in the latest Google Chrome. Click **Start listening** and allow microphone access when prompted. The app now checks microphone permission before starting speech recognition, reports permission/device/network errors visibly, confirms when recognition starts, and retries transient browser restarts. The browser Web Speech API is used for continuous speech recognition. The only sound is a short, soft beep when a tip appears; there is no text-to-speech or spoken AI output. Browser speech recognition may process audio using the browser vendor's speech service; do not assume audio stays on-device. Obtain consent from everyone in the room and follow institutional privacy requirements. If speech recognition is unsupported, the interface explains that limitation. Speech recognition is not uniformly supported across browsers; use Chrome for this prototype. If the browser says the microphone is active but no transcript appears, check the OS microphone privacy setting, Chrome site permissions, selected input device, and whether another application is holding the microphone.

`npm run setup` runs `uv sync`, installs npm dependencies, and trains the separate synthetic WSP toy model. The conversation engine and CDS Hooks service do not depend on the WSP experiment. The WSP toy network is a separate synthetic chest-pain urgency classifier; its results do not validate the disease-specific conversation prompts.

## Deploy on Render

The root `render.yaml` deploys a single free Python web service. FastAPI serves
the frontend, orb image, evidence page, and API from the same HTTPS origin; Node
and Vite are only needed for local frontend development.

1. Push these deployment changes to GitHub.
2. In Render, choose **New > Blueprint**, connect this repository, and select `main`.
3. Apply the Blueprint and wait for `/health` to pass, then open the service URL.
4. Optionally add `OPENAI_API_KEY` and `LLM_MODEL` in the service's Environment
   settings, using a model ID available to your account. Without a key the demo
   uses its heuristic judge. Never commit API keys.

For manual service creation, select Python, use `uv sync --frozen` as the build
command and `uv run uvicorn server:app --host 0.0.0.0 --port $PORT --workers 1`
as the start command. Set the health check path to `/health`.

Keep one worker and one instance because sessions live in process memory. Visits
are lost on restarts/redeploys. Free services can sleep when idle. Use synthetic
demo data only; this deployment does not add authentication or clinical readiness.

## Run tests

```bash
npm test
```

This runs the offline engine, WSP quality-gate, and FastAPI smoke tests; it does not call the OpenAI API. The Astra judge evaluation is a separate command.

## API

- `POST /api/sessions` creates an in-memory conversation session.
- `POST /api/sessions/{session_id}/turn` accepts `{"text":"..."}`. During the visit it returns a short tip (or no reply); after `/end` it returns the next debrief question. The response includes `phase` (`visit`, `debrief`, or `closed`).
- `POST /api/sessions/{session_id}/end` ends the visit and returns the first debrief question (or closes quietly if nothing was flagged).
- `GET /api/sessions/{session_id}/note` returns the reasoning note after the dialogue closes.
- `DELETE /api/sessions/{session_id}` removes a session.
- `GET /health` reports service health and the number of loaded condition profiles.

The LLM request timeout is controlled by `LLM_TIMEOUT_SECONDS` (default 8 seconds). Automatic retries are disabled to keep latency bounded. If a model call fails, the engine falls back to its deterministic heuristic judge. The speech transcript can contain both clinician and patient speech: browser speech recognition does not reliably identify speakers, so this prototype cannot know who said a diagnosis or distinguish patient statements from clinician assessments. It may therefore flag a condition mentioned by either speaker.

## CDS Hooks / Epic integration path

This prototype exposes the CDS Hooks discovery endpoint:

`GET /cds-services`

It advertises one service:

- Hook: `patient-view`
- Service ID: `clinician-reasoning-check`
- Prefetch: `Condition?patient={{context.patientId}}&_count=100`

The hook endpoint is:

`POST /cds-services/clinician-reasoning-check`

Example request (abbreviated):

```json
{
  "hook": "patient-view",
  "hookInstance": "demo-instance-001",
  "context": {"patientId": "example-patient"},
  "prefetch": {
    "conditions": {
      "resourceType": "Bundle",
      "entry": [{
        "resource": {
          "resourceType": "Condition",
          "code": {"coding": [{
            "system": "http://hl7.org/fhir/sid/icd-10-cm",
            "code": "D57.00"
          }]}
        }
      }]
    }
  }
}
```

A matching Condition returns an informational CDS Hooks card. The card does not prescribe treatment or make a diagnosis. The endpoint currently supports the `patient-view` hook only. It is a protocol-shaped demo endpoint, not a completed Epic integration: real deployment requires an approved CDS Hooks registration/configuration, EHR-specific validation, authentication, authorization, security review, privacy controls, clinical governance, and workflow testing. The service currently stores sessions in process memory and is not production-ready.

## Ambient listening boundaries

- Listening starts only after the user presses **Start listening**. The user must obtain appropriate consent from all people whose speech may be captured.
- The browser converts speech to text. The app sends recognized transcript segments to the local demo API; the browser's speech-recognition implementation may process audio outside the device. Check browser/provider terms and institutional policies before any real-world use.
- Speaker diarization is not implemented. Clinician and patient speech may be mixed, and a condition mentioned by either person may trigger the reasoning check. This is a known prototype limitation, not reliable clinician-intent detection.
- No AI speech is played. Tips and questions are displayed for the clinician to read; a soft beep marks a new tip.

## WSP evaluation

`wsp_edit.py` evaluates edits on four deterministic synthetic holdout populations. An edit is accepted only if it reduces mean maximum counterfactual disparity by at least 50% and 0.01 absolute, reduces accuracy by no more than 0.02, increases Brier score by no more than 0.02, and keeps absolute weights below 25. Rejected edits are reported and do not overwrite `weights_patched.json`. `npm run wsp:fallback` is a hand-written baseline that zeros protected-feature input weights; it is explicitly **not WSP**. Only an Astra-produced weight edit should be described as the WSP experiment. These are toy-model metrics, not clinical validation or proof that bias has been removed. Run `npm run eval:judge` with a valid Astra API key to evaluate the LLM answer judge on 12 labelled synthetic examples; inspect precision, recall, F1, and the saved case report. The benchmark is small and hand-authored, so it is a prompt regression check rather than evidence of clinical performance.

## Clinical and safety boundaries

- The assistant asks for reasoning in a warm, indirect style; it does not decide whether the diagnosis is correct or recommend treatment.
- Disease triggers and prompts are deterministic and defined in the JSON profiles. The LLM is not allowed to invent clinical questions.
- Clinical profile notes are marked DRAFT and must be reviewed against current authoritative guidance by qualified clinicians.
- A prompt or logged answer is not proof that bias was absent, that care was equitable, or that liability was reduced. This tool does not replace implicit-bias training, clinical governance, or legal/compliance review.
- This prototype does not replace implicit-bias education, clinical judgment, institutional quality programs, or legal/compliance review.
- Do not use real patient information in this hackathon build.

## WSP experiment

The separate `npm run wsp` command attempts a weight-space edit of a synthetic toy classifier. `npm run wsp:fallback` applies a manually defined baseline edit and must not be presented as an LLM WSP result. Report failed edits honestly and compare any claimed improvement against the counterfactual test and accuracy metrics.
