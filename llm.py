"""OpenAI-compatible JSON helper with bounded request timeouts.

The LLM is optional for the demo. Engine-level heuristic judging is used if it times out
or the API is unavailable.
"""
import json
import os
import re

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-6-astra")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "8"))
_client = None


def client():
    global _client
    if _client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install project dependencies with `uv sync` to enable Astra calls.") from exc
        _client = OpenAI(
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )
    return _client


def parse_json(text):
    text = re.sub(r"```(?:json)?", "", (text or "")).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def chat_json(system, user, model=None):
    response = client().chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    return parse_json(response.choices[0].message.content)
