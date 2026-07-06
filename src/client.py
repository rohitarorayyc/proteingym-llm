"""Native multi-provider client for pg-agent (NO OpenRouter).

Dispatches each model to its provider's native API at highest reasoning:
  openai                    -> Responses API (reasoning effort=high)
  anthropic                 -> Messages (extended thinking)
  google                    -> Vertex AI REST (GCP_KEY_JSON service account; thinkingLevel high)
  deepinfra/deepseek/moonshot/alibaba -> OpenAI-compatible chat.completions
                                  (per-provider base_url)

Keys load from process env, falling back to a local .env in the repo root. A
collaborator adds a model with one config/models.py row (provider + model_id).
Returns a uniform dict: {text, error, elapsed_s}.
"""
from __future__ import annotations
import base64
import json
import os
import time
from functools import lru_cache
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

# OpenAI-compatible providers: {base_url, key env}. base_url None = api.openai.com.
OAI_COMPAT = {
    "deepinfra": {"base_url": "https://api.deepinfra.com/v1/openai", "key": "DEEPINFRA_API_KEY"},
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "key": "DEEPSEEK_API_KEY"},
    "moonshot": {"base_url": "https://api.moonshot.ai/v1", "key": "MOONSHOT_API_KEY"},
    "alibaba":  {"base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "key": "ALIBABA_API_KEY"},
}


@lru_cache(maxsize=1)
def _env() -> dict:
    d: dict = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                d[k.strip()] = v.strip().strip('"').strip("'")
    d.update(os.environ)                      # process env wins
    return d


def _key(name: str) -> str:
    v = _env().get(name)
    if not v:
        raise RuntimeError(f"{name} not set (export it, or add it to {ENV_FILE.name})")
    return v


# ---- provider calls ---------------------------------------------------------

def _openai(model_id, system, user, max_tokens, reasoning, timeout) -> str:
    from openai import OpenAI
    cli = OpenAI(api_key=_key("OPENAI_API_KEY"), timeout=timeout)
    r = cli.responses.create(model=model_id, instructions=system, input=user,
                             reasoning={"effort": reasoning}, max_output_tokens=max_tokens)
    return r.output_text or ""


def _oai_compat(provider, model_id, system, user, max_tokens, reasoning, timeout) -> str:
    from openai import OpenAI
    cfg = OAI_COMPAT[provider]
    cli = OpenAI(api_key=_key(cfg["key"]), base_url=cfg["base_url"], timeout=timeout)
    kw = dict(model=model_id,
              messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
              max_tokens=max_tokens)
    if provider == "deepinfra" and model_id.startswith("zai-org/GLM"):
        kw["reasoning_effort"] = reasoning
    if provider == "alibaba":                 # DashScope Qwen thinking toggle
        kw["extra_body"] = {"enable_thinking": True}
    r = cli.chat.completions.create(**kw)
    return r.choices[0].message.content or ""


def _anthropic(model_id, system, user, max_tokens, timeout) -> str:
    import anthropic
    cli = anthropic.Anthropic(api_key=_key("ANTHROPIC_API_KEY"))
    if "opus" in model_id or "sonnet-4-6" in model_id or "sonnet-5" in model_id:
        thinking = {"type": "adaptive", "display": "summarized"}
    else:
        thinking = {"type": "enabled", "budget_tokens": max(1024, max_tokens - 4000)}
    msg = cli.messages.create(model=model_id, max_tokens=max_tokens, system=system,
                              messages=[{"role": "user", "content": user}],
                              thinking=thinking, timeout=timeout)
    if getattr(msg, "stop_reason", None) == "refusal":     # safety refusal: flag, no retry
        raise RuntimeError("anthropic refusal (stop_reason=refusal): safety policy")
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def _google(model_id, system, user, max_tokens, reasoning, timeout) -> str:
    import requests
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account
    info = json.loads(base64.b64decode(_key("GCP_KEY_JSON")).decode())
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(Request())
    project = info["project_id"]
    url = (f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global"
           f"/publishers/google/models/{model_id}:generateContent")
    body = {"contents": [{"role": "user", "parts": [{"text": user}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 1,
                                 "thinkingConfig": {"thinkingLevel": reasoning, "includeThoughts": True}}}
    r = requests.post(url, headers={"Authorization": f"Bearer {creds.token}"}, json=body, timeout=timeout)
    r.raise_for_status()
    parts = r.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def chat(spec: dict, system: str, user: str, *, timeout: int = 900, retries: int = 3) -> dict:
    """Call one model by its config spec. Uniform return; policy errors flagged (no retry)."""
    provider, model_id = spec["provider"], spec["model_id"]
    mt = spec.get("max_tokens", 32000)
    reasoning = spec.get("reasoning", "high")
    last = ""
    for attempt in range(retries):
        t0 = time.time()
        try:
            if provider == "openai":
                text = _openai(model_id, system, user, mt, reasoning, timeout)
            elif provider == "anthropic":
                text = _anthropic(model_id, system, user, mt, timeout)
            elif provider == "google":
                text = _google(model_id, system, user, mt, reasoning, timeout)
            elif provider in OAI_COMPAT:
                text = _oai_compat(provider, model_id, system, user, mt, reasoning, timeout)
            else:
                return {"text": "", "error": f"unknown provider {provider}", "elapsed_s": 0}
            return {"text": text, "error": None, "elapsed_s": round(time.time() - t0, 1)}
        except Exception as e:  # noqa
            last = f"{type(e).__name__}: {e}"
            low = last.lower()
            if (any(s in low for s in ("safety", "policy", "content_filter", "invalid_request",
                                       "permission", "not found", "does not exist"))
                    and "rate" not in low and "overload" not in low):
                return {"text": "", "error": last, "elapsed_s": round(time.time() - t0, 1)}
            time.sleep(2 ** attempt * 2)
    return {"text": "", "error": last, "elapsed_s": 0}


def estimate_tokens(text: str) -> int:
    return len(text) // 4
