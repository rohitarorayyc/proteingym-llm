"""Model lineup — native providers (NO OpenRouter). Add a model = add one row.

Each row: provider (openai|anthropic|google|deepinfra|deepseek|moonshot|alibaba),
provider-native model_id, max output tokens, context window, reasoning effort.
Confirm reachability with `python -m src.verify_models`.

Keys (process env, or a local .env in the repo root): OPENAI_API_KEY,
ANTHROPIC_API_KEY, GCP_KEY_JSON (Gemini via Vertex), DEEPSEEK_API_KEY,
DEEPINFRA_API_KEY (GLM), MOONSHOT_API_KEY (Kimi), ALIBABA_API_KEY (Qwen/DashScope).
"""

MODELS: dict[str, dict] = {
    "gpt-5.5":               {"provider": "openai",    "model_id": "gpt-5.5",                       "max_tokens": 64000, "ctx": 400000,  "reasoning": "high"},
    "gpt-5.5-low":           {"provider": "openai",    "model_id": "gpt-5.5",                       "max_tokens": 64000, "ctx": 400000,  "reasoning": "low"},
    "gpt-5.5-medium":        {"provider": "openai",    "model_id": "gpt-5.5",                       "max_tokens": 64000, "ctx": 400000,  "reasoning": "medium"},
    "gpt-5.5-xhigh":         {"provider": "openai",    "model_id": "gpt-5.5",                       "max_tokens": 64000, "ctx": 400000,  "reasoning": "xhigh"},
    "gpt-5.4":               {"provider": "openai",    "model_id": "gpt-5.4",                       "max_tokens": 64000, "ctx": 400000,  "reasoning": "high"},
    "gpt-5.4-mini":          {"provider": "openai",    "model_id": "gpt-5.4-mini",                  "max_tokens": 64000, "ctx": 400000,  "reasoning": "high"},
    "gpt-5.4-nano":          {"provider": "openai",    "model_id": "gpt-5.4-nano",                  "max_tokens": 64000, "ctx": 400000,  "reasoning": "high"},
    "claude-opus-4.8":       {"provider": "anthropic", "model_id": "claude-opus-4-8",               "max_tokens": 64000, "ctx": 200000,  "reasoning": "high"},
    "claude-opus-4.8-low":    {"provider": "anthropic", "model_id": "claude-opus-4-8",              "max_tokens": 64000, "ctx": 200000,  "reasoning": "low"},
    "claude-opus-4.8-medium": {"provider": "anthropic", "model_id": "claude-opus-4-8",              "max_tokens": 64000, "ctx": 200000,  "reasoning": "medium"},
    "claude-opus-4.8-high":   {"provider": "anthropic", "model_id": "claude-opus-4-8",              "max_tokens": 64000, "ctx": 200000,  "reasoning": "high"},
    "claude-opus-4.8-xhigh":  {"provider": "anthropic", "model_id": "claude-opus-4-8",              "max_tokens": 64000, "ctx": 200000,  "reasoning": "xhigh"},
    "claude-opus-4.8-max":    {"provider": "anthropic", "model_id": "claude-opus-4-8",              "max_tokens": 64000, "ctx": 200000,  "reasoning": "max"},
    "claude-opus-4.7":       {"provider": "anthropic", "model_id": "claude-opus-4-7",               "max_tokens": 64000, "ctx": 200000,  "reasoning": "high"},
    "claude-sonnet-4.6":     {"provider": "anthropic", "model_id": "claude-sonnet-4-6",             "max_tokens": 64000, "ctx": 200000,  "reasoning": "high"},
    "claude-sonnet-5":       {"provider": "anthropic", "model_id": "claude-sonnet-5",               "max_tokens": 64000, "ctx": 200000,  "reasoning": "high"},
    "gemini-3.5-flash":      {"provider": "google",    "model_id": "gemini-3.5-flash",              "max_tokens": 64000, "ctx": 1000000, "reasoning": "high"},
    "gemini-3.5-flash-low":  {"provider": "google",    "model_id": "gemini-3.5-flash",              "max_tokens": 64000, "ctx": 1000000, "reasoning": "low"},
    "gemini-3.5-flash-medium": {"provider": "google",  "model_id": "gemini-3.5-flash",              "max_tokens": 64000, "ctx": 1000000, "reasoning": "medium"},
    "gemini-3.5-flash-high": {"provider": "google",    "model_id": "gemini-3.5-flash",              "max_tokens": 64000, "ctx": 1000000, "reasoning": "high"},
    "gemini-3.1-pro":        {"provider": "google",    "model_id": "gemini-3.1-pro-preview",        "max_tokens": 64000, "ctx": 1000000, "reasoning": "high"},
    "gemini-3.1-flash-lite": {"provider": "google",    "model_id": "gemini-3.1-flash-lite-preview", "max_tokens": 64000, "ctx": 1000000, "reasoning": "high"},
    "glm-5.2":               {"provider": "deepinfra", "model_id": "zai-org/GLM-5.2",               "max_tokens": 64000, "ctx": 1048576, "reasoning": "high"},
    # ids below are best-guess for the lineup names; confirm with verify_models / a 1-call test
    "kimi-k2.6":             {"provider": "moonshot",  "model_id": "kimi-k2.6",                     "max_tokens": 32000, "ctx": 256000,  "reasoning": "high"},
    "qwen-3.7-max":          {"provider": "alibaba",   "model_id": "qwen3.7-max",                   "max_tokens": 32000, "ctx": 256000,  "reasoning": "high"},
    "deepseek-v4-pro":       {"provider": "deepseek",  "model_id": "deepseek-reasoner",             "max_tokens": 32000, "ctx": 160000,  "reasoning": "high"},
}

PILOT_MODELS = ["gemini-3.5-flash", "gpt-5.4-nano", "claude-opus-4.8", "gpt-5.5", "deepseek-v4-pro"]
SIZES = [10, 50, 100]
N_BATCHES = 3
