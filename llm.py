"""OpenAI-compatible chat driver for the local vLLM server."""
import json
import time

import requests


class LLMError(Exception):
    pass


class LLM:
    def __init__(self, cfg):
        self.base = cfg.get("base_url", "http://localhost:8000/v1").rstrip("/")
        self.model = cfg.get("model", "qwen3.8-27b")
        self.temperature = float(cfg.get("temperature", 0.3))
        self.max_tokens = int(cfg.get("max_tokens", 4096))
        self.timeout = int(cfg.get("timeout_sec", 600))

    def ping(self):
        try:
            r = requests.get(self.base + "/models", timeout=5)
            return r.ok
        except Exception:
            return False

    def chat(self, system, user, retries=3, temperature=None):
        last = None
        for i in range(retries):
            try:
                r = requests.post(
                    self.base + "/chat/completions",
                    json={
                        "model": self.model,
                        "temperature": self.temperature if temperature is None else temperature,
                        "max_tokens": self.max_tokens,
                        # qwen3: suppress thinking trace leaking into content
                        "chat_template_kwargs": {"enable_thinking": False},
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    },
                    timeout=self.timeout,
                )
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                content = msg.get("content")
                if not content:  # qwen3 reasoning-parser may leave content null
                    content = msg.get("reasoning_content") or msg.get("reasoning") or ""
                if not content.strip():
                    raise RuntimeError(
                        f"empty LLM content (refusal={msg.get('refusal')!r})")
                return content
            except Exception as e:
                last = e
                if i < retries - 1:
                    time.sleep(5 * (i + 1))
        raise LLMError(f"vllm chat failed after {retries} tries: {last}")

    def chat_json(self, system, user, temperature=None):
        """Chat and robustly extract JSON from the reply (LLMs may add prose)."""
        txt = self.chat(system, user + "\nRespond ONLY with the JSON, no prose.",
                        temperature=temperature)
        return extract_json(txt)


def extract_json(txt):
    """Find the first balanced {...} or [...] in txt and parse it."""
    for ch in ("{", "["):
        i = txt.find(ch)
        while i != -1:
            try:
                obj, _ = json.JSONDecoder().raw_decode(txt[i:])
                return obj
            except json.JSONDecodeError:
                i = txt.find(ch, i + 1)
    raise LLMError(f"no JSON in LLM output:\n{txt[:800]}")
