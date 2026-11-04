"""A minimal OpenAI-compatible chat client, plus the factory that picks offline vs. real mode."""
from __future__ import annotations

import json
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field

from .config import llm_settings


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class ChatResult(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class LLM(Protocol):
    name: str

    def chat(self, messages: list[dict], tools: list[dict] | None = None, json_mode: bool = False) -> ChatResult: ...


class OpenAICompatLLM:
    """Talks to any /chat/completions endpoint that follows the OpenAI wire format."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"openai-compat:{model}"
        self._client = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})

    def chat(self, messages, tools=None, json_mode=False) -> ChatResult:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        resp = self._client.post(f"{self.base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        calls = [
            ToolCall(id=c["id"], name=c["function"]["name"], arguments=json.loads(c["function"]["arguments"] or "{}"))
            for c in msg.get("tool_calls") or []
        ]
        return ChatResult(content=msg.get("content"), tool_calls=calls)

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        resp = self._client.post(f"{self.base_url}/embeddings", json={"model": model, "input": texts})
        resp.raise_for_status()
        return [d["embedding"] for d in resp.json()["data"]]


def get_llm(mode: str | None = None) -> LLM:
    s = llm_settings()
    mode = mode or s["mode"]
    if mode == "offline":
        from .offline import ScriptedLLM

        return ScriptedLLM()
    if not s["api_key"]:
        raise RuntimeError("PERKPILOT_LLM_API_KEY is not set; use PERKPILOT_LLM_MODE=offline")
    return OpenAICompatLLM(s["base_url"], s["api_key"], s["model"])


def assistant_message(result: ChatResult) -> dict:
    """Convert a ChatResult back into an OpenAI-format assistant message for the transcript."""
    msg: dict[str, Any] = {"role": "assistant", "content": result.content}
    if result.tool_calls:
        msg["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
            for c in result.tool_calls
        ]
    return msg
