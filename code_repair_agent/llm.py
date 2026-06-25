"""Optional LLM backends.

The default experiments are offline and deterministic. DeepSeek support is kept
behind an explicit environment variable to avoid persisting API keys in the repo.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    estimated_cost_usd: float = 0.0


class DeepSeekChatClient:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: str = "https://api.deepseek.com/chat/completions",
        api_key_env: str = "DEEPSEEK_API_KEY",
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ):
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.timeout = int(timeout or os.environ.get("DEEPSEEK_TIMEOUT", "300"))
        self.max_tokens = int(max_tokens or os.environ.get("DEEPSEEK_MAX_TOKENS", "8192"))
        self.retries = int(os.environ.get("DEEPSEEK_RETRIES", "0"))
        self.retry_backoff = float(os.environ.get("DEEPSEEK_RETRY_BACKOFF", "2.0"))
        self.transport_retries = int(os.environ.get("DEEPSEEK_TRANSPORT_RETRIES", "2"))
        self.transport_retry_backoff = float(os.environ.get("DEEPSEEK_TRANSPORT_RETRY_BACKOFF", "1.0"))
        self.empty_content_retries = int(os.environ.get("DEEPSEEK_EMPTY_CONTENT_RETRIES", "1"))
        self.empty_retry_max_tokens = int(
            os.environ.get("DEEPSEEK_EMPTY_RETRY_MAX_TOKENS", str(max(self.max_tokens, 16384)))
        )
        self.empty_retry_timeout = int(os.environ.get("DEEPSEEK_EMPTY_RETRY_TIMEOUT", str(max(self.timeout, 360))))
        self.empty_retry_prompt_chars = int(os.environ.get("DEEPSEEK_EMPTY_RETRY_PROMPT_CHARS", "18000"))
        self.empty_retry_json_mode = os.environ.get("DEEPSEEK_EMPTY_RETRY_JSON_MODE", "0") != "0"
        self.json_mode = os.environ.get("DEEPSEEK_JSON_MODE", "1") != "0"
        self.fallback_no_json_mode = os.environ.get("DEEPSEEK_FALLBACK_NO_JSON_MODE", "1") != "0"
        self.usage = LLMUsage()

    def enabled(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    def complete(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is not set")
        last_error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            retry_messages = messages
            for empty_round in range(self.empty_content_retries + 1):
                request_spec = self._request_spec(
                    retry_messages,
                    temperature=temperature,
                    max_tokens=self.empty_retry_max_tokens if empty_round else self.max_tokens,
                    timeout=self.empty_retry_timeout if empty_round else self.timeout,
                    json_mode=self.json_mode and (empty_round == 0 or self.empty_retry_json_mode),
                )
                specs = [request_spec]
                if "response_format" in request_spec["body"] and self.fallback_no_json_mode:
                    fallback_spec = json.loads(json.dumps(request_spec))
                    fallback_spec["body"].pop("response_format", None)
                    specs.append(fallback_spec)
                for spec_idx, spec in enumerate(specs):
                    try:
                        payload = self._post_with_transport_retries(spec)
                    except Exception as exc:
                        last_error = exc
                        if spec_idx + 1 < len(specs) and _is_retryable_transport_error(str(exc)):
                            continue
                        if attempt < self.retries and _is_retryable_transport_error(str(exc)):
                            time.sleep(self.retry_backoff * (attempt + 1))
                            break
                        raise
                    self.record_usage(payload.get("usage", {}))
                    message = payload["choices"][0].get("message", {})
                    content = str(message.get("content") or "")
                    if content.strip():
                        return content
                    fallback_content = _extract_json_fallback(message)
                    if fallback_content:
                        return fallback_content
                    last_error = RuntimeError(
                        "DeepSeek returned empty message content"
                        + (" with json_mode" if "response_format" in spec.get("body", {}) else " without json_mode")
                    )
                    if spec_idx + 1 < len(specs):
                        continue
                if empty_round < self.empty_content_retries:
                    retry_messages = _append_empty_retry_instruction(
                        messages,
                        empty_round + 1,
                        max_user_chars=self.empty_retry_prompt_chars,
                    )
                    continue
                if attempt < self.retries:
                    time.sleep(self.retry_backoff * (attempt + 1))
                    break
            if attempt < self.retries:
                continue
            raise last_error
        assert last_error is not None
        raise last_error

    def _request_spec(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        timeout: int,
        json_mode: bool,
    ) -> Dict[str, object]:
        request_spec = {
            "api_key_env": self.api_key_env,
            "base_url": self.base_url,
            "timeout": timeout,
            "body": {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        }
        if json_mode:
            request_spec["body"]["response_format"] = {"type": "json_object"}
        return request_spec

    def _post_with_transport_retries(self, request_spec: Dict[str, object]) -> Dict[str, object]:
        last_error: Optional[Exception] = None
        for attempt in range(self.transport_retries + 1):
            try:
                return self._post_with_deadline(request_spec)
            except Exception as exc:
                last_error = exc
                if attempt >= self.transport_retries or not _is_retryable_transport_error(str(exc)):
                    raise
                time.sleep(self.transport_retry_backoff * (attempt + 1))
        assert last_error is not None
        raise last_error

    def _post_with_deadline(self, request_spec: Dict[str, object]) -> Dict[str, object]:
        timeout_seconds = float(request_spec.get("timeout", self.timeout))
        with tempfile.TemporaryDirectory(prefix="code-repair-deepseek-") as tmpdir:
            tmp = Path(tmpdir)
            request_path = tmp / "request.json"
            stdout_path = tmp / "response.json"
            stderr_path = tmp / "stderr.txt"
            request_path.write_text(json.dumps(request_spec), encoding="utf-8")
            with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
                process = subprocess.Popen(
                    [sys.executable, "-c", _DEEPSEEK_CHILD, str(request_path)],
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=True,
                    text=True,
                )
                deadline = time.monotonic() + timeout_seconds
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        _kill_process_group(process)
                        raise TimeoutError(f"DeepSeek subprocess exceeded {timeout_seconds:g}s deadline")
                    time.sleep(0.2)
            stdout = stdout_path.read_text(encoding="utf-8")
            stderr = stderr_path.read_text(encoding="utf-8")
            if process.returncode != 0:
                raise RuntimeError((stderr.strip() or stdout.strip() or f"DeepSeek child exited {process.returncode}")[:1000])
        loaded = json.loads(stdout)
        if not isinstance(loaded, dict):
            raise RuntimeError("DeepSeek response must be a JSON object")
        return loaded

    def record_usage(self, usage: Dict[str, object]) -> None:
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        self.usage.calls += 1
        self.usage.prompt_tokens += prompt_tokens
        self.usage.completion_tokens += completion_tokens
        input_rate = float(os.environ.get("DEEPSEEK_INPUT_USD_PER_MTOK", "0") or 0)
        output_rate = float(os.environ.get("DEEPSEEK_OUTPUT_USD_PER_MTOK", "0") or 0)
        self.usage.estimated_cost_usd += (prompt_tokens / 1_000_000) * input_rate
        self.usage.estimated_cost_usd += (completion_tokens / 1_000_000) * output_rate


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _is_retryable_transport_error(error: str) -> bool:
    lowered = error.lower()
    return any(
        marker in lowered
        for marker in (
            "timed out",
            "timeout",
            "temporarily unavailable",
            "connection reset",
            "remote end closed",
            "remote disconnected",
            "ssl",
            "eof",
            "urlerror",
            "http 429",
            "too many requests",
        )
    )


def _extract_json_fallback(message: Dict[str, object]) -> str:
    for key in ("reasoning_content", "reasoning", "text"):
        value = message.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = _last_json_object(value)
        if candidate:
            return candidate
    return ""


def _append_empty_retry_instruction(
    messages: List[Dict[str, str]],
    retry_index: int,
    *,
    max_user_chars: int,
) -> List[Dict[str, str]]:
    retry_messages = json.loads(json.dumps(messages))
    if max_user_chars > 0:
        for message in retry_messages:
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                message["content"] = _compact_recovery_prompt(str(message["content"]), max_user_chars)
    retry_messages.append(
        {
            "role": "user",
            "content": (
                f"The previous DeepSeek call returned empty final content on recovery round {retry_index}. "
                "Return the final JSON object immediately. Do not include hidden reasoning, markdown, prose, or analysis. "
                "Keep diagnosis and final_explanation concise. Include concrete patch_hunks copied exactly from the current source snippets."
            ),
        }
    )
    return retry_messages


def _compact_recovery_prompt(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return _clip_middle_text(content, max_chars)
    if not isinstance(payload, dict):
        return _clip_middle_text(content, max_chars)
    compact = dict(payload)
    compact["empty_response_recovery"] = (
        "The prior model call produced no final content. This prompt is compressed; return only final JSON."
    )
    compact["failing_output"] = _clip_middle_text(str(compact.get("failing_output", "")), 1200)
    compact["failing_summary"] = _clip_middle_text(str(compact.get("failing_summary", "")), 900)
    compact["current_diff"] = _clip_middle_text(str(compact.get("current_diff", "")), 1200)
    compact["previous_attempt_failures"] = _clip_list(compact.get("previous_attempt_failures"), 3, 360)
    compact["relevant_failure_reflections"] = _clip_list(compact.get("relevant_failure_reflections"), 3, 360)
    compact["memory_successful_strategies"] = _clip_list(compact.get("memory_successful_strategies"), 4, 300)
    compact["memory_regression_warnings"] = _clip_list(compact.get("memory_regression_warnings"), 3, 300)
    compact["source_snippets"] = _clip_mapping(compact.get("source_snippets"), max(2200, int(max_chars * 0.42)))
    compact["source_snippet_line_numbers"] = _clip_mapping(
        compact.get("source_snippet_line_numbers"),
        max(1200, int(max_chars * 0.22)),
    )
    rendered = json.dumps(compact, ensure_ascii=False, indent=2)
    if len(rendered) <= max_chars:
        return rendered
    compact["source_snippets"] = _clip_mapping(compact.get("source_snippets"), max(1200, int(max_chars * 0.28)))
    compact["source_snippet_line_numbers"] = _clip_mapping(
        compact.get("source_snippet_line_numbers"),
        max(800, int(max_chars * 0.14)),
    )
    rendered = json.dumps(compact, ensure_ascii=False, indent=2)
    return _clip_middle_text(rendered, max_chars)


def _clip_mapping(value: object, budget: int) -> object:
    if not isinstance(value, dict) or budget <= 0:
        return value
    items = [(str(key), str(item)) for key, item in value.items()]
    if not items:
        return {}
    per_item = max(300, budget // len(items))
    return {key: _clip_middle_text(item, per_item) for key, item in items}


def _clip_list(value: object, max_items: int, item_limit: int) -> object:
    if not isinstance(value, list):
        return value
    return [_clip_middle_text(str(item), item_limit) for item in value[-max_items:]]


def _clip_middle_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n...[truncated for empty-response recovery]...\n"
    if limit <= len(marker) + 20:
        return text[:limit]
    body_budget = limit - len(marker)
    head = max(1, int(body_budget * 0.65))
    tail = max(1, body_budget - head)
    return text[:head] + marker + text[-tail:]


def _last_json_object(text: str) -> str:
    candidates: List[str] = []
    depth = 0
    start: Optional[int] = None
    in_string = False
    escape = False
    for idx, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : idx + 1])
                start = None
    for candidate in reversed(candidates):
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if re.search(r'"patch_hunks"|\"diagnosis\"', candidate):
            return candidate
    return ""


_DEEPSEEK_CHILD = r"""
import json
import os
import sys
import urllib.request

try:
    spec = json.loads(open(sys.argv[1], encoding="utf-8").read())
    api_key = os.environ.get(str(spec["api_key_env"]))
    if not api_key:
        raise RuntimeError("api key env is not set")
    request = urllib.request.Request(
        str(spec["base_url"]),
        data=json.dumps(spec["body"]).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=float(spec["timeout"])) as response:
        sys.stdout.write(response.read().decode("utf-8"))
except Exception as exc:
    sys.stderr.write(f"{type(exc).__name__}: {exc}")
    raise SystemExit(2)
"""
