"""DeepSeek-backed repair planning and response parsing."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, cast

from .safe_patch import PatchHunk


@dataclass
class LocalizationRange:
    file: str
    line_start: int
    line_end: int
    method_name: Optional[str] = None
    intent: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "file": self.file,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "method_name": self.method_name,
            "intent": self.intent,
        }


@dataclass
class LocalizationPlan:
    files_to_read: List[str] = field(default_factory=list)
    line_ranges: List[LocalizationRange] = field(default_factory=list)
    hypothesis: str = ""
    patch_intent: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "files_to_read": self.files_to_read,
            "line_ranges": [item.as_dict() for item in self.line_ranges],
            "hypothesis": self.hypothesis,
            "patch_intent": self.patch_intent,
        }


@dataclass
class RepairPlan:
    diagnosis: str
    files_to_read: List[str] = field(default_factory=list)
    patch_hunks: List[PatchHunk] = field(default_factory=list)
    tests_to_run_next: List[str] = field(default_factory=list)
    confidence: float = 0.0
    final_explanation: str = ""
    patch_style: str = "direct"

    def as_dict(self) -> Dict[str, object]:
        return {
            "diagnosis": self.diagnosis,
            "files_to_read": self.files_to_read,
            "patch_hunks": [hunk.__dict__ for hunk in self.patch_hunks],
            "tests_to_run_next": self.tests_to_run_next,
            "confidence": self.confidence,
            "final_explanation": self.final_explanation,
            "patch_style": self.patch_style,
        }


def parse_repair_plan(text: str, *, allow_empty: bool = False) -> RepairPlan:
    try:
        payload = _loads_jsonish(text)
    except (json.JSONDecodeError, ValueError, SyntaxError):
        payload = _loads_partial_repair_json(text)
    raw_hunks = payload.get("patch_hunks", [])
    if not isinstance(raw_hunks, list):
        raw_hunks = []
    hunks = [PatchHunk.from_dict(item) for item in raw_hunks if isinstance(item, dict)]
    if not hunks and not allow_empty:
        raise ValueError("response contains no patch_hunks")
    diagnosis = str(payload.get("diagnosis", "")).strip()
    if not diagnosis:
        raise ValueError("response requires diagnosis")
    raw_files = payload.get("files_to_read", [])
    if not isinstance(raw_files, list):
        raw_files = []
    raw_tests = payload.get("tests_to_run_next", [])
    if not isinstance(raw_tests, list):
        raw_tests = []
    return RepairPlan(
        diagnosis=diagnosis,
        files_to_read=[str(item) for item in raw_files],
        patch_hunks=hunks,
        tests_to_run_next=[str(item) for item in raw_tests],
        confidence=_optional_float(payload.get("confidence", 0.0)),
        final_explanation=str(payload.get("final_explanation", "")),
        patch_style=str(payload.get("patch_style", "direct") or "direct"),
    )


def parse_localization_plan(text: str) -> LocalizationPlan:
    payload = _loads_jsonish(text)
    raw_ranges = payload.get("line_ranges", payload.get("ranges", []))
    if not isinstance(raw_ranges, list):
        raise ValueError("localization plan requires line_ranges array")
    ranges: List[LocalizationRange] = []
    for item in raw_ranges:
        if not isinstance(item, dict):
            continue
        file_value = item.get("file", item.get("path"))
        line_start = _required_int(item.get("line_start", item.get("start_line")), "line_start")
        line_end = _required_int(item.get("line_end", item.get("end_line")), "line_end")
        if not isinstance(file_value, str) or not file_value:
            raise ValueError("localization range requires non-empty file/path")
        if line_start < 1 or line_end < line_start:
            raise ValueError("localization range has invalid line_start/line_end")
        method_name = item.get("method_name")
        intent = item.get("intent", payload.get("patch_intent", ""))
        ranges.append(
            LocalizationRange(
                file=file_value,
                line_start=line_start,
                line_end=line_end,
                method_name=str(method_name) if isinstance(method_name, str) and method_name else None,
                intent=str(intent) if intent is not None else "",
            )
        )
    if not ranges:
        raise ValueError("localization plan contains no valid line_ranges")
    files_to_read = payload.get("files_to_read", [])
    if not isinstance(files_to_read, list):
        files_to_read = []
    return LocalizationPlan(
        files_to_read=[str(item) for item in files_to_read],
        line_ranges=ranges,
        hypothesis=str(payload.get("hypothesis", payload.get("diagnosis", ""))).strip(),
        patch_intent=str(payload.get("patch_intent", payload.get("intent", ""))).strip(),
    )


def build_localization_prompt(
    *,
    project: str,
    bug_id: int,
    metadata: Dict[str, str],
    failing_output: str,
    snippets: Dict[str, str],
    current_diff: str = "",
    memory_preferences: Optional[List[str]] = None,
    visible_test_assertions: Optional[List[str]] = None,
    derived_repair_constraints: Optional[List[str]] = None,
    snippet_line_numbers: Optional[Dict[str, str]] = None,
    context_budget_chars: Optional[int] = None,
) -> List[Dict[str, str]]:
    system = (
        "You localize Defects4J Java bugs before patch generation. Return only JSON. "
        "Identify exact source files and 1-based inclusive line ranges to read for patching. "
        "Do not propose old/new replacement text in this step; the system will read exact source from your ranges. "
        "Prefer small contiguous ranges inside the named method that contain the buggy logic and enough context to edit."
    )
    schema = {
        "files_to_read": ["relative/source/File.java"],
        "line_ranges": [
            {
                "file": "relative/source/File.java",
                "line_start": "1-based inclusive start line",
                "line_end": "1-based inclusive end line",
                "method_name": "optional method name containing the range",
                "intent": "short intended semantic edit for this range; no old/new text",
            }
        ],
        "hypothesis": "short root-cause hypothesis",
        "patch_intent": "overall intended fix, without replacement text",
    }
    budget = max(3000, int(context_budget_chars)) if context_budget_chars else None
    if budget:
        snippets_for_prompt = _trim_dict_values_by_budget(snippets, max(1200, int(budget * 0.55)))
        line_numbers_for_prompt = _trim_dict_values_by_budget(snippet_line_numbers or {}, max(800, int(budget * 0.20)))
        failing_output_limit = max(700, min(4500, budget // 6))
    else:
        snippets_for_prompt = snippets
        line_numbers_for_prompt = snippet_line_numbers or {}
        failing_output_limit = 5000
    user = {
        "project": project,
        "bug_id": bug_id,
        "metadata": metadata,
        "failing_summary": _compact_failure_summary(failing_output),
        "failing_output": _clip_middle(failing_output, failing_output_limit),
        "visible_test_assertions": visible_test_assertions or [],
        "derived_repair_constraints": derived_repair_constraints or [],
        "diagnostic_snippets": snippets_for_prompt,
        "diagnostic_snippet_line_numbers": line_numbers_for_prompt,
        "current_diff": current_diff[-4000:],
        "memory_preferred_patch_styles": memory_preferences or [],
        "localization_instructions": (
            "Return files_to_read and line_ranges only. Each line range must be contiguous, 1-based, inclusive, "
            "and suitable for a verified read before patching. Do not include old/new replacement text."
        ),
        "required_json_schema": schema,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
    ]


def build_repair_prompt(
    *,
    project: str,
    bug_id: int,
    metadata: Dict[str, str],
    failing_output: str,
    snippets: Dict[str, str],
    current_diff: str,
    attempt: int,
    memory_preferences: List[str],
    visible_test_assertions: Optional[List[str]] = None,
    derived_repair_constraints: Optional[List[str]] = None,
    snippet_line_numbers: Optional[Dict[str, str]] = None,
    repair_skills: Optional[List[str]] = None,
    test_skills: Optional[List[str]] = None,
    regression_warnings: Optional[List[str]] = None,
    success_strategies: Optional[List[str]] = None,
    previous_attempt_failures: Optional[List[str]] = None,
    reflections: Optional[List[str]] = None,
    context_budget_chars: Optional[int] = None,
) -> List[Dict[str, str]]:
    system = (
        "You are a code repair agent for Defects4J Java bugs. "
        "Return only JSON. Do not edit tests, build files, generated files, secrets, or paths outside the checkout. "
        "Snippets marked [read-only-test] are visible tests for diagnosis only and must not be patched. "
        "Prefer exact old/new replacement: copy the old text verbatim from source_snippets, then provide the new text. "
        "If you cannot match the old text exactly, provide line_start, line_end, and new for range-grounded replacement. "
        "The old field is optional but strongly preferred when you can copy it verbatim from source_snippets. "
        "If using line ranges, ensure the new text preserves Java syntax and brace balance. "
        "Prefer a short precise line range over a long uncertain block; do not use line 1 as a placeholder anchor. "
        "Prefer the smallest semantic source change that fixes the root cause. "
        "Do not delete existing validation, parsing, serialization, or boundary logic unless the failing evidence specifically shows it is wrong. "
        "If current_diff is empty, previous failed patches are not present in the workspace; never use failed patch text as old. "
        "Prioritize failing_summary and visible test assertions over long stack traces. "
        "Do not return empty patch_hunks; if you request files_to_read, still provide the best grounded patch from current snippets. "
        "Output the final JSON object immediately; do not spend tokens on hidden reasoning."
    )
    schema = {
        "diagnosis": "short root-cause hypothesis",
        "files_to_read": ["optional additional source files"],
        "patch_hunks": [
            {
                "file": "relative/source/File.java",
                "old": "optional: copy verbatim from source_snippets for exact old/new replacement",
                "new": "replacement text without line-number prefixes",
                "line_start": "optional: 1-based source line for range-grounded replacement start",
                "line_end": "optional: inclusive 1-based source line for range-grounded replacement end",
                "method_name": "optional method name for anchoring (e.g., 'createNumber')",
                "intent": "short reason for this hunk",
                "anchor_before": "optional unique text that appears just before the old text in source",
                "anchor_after": "optional unique text that appears just after the old text in source",
            }
        ],
        "tests_to_run_next": ["Defects4J test names or scopes"],
        "confidence": 0.0,
        "final_explanation": "short patch explanation",
        "patch_style": "one label such as guard, boundary, null-check, arithmetic, API-contract",
    }
    budget = max(4000, int(context_budget_chars)) if context_budget_chars else None
    if budget:
        failing_output_limit = max(900, min(6000, budget // 8))
        snippet_budget = max(1800, int(budget * 0.52))
        line_number_budget = max(900, int(budget * 0.18))
        aux_items = max(2, min(6, budget // 10000))
        aux_limit = max(180, min(700, budget // 60))
        previous_failure_limit = max(aux_limit, min(1100, budget // 30))
        snippets_for_prompt = _trim_dict_values_by_budget(snippets, snippet_budget)
        line_numbers_for_prompt = _trim_dict_values_by_budget(snippet_line_numbers or {}, line_number_budget)
        previous_failures_for_prompt = _trim_list_values(
            previous_attempt_failures or [],
            aux_items,
            previous_failure_limit,
            tail=True,
        )
        reflections_for_prompt = _trim_list_values(reflections or [], aux_items, aux_limit)
        warnings_for_prompt = _trim_list_values(regression_warnings or [], aux_items, aux_limit)
        strategies_for_prompt = _trim_list_values(success_strategies or [], aux_items, aux_limit)
    else:
        failing_output_limit = 6000
        snippets_for_prompt = snippets
        line_numbers_for_prompt = snippet_line_numbers or {}
        previous_failures_for_prompt = previous_attempt_failures or []
        reflections_for_prompt = reflections or []
        warnings_for_prompt = regression_warnings or []
        strategies_for_prompt = success_strategies or []

    user = {
        "project": project,
        "bug_id": bug_id,
        "attempt": attempt,
        "metadata": metadata,
        "failing_summary": _compact_failure_summary(failing_output),
        "failing_output": _clip_middle(failing_output, failing_output_limit),
        "visible_test_assertions": visible_test_assertions or [],
        "derived_repair_constraints": derived_repair_constraints or [],
        "source_snippets": snippets_for_prompt,
        "source_snippet_line_numbers": line_numbers_for_prompt,
        "patch_grounding_instructions": (
            "Prefer exact old/new replacement: copy old verbatim from source_snippets, then provide new. "
            "If you cannot match old exactly, provide line_start, line_end, and new for range-grounded replacement. "
            "When using line ranges, ensure new text preserves Java syntax and brace balance. "
            "Never include line-number prefixes in new. Previous failed hunk text is diagnostic only; "
            "do not use it as old text unless current_diff or source_snippets contain it."
        ),
        "current_diff": current_diff[-6000:],
        "memory_preferred_patch_styles": memory_preferences,
        "memory_preferred_repair_skills": repair_skills or [],
        "memory_preferred_test_skills": test_skills or [],
        "memory_regression_warnings": warnings_for_prompt,
        "memory_successful_strategies": strategies_for_prompt,
        "previous_attempt_failures": previous_failures_for_prompt,
        "relevant_failure_reflections": reflections_for_prompt,
        "context_budget_chars": budget,
        "required_json_schema": schema,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
    ]


def _trim_dict_values_by_budget(values: Dict[str, str], total_budget: int) -> Dict[str, str]:
    if total_budget <= 0 or not values:
        return {str(key): "" for key in values}
    items = [(str(key), str(value)) for key, value in values.items()]
    weights = _snippet_key_weights([key for key, _ in items])
    total_weight = sum(weights) or len(items)
    remaining = total_budget
    trimmed: Dict[str, str] = {}
    for index, (key, value) in enumerate(items):
        slots_left = len(items) - index
        if slots_left == 1:
            limit = remaining
        else:
            limit = int(total_budget * (weights[index] / total_weight))
            limit = min(limit, remaining - 300 * (slots_left - 1))
        limit = max(300, limit)
        clipped = _clip_middle(value, limit)
        trimmed[key] = clipped
        remaining = max(0, remaining - len(clipped))
    return trimmed


def _snippet_key_weights(keys: List[str]) -> List[int]:
    weights: List[int] = []
    first_editable_seen = False
    for key in keys:
        lowered = key.lower()
        if "[read-only-test]" in lowered or "/test/" in lowered or "src/test" in lowered:
            weights.append(1)
            continue
        if not first_editable_seen:
            weights.append(8)
            first_editable_seen = True
            continue
        weights.append(2)
    return weights


def _trim_list_values(values: List[str], max_items: int, item_limit: int, *, tail: bool = False) -> List[str]:
    selected = values[-max_items:] if tail else values[:max_items]
    return [_clip_middle(str(item), item_limit) for item in selected]


def _clip_middle(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    marker = "\n...[truncated]...\n"
    if limit <= len(marker) + 20:
        return text[:limit]
    body_budget = limit - len(marker)
    head = max(1, int(body_budget * 0.65))
    tail = max(1, body_budget - head)
    return text[:head] + marker + text[-tail:]


def _compact_failure_summary(output: str, limit: int = 1200) -> str:
    if not output:
        return ""
    failing_block = output.split("[failing_tests]", 1)[-1] if "[failing_tests]" in output else output
    lines = [line.strip() for line in failing_block.splitlines() if line.strip()]
    selected: List[str] = []
    for line in lines:
        lowered = line.lower()
        if (
            line.startswith("---")
            or "assertion" in lowered
            or "expected" in lowered
            or "but was" in lowered
            or "error:" in lowered
            or re.search(r"\b[A-Za-z_][A-Za-z0-9_.]*(Exception|Error)\b", line)
        ):
            selected.append(line)
        if len(selected) >= 8:
            break
    if not selected:
        selected = lines[:8]
    return "\n".join(selected)[:limit]


def _loads_jsonish(text: str) -> Dict[str, object]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty response")
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
        if match:
            stripped = match.group(1).strip()
    patch_match = re.search(r"<<<PATCH_JSON>>>\s*(.*?)\s*<<<END_PATCH_JSON>>>", stripped, re.DOTALL)
    if patch_match:
        stripped = patch_match.group(1).strip()
    segments = []
    if "<｜end▁of▁thinking｜>" in stripped:
        segments.append(stripped.rsplit("<｜end▁of▁thinking｜>", 1)[-1].strip())
    segments.append(stripped)
    for segment in segments:
        for candidate in reversed(_json_object_candidates(segment)):
            try:
                payload = _parse_jsonish_candidate(candidate)
            except (json.JSONDecodeError, ValueError, SyntaxError):
                continue
            if isinstance(payload, dict):
                return cast(Dict[str, object], payload)
    try:
        payload = _parse_jsonish_candidate(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = _parse_jsonish_candidate(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("response JSON must be an object")
    return cast(Dict[str, object], payload)


def _required_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be an integer") from None


def _optional_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_jsonish_candidate(candidate: str) -> object:
    variants = [candidate]
    escaped = _escape_raw_control_chars_in_json_strings(candidate)
    if escaped != candidate:
        variants.append(escaped)
    stripped_trailing = _strip_trailing_commas(candidate)
    if stripped_trailing != candidate:
        variants.append(stripped_trailing)
    if escaped != candidate and stripped_trailing != candidate:
        combined = _strip_trailing_commas(escaped)
        if combined != escaped:
            variants.append(combined)
    last_error: Optional[Exception] = None
    for variant in variants:
        try:
            return json.loads(variant)
        except json.JSONDecodeError as exc:
            last_error = exc
    for variant in variants:
        try:
            return ast.literal_eval(variant)
        except (SyntaxError, ValueError) as exc:
            last_error = exc
    if isinstance(last_error, json.JSONDecodeError):
        raise last_error
    raise ValueError(str(last_error) if last_error else "invalid JSON object")


def _escape_raw_control_chars_in_json_strings(text: str) -> str:
    output: List[str] = []
    in_string = False
    escape = False
    for char in text:
        if in_string:
            if escape:
                output.append(char)
                escape = False
                continue
            if char == "\\":
                output.append(char)
                escape = True
                continue
            if char == '"':
                output.append(char)
                in_string = False
                continue
            if char == "\n":
                output.append("\\n")
                continue
            if char == "\r":
                output.append("\\r")
                continue
            if char == "\t":
                output.append("\\t")
                continue
            output.append(char)
            continue
        output.append(char)
        if char == '"':
            in_string = True
    return "".join(output)


def _strip_trailing_commas(text: str) -> str:
    cleaned = re.sub(r",(\s*[}\]])", r"\1", text)
    if cleaned != text:
        return cleaned
    return text


def _json_object_candidates(text: str) -> List[str]:
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
    return candidates


def _loads_partial_repair_json(text: str) -> Dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
        if match:
            stripped = match.group(1).strip()
    hunks_text = _balanced_json_array_after_key(stripped, "patch_hunks")
    if not hunks_text:
        raise ValueError("response JSON is truncated before a complete patch_hunks array")
    hunks = _parse_jsonish_candidate(hunks_text)
    if not isinstance(hunks, list):
        raise ValueError("patch_hunks must be a JSON array")
    payload: Dict[str, object] = {"patch_hunks": hunks}
    diagnosis = _json_string_value_after_key(stripped, "diagnosis")
    if diagnosis:
        payload["diagnosis"] = diagnosis
    files_to_read = _balanced_json_array_after_key(stripped, "files_to_read")
    if files_to_read:
        try:
            parsed_files = _parse_jsonish_candidate(files_to_read)
            if isinstance(parsed_files, list):
                payload["files_to_read"] = parsed_files
        except (json.JSONDecodeError, ValueError, SyntaxError):
            pass
    tests_to_run = _balanced_json_array_after_key(stripped, "tests_to_run_next")
    if tests_to_run:
        try:
            parsed_tests = _parse_jsonish_candidate(tests_to_run)
            if isinstance(parsed_tests, list):
                payload["tests_to_run_next"] = parsed_tests
        except (json.JSONDecodeError, ValueError, SyntaxError):
            pass
    patch_style = _json_string_value_after_key(stripped, "patch_style")
    if patch_style:
        payload["patch_style"] = patch_style
    final_explanation = _json_string_value_after_key(stripped, "final_explanation")
    if final_explanation:
        payload["final_explanation"] = final_explanation
    return payload


def _balanced_json_array_after_key(text: str, key: str) -> str:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[', text)
    if not match:
        return ""
    start = match.end() - 1
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
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
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return ""


def _json_string_value_after_key(text: str, key: str) -> str:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"', text)
    if not match:
        return ""
    start = match.end() - 1
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if idx == start:
            in_string = True
            continue
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                candidate = text[start : idx + 1]
                try:
                    return str(json.loads(candidate))
                except json.JSONDecodeError:
                    return ""
    return ""
