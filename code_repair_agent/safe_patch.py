"""Safe patch application for benchmark checkouts."""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


BLOCKED_FILENAMES = {
    "pom.xml",
    "build.xml",
    "build.gradle",
    "settings.gradle",
    "gradlew",
    "maven-wrapper.properties",
}
MAX_FUZZY_OLD_CHARS = int(os.environ.get("SAFE_PATCH_MAX_FUZZY_OLD_CHARS", "3000"))


@dataclass(frozen=True)
class RangeHunk:
    file: str
    line_start: int
    line_end: int
    new: str
    method_name: Optional[str] = None
    intent: Optional[str] = None
    line_offset: int = 0

    @classmethod
    def from_dict(cls, raw: Dict[str, object]) -> "RangeHunk":
        file_value = raw.get("file", raw.get("path"))
        if not isinstance(file_value, str) or not file_value:
            raise ValueError("range hunk requires non-empty file/path")
        new = raw.get("new", raw.get("new_text"))
        if not isinstance(new, str):
            raise ValueError("range hunk requires string new/new_text")
        line_start = _optional_int(raw.get("line_start", raw.get("start_line")))
        line_end = _optional_int(raw.get("line_end", raw.get("end_line")))
        if line_start is None or line_end is None:
            raise ValueError("range hunk requires line_start/line_end")
        method_name = raw.get("method_name")
        intent = raw.get("intent")
        line_offset = _optional_int(raw.get("line_offset", raw.get("offset"))) or 0
        return cls(
            file=file_value,
            line_start=line_start,
            line_end=line_end,
            new=new,
            method_name=str(method_name) if isinstance(method_name, str) and method_name else None,
            intent=str(intent) if isinstance(intent, str) and intent else None,
            line_offset=line_offset,
        )


@dataclass(frozen=True)
class PatchHunk:
    file: str
    old: str
    new: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    method_name: Optional[str] = None
    anchor_before: Optional[str] = None
    anchor_after: Optional[str] = None
    intent: Optional[str] = None
    range_grounded: bool = False
    original_line_start: Optional[int] = None
    original_line_end: Optional[int] = None
    line_offset: int = 0

    @classmethod
    def from_dict(cls, raw: Dict[str, object]) -> "PatchHunk":
        file_value = raw.get("file", raw.get("path"))
        if not isinstance(file_value, str) or not file_value:
            raise ValueError("patch hunk requires non-empty file/path")
        old = raw.get("old", raw.get("old_text"))
        new = raw.get("new", raw.get("new_text"))
        line_start = _optional_int(raw.get("line_start", raw.get("start_line")))
        line_end = _optional_int(raw.get("line_end", raw.get("end_line")))
        if not isinstance(new, str):
            raise ValueError("patch hunk requires string new/new_text")
        if not isinstance(old, str):
            if line_start is None or line_end is None:
                raise ValueError("patch hunk requires string old/old_text unless line_start/line_end are provided")
            old = ""
        method_name = raw.get("method_name")
        anchor_before = raw.get("anchor_before")
        anchor_after = raw.get("anchor_after")
        intent = raw.get("intent")
        line_offset = _optional_int(raw.get("line_offset", raw.get("offset"))) or 0
        original_line_start = _optional_int(raw.get("original_line_start"))
        original_line_end = _optional_int(raw.get("original_line_end"))
        range_grounded = bool(raw.get("range_grounded", False))
        return cls(
            file=file_value,
            old=old,
            new=new,
            line_start=line_start,
            line_end=line_end,
            method_name=str(method_name) if isinstance(method_name, str) and method_name else None,
            anchor_before=str(anchor_before) if isinstance(anchor_before, str) and anchor_before else None,
            anchor_after=str(anchor_after) if isinstance(anchor_after, str) and anchor_after else None,
            intent=str(intent) if isinstance(intent, str) and intent else None,
            range_grounded=range_grounded,
            original_line_start=original_line_start,
            original_line_end=original_line_end,
            line_offset=line_offset,
        )

    @classmethod
    def from_range(
        cls,
        file: str,
        line_start: int,
        line_end: int,
        new: str,
        current_text: str,
        method_name: Optional[str] = None,
        intent: Optional[str] = None,
        line_offset: int = 0,
    ) -> "PatchHunk":
        if not isinstance(file, str) or not file:
            raise ValueError("range hunk requires non-empty file/path")
        if not isinstance(new, str) or not isinstance(current_text, str):
            raise ValueError("range hunk requires string new and current_text")
        if line_start < 1 or line_end < line_start:
            raise ValueError("range hunk has invalid line_start/line_end")
        current_line_start = line_start + line_offset
        current_line_end = line_end + line_offset
        span = _exact_line_range_span(current_text, current_line_start, current_line_end)
        if span is None:
            raise ValueError("range hunk line range is outside current source")
        start, end = span
        old = current_text[start:end]
        if not old.strip():
            raise ValueError("range hunk source slice is empty")
        if method_name:
            method_span = _method_brace_span(current_text, method_name)
            if method_span is None:
                # Method name is a hint; if it doesn't match, warn and proceed
                # with the verified line range rather than failing grounding.
                pass
            else:
                method_start, method_end = method_span
                trailing = current_text[method_end:end] if end > method_end else ""
                if start < method_start or (end > method_end and trailing.strip()):
                    # Lines span outside the named method; warn but still ground
                    # since the line range itself was verified against source.
                    pass
        return cls(
            file=file,
            old=old,
            new=new,
            line_start=current_line_start,
            line_end=current_line_end,
            method_name=method_name,
            intent=intent,
            range_grounded=True,
            original_line_start=line_start,
            original_line_end=line_end,
            line_offset=line_offset,
        )


@dataclass
class PatchApplyResult:
    ok: bool
    unsafe: bool
    changed_files: List[str]
    diff: str
    errors: List[str]

    @property
    def patch_size(self) -> int:
        return sum(
            1
            for line in self.diff.splitlines()
            if (line.startswith("+") and not line.startswith("+++"))
            or (line.startswith("-") and not line.startswith("---"))
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "unsafe": self.unsafe,
            "changed_files": self.changed_files,
            "diff": self.diff,
            "errors": self.errors,
            "patch_size": self.patch_size,
        }


class SafePatchApplier:
    def __init__(
        self,
        root: Path,
        *,
        source_dirs: Optional[Iterable[str]] = None,
        test_dirs: Optional[Iterable[str]] = None,
        allow_build_edits: bool = False,
    ):
        self.root = root.resolve()
        self.source_dirs = tuple(source_dirs or [])
        self.test_dirs = tuple(test_dirs or [])
        self.allow_build_edits = allow_build_edits

    def apply(self, hunks: Iterable[PatchHunk]) -> PatchApplyResult:
        originals: Dict[Path, str] = {}
        replacements: Dict[Path, List[tuple[int, int, PatchHunk]]] = {}
        changed: List[Path] = []
        errors: List[str] = []
        unsafe = False

        for hunk in list(hunks):
            safe_path, path_error, path_unsafe = self._resolve_safe_path(hunk.file)
            if path_error:
                errors.append(path_error)
                unsafe = unsafe or path_unsafe
                continue
            assert safe_path is not None
            if safe_path not in originals:
                originals[safe_path] = safe_path.read_text(encoding="utf-8", errors="replace")
            text = originals[safe_path]
            if hunk.old == hunk.new:
                errors.append(f"{hunk.file}: no-op hunk")
                continue
            if not hunk.range_grounded and hunk.new.strip() and hunk.new in text and hunk.old in text:
                errors.append(f"{hunk.file}: replacement text already exists")
                continue
            span = _replacement_span(
                text,
                hunk.old,
                line_start=hunk.line_start,
                line_end=hunk.line_end,
                method_name=hunk.method_name,
                anchor_before=hunk.anchor_before,
                anchor_after=hunk.anchor_after,
                range_grounded=hunk.range_grounded,
            )
            if span is None:
                errors.append(f"{hunk.file}: old text not found")
                continue
            replacements.setdefault(safe_path, []).append((span[0], span[1], hunk))
            if safe_path not in changed:
                changed.append(safe_path)

        working_texts: Dict[Path, str] = {}
        for path, file_replacements in replacements.items():
            ordered = sorted(file_replacements, key=lambda item: (item[0], item[1]))
            for previous, current in zip(ordered, ordered[1:]):
                if current[0] < previous[1]:
                    errors.append(f"{path.relative_to(self.root)}: overlapping patch ranges are blocked")
                    break
            if errors:
                continue
            patched_text = originals[path]
            for start, end, hunk in sorted(file_replacements, key=lambda item: item[0], reverse=True):
                patched_text = _replace_span(patched_text, (start, end), hunk.new)
            working_texts[path] = patched_text

        for path, patched_text in working_texts.items():
            if path.suffix == ".java" and _java_braces_balanced(originals[path]) and not _java_braces_balanced(patched_text):
                errors.append(f"{path.relative_to(self.root)}: java brace imbalance after patch")

        if errors:
            return PatchApplyResult(ok=False, unsafe=unsafe, changed_files=[], diff="", errors=errors)

        written: List[Path] = []
        try:
            for path, patched_text in working_texts.items():
                path.write_text(patched_text, encoding="utf-8")
                written.append(path)
        except OSError as exc:
            for path in written:
                path.write_text(originals[path], encoding="utf-8")
            return PatchApplyResult(ok=False, unsafe=unsafe, changed_files=[], diff="", errors=[f"atomic patch write failed: {exc}"])
        diff = self._diff(originals)
        changed_files = [str(path.relative_to(self.root)) for path in changed]
        return PatchApplyResult(ok=bool(changed), unsafe=False, changed_files=changed_files, diff=diff, errors=[])

    def _resolve_safe_path(self, rel_path: str) -> tuple[Optional[Path], Optional[str], bool]:
        candidate = Path(rel_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            return None, f"{rel_path}: path traversal or absolute path is blocked", True
        resolved = (self.root / candidate).resolve()
        if self.root not in [resolved, *resolved.parents]:
            return None, f"{rel_path}: outside checkout is blocked", True
        if not resolved.exists() or not resolved.is_file():
            return None, f"{rel_path}: target file does not exist", False
        normalized = str(candidate)
        if any(normalized == test_dir or normalized.startswith(test_dir.rstrip("/") + "/") for test_dir in self.test_dirs):
            return None, f"{rel_path}: editing tests is blocked", True
        if not self.allow_build_edits and candidate.name in BLOCKED_FILENAMES:
            return None, f"{rel_path}: build/config edits are blocked", True
        if self.source_dirs and not any(
            normalized == source_dir or normalized.startswith(source_dir.rstrip("/") + "/")
            for source_dir in self.source_dirs
        ):
            return None, f"{rel_path}: file is outside configured source dirs", True
        return resolved, None, False

    def _diff(self, originals: Dict[Path, str]) -> str:
        chunks: List[str] = []
        for path, old_text in sorted(originals.items(), key=lambda item: str(item[0])):
            new_text = path.read_text(encoding="utf-8", errors="replace")
            chunks.extend(
                difflib.unified_diff(
                    old_text.splitlines(),
                    new_text.splitlines(),
                    fromfile=f"a/{path.relative_to(self.root)}",
                    tofile=f"b/{path.relative_to(self.root)}",
                    lineterm="",
                )
            )
        return "\n".join(chunks)


def _replace_exact_or_normalized(
    text: str,
    old: str,
    new: str,
    *,
    line_start: Optional[int] = None,
    line_end: Optional[int] = None,
    method_name: Optional[str] = None,
    anchor_before: Optional[str] = None,
    anchor_after: Optional[str] = None,
    range_grounded: bool = False,
) -> Optional[str]:
    span = _replacement_span(
        text,
        old,
        line_start=line_start,
        line_end=line_end,
        method_name=method_name,
        anchor_before=anchor_before,
        anchor_after=anchor_after,
        range_grounded=range_grounded,
    )
    if span is None:
        return None
    return _replace_span(text, span, new)


def _replacement_span(
    text: str,
    old: str,
    *,
    line_start: Optional[int] = None,
    line_end: Optional[int] = None,
    method_name: Optional[str] = None,
    anchor_before: Optional[str] = None,
    anchor_after: Optional[str] = None,
    range_grounded: bool = False,
) -> Optional[tuple[int, int]]:
    if range_grounded:
        span = _exact_line_range_span(text, line_start, line_end)
        if span is None:
            return None
        start, end = span
        if text[start:end] != old:
            return None
        return span
    if text.count(old) == 1:
        start = text.find(old)
        return start, start + len(old)
    span: Optional[tuple[int, int]] = None
    if method_name:
        span = _method_scoped_span(text, old, method_name)
        if span is not None:
            return span
    if anchor_before and anchor_after:
        span = _anchor_scoped_span(text, old, anchor_before, anchor_after)
        if span is not None:
            return span
    if line_start is not None or line_end is not None:
        span = _line_range_span(text, old, line_start, line_end)
        if span is None:
            span = _near_line_span(text, old, line_start, line_end)
        if span is not None:
            return span
    span = _unique_normalized_span(text, old)
    if span is None:
        span = _unique_similar_line_span(text, old)
    return span


def _replace_span(text: str, span: tuple[int, int], new: str) -> str:
    start, end = span
    replacement = new
    if end > start and text[end - 1] in "\n\r" and not replacement.endswith(("\n", "\r")):
        replacement += "\n"
    return text[:start] + replacement + text[end:]


def _exact_line_range_span(text: str, line_start: Optional[int], line_end: Optional[int]) -> Optional[tuple[int, int]]:
    if line_start is None or line_end is None:
        return None
    if line_start < 1 or line_end < line_start:
        return None
    lines = text.splitlines(keepends=True)
    if line_end > len(lines):
        return None
    offsets: List[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)
    offsets.append(cursor)
    return offsets[line_start - 1], offsets[line_end]


def _method_brace_span(text: str, method_name: str) -> Optional[tuple[int, int]]:
    if not method_name:
        return None
    import re as _re

    method_pattern = _re.compile(
        r'(?:public|private|protected|static|final|synchronized|native|abstract|\s)+[\w<>\[\],\s]+\s+' + _re.escape(method_name) + r'\s*\([^)]*\)\s*(?:throws\s+[\w\s,]+)?\s*\{',
        _re.MULTILINE,
    )
    match = method_pattern.search(text)
    if not match:
        return None
    method_start = match.start()
    brace_pos = match.end() - 1
    depth = 1
    pos = brace_pos + 1
    while pos < len(text) and depth > 0:
        if text[pos] == '{':
            depth += 1
        elif text[pos] == '}':
            depth -= 1
        pos += 1
    if depth != 0:
        return None
    return method_start, pos


def _method_scoped_span(text: str, old: str, method_name: str) -> Optional[tuple[int, int]]:
    if not method_name or not old.strip():
        return None
    import re as _re
    method_pattern = _re.compile(
        r'(?:public|private|protected|static|\s)+[\w<>\[\],\s]+\s+' + _re.escape(method_name) + r'\s*\([^)]*\)\s*(?:throws\s+[\w\s,]+)?\s*\{',
        _re.MULTILINE,
    )
    match = method_pattern.search(text)
    if not match:
        return None
    method_start = match.end() - 1
    depth = 1
    pos = method_start + 1
    while pos < len(text) and depth > 0:
        if text[pos] == '{':
            depth += 1
        elif text[pos] == '}':
            depth -= 1
        pos += 1
    method_body = text[method_start + 1:pos - 1]
    count = method_body.count(old)
    if count != 1:
        return None
    idx = method_body.find(old)
    return method_start + 1 + idx, method_start + 1 + idx + len(old)


def _anchor_scoped_span(text: str, old: str, anchor_before: str, anchor_after: str) -> Optional[tuple[int, int]]:
    if not anchor_before or not anchor_after or not old.strip():
        return None
    before_idx = text.find(anchor_before)
    if before_idx < 0:
        return None
    search_start = before_idx + len(anchor_before)
    after_idx = text.find(anchor_after, search_start)
    if after_idx < 0:
        return None
    scoped = text[search_start:after_idx]
    count = scoped.count(old)
    if count != 1:
        return None
    idx = scoped.find(old)
    return search_start + idx, search_start + idx + len(old)


def _line_range_span(text: str, old: str, line_start: Optional[int], line_end: Optional[int]) -> Optional[tuple[int, int]]:
    if line_start is None or line_end is None:
        return None
    if line_start < 1 or line_end < line_start:
        return None
    lines = text.splitlines(keepends=True)
    if line_end > len(lines):
        return None
    offsets: List[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)
    offsets.append(cursor)
    start = offsets[line_start - 1]
    end = offsets[line_end]
    candidate = text[start:end]
    if not candidate.strip():
        return None
    old_norm = _normalize_for_similarity(old)
    candidate_norm = _normalize_for_similarity(candidate)
    if not old_norm or not candidate_norm:
        return None
    threshold = 0.78
    if _similarity_ratio(old_norm, candidate_norm, threshold) < threshold:
        return None
    return start, end


def _near_line_span(
    text: str,
    old: str,
    line_start: Optional[int],
    line_end: Optional[int],
    *,
    radius: int = 40,
) -> Optional[tuple[int, int]]:
    if line_start is None:
        return None
    lines = text.splitlines(keepends=True)
    if not lines or line_start < 1 or line_start > len(lines):
        return None
    old_lines = [line for line in old.splitlines() if line.strip()]
    if not old_lines:
        return None
    offsets: List[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)
    offsets.append(cursor)

    old_norm = _normalize_for_similarity(old)
    if not old_norm:
        return None
    anchor = line_start - 1
    if len(old_norm) < 32:
        return _near_line_exact_span(lines, offsets, old_norm, anchor, radius)

    hinted_size = (line_end - line_start + 1) if line_end is not None and line_end >= line_start else len(old_lines)
    min_size = max(1, min(len(old_lines), hinted_size) - 2)
    max_size = min(len(lines), max(len(old_lines), hinted_size) + 2)
    first_line = max(0, anchor - radius)
    last_line = min(len(lines), anchor + radius + 1)
    threshold = 0.88 if len(old_norm) < 120 else 0.82
    candidates: List[tuple[float, int, int, int]] = []
    for window_size in range(min_size, max_size + 1):
        for start_line in range(first_line, max(first_line, last_line - window_size + 1)):
            end_line = start_line + window_size
            candidate = "".join(lines[start_line:end_line])
            candidate_norm = _normalize_for_similarity(candidate)
            if not candidate_norm:
                continue
            score = _similarity_ratio(old_norm, candidate_norm, threshold)
            if score >= threshold:
                distance = abs(start_line - anchor)
                candidates.append((score, -distance, offsets[start_line], offsets[end_line]))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    best = candidates[0]
    if len(candidates) > 1 and best[0] - candidates[1][0] < 0.03 and best[1] == candidates[1][1]:
        return None
    return best[2], best[3]


def _near_line_exact_span(
    lines: List[str],
    offsets: List[int],
    old_norm: str,
    anchor: int,
    radius: int,
) -> Optional[tuple[int, int]]:
    candidates: List[tuple[int, int, int]] = []
    first_line = max(0, anchor - radius)
    last_line = min(len(lines), anchor + radius + 1)
    for idx in range(first_line, last_line):
        if _normalize_for_similarity(lines[idx]) == old_norm:
            candidates.append((abs(idx - anchor), offsets[idx], offsets[idx + 1]))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None
    return candidates[0][1], candidates[0][2]


def _unique_normalized_span(text: str, old: str) -> Optional[tuple[int, int]]:
    normalized_old = "".join(old.split())
    if not normalized_old:
        return None
    matches: List[tuple[int, int]] = []
    token_index = 0
    start: Optional[int] = None
    for idx, char in enumerate(text):
        if char.isspace():
            continue
        if char == normalized_old[token_index]:
            if token_index == 0:
                start = idx
            token_index += 1
            if token_index == len(normalized_old):
                assert start is not None
                matches.append((start, idx + 1))
                if len(matches) > 1:
                    return None
                token_index = 0
                start = None
            continue
        token_index = 0
        start = None
        if char == normalized_old[0]:
            token_index = 1
            start = idx
            if len(normalized_old) == 1:
                matches.append((idx, idx + 1))
                if len(matches) > 1:
                    return None
                token_index = 0
                start = None
    if len(matches) != 1:
        return None
    return matches[0]


def _unique_similar_line_span(text: str, old: str) -> Optional[tuple[int, int]]:
    old_lines = [line for line in old.splitlines() if line.strip()]
    if not old_lines:
        return None
    old_norm = _normalize_for_similarity("\n".join(old_lines))
    if len(old_norm) < 16:
        return None
    if len(old_norm) > MAX_FUZZY_OLD_CHARS:
        return None
    text_lines = text.splitlines(keepends=True)
    if not text_lines:
        return None
    offsets: List[int] = []
    cursor = 0
    for line in text_lines:
        offsets.append(cursor)
        cursor += len(line)
    offsets.append(cursor)
    base = len(old_lines)
    window_sizes = range(max(1, base - 2), min(len(text_lines), base + 2) + 1)
    threshold = 0.92 if len(old_norm) < 80 else 0.86
    min_len = max(1, int(len(old_norm) * 0.55))
    max_len = max(min_len, int(len(old_norm) * 1.8))
    anchor = _anchor_line(old_lines)
    candidates: List[tuple[float, int, int]] = []
    for window_size in window_sizes:
        for start_line in range(0, len(text_lines) - window_size + 1):
            end_line = start_line + window_size
            candidate = "".join(text_lines[start_line:end_line])
            candidate_norm = _normalize_for_similarity(candidate)
            if not candidate_norm:
                continue
            if len(candidate_norm) < min_len or len(candidate_norm) > max_len:
                continue
            if anchor and anchor not in candidate_norm:
                continue
            score = _similarity_ratio(old_norm, candidate_norm, threshold)
            if score >= threshold:
                candidates.append((score, offsets[start_line], offsets[end_line]))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_start, best_end = candidates[0]
    competing = [
        item for item in candidates[1:]
        if item[2] <= best_start or item[1] >= best_end
    ]
    if competing and best_score - competing[0][0] < 0.04:
        return None
    return best_start, best_end


def _normalize_for_similarity(value: str) -> str:
    return " ".join(value.split())


def _similarity_ratio(left: str, right: str, threshold: float) -> float:
    matcher = difflib.SequenceMatcher(None, left, right)
    if matcher.real_quick_ratio() < threshold:
        return 0.0
    if matcher.quick_ratio() < threshold:
        return 0.0
    return matcher.ratio()


def _anchor_line(lines: List[str]) -> str:
    normalized = [_normalize_for_similarity(line) for line in lines]
    candidates = [line for line in normalized if len(line) >= 24]
    if not candidates:
        candidates = [line for line in normalized if len(line) >= 12]
    if not candidates:
        return ""
    return max(candidates, key=len)[:160]


def _optional_int(value: object) -> Optional[int]:
    if value in (None, ""):
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _java_braces_balanced(text: str) -> bool:
    balance = 0
    in_line_comment = False
    in_block_comment = False
    in_string = False
    in_char = False
    escape = False
    idx = 0
    while idx < len(text):
        char = text[idx]
        nxt = text[idx + 1] if idx + 1 < len(text) else ""
        if in_line_comment:
            if char in "\r\n":
                in_line_comment = False
            idx += 1
            continue
        if in_block_comment:
            if char == "*" and nxt == "/":
                in_block_comment = False
                idx += 2
            else:
                idx += 1
            continue
        if in_string or in_char:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif in_string and char == '"':
                in_string = False
            elif in_char and char == "'":
                in_char = False
            idx += 1
            continue
        if char == "/" and nxt == "/":
            in_line_comment = True
            idx += 2
            continue
        if char == "/" and nxt == "*":
            in_block_comment = True
            idx += 2
            continue
        if char == '"':
            in_string = True
        elif char == "'":
            in_char = True
        elif char == "{":
            balance += 1
        elif char == "}":
            balance -= 1
            if balance < 0:
                return False
        idx += 1
    return balance == 0 and not in_block_comment and not in_string and not in_char
