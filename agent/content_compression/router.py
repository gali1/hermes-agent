"""Content-type router for tool-output compression (Headroom-derived).

Pure, stdlib-only, deterministic. Ports the detection order and confidence
thresholds of Headroom's ``transforms/content_detector.py`` so each tool
output shape reaches the compressor written for it:

    json -> diff -> html -> search -> log -> tabular -> config -> code

with a weak tool-name prior as a tie-breaker before the plain-text fallback.
Detection is heuristic by design: the fallback is ``("text", 0.5)`` and
callers must treat every result as a hint, never as ground truth.
"""

from __future__ import annotations

import configparser
import json
import re

__all__ = ["detect_content_type"]

_JSON_MIN_BULK_FRACTION = 0.6

_SEARCH_RESULT_RE = re.compile(r"^[^\s:]+:\d+:")
_GREP_CONTEXT_RE = re.compile(r"^(?P<path>[^\s:]+?)-(?P<line>\d+)-")
_GREP_COLON_DASH_RE = re.compile(r"^(?P<path>[^\s:]+?):(?P<line>\d+)-")
_TIMESTAMP_ROW_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}[T ]")

_DIFF_HEADER_RE = re.compile(
    r"^("
    r"diff --git"
    r"|diff --combined "
    r"|diff --cc "
    r"|--- a/"
    r"|@@\s+-\d+,\d+\s+\+\d+,\d+\s+@@"
    r")"
)
_DIFF_CHANGE_RE = re.compile(r"^[+-][^+-]")

_HTML_DOCTYPE_RE = re.compile(r"^\s*<!doctype\s+html", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<html[\s>]", re.IGNORECASE)
_HTML_HEAD_RE = re.compile(r"<head[\s>]", re.IGNORECASE)
_HTML_BODY_RE = re.compile(r"<body[\s>]", re.IGNORECASE)
_HTML_STRUCTURAL_RE = re.compile(
    r"<(div|span|script|style|link|meta|nav|header|footer|aside|article|section|main)[\s>]",
    re.IGNORECASE,
)

_LOG_PATTERNS = [
    re.compile(r"\b(ERROR|FAIL|FAILED|FATAL|CRITICAL)\b", re.IGNORECASE),
    re.compile(r"\b(WARN|WARNING)\b", re.IGNORECASE),
    re.compile(r"\b(INFO|DEBUG|TRACE)\b", re.IGNORECASE),
    re.compile(r"^\s*\d{4}-\d{2}-\d{2}"),
    re.compile(r"^\s*\[\d{2}:\d{2}:\d{2}\]"),
    re.compile(r"^={3,}|^-{3,}"),
    re.compile(r"^\s*PASSED|^\s*FAILED|^\s*SKIPPED"),
    re.compile(r"^npm ERR!|^yarn error|^cargo error"),
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"^\w*(Error|Exception):"),
    re.compile(r"^\s*at\s+[\w.$/]+\("),
    re.compile(r"^\s*at async \S"),
    re.compile(r"^(panic|fatal error): "),
    re.compile(r"^goroutine \d+ \["),
    re.compile(r"^\t\S+\.go:\d+ \+0x"),
    re.compile(r"^thread '[^']*' panicked at"),
    re.compile(r"^stack backtrace:"),
    re.compile(r"^\s+\d+: \S"),
    re.compile(r"^Unhandled exception\."),
    re.compile(r"^Caused by: "),
]

_MD_SEP_CELL_RE = re.compile(r"^:?-{2,}:?$")

_CONFIG_SECTION_RE = re.compile(r"^\s*\[\[?[\w.\-\"' ]+\]\]?\s*$")
_TOML_ASSIGN_RE = re.compile(r"""^\s*(?:[\w.\-]+|"[^"]+"|'[^']+')\s*=\s*\S""")
_INI_ASSIGN_RE = re.compile(r"^\s*[\w.\-@ ]+?\s*[=:]\s*")
_YAML_KEY_RE = re.compile(r"""^\s*(?:-\s+)?(?:[\w.\-/]+|"[^"]+"|'[^']+')\s*:(?:\s|$)""")
_YAML_LIST_RE = re.compile(r"^\s*-\s+\S")
_YAML_DOC_RE = re.compile(r"^---\s*$|^\.\.\.\s*$")
_CONFIG_COMMENT_RE = re.compile(r"^\s*[#;]")

_CODE_PATTERNS = {
    "python": [
        re.compile(r"^\s*(def|class|import|from|async def)\s+\w+"),
        re.compile(r"^\s*@\w+"),
        re.compile(r'^\s*"""'),
        re.compile(r"^\s*if __name__\s*=="),
    ],
    "javascript": [
        re.compile(r"^\s*(function|const|let|var|class|import|export)\s+"),
        re.compile(r"^\s*(async\s+function|=>\s*\{)"),
        re.compile(r"^\s*module\.exports"),
    ],
    "typescript": [
        re.compile(r"^\s*(interface|type|enum|namespace)\s+\w+"),
        re.compile(r":\s*(string|number|boolean|any|void)\b"),
    ],
    "go": [
        re.compile(r"^\s*(func|type|package|import)\s+"),
        re.compile(r"^\s*func\s+\([^)]+\)\s+\w+"),
    ],
    "rust": [
        re.compile(r"^\s*(fn|struct|enum|impl|mod|use|pub)\s+"),
        re.compile(r"^\s*#\["),
    ],
    "java": [
        re.compile(r"^\s*(public|private|protected)\s+(class|interface|enum)"),
        re.compile(r"^\s*@\w+"),
        re.compile(r"^\s*package\s+[\w.]+;"),
    ],
    "csharp": [
        re.compile(r"^\s*using\s+[\w.]+\s*;"),
        re.compile(r"^\s*namespace\s+[\w.]+"),
        re.compile(
            r"^\s*(public|private|protected|internal|sealed|static|abstract|partial)\s+"
            r"(class|struct|record|interface|enum)\b"
        ),
    ],
    "php": [
        re.compile(r"<\?php\b"),
        re.compile(r"^\s*namespace\s+[\w\\]+\s*;"),
        re.compile(r"^\s*(public|private|protected|static|abstract|final)?\s*function\s+\w+\s*\("),
        re.compile(r"\$this->"),
    ],
}

_FIXED_WIDTH_LIST_RE = re.compile(r"^\s*(?:[-*+•]|\d{1,3}[.)])\s")
_FIXED_WIDTH_PROSE_END_RE = re.compile(r"[A-Za-z][.!?][\"')\]]?$")
_FIXED_WIDTH_CODE_ENDS = ("{", "}", ";", "(", ")", ",", ":", "\\")
_FIXED_WIDTH_CODE_STARTS = ("#", "//", "/*", "--")
_FIXED_WIDTH_MAX_COLS = 400


def detect_content_type(text, tool_name: str = "") -> tuple[str, float]:
    """Return ``(content_type, confidence)`` for a tool output.

    Types: ``json|log|search|diff|config|tabular|html|code|text``. Never
    raises; non-string/empty input is ``("text", 0.0)``. The tool-name prior
    only applies when structural detection found nothing (result ``text``),
    so it can never override a confident shape match.
    """
    if not isinstance(text, str) or not text.strip():
        return ("text", 0.0)

    for detect, threshold in (
        (_detect_json, None),
        (_detect_diff, 0.7),
        (_detect_html, 0.7),
        (_detect_search, 0.6),
        (_detect_log, 0.5),
        (_detect_tabular, 0.6),
        (_detect_config, 0.6),
        (_detect_code, 0.5),
    ):
        try:
            result = detect(text)
        except Exception:
            result = None
        if result and (threshold is None or result[1] >= threshold):
            return result

    try:
        fixed = _detect_fixed_width(text)
    except Exception:
        fixed = None
    if fixed:
        return fixed

    prior = _tool_name_prior(text, tool_name)
    if prior:
        return prior
    return ("text", 0.5)


def _decode_concatenated_json(content: str):
    decoder = json.JSONDecoder()
    idx, length = 0, len(content)
    items = []
    while idx < length:
        while idx < length and content[idx].isspace():
            idx += 1
        if idx >= length:
            break
        try:
            value, idx = decoder.raw_decode(content, idx)
        except ValueError:
            return None
        items.append(value)
    return items or None


def _detect_json(content: str) -> tuple[str, float] | None:
    stripped = content.strip()
    if not stripped:
        return None
    try:
        value = json.loads(stripped)
    except RecursionError:
        return None
    except ValueError:
        if stripped.startswith("{"):
            try:
                items = _decode_concatenated_json(stripped)
            except RecursionError:
                return None
            if items and len(items) >= 2 and all(isinstance(i, dict) for i in items):
                return ("json", 1.0)
        start = min(
            (i for i in (stripped.find("{"), stripped.find("[")) if i >= 0),
            default=-1,
        )
        if start < 0:
            return None
        try:
            value, end = json.JSONDecoder().raw_decode(stripped, start)
        except (ValueError, RecursionError):
            return None
        if (end - start) < len(stripped) * _JSON_MIN_BULK_FRACTION:
            return None

    if isinstance(value, list):
        is_dict_array = bool(value) and all(isinstance(i, dict) for i in value)
        return ("json", 1.0 if is_dict_array else 0.8)
    if isinstance(value, dict):
        return ("json", 0.9)
    return None


def _detect_diff(content: str) -> tuple[str, float] | None:
    lines = content.split("\n")[:500]
    headers = sum(1 for line in lines if _DIFF_HEADER_RE.match(line))
    if headers == 0:
        return None
    changes = sum(1 for line in lines if _DIFF_CHANGE_RE.match(line))
    confidence = min(1.0, 0.5 + headers * 0.2 + changes * 0.05)
    return ("diff", confidence)


def _detect_html(content: str) -> tuple[str, float] | None:
    sample = content[:3000]
    has_doctype = bool(_HTML_DOCTYPE_RE.search(sample))
    has_html_tag = bool(_HTML_TAG_RE.search(sample))
    has_head = bool(_HTML_HEAD_RE.search(sample))
    has_body = bool(_HTML_BODY_RE.search(sample))
    structural = len(_HTML_STRUCTURAL_RE.findall(sample))
    if not has_doctype and not has_html_tag and structural < 3:
        return None
    confidence = 0.0
    if has_doctype:
        confidence += 0.5
    if has_html_tag:
        confidence += 0.3
    if has_head:
        confidence += 0.1
    if has_body:
        confidence += 0.1
    confidence += min(0.3, structural * 0.03)
    confidence = min(1.0, confidence)
    if confidence < 0.5:
        return None
    return ("html", confidence)


def _prefix_looks_like_path(prefix: str) -> bool:
    return "<" not in prefix and ">" not in prefix and "=" not in prefix


def _is_grep_context_line(line: str) -> bool:
    match = _GREP_CONTEXT_RE.match(line)
    if not match:
        return False
    prefix = match.group("path")
    if not _prefix_looks_like_path(prefix):
        return False
    return "/" in prefix or "." in prefix


def _is_search_result_line(line: str) -> bool:
    if _TIMESTAMP_ROW_RE.match(line):
        return False
    if _SEARCH_RESULT_RE.match(line) or _GREP_COLON_DASH_RE.match(line):
        return _prefix_looks_like_path(line.split(":", 1)[0])
    return _is_grep_context_line(line)


def _detect_search(content: str) -> tuple[str, float] | None:
    lines = content.split("\n")[:100]
    matching = sum(1 for line in lines if line.strip() and _is_search_result_line(line))
    if matching < 2:
        return None
    non_empty = sum(1 for line in lines if line.strip())
    if non_empty == 0:
        return None
    ratio = matching / non_empty
    if ratio < 0.3:
        return None
    confidence = min(1.0, 0.4 + ratio * 0.6)
    return ("search", confidence)


def _detect_log(content: str) -> tuple[str, float] | None:
    lines = content.split("\n")[:200]
    pattern_matches = 0
    error_matches = 0
    for line in lines:
        for i, pattern in enumerate(_LOG_PATTERNS):
            if pattern.search(line):
                pattern_matches += 1
                if i < 2:
                    error_matches += 1
                break
    if pattern_matches == 0:
        return None
    non_empty = sum(1 for line in lines if line.strip())
    if non_empty == 0:
        return None
    ratio = pattern_matches / non_empty
    if ratio < 0.1:
        return None
    confidence = min(1.0, 0.3 + ratio * 0.5 + error_matches * 0.05)
    return ("log", confidence)


def _md_cell_count(row: str) -> int:
    return len(row.strip().strip("|").split("|"))


def _is_md_separator(row: str) -> bool:
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    cells = [c for c in cells if c != ""]
    if len(cells) < 2:
        return False
    return all(_MD_SEP_CELL_RE.match(c) for c in cells)


def _detect_md_table(lines: list[str]) -> tuple[str, float] | None:
    for i in range(len(lines) - 1):
        header, sep = lines[i], lines[i + 1]
        if "|" in header and _is_md_separator(sep):
            cols = _md_cell_count(header)
            if cols >= 2:
                return ("tabular", 0.95)
    return None


def _looks_like_prose(sample: list[str], delim: str) -> bool:
    enders = sum(1 for r in sample if r.rstrip().endswith((".", "!", "?")))
    if enders / len(sample) >= 0.5:
        return True
    cells = [c.strip() for r in sample for c in r.split(delim)]
    if not cells:
        return True
    avg_words = sum(len(c.split()) for c in cells) / len(cells)
    return avg_words > 3


def _detect_delimited(lines: list[str]) -> tuple[str, float] | None:
    from collections import Counter

    sample = lines[:20]
    if len(sample) < 3:
        return None
    best = None
    for delim, min_consistency in ((",", 0.85), ("\t", 0.7), (";", 0.85), ("|", 0.85)):
        counts = [row.count(delim) for row in sample]
        if counts[0] == 0:
            continue
        common_count, freq = Counter(counts).most_common(1)[0]
        if common_count == 0:
            continue
        consistency = freq / len(sample)
        ncols = common_count + 1
        if ncols < 2 or consistency < min_consistency:
            continue
        if _looks_like_prose(sample, delim):
            continue
        confidence = min(0.95, 0.5 + consistency * 0.3 + min(ncols, 5) * 0.03)
        if best is None or confidence > best[1]:
            best = ("tabular", confidence)
    return best


def _detect_tabular(content: str) -> tuple[str, float] | None:
    lines = [ln for ln in content.split("\n") if ln.strip()][:50]
    if len(lines) < 3:
        return None
    md = _detect_md_table(lines)
    if md:
        return md
    return _detect_delimited(lines)


def _try_parse_toml(content: str) -> bool:
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            return False
    try:
        tomllib.loads(content)
        return True
    except Exception:
        return False


def _parse_config_flavor(content: str) -> str | None:
    if len(content) > 1_000_000:
        return None
    if _try_parse_toml(content):
        return "toml"
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    try:
        parser.read_string(content)
    except Exception:
        return None
    return "ini" if parser.sections() else None


def _detect_config(content: str) -> tuple[str, float] | None:
    head = content.lstrip()[:1]
    if not head or head in "{<":
        return None
    lines = content.split("\n")[:200]
    non_empty = [ln for ln in lines if ln.strip()]
    if len(non_empty) < 3:
        return None
    body = [ln for ln in non_empty if not _CONFIG_COMMENT_RE.match(ln)]
    if len(body) < 3:
        return None

    sections = sum(1 for ln in body if _CONFIG_SECTION_RE.match(ln))
    if sections >= 1:
        assigns = sum(1 for ln in body if _TOML_ASSIGN_RE.match(ln) or _INI_ASSIGN_RE.match(ln))
        if assigns >= 2 and (sections + assigns) / len(body) >= 0.6:
            flavor = _parse_config_flavor(content)
            if flavor is not None:
                share = (sections + assigns) / len(body)
                return ("config", min(0.95, 0.7 + share * 0.25))

    if lines and lines[0].strip() == "---":
        for idx in range(1, min(len(lines), 60)):
            if lines[idx].strip() in ("---", "..."):
                tail = [ln for ln in lines[idx + 1 :] if ln.strip()]
                tail_yaml = sum(
                    1 for ln in tail if _YAML_KEY_RE.match(ln) or _YAML_LIST_RE.match(ln)
                )
                if tail and tail_yaml / len(tail) < 0.3:
                    return None
                break

    yaml_keys = sum(1 for ln in body if _YAML_KEY_RE.match(ln))
    yaml_lists = sum(
        1 for ln in body if _YAML_LIST_RE.match(ln) and not _YAML_KEY_RE.match(ln)
    )
    doc_marks = sum(1 for ln in body if _YAML_DOC_RE.match(ln.strip()))
    if yaml_keys < 3:
        return None
    share = (yaml_keys + yaml_lists + doc_marks) / len(body)
    if share < 0.6:
        return None
    enders = sum(1 for ln in body if ln.rstrip().endswith((".", "!", "?")))
    if enders / len(body) >= 0.5:
        return None
    avg_words = sum(len(ln.split()) for ln in body) / len(body)
    if avg_words > 8:
        return None
    indents = {
        len(ln) - len(ln.lstrip(" "))
        for ln in body
        if _YAML_KEY_RE.match(ln) or _YAML_LIST_RE.match(ln)
    }
    if len(indents) < 2 and doc_marks == 0 and yaml_lists < 3:
        return None
    return ("config", min(0.9, 0.55 + share * 0.35))


def _detect_code(content: str) -> tuple[str, float] | None:
    lines = content.split("\n")[:100]
    scores: dict[str, int] = {}
    for line in lines:
        for lang, patterns in _CODE_PATTERNS.items():
            for pattern in patterns:
                if pattern.match(line):
                    scores[lang] = scores.get(lang, 0) + 1
                    break
    if not scores:
        return None
    best_lang = max(scores, key=lambda k: scores[k])
    best_score = scores[best_lang]
    if best_score < 3:
        return None
    non_empty = sum(1 for line in lines if line.strip())
    ratio = best_score / max(non_empty, 1)
    confidence = min(1.0, 0.4 + ratio * 0.4 + best_score * 0.02)
    return ("code", confidence)


def _fixed_width_gutters(lines: list[str]) -> int:
    need = -(-9 * len(lines) // 10)
    hits = [0] * _FIXED_WIDTH_MAX_COLS
    for ln in lines:
        start = len(ln) - len(ln.lstrip(" "))
        for i in range(start, min(len(ln), _FIXED_WIDTH_MAX_COLS)):
            if ln[i] == " ":
                hits[i] += 1
    gaps = 0
    in_gap = False
    for count in hits:
        is_gutter = count >= need
        if is_gutter and not in_gap:
            gaps += 1
        in_gap = is_gutter
    return gaps


def _detect_fixed_width(content: str) -> tuple[str, float] | None:
    lines = [ln.rstrip() for ln in content.split("\n") if ln.strip()][:50]
    if len(lines) < 4 or any("\t" in ln for ln in lines):
        return None
    n = len(lines)
    if sum(1 for ln in lines if _FIXED_WIDTH_LIST_RE.match(ln)) / n >= 0.5:
        return None
    if sum(1 for ln in lines if _FIXED_WIDTH_PROSE_END_RE.search(ln)) / n >= 0.3:
        return None
    code_like = sum(
        1
        for ln in lines
        if ln.endswith(_FIXED_WIDTH_CODE_ENDS)
        or ln.lstrip().startswith(_FIXED_WIDTH_CODE_STARTS)
    )
    if code_like / n >= 0.3:
        return None
    gaps = _fixed_width_gutters(lines)
    if gaps < 2 and n >= 5:
        gaps = _fixed_width_gutters(lines[1:])
    if gaps < 2:
        return None
    return ("tabular", 0.7)


def _tool_name_prior(text: str, tool_name: str) -> tuple[str, float] | None:
    name = (tool_name or "").strip().lower()
    if not name:
        return None
    if "search" in name:
        return ("search", 0.55)
    if name in ("terminal", "shell", "bash", "sh", "exec", "run_command") or "terminal" in name:
        return ("log", 0.55)
    if "read_file" in name or name in ("read", "cat", "view"):
        if _looks_like_code(text):
            return ("code", 0.55)
        return ("text", 0.5)
    return None


def _looks_like_code(text: str) -> bool:
    signals = 0
    for pattern in (
        re.compile(r"^\s*(def|class|import|from)\s+\w+", re.MULTILINE),
        re.compile(r"^\s*(function|const|let|var|export)\s+", re.MULTILINE),
        re.compile(r"^\s*(public|private|protected|static)\s+\w+", re.MULTILINE),
        re.compile(r"#include\s*[<\"]", re.MULTILINE),
        re.compile(r"^\s*(fn|impl|struct|use)\s+\w+", re.MULTILINE),
    ):
        if pattern.search(text):
            signals += 1
    return signals >= 2
