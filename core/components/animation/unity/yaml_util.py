# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

"""Minimal parser for Unity's YAML text serialization format.

Unity serializes animation assets (.anim, .controller) as a stream of
``--- !u!<classId> &<fileId>`` document blocks. This module parses the
subset those assets use: block mappings, block sequences (including the
Unity idiom ``- key: value`` with continuation keys indented +2), flow
maps ``{key: value, ...}``, inline scalars and quoted strings. No
external YAML dependency is required.

Returns a list of documents::

    [{"class_id": int|None, "file_id": int|None, "name": str, "data": Any}]
"""

from __future__ import annotations
import re
from typing import Any, Optional

_DOC_RE = re.compile(r"^---\s*!u!\s*(\d+)\s*&\s*(-?\d+)(\s+stripped)?\s*$")
_DOC_BARE_RE = re.compile(r"^---\s*$")
_HEX_RE = re.compile(r"^[-+]?0x[0-9a-fA-F]+$")


def _parse_scalar_raw(raw: str) -> Any:
    s = raw.strip()
    if not s:
        return ""
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        inner = s[1:-1]
        return inner.replace('\\"', '"').replace("\\\\", "\\")
    if s.startswith("'") and s.endswith("'") and len(s) >= 2:
        return s[1:-1]
    if s in ("1.#INF", "+inf", "inf"):
        return float("inf")
    if s in ("-1.#INF", "-inf"):
        return float("-inf")
    if s in ("1.#IND", "-1.#IND", "nan"):
        return float("nan")
    if _HEX_RE.match(s):
        return int(s, 16)
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _split_flow_entries(body: str) -> list[str]:
    entries: list[str] = []
    depth = 0
    current: list[str] = []
    in_str = False
    esc = False
    for ch in body:
        if in_str:
            current.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            current.append(ch)
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        if ch == "," and depth == 0:
            entries.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        entries.append("".join(current).strip())
    return [e for e in entries if e]


def _parse_flow(data: str) -> Any:
    data = data.strip()
    if not data:
        return None
    if data.startswith("{") and data.endswith("}"):
        result: dict[str, Any] = {}
        for entry in _split_flow_entries(data[1:-1]):
            if not entry:
                continue
            if ":" in entry:
                key, _, val = entry.partition(":")
                result[key.strip().strip('"')] = _parse_flow(val.strip())
            else:
                result[f"_item{len(result)}"] = _parse_flow(entry)
        return result
    if data.startswith("[") and data.endswith("]"):
        body = data[1:-1]
        if not body.strip():
            return []
        out: list[Any] = []
        for entry in _split_flow_entries(body):
            if entry:
                out.append(_parse_flow(entry))
        return out
    return _parse_scalar_raw(data)


def _find_colon(line: str) -> int:
    in_str = False
    esc = False
    for idx, ch in enumerate(line):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == ":":
            return idx
    return -1


def _split_key_value(content: str) -> tuple[str, str]:
    idx = _find_colon(content)
    if idx < 0:
        return content.strip(), ""
    return content[:idx].strip(), content[idx + 1:].strip()


def _indent_of(raw: str) -> tuple[int, str]:
    stripped = raw.lstrip(" ")
    return len(raw) - len(stripped), stripped


def _parse_scalar_or_flow(rest: str) -> Any:
    rest = rest.strip()
    if rest in ("[]", "{}"):
        return _parse_flow(rest)
    if rest.startswith("{") or rest.startswith("["):
        return _parse_flow(rest)
    return _parse_scalar_raw(rest)


def _parse_mapping(tokens: list[tuple[int, str]], start: int, indent: int) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    i = start
    n = len(tokens)
    while i < n:
        ind, content = tokens[i]
        if ind != indent or content.startswith("-"):
            break
        key, rest = _split_key_value(content)
        i += 1
        rest = rest.strip()
        if rest.startswith("|") or rest.startswith(">"):
            text_lines: list[str] = []
            while i < n and tokens[i][0] > indent:
                text_lines.append(tokens[i][1])
                i += 1
            result[key] = "\n".join(text_lines)
            continue
        if rest == "":
            if i < n:
                nind = tokens[i][0]
                if nind > indent or (nind == indent and tokens[i][1].startswith("-")):
                    value, i = _parse_node(tokens, i, nind)
                    result[key] = value
                else:
                    result[key] = None
            else:
                result[key] = None
        else:
            result[key] = _parse_scalar_or_flow(rest)
    return result, i


def _parse_sequence(tokens: list[tuple[int, str]], start: int, indent: int) -> tuple[list[Any], int]:
    result: list[Any] = []
    i = start
    n = len(tokens)
    while i < n:
        ind, content = tokens[i]
        if ind != indent or not content.startswith("-"):
            break
        item_text = content[1:].strip()
        i += 1
        if item_text == "":
            if i < n and tokens[i][0] > indent:
                child_indent = tokens[i][0]
                value, i = _parse_node(tokens, i, child_indent)
                result.append(value)
            else:
                result.append(None)
        elif item_text.startswith("{") or item_text.startswith("["):
            result.append(_parse_flow(item_text))
            if i < n and tokens[i][0] > indent and tokens[i][1].startswith("-") is False:
                # Unity sometimes continues an inline item mapping below.
                cont_indent = tokens[i][0]
                if cont_indent > indent and _find_colon(tokens[i][1]) >= 0:
                    extra, i = _parse_item_continuation(tokens, i, cont_indent, result[-1])
                    result[-1] = extra
        elif _find_colon(item_text) >= 0:
            # Item is a mapping with its first key on the '-' line.
            item: dict[str, Any] = {}
            key, rest = _split_key_value(item_text)
            rest = rest.strip()
            if rest.startswith("|") or rest.startswith(">"):
                item[key] = ""
                while i < n and tokens[i][0] > indent:
                    i += 1
            elif rest == "":
                if i < n:
                    nind = tokens[i][0]
                    if nind > indent or (nind == indent and tokens[i][1].startswith("-")):
                        value, i = _parse_node(tokens, i, nind)
                        item[key] = value
                    else:
                        item[key] = None
                else:
                    item[key] = None
            else:
                item[key] = _parse_scalar_or_flow(rest)
            cont_indent = indent + 2
            if i < n and tokens[i][0] > indent and not tokens[i][1].startswith("-"):
                cont_indent = tokens[i][0]
            item, i = _parse_item_continuation(tokens, i, cont_indent, item)
            result.append(item)
        else:
            result.append(_parse_scalar_raw(item_text))
    return result, i


def _parse_item_continuation(
    tokens: list[tuple[int, str]], i: int, cont_indent: int, item: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    n = len(tokens)
    while i < n:
        ind, content = tokens[i]
        if ind != cont_indent or content.startswith("-"):
            break
        key, rest = _split_key_value(content)
        i += 1
        rest = rest.strip()
        if rest.startswith("|") or rest.startswith(">"):
            while i < n and tokens[i][0] > cont_indent:
                i += 1
            item[key] = ""
            continue
        if rest == "":
            if i < n:
                nind = tokens[i][0]
                if nind > cont_indent or (nind == cont_indent and tokens[i][1].startswith("-")):
                    value, i = _parse_node(tokens, i, nind)
                    item[key] = value
                else:
                    item[key] = None
            else:
                item[key] = None
        else:
            item[key] = _parse_scalar_or_flow(rest)
    return item, i


def _parse_node(tokens: list[tuple[int, str]], start: int, indent: int) -> tuple[Any, int]:
    if start >= len(tokens):
        return None, start
    if tokens[start][1].startswith("-") and tokens[start][0] == indent:
        return _parse_sequence(tokens, start, indent)
    return _parse_mapping(tokens, start, indent)


def parse_unity_documents(text: str) -> list[dict[str, Any]]:
    """Parse Unity-YAML text into documents ``[{class_id, file_id, name, data}]``."""
    raw_lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        indent, content = _indent_of(raw)
        if content.startswith("%"):
            continue
        raw_lines.append((indent, content))

    n = len(raw_lines)
    docs: list[dict[str, Any]] = []
    i = 0
    while i < n:
        indent, content = raw_lines[i]
        class_id: Optional[int] = None
        file_id: Optional[int] = None
        m = _DOC_RE.match(content)
        if m:
            class_id = int(m.group(1))
            file_id = int(m.group(2))
            i += 1
        elif _DOC_BARE_RE.match(content):
            i += 1
        elif content.startswith("---"):
            # Unknown/partial document marker (e.g. stripped variants) —
            # always advance so the loop cannot stall.
            i += 1
        start = i
        while i < n and not (raw_lines[i][0] == 0 and raw_lines[i][1].startswith("---")):
            i += 1
        if start == i:
            continue
        tokens = raw_lines[start:i]
        value, _ = _parse_node(tokens, 0, tokens[0][0])
        name = ""
        data: Any = value
        if isinstance(value, dict):
            if len(value) == 1:
                name, data = next(iter(value.items()))
            else:
                for k in list(value.keys()):
                    if not (isinstance(k, str) and k.startswith("_")):
                        name = k
                        data = value[k]
                        break
        docs.append({"class_id": class_id, "file_id": file_id, "name": name, "data": data})
    return docs


def parse_document_tree(text: str) -> Any:
    docs = parse_unity_documents(text)
    return docs[0]["data"] if docs else None