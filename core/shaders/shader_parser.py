# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import re
import os
from typing import Optional, Any
from dataclasses import dataclass, field


@dataclass
class ShaderProperty:
    name: str
    display_name: str
    prop_type: str
    default_value: Any
    attributes: list[str] = field(default_factory=list)
    range_min: float = 0.0
    range_max: float = 1.0


@dataclass
class ShaderPass:
    vertex_source: str
    fragment_source: str
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class ShaderData:
    name: str
    properties: list[ShaderProperty]
    passes: list[ShaderPass]
    fallback: str = ""


_SHADER_PATTERN = re.compile(
    r'Shader\s+"([^"]*)"\s*\{',
    re.DOTALL
)


def _find_closing_brace(text: str, start: int) -> int:
    depth = 0
    i = start
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _parse_properties_block(text: str) -> list[ShaderProperty]:
    props = []
    prop_pattern = re.compile(
        r'\[?([^\]]*)\]?\s*'
        r'(\w+)\s*'
        r'\("([^"]*)"\s*,\s*'
        r'(\w+(?:\s*\(\s*[^)]*\s*\))?)\s*\)\s*'
        r'=\s*(.+?)\s*(?=\n\s*\[|\n\s*\w|\n\s*\})',
        re.DOTALL
    )
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped == '{' or stripped == '}':
            i += 1
            continue
        attrs = []
        while stripped.startswith('['):
            bracket_end = stripped.find(']')
            if bracket_end >= 0:
                attrs.append(stripped[1:bracket_end].strip())
                stripped = stripped[bracket_end + 1:].strip()
        prop_match = re.match(
            r'(\w+)\s*'
            r'\("([^"]*)"\s*,\s*'
            r'(\w+)'
            r'(?:\s*\(\s*([^)]*)\s*\))?\s*\)\s*'
            r'=\s*(.+?)\s*$',
            stripped
        )
        if prop_match:
            prop_name = prop_match.group(1)
            display_name = prop_match.group(2)
            prop_type = prop_match.group(3)
            type_args = prop_match.group(4)
            default_raw = prop_match.group(5).strip()
            default_value: Any = None
            range_min, range_max = 0.0, 1.0
            if prop_type == "Color":
                match = re.match(r'\(([^)]+)\)', default_raw)
                if match:
                    parts = match.group(1).split(',')
                    default_value = [float(p.strip()) for p in parts]
            elif prop_type == "Float" or prop_type == "Range":
                match = re.match(r'([0-9.]+)', default_raw)
                default_value = float(match.group(1)) if match else 0.0
                if prop_type == "Range" and type_args:
                    range_parts = type_args.split(',')
                    if len(range_parts) >= 2:
                        range_min = float(range_parts[0].strip())
                        range_max = float(range_parts[1].strip())
            elif prop_type == "Int":
                match = re.match(r'([0-9]+)', default_raw)
                default_value = int(match.group(1)) if match else 0
            elif prop_type == "2D":
                default_value = ""
            elif prop_type == "Vector":
                match = re.match(r'\(([^)]+)\)', default_raw)
                if match:
                    parts = match.group(1).split(',')
                    default_value = [float(p.strip()) for p in parts]
            props.append(ShaderProperty(
                name=prop_name,
                display_name=display_name,
                prop_type=prop_type,
                default_value=default_value,
                attributes=attrs,
                range_min=range_min,
                range_max=range_max
            ))
        i += 1
    return props


_GLSL_BLOCK = re.compile(
    r'GLSLPROGRAM\s*(.*?)\s*ENDGLSL',
    re.DOTALL
)


def _parse_glsl_block(glsl_source: str) -> tuple[str, str]:
    frag_marker = '// @FRAGMENT'
    frag_idx = glsl_source.find(frag_marker)
    if frag_idx < 0:
        frag_idx = glsl_source.find('#ifdef FRAGMENT_SHADER')
        if frag_idx >= 0:
            vert = glsl_source[:frag_idx]
            frag = glsl_source[frag_idx:]
            vert = '#define VERTEX_SHADER\n' + vert
            frag = '#define FRAGMENT_SHADER\n' + frag
            return vert, frag
        return glsl_source, glsl_source
    vert_source = glsl_source[:frag_idx].strip()
    frag_source = glsl_source[frag_idx + len(frag_marker):].strip()
    return vert_source, frag_source


def parse_shader_file(path: str) -> Optional[ShaderData]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return None

    name_match = re.search(r'Shader\s+"([^"]*)"', content)
    if not name_match:
        return None
    shader_name = name_match.group(1)

    shader_start = content.find('{', name_match.end())
    if shader_start < 0:
        return None

    props = []
    prop_match = re.search(r'Properties\s*\{', content)
    if prop_match:
        props_start = prop_match.end()
        props_end = _find_closing_brace(content, props_start - 1)
        if props_end > props_start:
            props_block = content[props_start:props_end]
            props = _parse_properties_block(props_block)

    passes = []
    subshader_match = re.search(r'SubShader\s*\{', content)
    if subshader_match:
        sub_start = subshader_match.end()
        sub_end = _find_closing_brace(content, sub_start - 1)
        sub_content = content[sub_start:sub_end] if sub_end > sub_start else ""

        pass_pattern = re.compile(r'Pass\s*\{', re.DOTALL)
        for pm in pass_pattern.finditer(sub_content):
            pass_start = pm.end()
            pass_end = _find_closing_brace(sub_content, pass_start - 1)
            pass_body = sub_content[pass_start:pass_end] if pass_end > pass_start else ""

            glsl_match = _GLSL_BLOCK.search(pass_body)
            if glsl_match:
                vert_src, frag_src = _parse_glsl_block(glsl_match.group(1))
                pass_tags = {}
                tag_match = re.search(r'Tags\s*\{([^}]*)\}', pass_body)
                if tag_match:
                    for kv in re.finditer(r'"([^"]*)"\s*=\s*"([^"]*)"', tag_match.group(1)):
                        pass_tags[kv.group(1)] = kv.group(2)
                passes.append(ShaderPass(
                    vertex_source=vert_src,
                    fragment_source=frag_src,
                    tags=pass_tags
                ))

    fallback_match = re.search(r'Fallback\s+"([^"]*)"', content)
    fallback = fallback_match.group(1) if fallback_match else ""

    return ShaderData(
        name=shader_name,
        properties=props,
        passes=passes,
        fallback=fallback
    )


def shader_properties_to_material_dict(properties: list[ShaderProperty]) -> dict[str, Any]:
    result = {}
    for prop in properties:
        if prop.prop_type == "Color":
            result[prop.name] = list(prop.default_value) if prop.default_value else [1.0, 1.0, 1.0, 1.0]
        elif prop.prop_type == "Float" or prop.prop_type == "Range":
            result[prop.name] = float(prop.default_value) if prop.default_value is not None else 0.0
        elif prop.prop_type == "Int":
            result[prop.name] = int(prop.default_value) if prop.default_value is not None else 0
        elif prop.prop_type == "2D":
            result[prop.name] = ""
        elif prop.prop_type == "Vector":
            result[prop.name] = list(prop.default_value) if prop.default_value else [0.0, 0.0, 0.0, 0.0]
    return result
