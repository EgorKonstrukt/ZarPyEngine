# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import json
import re
import os
from typing import Optional, Any
from core.foundation.logger import Logger


def _project_root():
    cur = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.basename(cur) == "core":
            return os.path.dirname(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_SHADER_PROP_RE = re.compile(
    r'(?:\[(\w+)\]\s*)?'              # optional attribute like [MainColor]
    r'_(\w+)'                          # property name _BaseColor
    r'\s*\(\s*"([^"]+)"'             # display name "Base Color"
    r'\s*,\s*(\w+(?:\s*\([^)]*\))?)' # type: Color, Float, Range(0,1), 2D, Int, Vector
    r'\s*\)\s*=\s*(.+)'               # default value
)


class ShaderProperty:
    def __init__(self, name: str, display_name: str, prop_type: str,
                 default_value: Any, attributes: list[str] | None = None,
                 range_min: float = 0.0, range_max: float = 1.0):
        self.name = name
        self.display_name = display_name
        self.prop_type = prop_type
        self.default_value = default_value
        self.attributes = attributes or []
        self.range_min = range_min
        self.range_max = range_max

    def __repr__(self):
        return f"ShaderProperty({self.name}, {self.display_name}, {self.prop_type})"


def _parse_shader_properties_block(text: str) -> list[ShaderProperty]:
    """Parse the Properties { ... } block from a .shader file."""
    props = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line == "{" or line == "}":
            continue
        m = _SHADER_PROP_RE.match(line)
        if not m:
            continue
        attr = m.group(1)
        name = "_" + m.group(2)
        display_name = m.group(3)
        raw_type = m.group(4).strip()
        raw_default = m.group(5).strip()

        attributes = [attr] if attr else []
        prop_type = raw_type
        range_min, range_max = 0.0, 1.0

        # Parse Range(min, max)
        range_match = re.match(r'Range\s*\(([^,]+),([^)]+)\)', raw_type, re.IGNORECASE)
        if range_match:
            prop_type = "Range"
            range_min = float(range_match.group(1))
            range_max = float(range_match.group(2))

        default_value = _parse_shader_default(raw_type, raw_default)

        props.append(ShaderProperty(
            name=name,
            display_name=display_name,
            prop_type=prop_type,
            default_value=default_value,
            attributes=attributes,
            range_min=range_min,
            range_max=range_max,
        ))
    return props


def _parse_shader_default(raw_type: str, raw_default: str) -> Any:
    """Parse the default value string based on property type."""
    t = raw_type.lower().strip()

    if t == "color" or t.startswith("vector"):
        # (1, 1, 1, 1)
        vals = re.findall(r'[\d.]+(?:e[+-]?\d+)?', raw_default)
        return [float(v) for v in vals]

    if t == "float" or t.startswith("range"):
        return float(raw_default.strip())

    if t == "int":
        return int(raw_default.strip())

    if t == "2d" or t == "cube":
        # "white" {} or "bump" {}
        m = re.match(r'"([^"]*)"', raw_default)
        return m.group(1) if m else ""

    return raw_default.strip()


def _extract_glsl_from_shader(text: str) -> tuple[str, str] | None:
    """Extract vertex and fragment GLSL source from a .shader file.
    Returns (vert_src, frag_src) or None."""
    glsl_start = text.find("GLSLPROGRAM")
    if glsl_start < 0:
        return None
    glsl_start += len("GLSLPROGRAM")
    glsl_end = text.find("ENDGLSL", glsl_start)
    if glsl_end < 0:
        return None
    glsl_body = text[glsl_start:glsl_end].strip()

    frag_marker = "// @FRAGMENT"
    frag_pos = glsl_body.find(frag_marker)
    if frag_pos < 0:
        return None

    vert_src = glsl_body[:frag_pos].strip()
    frag_src = glsl_body[frag_pos + len(frag_marker):].strip()
    return (vert_src, frag_src)


def _parse_shader_file(path: str) -> tuple[list[ShaderProperty], str, str] | None:
    """Parse a .shader file and return (properties, vert_src, frag_src) or None."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        Logger.error(f"Failed to read shader '{path}': {e}", e)
        return None

    # Extract Properties block
    props_start = text.find("Properties")
    if props_start < 0:
        Logger.error(f"No Properties block in shader '{path}'")
        return None
    brace = text.find("{", props_start)
    if brace < 0:
        return None
    depth = 0
    props_end = -1
    for i in range(brace, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                props_end = i
                break
    if props_end < 0:
        return None
    props_text = text[brace + 1:props_end]
    properties = _parse_shader_properties_block(props_text)

    # Extract GLSL
    glsl = _extract_glsl_from_shader(text)
    if not glsl:
        return None
    vert_src, frag_src = glsl
    return (properties, vert_src, frag_src)


class ShaderSource:
    def __init__(self, vert: str, frag: str, geom: str = ""):
        self.vert = vert
        self.frag = frag
        self.geom = geom


class Material:
    def __init__(self, name: str = "Material"):
        self.name: str = name
        self.shader_path: str = ""
        self.properties: dict[str, Any] = {}
        self._shader_properties: list[ShaderProperty] = []
        self._gl_program: Optional[Any] = None

    def set_property(self, name: str, value: Any): self.properties[name] = value
    def get_property(self, name: str, default: Any = None) -> Any: return self.properties.get(name, default)

    @staticmethod
    def _resolve_shader_path(shader_path: str, project_root: str) -> str:
        if os.path.isabs(shader_path):
            return shader_path
        candidates = []
        if project_root:
            candidates.append(os.path.normpath(os.path.join(project_root, shader_path)))
        engine_root = _project_root()
        candidates.append(os.path.normpath(os.path.join(engine_root, shader_path)))
        candidates.append(os.path.normpath(os.path.join(engine_root, "core", "shaders", shader_path)))
        shader_name = os.path.basename(shader_path)
        if shader_name != shader_path:
            candidates.append(os.path.normpath(os.path.join(engine_root, "core", "shaders", shader_name)))
        for c in candidates:
            if os.path.exists(c):
                return c
        return candidates[0] if candidates else shader_path

    def load_shader_properties(self, shader_path: str, project_root: str = "") -> bool:
        """Parse a .shader file and populate properties from its Properties block.
        project_root: project root directory, used to resolve relative shader paths.
        Returns True if properties were loaded from a .shader file."""
        if not shader_path or shader_path in ("default", ""):
            self._shader_properties = []
            return False
        abs_shader = self._resolve_shader_path(shader_path, project_root)
        ext = os.path.splitext(abs_shader)[1].lower()
        if ext != ".shader":
            self._shader_properties = []
            return False
        result = _parse_shader_file(abs_shader)
        if not result:
            return False
        shader_props, vert_src, frag_src = result
        self._shader_properties = shader_props
        for p in shader_props:
            if p.name not in self.properties:
                self.properties[p.name] = p.default_value
        return True

    def save(self, path: str, project_root: str = ""):
        root = project_root or os.getcwd()
        abs_path = path if os.path.isabs(path) else os.path.normpath(os.path.join(root, path))
        abs_path = os.path.normpath(abs_path)
        rel_shader = self._rel_path(self.shader_path, root)
        rel_props = dict(self.properties)
        for k, v in list(rel_props.items()):
            if isinstance(v, str) and v:
                rel_props[k] = self._rel_path(v, root)
        data = {"name": self.name, "shader_path": rel_shader, "properties": rel_props}
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        MaterialLibrary._materials[abs_path] = self

    @classmethod
    def load(cls, path: str, project_root: str = "") -> Optional[Material]:
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext in (".mat", ".zpem"):
                return cls._load_mat_file(path, project_root)
            elif ext == ".json":
                return cls._load_json_file(path, project_root)
            else:
                return None
        except Exception as e:
            Logger.error(f"Failed to load material '{path}': {e}", e)
            return None

    @classmethod
    def _load_mat_file(cls, path: str, project_root: str = "") -> Optional[Material]:
        try:
            mat_dir = os.path.dirname(os.path.abspath(path))

            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()

            name = os.path.splitext(os.path.basename(path))[0]
            m = cls(name)

            if raw.startswith("{"):
                try:
                    data = json.loads(raw)
                    m.shader_path = data.get("shader_path", "")
                    saved_props = data.get("properties", {})
                    m.properties.update(saved_props)
                    if "name" in data:
                        m.name = data["name"]
                except json.JSONDecodeError:
                    pass

            if not m.shader_path:
                m.shader_path = "default"

            # Repair corrupted "false" key (old bug stored color as boolean key)
            false_val = m.properties.pop("false", None)
            if false_val is not None and isinstance(false_val, list) and len(false_val) >= 3:
                m.properties.setdefault("albedo_color", false_val)

            for old_key, new_key in _PROP_MIGRATION.items():
                if old_key in m.properties and new_key not in m.properties:
                    m.properties[new_key] = m.properties[old_key]

            m.load_shader_properties(m.shader_path, project_root)

            for k, v in list(m.properties.items()):
                if isinstance(v, str) and v and not os.path.isabs(v):
                    resolved = cls._resolve_texture(v, mat_dir)
                    if not resolved:
                        resolved = cls._resolve_tex_path(v, mat_dir, project_root or os.getcwd())
                    if resolved:
                        m.properties[k] = resolved

            MaterialLibrary._materials[os.path.normpath(path)] = m
            return m
        except Exception as e:
            Logger.error(f"Failed to load .mat file '{path}': {e}", e)
            return None

    @classmethod
    def _load_json_file(cls, path: str, project_root: str = "") -> Optional[Material]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            m = cls(data.get("name", "Material"))
            m.shader_path = data.get("shader_path", "")
            if not m.shader_path:
                m.shader_path = "default"
            m.properties.update(data.get("properties", {}))

            false_val = m.properties.pop("false", None)
            if false_val is not None and isinstance(false_val, list) and len(false_val) >= 3:
                m.properties.setdefault("albedo_color", false_val)

            for old_key, new_key in _PROP_MIGRATION.items():
                if old_key in m.properties and new_key not in m.properties:
                    m.properties[new_key] = m.properties[old_key]

            root = project_root or os.getcwd()
            mat_dir = os.path.dirname(path)
            for k, v in list(m.properties.items()):
                if isinstance(v, str) and v:
                    resolved = cls._resolve_tex_path(v, mat_dir, root)
                    if resolved:
                        m.properties[k] = resolved
            m.load_shader_properties(m.shader_path, project_root)
            MaterialLibrary._materials[os.path.normpath(path)] = m
            return m
        except Exception as e:
            Logger.error(f"Failed to load material '{path}': {e}", e)
            return None

    @classmethod
    def _resolve_texture(cls, tex_name: str, mat_dir: str) -> str:
        if not tex_name:
            return ""

        for ext in ["", ".png", ".jpg", ".jpeg", ".tga", ".bmp", ".dds"]:
            candidate = os.path.join(mat_dir, tex_name + ext)
            if os.path.exists(candidate):
                return candidate

        candidates = [
            os.path.join(mat_dir, tex_name),
            os.path.join(mat_dir, tex_name + ".png"),
            os.path.join(mat_dir, tex_name + ".jpg"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c

        return ""

    @staticmethod
    def _resolve_tex_path(val: str, mat_dir: str, root: str) -> str:
        if os.path.exists(val):
            return val
        # Relative to material directory
        candidate = os.path.normpath(os.path.join(mat_dir, val))
        if os.path.exists(candidate):
            return candidate.replace("\\", "/")
        # Relative to project root
        candidate = os.path.normpath(os.path.join(root, val))
        if os.path.exists(candidate):
            return candidate.replace("\\", "/")
        # Windows absolute path вЂ” extract subpath
        if len(val) > 1 and val[1] == ":":
            parts = val.replace("\\", "/").split("/")
            for i in range(len(parts)):
                sub = "/".join(parts[i:])
                if sub:
                    c = os.path.normpath(os.path.join(root, sub))
                    if os.path.exists(c):
                        return c.replace("\\", "/")
        return val

    @staticmethod
    def _rel_path(val: str, root: str) -> str:
        if not val:
            return ""
        if not os.path.isabs(val):
            return val.replace("\\", "/")
        try:
            rel = os.path.relpath(val, root)
            if not rel.startswith(".."):
                return rel.replace("\\", "/")
        except ValueError:
            pass
        return val


_PROP_MIGRATION = {
    "albedo_color": "_BaseColor",
    "metallic": "_Metallic",
    "smoothness": "_Smoothness",
    "emission_color": "_EmissionColor",
    "emission_intensity": "_EmissionIntensity",
    "albedo_texture": "_BaseMap",
    "normal_texture": "_NormalMap",
    "roughness_texture": "_RoughnessMap",
    "occlusion_texture": "_OcclusionMap",
    "emission_texture": "_EmissionMap",
}


class MaterialLibrary:
    _materials: dict[str, Material] = {}

    @classmethod
    def register(cls, name: str, mat: Material): cls._materials[name] = mat

    @classmethod
    def get(cls, name: str) -> Optional[Material]: return cls._materials.get(name)

    @classmethod
    def load(cls, path: str, project_root: str = "") -> Optional[Material]:
        if path in cls._materials: return cls._materials[path]
        m = Material.load(path, project_root) if project_root else Material.load(path)
        if m: cls._materials[path] = m
        return m

    @classmethod
    def create_default(cls) -> Material:
        m = Material("Default")
        m.shader_path = "default"
        cls._materials["Default"] = m
        return m
