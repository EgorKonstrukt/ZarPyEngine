# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import moderngl
import numpy as np

from core.foundation.logger import Logger
from core.foundation.progress import notify_error, task_complete, task_start


def compile_compute_shader(ctx: moderngl.Context, source: str,
                           title_hint: str = "compute") -> moderngl.ComputeShader | None:
    label = os.path.basename(title_hint)
    task_id = f"compute:{title_hint}"
    task_start(task_id, f"Compiling compute {label}...", fraction=None)
    try:
        return ctx.compute_shader(source)
    except moderngl.Error as e:
        Logger.error(f"Failed to compile compute shader '{label}': {e}", e)
        notify_error(f"Failed to compile compute {label}")
        return None
    finally:
        task_complete(task_id)


@dataclass
class ComputeProperty:
    name: str
    display_name: str
    prop_type: str
    default_value: Any
    attributes: list[str] = field(default_factory=list)
    range_min: float = 0.0
    range_max: float = 1.0


@dataclass
class ComputeData:
    name: str
    properties: list[ComputeProperty]
    source: str


class ComputeBuffer:
    def __init__(self, ctx: moderngl.Context, binding: int, data: np.ndarray):
        self._ctx = ctx
        self._binding = binding
        self._data = data
        self._buffer: Optional[moderngl.Buffer] = None

    def upload(self):
        if self._buffer is None:
            self._buffer = self._ctx.buffer(self._data.tobytes())
        else:
            self._buffer.write(self._data.tobytes())
        self._buffer.bind_to_storage_buffer(self._binding)

    def download(self) -> np.ndarray:
        if self._buffer is None:
            return self._data
        data = self._buffer.read()
        return np.frombuffer(data, dtype=self._data.dtype).reshape(self._data.shape)

    def download_into(self):
        if self._buffer is None:
            return
        data = self._buffer.read()
        arr = np.frombuffer(data, dtype=self._data.dtype).reshape(self._data.shape)
        self._data[:] = arr[:]

    def read(self) -> np.ndarray:
        return self.download()

    def write(self, data: np.ndarray):
        self._data = data
        self.upload()

    def bind(self, prog: moderngl.Program, name: str):
        if self._buffer is not None:
            self._buffer.bind_to_storage_buffer(self._binding)

    def release(self):
        if self._buffer:
            self._buffer.release()
            self._buffer = None


class ComputeShader:
    def __init__(self, ctx: moderngl.Context, name: str = "ComputeShader"):
        self._ctx = ctx
        self._name = name
        self._program: Optional[moderngl.ComputeShader] = None
        self._source: str = ""
        self._properties: list[ComputeProperty] = []
        self._buffers: dict[str, ComputeBuffer] = {}
        self._uniforms: dict[str, Any] = {}
        self._group_x: int = 1
        self._group_y: int = 1
        self._group_z: int = 1

    @property
    def name(self) -> str: return self._name

    @property
    def program(self) -> Optional[moderngl.ComputeShader]:
        return self._program

    @classmethod
    def load_from_file(cls, ctx: moderngl.Context, path: str) -> Optional[ComputeShader]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            Logger.error(f"Failed to read compute shader '{path}': {e}", e)
            return None

        name_match = re.search(r'ComputeShader\s+"([^"]*)"', content)
        if not name_match:
            Logger.error(f"Invalid .compute file '{path}': no ComputeShader declaration")
            return None
        shader_name = name_match.group(1)

        props: list[ComputeProperty] = []
        prop_match = re.search(r'Properties\s*\{', content)
        if prop_match:
            brace = content.index('{', prop_match.start())
            depth = 0
            end = -1
            for i in range(brace, len(content)):
                if content[i] == '{': depth += 1
                elif content[i] == '}':
                    depth -= 1
                    if depth == 0: end = i; break
            if end > brace:
                props_text = content[brace + 1:end]
                for line in props_text.splitlines():
                    line = line.strip()
                    if not line or line == '{' or line == '}' or line.startswith('//'):
                        continue
                    prop_match_line = re.match(
                        r'(\w+)\s*\(\s*"([^"]*)"\s*,\s*(\w+)\s*\)\s*=\s*(.+?)\s*$',
                        line
                    )
                    if prop_match_line:
                        pname = prop_match_line.group(1)
                        display = prop_match_line.group(2)
                        ptype = prop_match_line.group(3)
                        default_raw = prop_match_line.group(4).strip()
                        default_value: Any = None
                        if ptype == "Int":
                            default_value = int(default_raw)
                        elif ptype == "Float":
                            default_value = float(default_raw)
                        elif ptype == "Int2":
                            vals = re.findall(r'[\d-]+', default_raw)
                            default_value = [int(v) for v in vals]
                        elif ptype == "Int4":
                            vals = re.findall(r'[\d-]+', default_raw)
                            default_value = [int(v) for v in vals]
                        elif ptype == "Float2":
                            vals = re.findall(r'[\d.e-]+', default_raw)
                            default_value = [float(v) for v in vals]
                        elif ptype == "Float4":
                            vals = re.findall(r'[\d.e-]+', default_raw)
                            default_value = [float(v) for v in vals]
                        else:
                            default_value = default_raw
                        props.append(ComputeProperty(
                            name=pname, display_name=display,
                            prop_type=ptype, default_value=default_value
                        ))

        glsl_start = content.find('GLSLPROGRAM')
        if glsl_start < 0:
            Logger.error(f"No GLSLPROGRAM block in '{path}'")
            return None
        glsl_start += len('GLSLPROGRAM')
        glsl_end = content.find('ENDGLSL', glsl_start)
        if glsl_end < 0:
            Logger.error(f"No ENDGLSL terminator in '{path}'")
            return None
        source = content[glsl_start:glsl_end].strip()

        cs = cls(ctx, shader_name)
        cs._properties = props
        cs._source = source

        cs._program = compile_compute_shader(ctx, source, path)
        if cs._program is None:
            return None
        cs._parse_local_size(source)

        return cs

    def _parse_local_size(self, source: str):
        m = re.search(r'layout\s*\(\s*local_size_x\s*=\s*(\d+)\s*,\s*local_size_y\s*=\s*(\d+)\s*,\s*local_size_z\s*=\s*(\d+)', source)
        if m:
            self._group_x = int(m.group(1))
            self._group_y = int(m.group(2))
            self._group_z = int(m.group(3))

    def set_buffer(self, name: str, binding: int, data: np.ndarray) -> ComputeBuffer:
        buf = ComputeBuffer(self._ctx, binding, data)
        self._buffers[name] = buf
        return buf

    def get_buffer(self, name: str) -> Optional[ComputeBuffer]:
        return self._buffers.get(name)

    def set_uniform(self, name: str, value: Any):
        self._uniforms[name] = value

    def set_uniform_buffer(self, name: str, binding: int, data: np.ndarray):
        buf = self._ctx.buffer(data.tobytes())
        buf.bind_to_uniform_buffer(binding)
        if name in self._program:
            ubo = self._program[name]
            ubo.binding = binding

    def dispatch(self, count_x: int, count_y: int = 1, count_z: int = 1):
        if not self._program:
            Logger.error(f"Compute shader '{self._name}' not compiled")
            return

        for buf_name, buf in self._buffers.items():
            buf.upload()
            buf.bind(self._program, buf_name)

        for uname, uval in self._uniforms.items():
            if uname in self._program:
                try:
                    self._program[uname] = uval
                except Exception:
                    try:
                        arr = np.array(uval, dtype=np.float32)
                        self._program[uname].write(arr.tobytes())
                    except Exception:
                        pass

        groups_x = (count_x + self._group_x - 1) // self._group_x
        groups_y = (count_y + self._group_y - 1) // self._group_y
        groups_z = (count_z + self._group_z - 1) // self._group_z

        self._program.run(groups_x, groups_y, groups_z)

    def dispatch_groups(self, groups_x: int, groups_y: int = 1, groups_z: int = 1):
        if not self._program:
            Logger.error(f"Compute shader '{self._name}' not compiled")
            return
        for buf_name, buf in self._buffers.items():
            buf.upload()
            buf.bind(self._program, buf_name)
        for uname, uval in self._uniforms.items():
            if uname in self._program:
                try:
                    self._program[uname] = uval
                except Exception:
                    try:
                        arr = np.array(uval, dtype=np.float32)
                        self._program[uname].write(arr.tobytes())
                    except Exception:
                        pass
        self._program.run(groups_x, groups_y, groups_z)

    def release(self):
        for buf in self._buffers.values():
            buf.release()
        self._buffers.clear()
        if self._program:
            self._program.release()
            self._program = None


class ComputeManager:
    def __init__(self, ctx: moderngl.Context):
        self._ctx = ctx
        self._cache: dict[str, ComputeShader] = {}

    def load(self, path: str) -> Optional[ComputeShader]:
        abs_path = os.path.abspath(path)
        if abs_path in self._cache:
            return self._cache[abs_path]
        cs = ComputeShader.load_from_file(self._ctx, abs_path)
        if cs:
            self._cache[abs_path] = cs
        return cs

    def get(self, name_or_path: str) -> Optional[ComputeShader]:
        if name_or_path in self._cache:
            return self._cache[name_or_path]
        abs_path = os.path.abspath(name_or_path)
        return self._cache.get(abs_path)

    def register(self, key: str, cs: ComputeShader):
        self._cache[key] = cs

    def release(self):
        for cs in self._cache.values():
            cs.release()
        self._cache.clear()
