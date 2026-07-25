# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import struct
import numpy as np
from dataclasses import dataclass
from typing import Optional

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2_0 = 1.0925484305920792
SH_C2_1 = -1.0925484305920792
SH_C2_2 = 0.31539156525252005
SH_C2_3 = -1.0925484305920792
SH_C2_4 = 0.5462742152960396
SH_C3_0 = -0.5900435899266435
SH_C3_1 = 2.890611442640554
SH_C3_2 = -0.4570457994644658
SH_C3_3 = 0.3731763325901154
SH_C3_4 = -0.4570457994644658
SH_C3_5 = 1.4453057213202769
SH_C3_6 = -0.5900435899266435


def _splat_dtype(num_sh: int):
    n_rest = (num_sh * num_sh - 1) * 3
    fields = [
        ("x", np.float32), ("y", np.float32), ("z", np.float32),
        ("nx", np.float32), ("ny", np.float32), ("nz", np.float32),
        ("f_dc_0", np.float32), ("f_dc_1", np.float32), ("f_dc_2", np.float32),
    ]
    for i in range(n_rest):
        fields.append((f"f_rest_{i}", np.float32))
    fields.append(("opacity", np.float32))
    fields.append(("scale_0", np.float32))
    fields.append(("scale_1", np.float32))
    fields.append(("scale_2", np.float32))
    fields.append(("rot_0", np.float32))
    fields.append(("rot_1", np.float32))
    fields.append(("rot_2", np.float32))
    fields.append(("rot_3", np.float32))
    return np.dtype(fields)


@dataclass
class GaussianSplatData:
    positions: np.ndarray
    normals: np.ndarray
    sh_coeffs: np.ndarray
    opacity: np.ndarray
    scales: np.ndarray
    quaternions: np.ndarray
    num_sh: int

    @property
    def num_splats(self) -> int:
        return len(self.positions)


def _parse_header(lines: list[str]):
    vertex_count = 0
    properties = []
    fmt = "binary_little_endian"
    in_vertex = False
    for line in lines:
        s = line.strip()
        if s.startswith("format"):
            parts = s.split()
            if len(parts) >= 2:
                fmt = parts[1]
        elif s.startswith("element vertex"):
            parts = s.split()
            vertex_count = int(parts[2])
            in_vertex = True
        elif s.startswith("element"):
            in_vertex = False
        elif s.startswith("property") and in_vertex:
            parts = s.split()
            dtype_str = parts[1]
            name = parts[2]
            properties.append((dtype_str, name))
    return vertex_count, properties, fmt


def _ply_type(s: str):
    m = {
        "float": np.float32, "float32": np.float32, "double": np.float64,
        "uchar": np.uint8, "uint8": np.uint8,
        "short": np.int16, "int16": np.int16,
        "ushort": np.uint16, "uint16": np.uint16,
        "int": np.int32, "int32": np.int32,
        "uint": np.uint32, "uint32": np.uint32,
    }
    return m.get(s, np.float32)


def load_ply_gaussian_splat(path: str) -> Optional[GaussianSplatData]:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            header_lines = []
            while True:
                line = f.readline()
                if not line:
                    return None
                decoded = line.decode("ascii", errors="ignore").strip()
                header_lines.append(decoded)
                if decoded == "end_header":
                    break

            vertex_count, properties, fmt = _parse_header(header_lines)

            prop_names = [p[1] for p in properties]
            has_normals = "nx" in prop_names

            if fmt == "binary_little_endian":
                dt = np.dtype([(p[1], _ply_type(p[0])) for p in properties])
                raw = np.frombuffer(f.read(vertex_count * dt.itemsize), dtype=dt)
            elif fmt == "binary_big_endian":
                dt = np.dtype([(p[1], _ply_type(p[0])) for p in properties])
                raw = np.frombuffer(f.read(vertex_count * dt.itemsize), dtype=dt)
                raw = raw.astype(dt.newbyteorder("<"))
            else:
                rows = []
                for _ in range(vertex_count):
                    line = f.readline().decode("ascii", errors="ignore").strip()
                    vals = line.split()
                    row = []
                    for i, (_, n) in enumerate(properties):
                        v = vals[i] if i < len(vals) else "0"
                        t = _ply_type(properties[i][0])
                        if t == np.uint8:
                            row.append(int(v))
                        elif t in (np.int16, np.int32):
                            row.append(int(v))
                        else:
                            row.append(float(v))
                    rows.append(row)
                dt = np.dtype([(p[1], _ply_type(p[0])) for p in properties])
                raw = np.array(rows, dtype=dt)

            positions = np.column_stack([raw["x"].astype(np.float32),
                                          raw["y"].astype(np.float32),
                                          raw["z"].astype(np.float32)])

            if has_normals:
                normals = np.column_stack([raw["nx"].astype(np.float32),
                                            raw["ny"].astype(np.float32),
                                            raw["nz"].astype(np.float32)])
            else:
                normals = np.zeros_like(positions)

            dc = np.column_stack([raw["f_dc_0"].astype(np.float32),
                                   raw["f_dc_1"].astype(np.float32),
                                   raw["f_dc_2"].astype(np.float32)])
            dc = dc * SH_C0 + 0.5

            rest_keys = [k for k in prop_names if k.startswith("f_rest_")]
            num_rest = len(rest_keys)
            if num_rest > 0:
                rest = np.column_stack([raw[k].astype(np.float32) for k in rest_keys])
            else:
                rest = np.zeros((vertex_count, 0), dtype=np.float32)

            sh_coeffs = np.concatenate([dc, rest], axis=1)

            logit = raw["opacity"].astype(np.float32)
            opacity = 1.0 / (1.0 + np.exp(-logit))

            scales = np.exp(np.column_stack([raw["scale_0"].astype(np.float32),
                                              raw["scale_1"].astype(np.float32),
                                              raw["scale_2"].astype(np.float32)]))

            quaternions = np.column_stack([raw["rot_0"].astype(np.float32),
                                            raw["rot_1"].astype(np.float32),
                                            raw["rot_2"].astype(np.float32),
                                            raw["rot_3"].astype(np.float32)])
            q_len = np.linalg.norm(quaternions, axis=1, keepdims=True)
            q_len = np.maximum(q_len, 1e-8)
            quaternions = quaternions / q_len

            num_sh = 1
            rest_per_sh = 3
            if num_rest > 0:
                num_sh_approx = int(np.sqrt(num_rest // rest_per_sh + 1))
                num_sh = max(1, min(num_sh_approx, 4))

            return GaussianSplatData(
                positions=positions,
                normals=normals,
                sh_coeffs=sh_coeffs,
                opacity=opacity,
                scales=scales,
                quaternions=quaternions,
                num_sh=num_sh,
            )
    except Exception:
        return None
