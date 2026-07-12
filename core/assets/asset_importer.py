# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import sys
import os
import ctypes
from asyncio import Future

import numpy as np
from typing import Optional, Callable
from core.ecs.pool import asset as _get_asset_pool

if sys.platform == "win32":
    _ASSIMP_LIB_NAME = "assimp-vc143-mt.dll"
else:
    _ASSIMP_LIB_NAME = "libassimp.so.6.0.5"


def _find_project_root():
    cur = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.basename(cur) == "core":
            return os.path.dirname(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_PROJECT_ROOT = _find_project_root()
_DLL_PATH = os.path.join(_PROJECT_ROOT, _ASSIMP_LIB_NAME)


class aiVector3D(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]


class aiVector2D(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class aiColor4D(ctypes.Structure):
    _fields_ = [("r", ctypes.c_float), ("g", ctypes.c_float), ("b", ctypes.c_float), ("a", ctypes.c_float)]


class aiMatrix4x4(ctypes.Structure):
    _fields_ = [
        ("a1", ctypes.c_float), ("a2", ctypes.c_float), ("a3", ctypes.c_float), ("a4", ctypes.c_float),
        ("b1", ctypes.c_float), ("b2", ctypes.c_float), ("b3", ctypes.c_float), ("b4", ctypes.c_float),
        ("c1", ctypes.c_float), ("c2", ctypes.c_float), ("c3", ctypes.c_float), ("c4", ctypes.c_float),
        ("d1", ctypes.c_float), ("d2", ctypes.c_float), ("d3", ctypes.c_float), ("d4", ctypes.c_float),
    ]


class aiString(ctypes.Structure):
    _fields_ = [("length", ctypes.c_size_t), ("data", ctypes.c_char * 1024)]


class aiFace(ctypes.Structure):
    _fields_ = [("mNumIndices", ctypes.c_uint), ("mIndices", ctypes.POINTER(ctypes.c_uint))]


class aiMesh(ctypes.Structure):
    _fields_ = [
        ("mPrimitiveTypes", ctypes.c_uint),
        ("mNumVertices", ctypes.c_uint),
        ("mNumFaces", ctypes.c_uint),
        ("mVertices", ctypes.POINTER(aiVector3D)),
        ("mNormals", ctypes.POINTER(aiVector3D)),
        ("mTangents", ctypes.POINTER(aiVector3D)),
        ("mBitangents", ctypes.POINTER(aiVector3D)),
        ("mColorss", ctypes.POINTER(aiColor4D) * 8),
        ("mTextureCoords", ctypes.POINTER(aiVector3D) * 8),
        ("mNumUVComponents", ctypes.c_uint * 8),
        ("mFaces", ctypes.POINTER(aiFace)),
        ("mNumBones", ctypes.c_uint),
        ("mBones", ctypes.c_void_p),
        ("mMaterialIndex", ctypes.c_uint),
    ]


class aiNode(ctypes.Structure):
    _fields_ = [
        ("mName", aiString),
        ("mTransformation", aiMatrix4x4),
        ("mParent", ctypes.c_void_p),
        ("mNumChildren", ctypes.c_uint),
        ("mChildren", ctypes.c_void_p),
        ("mNumMeshes", ctypes.c_uint),
        ("mMeshes", ctypes.POINTER(ctypes.c_uint)),
    ]


class aiScene(ctypes.Structure):
    _fields_ = [
        ("mFlags", ctypes.c_uint),
        ("mRootNode", ctypes.c_void_p),
        ("mNumMeshes", ctypes.c_uint),
        ("mMeshes", ctypes.POINTER(ctypes.POINTER(aiMesh))),
        ("mNumMaterials", ctypes.c_uint),
        ("mMaterials", ctypes.c_void_p),
    ]


_dll = None


def _get_dll():
    global _dll
    if _dll is not None:
        return _dll

    candidates = [_DLL_PATH, _ASSIMP_LIB_NAME]

    if sys.platform == "win32":
        ctypes.windll.kernel32.SetErrorMode(0x8007)

    for p in candidates:
        try:
            d = ctypes.CDLL(p)
            if hasattr(d, 'aiImportFile'):
                d.aiImportFile.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
                d.aiImportFile.restype = ctypes.POINTER(aiScene)
                d.aiReleaseImport.argtypes = [ctypes.POINTER(aiScene)]
                if hasattr(d, 'aiGetMaterialString'):
                    d.aiGetMaterialString.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                                      ctypes.c_uint, ctypes.c_uint,
                                                      ctypes.POINTER(aiString)]
                    d.aiGetMaterialString.restype = ctypes.c_uint
                _dll = d
                return d
        except Exception:
            pass

    raise RuntimeError("Assimp library not found")


def _read_material_name(scene, mat_idx):
    try:
        if not scene.mMaterials or mat_idx >= scene.mNumMaterials or mat_idx < 0:
            return ""
        dll = _get_dll()
        if not hasattr(dll, 'aiGetMaterialString'):
            return ""
        mats = ctypes.cast(scene.mMaterials, ctypes.POINTER(ctypes.c_void_p))
        s = aiString()
        dll.aiGetMaterialString(ctypes.c_void_p(mats[mat_idx]), b"?mat.name", 0, 0, ctypes.byref(s))
        return s.data.decode("utf-8", "ignore")[:s.length]
    except Exception:
        return ""


def _collect_meshes(node_ptr, scene, mesh_parts):
    if not node_ptr:
        return
    node = ctypes.cast(node_ptr, ctypes.POINTER(aiNode)).contents
    node_name = ""
    try:
        node_name = node.mName.data.decode("utf-8", "ignore")[:node.mName.length]
    except Exception:
        node_name = ""
    for i in range(node.mNumMeshes):
        mesh_idx = node.mMeshes[i]
        mesh_ptr = scene.mMeshes[mesh_idx]
        if not mesh_ptr:
            continue
        mesh = mesh_ptr.contents
        nv = mesh.mNumVertices
        nf = mesh.mNumFaces
        name = _read_material_name(scene, mesh.mMaterialIndex)
        if not name:
            name = node_name
        if not name:
            try:
                name = mesh.mName.data.decode("utf-8", "ignore")[:mesh.mName.length]
            except Exception:
                name = ""
        if nv == 0 or not mesh.mVertices:
            mesh_parts.append((np.zeros(nv * 3, dtype=np.float32),
                               np.full(nv * 3, 1.0, dtype=np.float32),
                               np.zeros(nv * 2, dtype=np.float32),
                               np.array([], dtype=np.uint32), nv, name))
            continue
        verts_ptr = ctypes.cast(mesh.mVertices, ctypes.POINTER(aiVector3D * nv)).contents
        verts = np.frombuffer(verts_ptr, dtype=np.float32).copy()
        if mesh.mNormals:
            norms_ptr = ctypes.cast(mesh.mNormals, ctypes.POINTER(aiVector3D * nv)).contents
            norms = np.frombuffer(norms_ptr, dtype=np.float32).copy()
        else:
            norms = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), nv)
        tc = mesh.mTextureCoords[0]
        if tc and mesh.mNumUVComponents[0] >= 2:
            uvs_ptr = ctypes.cast(tc, ctypes.POINTER(aiVector3D * nv)).contents
            uvs_raw = np.frombuffer(uvs_ptr, dtype=np.float32)
            uvs = uvs_raw.reshape(-1, 3)[:, :2].copy().flatten()
        else:
            uvs = np.zeros(nv * 2, dtype=np.float32)
        all_idxs = []
        if mesh.mFaces and nf > 0:
            faces_arr = ctypes.cast(mesh.mFaces, ctypes.POINTER(aiFace * nf)).contents
            for j in range(nf):
                face = faces_arr[j]
                idx_ptr = ctypes.cast(face.mIndices, ctypes.POINTER(ctypes.c_uint * face.mNumIndices)).contents
                for k in range(face.mNumIndices):
                    all_idxs.append(idx_ptr[k])
        indices = np.array(all_idxs, dtype=np.uint32)
        mesh_parts.append((verts, norms, uvs, indices, nv, name))
    children_ptr = ctypes.cast(node.mChildren, ctypes.POINTER(ctypes.c_void_p * node.mNumChildren))
    for i in range(node.mNumChildren):
        child_addr = children_ptr.contents[i]
        if child_addr:
            _collect_meshes(child_addr, scene, mesh_parts)


def _read_mesh_import(path: str) -> dict:
    import_path = path + ".import"
    settings = {
        "scale": 1.0,
        "center_pivot": False,
        "flip_uvs": True,
        "smooth_angle": 30.0,
        "gen_normals": True,
        "gen_uvs": True,
    }
    if os.path.exists(import_path):
        try:
            with open(import_path) as _f:
                _data = json.load(_f)
            for _k in settings:
                if _k in _data:
                    settings[_k] = _data[_k]
        except Exception:
            pass
    return settings


def _compute_smooth_normals(verts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    n = len(verts)
    normals = np.zeros((n, 3), dtype=np.float32)
    if len(indices) == 0:
        return normals
    tri = verts[indices].reshape(-1, 3, 3)
    f0 = tri[:, 1] - tri[:, 0]
    f1 = tri[:, 2] - tri[:, 0]
    face_n = np.cross(f0, f1)
    lens = np.linalg.norm(face_n, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    face_n = face_n / lens
    for i in range(0, len(indices), 3):
        for j in range(3):
            normals[indices[i + j]] += face_n[i // 3]
    nl = np.linalg.norm(normals, axis=1, keepdims=True)
    nl[nl == 0] = 1.0
    return (normals / nl).astype(np.float32)


def _generate_planar_uvs(verts: np.ndarray) -> np.ndarray:
    if len(verts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    span = (maxs - mins)
    span[span == 0] = 1.0
    uvs = (verts - mins) / span
    uvs = uvs[:, :2]
    uvs[:, 1] = 1.0 - uvs[:, 1]
    return uvs.astype(np.float32)


def load_mesh(path: str, import_settings: Optional[dict] = None) -> Optional[MeshImportData]:
    eng = None
    try:
        from core.engine.engine import Engine
        eng = Engine.instance()
    except Exception: pass
    prof = eng._profiler if eng and hasattr(eng, '_profiler') else None
    if prof: prof.start("load_mesh")
    
    dll = _get_dll()
    try:
        c_path = ctypes.c_char_p(path.encode('utf-8'))
        AI_TRIANGULATE = 0x10000000
        AI_GEN_NORMALS = 0x2
        _settings = import_settings if import_settings is not None else _read_mesh_import(path)
        flags = AI_TRIANGULATE
        if _settings.get("gen_normals", True):
            flags |= AI_GEN_NORMALS
        scene_ptr = dll.aiImportFile(c_path, ctypes.c_uint32(flags))
        if not scene_ptr:
            if prof: prof.stop("load_mesh")
            return None
        scene = scene_ptr.contents
        mesh_parts = []
        if scene.mRootNode:
            _collect_meshes(scene.mRootNode, scene, mesh_parts)
        dll.aiReleaseImport(scene_ptr)
        data = MeshImportData()
        data.name = os.path.splitext(os.path.basename(path))[0]
        if mesh_parts:
            vert_offset = 0
            all_verts = []
            all_norms = []
            all_uvs = []
            all_idxs = []
            ranges = []
            names = []
            idx_offset = 0
            for verts, norms, uvs, idxs, nv, name in mesh_parts:
                all_verts.append(verts)
                all_norms.append(norms)
                all_uvs.append(uvs)
                if len(idxs) > 0:
                    offset_idxs = idxs + vert_offset
                    ranges.append((idx_offset, len(offset_idxs)))
                    names.append(name)
                    idx_offset += len(offset_idxs)
                    all_idxs.append(offset_idxs)
                vert_offset += nv
            verts_out = np.concatenate(all_verts).reshape(-1, 3)
            norms_out = np.concatenate(all_norms).reshape(-1, 3)

            z_up_to_y_up = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0]
            ], dtype=np.float32)

            verts_out = (verts_out @ z_up_to_y_up.T).ravel()
            norms_out = (norms_out @ z_up_to_y_up.T).ravel()

            data.vertices = verts_out
            data.normals = norms_out
            data.uvs = np.concatenate(all_uvs)
            if all_idxs:
                data.indices = np.concatenate(all_idxs)
                data.sub_mesh_ranges = ranges
                data.sub_mesh_names = names
        if prof: prof.stop("load_mesh")
        return data
    except Exception:
        if prof: prof.stop("load_mesh")
        return None


class MeshImportData:
    def __init__(self):
        self.name: str = ""
        self.vertices: np.ndarray = np.array([], dtype=np.float32)
        self.normals: np.ndarray = np.array([], dtype=np.float32)
        self.uvs: np.ndarray = np.array([], dtype=np.float32)
        self.indices: np.ndarray = np.array([], dtype=np.uint32)
        self.is_error_mesh: bool = False
        self.sub_mesh_ranges: list[tuple[int, int]] = []
        self.sub_mesh_names: list[str] = []


def load_obj(path: str, import_settings: Optional[dict] = None) -> Optional[MeshImportData]:
    eng = None
    try:
        from core.engine.engine import Engine
        eng = Engine.instance()
    except Exception: pass
    prof = eng._profiler if eng and hasattr(eng, '_profiler') else None
    if prof: prof.start("load_obj")
    
    positions, texcoords, normals = [], [], []
    face_pos, face_tex, face_nrm = [], [], []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("o ") or line.startswith("s ") or line.startswith("g "):
                    continue
                if line.startswith("usemtl ") or line.startswith("mtllib "):
                    continue
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "v":
                    positions.extend([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == "vt":
                    texcoords.extend([float(parts[1]), float(parts[2])])
                elif parts[0] == "vn":
                    normals.extend([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == "f":
                    for token in parts[1:]:
                        v = token.split("/")
                        vi = int(v[0]) - 1
                        face_pos.append(vi)
                        if len(v) > 1 and v[1]:
                            face_tex.append(int(v[1]) - 1)
                        if len(v) > 2 and v[2]:
                            face_nrm.append(int(v[2]) - 1)
    except Exception:
        if prof: prof.stop("load_obj")
        return None
    if not face_pos:
        if prof: prof.stop("load_obj")
        return None
    has_uv = len(face_tex) == len(face_pos) and len(texcoords) > 0
    has_nrm = len(face_nrm) == len(face_pos) and len(normals) > 0
    pos_arr = np.array(positions, dtype=np.float32)
    n_faces = len(face_pos)
    verts = np.empty(n_faces * 3, dtype=np.float32)
    norms_out = np.empty(n_faces * 3, dtype=np.float32)
    uvs_out = np.empty(n_faces * 2, dtype=np.float32)
    idx = np.empty(n_faces, dtype=np.uint32)
    seen: dict[tuple, int] = {}
    out_idx = 0
    normals_arr = np.array(normals, dtype=np.float32)
    texcoords_arr = np.array(texcoords, dtype=np.float32)
    for i in range(n_faces):
        pi = face_pos[i]
        ni = face_nrm[i] if has_nrm else 0
        ti = face_tex[i] if has_uv else 0
        key = (pi, ni, ti)
        if key not in seen:
            seen[key] = out_idx
            pi3 = pi * 3
            verts[out_idx * 3:out_idx * 3 + 3] = pos_arr[pi3:pi3 + 3]
            if has_nrm:
                ni3 = ni * 3
                norms_out[out_idx * 3:out_idx * 3 + 3] = normals_arr[ni3:ni3 + 3]
            else:
                norms_out[out_idx * 3:out_idx * 3 + 3] = [0.0, 0.0, 0.0]
            if has_uv:
                ti2 = ti * 2
                uvs_out[out_idx * 2:out_idx * 2 + 2] = texcoords_arr[ti2:ti2 + 2]
            else:
                uvs_out[out_idx * 2:out_idx * 2 + 2] = [0.0, 0.0]
            out_idx += 1
        idx[i] = seen[key]
    data = MeshImportData()
    data.name = os.path.splitext(os.path.basename(path))[0]
    if out_idx > 0:
        data.vertices = verts[:out_idx * 3].copy()
        data.normals = norms_out[:out_idx * 3].copy()
        data.uvs = uvs_out[:out_idx * 2].copy()
    data.indices = idx
    if out_idx > 0:
        _v = data.vertices.reshape(-1, 3)
        _settings = import_settings if import_settings is not None else _read_mesh_import(path)
        if (not has_nrm or len(data.normals) == 0) and _settings.get("gen_normals", True):
            data.normals = _compute_smooth_normals(_v, data.indices).ravel()
        if (not has_uv or len(data.uvs) == 0) and _settings.get("gen_uvs", True):
            data.uvs = _generate_planar_uvs(_v).ravel()
    if prof: prof.stop("load_obj")
    return data


def load_mesh_async(path: str, callback: Callable[[Optional[MeshImportData]], None]) -> None:
    def _task():
        result = load_mesh(path)
        callback(result)
    _get_asset_pool().submit(_task)


def load_obj_async(path: str, callback: Callable[[Optional[MeshImportData]], None]) -> None:
    def _task():
        result = load_obj(path)
        callback(result)
    _get_asset_pool().submit(_task)


def load_gif_frames(path: str) -> list[np.ndarray]:
    from PIL import Image
    gif = Image.open(path)
    frames = []
    try:
        while True:
            frame = gif.copy().convert("RGBA")
            frames.append(np.array(frame, dtype=np.uint8))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    if not frames:
        raise ValueError(f"No frames found in GIF: {path}")
    return frames


def gif_frames_to_flipbook(frames: list[np.ndarray], cols: int = None, rows: int = None) -> tuple[np.ndarray, int, int]:
    n = len(frames)
    if cols is None and rows is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    elif cols is None:
        cols = int(np.ceil(n / rows))
    elif rows is None:
        rows = int(np.ceil(n / cols))
    total = cols * rows
    fh, fw = frames[0].shape[:2]
    sheet = np.zeros((fh * rows, fw * cols, 4), dtype=np.uint8)
    for i in range(total):
        cx = (i % cols) * fw
        cy = (i // cols) * fh
        src = frames[i] if i < n else frames[-1]
        sheet[cy:cy + fh, cx:cx + fw] = src
    return sheet, cols, rows


def import_gif_to_flipbook(gif_path: str, output_path: str = None, cols: int = None, rows: int = None) -> tuple[int, int, int]:
    frames = load_gif_frames(gif_path)
    sheet, cols_out, rows_out = gif_frames_to_flipbook(frames, cols, rows)
    if output_path is None:
        base = os.path.splitext(gif_path)[0]
        output_path = base + "_flipbook.png"
    from PIL import Image
    Image.fromarray(sheet).save(output_path)
    return cols_out, rows_out, len(frames)


def import_gif_to_flipbook_async(gif_path: str, callback: Callable[[Optional[tuple[int, int, int]]], None] = None,
                                  output_path: str = None, cols: int = None, rows: int = None) -> Future:
    def _task():
        try:
            result = import_gif_to_flipbook(gif_path, output_path, cols, rows)
            if callback:
                callback(result)
            return result
        except Exception as e:
            if callback:
                callback(None)
            return None
    return _get_asset_pool().submit(_task)


def load_mesh_future(path: str) -> Future:
    return _get_asset_pool().submit(load_mesh, path)


def load_obj_future(path: str) -> Future:
    return _get_asset_pool().submit(load_obj, path)
