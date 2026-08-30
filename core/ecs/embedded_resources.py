# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import base64
import hashlib
import json
import os
import tempfile
import zlib
from typing import Callable, Optional

_CACHE_SUBDIR = os.path.join("Library", "Embedded")

_PATH_SUFFIX = "_path"
_MATERIAL_EXTS = (".mat", ".zpem")


def _compress(raw: bytes, level: int) -> bytes:
    if level <= 0 or len(raw) < 64:
        return raw
    comp = zlib.compress(raw, level)
    if len(comp) < len(raw):
        return comp
    return raw


def _decompress(data: bytes) -> bytes:
    if not data:
        return b""
    try:
        return zlib.decompress(data)
    except zlib.error:
        return data


def _is_material_file(abs_path: str) -> bool:
    return abs_path.lower().endswith(_MATERIAL_EXTS)


def _norm(p: str) -> str:
    return p.replace("\\", "/")


def _sanitize_name(name: str) -> str:
    name = "".join(c for c in name if c not in ':"/\\*?<>|') or "resource.bin"
    return name


def _key_candidates(val: str, root: str) -> set:
    v = _norm(val).lstrip("/")
    candidates = {v}
    if root:
        root_n = _norm(os.path.normpath(root)).rstrip("/") + "/"
        if v.lower().startswith(root_n.lower()):
            candidates.add(v[len(root_n):])
        if not os.path.isabs(val):
            for cand in (os.path.join(root, val), os.path.join(root, "assets", val)):
                candidates.add(_storage_key(_norm(os.path.normpath(cand)), root))
    absv = _abs_path(val, root)
    if absv:
        candidates.add(_storage_key(absv, root))
    return candidates


def _abs_path(val: str, root: str) -> Optional[str]:
    if not val:
        return None
    if os.path.isabs(val):
        if os.path.exists(val):
            return _norm(os.path.normpath(val))
        return None
    for cand in (os.path.join(root, val), os.path.join(root, "assets", val)):
        if os.path.exists(cand):
            return _norm(os.path.normpath(cand))
    return None


def _storage_key(abs_path: str, root: str) -> str:
    try:
        rel = os.path.relpath(abs_path, root)
        if not os.path.isabs(rel):
            return _norm(rel)
    except ValueError:
        pass
    return _norm(abs_path)


def _cache_dir(root: str, mode: str) -> str:
    if mode != "temp" and root:
        base = os.path.join(root, _CACHE_SUBDIR)
        try:
            os.makedirs(base, exist_ok=True)
            return base
        except OSError:
            pass
    base = os.path.join(tempfile.gettempdir(), "ZarinEngine", "Embedded")
    os.makedirs(base, exist_ok=True)
    return base


def _entry_cache_basename(entry: dict, storage: Optional[dict] = None) -> str:
    entry = _resolve_alias(entry, storage)
    digest = entry.get("digest")
    name = _sanitize_name(str(entry.get("name") or "resource.bin"))
    if not digest:
        raw = _entry_raw_bytes(entry)
        digest = hashlib.sha1(raw).hexdigest()[:16]
    return f"{digest}_{name}"


def _resolve_alias(entry: dict, storage: Optional[dict] = None) -> dict:
    if storage is None:
        return entry
    seen = 0
    while isinstance(entry.get("alias"), str):
        nxt = storage.get(entry["alias"])
        if nxt is None:
            break
        entry = nxt
        seen += 1
        if seen > 32:
            break
    return entry


def _entry_raw_bytes(entry: dict) -> bytes:
    data = base64.b64decode(entry["data"])
    if entry.get("compression"):
        return _decompress(data)
    return data


def _cache_path_for(storage_key: str, storage: dict, cache_dir: str) -> str:
    entry = storage.get(storage_key)
    if not entry:
        return ""
    entry = _resolve_alias(entry, storage)
    raw = _entry_raw_bytes(entry)
    digest = entry.get("digest")
    if not digest:
        digest = hashlib.sha1(raw).hexdigest()[:16]
    name = _sanitize_name(str(entry.get("name") or os.path.basename(storage_key)))
    path = os.path.join(cache_dir, f"{digest}_{name}")
    if not os.path.exists(path):
        try:
            with open(path, "wb") as f:
                f.write(raw)
        except OSError:
            return ""
    return _norm(os.path.abspath(path))


def _material_textures(mat_abs: str, root: str) -> dict:
    try:
        with open(mat_abs, "r", encoding="utf-8") as f:
            mat = json.load(f)
    except Exception:
        return {}
    out: dict = {}
    textures = mat.get("textures")
    if isinstance(textures, dict):
        for slot, tex in textures.items():
            if isinstance(tex, str) and tex:
                out[slot] = tex
    elif isinstance(textures, list):
        for i, tex in enumerate(textures):
            if isinstance(tex, str) and tex:
                out[f"_{i}"] = tex
    return out


def _iter_component_path_entries(data: dict):
    entities = data.get("entities", {})
    for ed in entities.values():
        comps = ed.get("components", [])
        if not comps:
            continue
        for comp in comps:
            for key, val in comp.items():
                if key.endswith(_PATH_SUFFIX) and isinstance(val, str) and val:
                    yield ed, comp, (key,), val
            materials = comp.get("materials")
            if isinstance(materials, list):
                for i, entry in enumerate(materials):
                    if isinstance(entry, dict) and isinstance(entry.get("path"), str) and entry["path"]:
                        yield ed, comp, ("materials", i, "path"), entry["path"]
            elif isinstance(materials, dict):
                for i, entry in enumerate(materials.values()):
                    if isinstance(entry, dict) and isinstance(entry.get("path"), str) and entry["path"]:
                        yield ed, comp, ("materials", i, "path"), entry["path"]


def _set_nested(comp: dict, path: tuple, value: str):
    obj = comp
    for p in path[:-1]:
        obj = obj[p]
    obj[path[-1]] = value


def _flagged_entity_ids(data: dict, entities: dict) -> set:
    if data.get("embed_all"):
        return set(entities.keys())
    flagged = {eid for eid, ed in entities.items() if ed.get("embed_resources")}
    if not flagged:
        return set()
    children: dict[str, list[str]] = {}
    for eid, ed in entities.items():
        pid = ed.get("parent")
        if pid:
            children.setdefault(pid, []).append(eid)
    stack = list(flagged)
    result = set(flagged)
    while stack:
        eid = stack.pop()
        for cid in children.get(eid, []):
            if cid not in result:
                result.add(cid)
                stack.append(cid)
    return result


def embed_scene_resources(data: dict, root: str, existing: Optional[dict] = None,
                          compress_level: int = 0, progress_cb: Optional[Callable] = None) -> int:
    entities = data.get("entities", {})
    flagged = _flagged_entity_ids(data, entities)
    storage: dict = {}
    if isinstance(existing, dict):
        storage.update(existing)
    else:
        old = data.get("embedded_resources")
        if isinstance(old, dict):
            storage.update(old)
    if not flagged:
        data.pop("embedded_resources", None)
        if progress_cb:
            progress_cb(0, 0, "")
        return 0
    use_compression = bool(data.get("compress_resources") and compress_level > 0)
    has_compressed = any(bool(e.get("compression")) for e in storage.values())
    has_raw_large = any(not e.get("compression") and int(e.get("size", 0)) >= 64 for e in storage.values())
    force_reencode = (use_compression and has_raw_large) or (not use_compression and has_compressed)
    fields = [val for eid in flagged
              for _, _, _, val in _iter_component_path_entries({"entities": {eid: entities[eid]}})]
    total = len(fields)
    done = 0
    progress_cb = progress_cb or (lambda *_: None)
    basename_map: dict[str, str] = {}
    digest_map: dict[str, str] = {}
    for k, entry in storage.items():
        basename_map[_entry_cache_basename(entry, storage)] = k
        dg = entry.get("digest")
        if dg and isinstance(dg, str):
            digest_map.setdefault(dg, k)

    def embed_file(val: str) -> str:
        bname = _sanitize_name(os.path.basename(_norm(val).rstrip("/")))
        if bname and bname in basename_map:
            return basename_map[bname]
        absv = _abs_path(val, root)
        storage_key = _storage_key(absv, root) if absv else None
        if storage_key is None:
            storage_key = next((k for k in _key_candidates(val, root) if k in storage), None)
        if storage_key is None:
            return ""
        if absv:
            try:
                with open(absv, "rb") as f:
                    raw = f.read()
            except OSError:
                return ""
            digest = hashlib.sha1(raw).hexdigest()[:16]
            if not force_reencode and digest in digest_map:
                alias_key = digest_map[digest]
                if alias_key != storage_key:
                    storage[storage_key] = {"key": storage_key, "name": os.path.basename(absv), "alias": alias_key}
                return alias_key
            existing_entry = storage.get(storage_key)
            if existing_entry and existing_entry.get("digest") == digest and not force_reencode:
                return storage_key
            payload = _compress(raw, compress_level)
            entry = {
                "key": storage_key,
                "name": os.path.basename(absv),
                "size": len(raw),
                "digest": digest,
                "data": base64.b64encode(payload).decode("ascii"),
            }
            if len(payload) < len(raw):
                entry["compression"] = "zlib"
                entry["csize"] = len(payload)
            elif not use_compression:
                entry.pop("compression", None)
                entry.pop("csize", None)
            storage[storage_key] = entry
            digest_map[digest] = storage_key
            basename_map[_entry_cache_basename(entry, storage)] = storage_key
            if _is_material_file(absv):
                for tex in _material_textures(absv, root).values():
                    embed_file(tex)
        return storage_key

    for val in fields:
        embed_file(val)
        done += 1
        progress_cb(done, total, os.path.basename(str(val)))
    if not storage:
        data.pop("embedded_resources", None)
        return 0
    data["embedded_resources"] = storage
    return len(storage)


def extract_embedded_resources(data: dict, root: str, cache_mode: str = "project",
                               progress_cb: Optional[Callable] = None) -> dict:
    storage = data.get("embedded_resources")
    if not isinstance(storage, dict) or not storage:
        data.pop("embedded_resources", None)
        if progress_cb:
            progress_cb(0, 0, "")
        return {}
    cache_dir = _cache_dir(root, cache_mode)
    entities = data.get("entities", {})
    fields = list(_iter_component_path_entries(data))
    total = len(fields)
    done = 0
    progress_cb = progress_cb or (lambda *_: None)
    for ed, comp, path, val in fields:
        if ed.get("id") not in _flagged_entity_ids(data, entities):
            done += 1
            continue
        storage_key = next((k for k in _key_candidates(val, root) if k in storage), None)
        if storage_key is None:
            bname = _sanitize_name(os.path.basename(_norm(val).rstrip("/")))
            storage_key = next((k for k, e in storage.items() if _entry_cache_basename(e, storage) == bname), None)
        if storage_key is not None:
            cache_path = _cache_path_for(storage_key, storage, cache_dir)
            if cache_path:
                _set_nested(comp, path, cache_path)
                if _is_material_file(storage_key):
                    _rewrite_material_textures(cache_path, storage, root, cache_dir)
        done += 1
        progress_cb(done, total, os.path.basename(str(val)))
    data.pop("embedded_resources", None)
    return storage


def _rewrite_material_textures(mat_cache_path: str, storage: dict, root: str, cache_dir: str):
    try:
        with open(mat_cache_path, "r", encoding="utf-8") as f:
            mat = json.load(f)
    except Exception:
        return
    textures = mat.get("textures")
    if not isinstance(textures, dict) or not textures:
        return
    changed = False
    for slot, tex in textures.items():
        if not isinstance(tex, str) or not tex:
            continue
        storage_key = next((k for k in _key_candidates(tex, root) if k in storage), None)
        if storage_key is None:
            continue
        tex_cache = _cache_path_for(storage_key, storage, cache_dir)
        if tex_cache:
            textures[slot] = tex_cache
            changed = True
    if changed:
        try:
            with open(mat_cache_path, "w", encoding="utf-8") as f:
                json.dump(mat, f, indent=2)
        except OSError:
            pass