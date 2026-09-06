# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

"""
Build a .zplugin distributable archive (zip + manifest.json at root).

The archive payload keeps the plugin importable as a package: the loader
adds the payload root to sys.path and imports manifest["module"], so the
package must use relative imports internally.

Usage:
    python tools/build_zplugin.py plugins/tracker_music_plugin
    python tools/build_zplugin.py plugins/tracker_music_plugin -o dist --mode source
    python tools/build_zplugin.py plugins/tracker_music_plugin --mode cython
    python tools/build_zplugin.py plugins/tracker_music_plugin --mode nuitka

Modes:
    source  - ship .py sources, manifest architecture "any" (default).
    cython  - compile package modules to extension modules in place
              (entry __init__.py files stay source so the loader can
              import the package), manifest architecture = current.
    nuitka  - same idea via Nuitka --module, architecture = current.

Compiled files sit next to their .py sources; CPython prefers the
extension module when ABI tags match and falls back to source otherwise.
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.foundation.plugin_manager import current_architecture
except Exception:
    import platform

    def current_architecture() -> str:
        sys_name = {"win32": "win", "darwin": "macos"}.get(sys.platform, sys.platform)
        machine = {"amd64": "x86_64", "x86_64": "x86_64", "x64": "x86_64",
                   "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine().lower(), platform.machine().lower())
        return f"{sys_name}_{machine}"

try:
    from core.foundation.ed25519 import (
        generate_hex as _ed_gen,
    )
    from core.foundation.ed25519 import (
        publickey_hex as _ed_pub,
    )
    from core.foundation.ed25519 import (
        sign_hex as _ed_sign,
    )
    from core.foundation.plugin_manager import (
        _canonical_manifest_bytes as _pm_canonical,
    )
    from core.foundation.plugin_manager import (
        _split_requirement as _pm_split_requirement,
    )
    from core.foundation.plugin_manager import (
        _split_spec as _pm_split_spec,
    )
    _PM_OK = True
except Exception:
    _PM_OK = False


DEFAULT_EXCLUDES = ["__pycache__", "*.pyc", "*.pyo", ".pytest_cache", "*.egg-info"]


def load_source_manifest(plugin_dir: str) -> dict:
    path = os.path.join(plugin_dir, "manifest.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No manifest.json in '{plugin_dir}'")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data.get("name"):
        raise ValueError(f"Invalid manifest.json in '{plugin_dir}' (need at least a 'name')")
    return data


def _excluded(rel_path: str, patterns: list[str]) -> bool:
    parts = rel_path.replace(os.sep, "/").split("/")
    for pat in patterns:
        if any(fnmatch.fnmatchcase(p, pat) for p in parts):
            return True
        if fnmatch.fnmatchcase(rel_path.replace(os.sep, "/"), pat):
            return True
    return False


def _iter_package_modules(pkg_dir: str) -> list[str]:
    mods = []
    for root, _dirs, files in os.walk(pkg_dir):
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, pkg_dir)
            if fn == "__init__.py":
                continue
            mods.append(rel)
    return mods


def _compile_tree_cython(stage_root: str, module: str, pkg_dir: str) -> bool:
    try:
        import Cython  # noqa: F401
    except ImportError:
        print("Cython not installed. Run: pip install cython")
        return False
    rels = _iter_package_modules(pkg_dir)
    if not rels:
        return True
    ext_lines = []
    for rel in rels:
        dotted = module + "." + rel[:-3].replace(os.sep, ".")
        src = os.path.join(module, rel).replace(os.sep, "/")
        ext_lines.append(f'Extension("{dotted}", [r"{src}"])')
    setup_path = os.path.join(stage_root, "_zpl_cython_setup.py")
    with open(setup_path, "w", encoding="utf-8") as f:
        f.write(
            "from setuptools import setup, Extension\n"
            "from Cython.Build import cythonize\n"
            "setup(ext_modules=cythonize([\n" + ",\n".join(ext_lines) + "\n]))\n"
        )
    try:
        result = subprocess.run(
            [sys.executable, setup_path, "build_ext", "--inplace"],
            capture_output=True, text=True, cwd=stage_root,
        )
    finally:
        if os.path.isfile(setup_path):
            os.remove(setup_path)
    if result.returncode != 0:
        print(f"Cython build failed:\n{result.stderr[-2000:]}")
        return False
    ok = True
    for rel in rels:
        base = os.path.splitext(os.path.basename(rel))[0]
        d = os.path.join(pkg_dir, os.path.dirname(rel))
        found = [f for f in os.listdir(d) if f.startswith(base + ".") and f.endswith((".pyd", ".so"))]
        if found:
            print(f"  cython: {rel} -> {found[0]}")
        else:
            print(f"  cython: {rel} produced no extension module")
            ok = False
    for root, dirs, _files in os.walk(stage_root):
        for d in [x for x in dirs if x == "build"]:
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)
    return ok


def _compile_tree_nuitka(stage_root: str, module: str, pkg_dir: str) -> bool:
    try:
        import nuitka  # noqa: F401
    except ImportError:
        print("Nuitka not installed. Run: pip install nuitka")
        return False
    rels = _iter_package_modules(pkg_dir)
    if not rels:
        return True
    ok = True
    for rel in rels:
        src = os.path.join(pkg_dir, rel)
        outdir = os.path.dirname(src)
        result = subprocess.run(
            [sys.executable, "-m", "nuitka", "--module", f"--output-dir={outdir}", src],
            capture_output=True, text=True, cwd=stage_root,
        )
        base = os.path.splitext(os.path.basename(rel))[0]
        found = [f for f in os.listdir(outdir) if f.startswith(base + ".") and f.endswith((".pyd", ".so"))]
        for d in os.listdir(outdir):
            if d.startswith(base + ".build") or d.startswith(base + ".dist"):
                shutil.rmtree(os.path.join(outdir, d), ignore_errors=True)
        if result.returncode == 0 and found:
            print(f"  nuitka: {rel} -> {found[0]}")
        else:
            print(f"  nuitka failed for {rel}:\n{(result.stderr or '')[-2000:]}")
            ok = False
    return ok


def lint_manifest(plugin_src: str) -> list:
    try:
        meta = load_source_manifest(plugin_src)
    except (FileNotFoundError, ValueError) as e:
        return [str(e)]
    errors = []
    name = str(meta.get("name", ""))
    if not name:
        errors.append("manifest needs a 'name'")
    ptype = str(meta.get("type", "plugin"))
    if ptype not in ("plugin", "library"):
        errors.append(f"unknown type '{ptype}' (want 'plugin' or 'library')")
    module = str(meta.get("module", name))
    entry = str(meta.get("entry", f"{module}/__init__.py" if ptype == "plugin" else ""))
    if ptype == "plugin" and entry:
        parent = os.path.dirname(os.path.abspath(plugin_src))
        if not os.path.isfile(os.path.join(parent, entry)) and not os.path.isfile(os.path.join(plugin_src, entry)):
            errors.append(f"entry '{entry}' not found")
    if ptype == "library" and not meta.get("provides") and not os.path.isdir(os.path.join(plugin_src, "libs")):
        errors.append("library pack needs 'provides' or a libs/ directory")
    if _PM_OK:
        for dep in meta.get("dependencies", []) or []:
            if not _pm_split_requirement(dep)[0]:
                errors.append(f"bad dependency '{dep}'")
        for spec in (meta.get("engine_api", ""), meta.get("python_requires", "")):
            if spec and not all(op in ("==", "!=", ">=", "<=", "~=", ">", "<", "=") and want for op, want in _pm_split_spec(spec)):
                errors.append(f"bad version spec '{spec}'")
        for req in meta.get("requires", []) or []:
            if isinstance(req, dict):
                good = bool(req.get("name"))
            else:
                good = bool(_pm_split_requirement(str(req))[0])
            if not good:
                errors.append(f"bad plugin requirement '{req}'")
        sig = meta.get("signature")
        if sig is not None and (not isinstance(sig, dict) or not sig.get("sig")):
            errors.append("malformed 'signature' (want {'by': ..., 'sig': ...})")
    return errors


def _dist_top_levels(dist) -> list:
    try:
        tl = dist.read_text("top_level.txt")
        if tl:
            return [t.strip() for t in tl.split() if t.strip()]
    except Exception:
        pass
    tops = []
    try:
        files = dist.files or []
    except Exception:
        return tops
    for f in files:
        parts = str(f).replace("\\", "/").split("/")
        if not parts or parts[0].startswith(("_", ".")):
            continue
        if len(parts) == 1:
            if parts[0].endswith(".py"):
                tops.append(parts[0][:-3])
        elif not parts[0].endswith((".dist-info", ".egg-info")):
            tops.append(parts[0])
    return sorted(set(tops))


def _find_distribution(top: str):
    from importlib import metadata as _md
    try:
        return _md.distribution(top)
    except Exception:
        pass
    try:
        for dist in _md.distributions():
            if top in _dist_top_levels(dist):
                return dist
    except Exception:
        pass
    return None


def _dist_version(top: str) -> str:
    try:
        dist = _find_distribution(top)
        if dist is not None:
            return dist.version
    except Exception:
        pass
    return "unknown"


def _copy_dist_info(top: str, libs: str):
    try:
        dist = _find_distribution(top)
        src = str(getattr(dist, "_path", "") or "") if dist is not None else ""
        if src and os.path.isdir(src) and src.endswith(".dist-info"):
            dst = os.path.join(libs, os.path.basename(src))
            if os.path.isdir(dst):
                shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
    except Exception as e:
        print(f"  bundle: dist-info for '{top}' skipped: {e}")


def _resolve_bundle_modules(names) -> list:
    out = []
    for raw in names or []:
        top = str(raw).strip().split(".")[0]
        if not top:
            continue
        try:
            found = importlib.util.find_spec(top) is not None
        except Exception:
            found = False
        if found:
            out.append(top)
            continue
        resolved = False
        try:
            from importlib import metadata as _md
            tops = _dist_top_levels(_md.distribution(top))
            if tops:
                out.extend(tops)
                resolved = True
        except Exception:
            pass
        if not resolved:
            out.append(top)
    seen = set()
    ordered = []
    for m in out:
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


def download_wheels(requirements, dest_dir, python_version=None, platform=None,
                    abi=None, implementation="cp") -> list:
    cmd = [sys.executable, "-m", "pip", "download", "--only-binary=:all:",
           "-d", dest_dir]
    if python_version or platform or abi:
        if python_version:
            cmd += ["--python-version", str(python_version)]
        if platform:
            cmd += ["--platform", str(platform)]
        if abi:
            cmd += ["--abi", str(abi)]
        cmd += ["--implementation", str(implementation or "cp")]
    cmd += list(requirements or [])
    if not cmd[-len(list(requirements or [])):]:
        raise ValueError("download_wheels needs at least one requirement")
    os.makedirs(dest_dir, exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = ((result.stderr or "") + "\n" + (result.stdout or "")).strip().splitlines()
        raise RuntimeError("pip download failed:\n" + "\n".join(tail[-15:]))
    wheels = sorted(f for f in os.listdir(dest_dir) if f.endswith(".whl"))
    if not wheels:
        raise RuntimeError("pip download produced no wheels")
    return [os.path.join(dest_dir, w) for w in wheels]


def _wheel_name_version(whl_path):
    base = os.path.basename(whl_path)
    if base.endswith(".whl"):
        base = base[:-4]
    parts = base.split("-")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return base, "unknown"


def _wheel_top_levels(whl_path) -> list:
    tops = []
    try:
        with zipfile.ZipFile(whl_path) as zf:
            for n in zf.namelist():
                if n.endswith("/"):
                    continue
                parts = n.split("/")
                if not parts or parts[0].startswith(("_", ".")):
                    continue
                if parts[0].endswith((".dist-info", ".egg-info", ".data")):
                    continue
                if len(parts) == 1:
                    if parts[0].endswith(".py"):
                        tops.append(parts[0][:-3])
                else:
                    tops.append(parts[0])
    except Exception:
        pass
    return sorted(set(tops))


def vendor_wheels(wheel_paths, libs_dir, exclude=()) -> dict:
    excluded = {_normalize_mod_name(e) for e in (exclude or [])}
    provides = {}
    os.makedirs(libs_dir, exist_ok=True)
    for whl in wheel_paths or []:
        dist_name, dist_version = _wheel_name_version(whl)
        if _normalize_mod_name(dist_name) in excluded:
            print(f"  vendor: {dist_name} excluded")
            continue
        try:
            with zipfile.ZipFile(whl) as zf:
                for member in zf.namelist():
                    if member.endswith("/"):
                        continue
                    parts = member.split("/")
                    if parts[0].endswith(".data"):
                        if len(parts) < 3 or parts[1] not in ("purelib", "platlib"):
                            continue
                        rel = "/".join(parts[2:])
                    else:
                        rel = member
                    if not rel:
                        continue
                    target = os.path.join(libs_dir, rel.replace("/", os.sep))
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with zf.open(member) as srcf, open(target, "wb") as dstf:
                        shutil.copyfileobj(srcf, dstf)
            for top in _wheel_top_levels(whl):
                provides.setdefault(top, dist_version)
            print(f"  vendor: {dist_name} ({dist_version})")
        except Exception as e:
            print(f"  vendor: '{os.path.basename(whl)}' failed: {e}")
    return provides


def _normalize_mod_name(name: str) -> str:
    return str(name or "").lower().replace("-", "_").replace(".", "_")


def _venv_site_packages(venv=None, python_exe=None) -> str:
    exe = python_exe
    if not exe and venv:
        cand = os.path.join(venv, "Scripts", "python.exe")
        if not os.path.isfile(cand):
            cand = os.path.join(venv, "bin", "python")
        exe = cand
    if not exe or not os.path.isfile(exe):
        raise ValueError("venv python not found (pass --venv or --python)")
    result = subprocess.run([exe, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"could not query site-packages: {(result.stderr or '').strip()[-300:]}")
    path = result.stdout.strip().splitlines()[-1]
    if not os.path.isdir(path):
        raise RuntimeError(f"site-packages not found: {path}")
    return path


def _receipts_dir(site_packages: str) -> str:
    d = os.path.join(site_packages, "zplugin_receipts")
    os.makedirs(d, exist_ok=True)
    return d


def _hash_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_zip_member(zf, member: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with zf.open(member) as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def install_pack(archive: str, venv=None, python_exe=None, force: bool = False):
    try:
        import tempfile as _tf
        site_packages = _venv_site_packages(venv, python_exe)
        with zipfile.ZipFile(archive) as zf:
            mname = next((n for n in zf.namelist() if n.endswith("manifest.json")), None)
            if mname is None:
                return (False, "archive has no manifest.json")
            manifest = json.loads(zf.open(mname).read().decode("utf-8"))
            members = [n for n in zf.namelist()
                       if not n.endswith("/") and n != mname and not n.endswith(".zpl_meta.json")]
            if not members:
                return (False, "nothing installable (empty payload)")
            clashes = []
            for member in members:
                rel = member[len("libs/"):] if member.startswith("libs/") else member
                target = os.path.join(site_packages, rel.replace("/", os.sep))
                if os.path.isfile(target):
                    try:
                        if _hash_file(target) != _hash_zip_member(zf, member):
                            clashes.append(rel.replace("/", os.sep))
                    except Exception:
                        clashes.append(rel.replace("/", os.sep))
            if clashes and not force:
                shown = ", ".join(clashes[:5])
                more = f" (+{len(clashes) - 5} more)" if len(clashes) > 5 else ""
                return (False, f"refusing to overwrite {len(clashes)} differing files "
                               f"(use --force): {shown}{more}")
            installed = []
            overwrote = 0
            for member in members:
                rel = member[len("libs/"):] if member.startswith("libs/") else member
                target = os.path.join(site_packages, rel.replace("/", os.sep))
                os.makedirs(os.path.dirname(target), exist_ok=True)
                if os.path.isfile(target):
                    overwrote += 1
                with zf.open(member) as srcf, open(target, "wb") as dstf:
                    shutil.copyfileobj(srcf, dstf)
                installed.append(rel.replace("/", os.sep))
            receipt = {"name": manifest.get("name", ""), "version": manifest.get("version", ""),
                       "archive": os.path.basename(archive), "files": sorted(installed)}
            with open(os.path.join(_receipts_dir(site_packages), receipt["name"] + ".json"),
                      "w", encoding="utf-8") as f:
                json.dump(receipt, f, indent=2)
            msg = f"{len(installed)} files -> {site_packages}"
            if overwrote:
                msg += f" ({overwrote} overwrote existing files)"
            return (True, msg)
    except Exception as e:
        return (False, str(e))


def uninstall_pack(name: str, venv=None, python_exe=None):
    try:
        site_packages = _venv_site_packages(venv, python_exe)
        receipt_path = os.path.join(_receipts_dir(site_packages), name + ".json")
        if not os.path.isfile(receipt_path):
            return (False, f"no install receipt for '{name}'")
        with open(receipt_path, encoding="utf-8") as f:
            receipt = json.load(f)
        removed, missing = 0, 0
        tops = set()
        for rel in receipt.get("files", []):
            target = os.path.join(site_packages, rel)
            tops.add(rel.split(os.sep)[0])
            try:
                if os.path.isfile(target):
                    os.remove(target)
                    removed += 1
                else:
                    missing += 1
            except Exception:
                missing += 1
        for top in sorted(tops):
            tdir = os.path.join(site_packages, top)
            for root, _dirs, _files in os.walk(tdir, topdown=False):
                if os.path.basename(root) == "__pycache__":
                    shutil.rmtree(root, ignore_errors=True)
            for root, dirs, files in os.walk(tdir, topdown=False):
                if not dirs and not files:
                    try:
                        os.rmdir(root)
                    except Exception:
                        pass
            try:
                os.rmdir(tdir)
            except Exception:
                pass
        try:
            os.remove(receipt_path)
        except Exception:
            pass
        return (True, f"removed {removed} files ({missing} already gone)")
    except Exception as e:
        return (False, str(e))


def list_installed_packs(venv=None, python_exe=None) -> list:
    try:
        site_packages = _venv_site_packages(venv, python_exe)
        d = os.path.join(site_packages, "zplugin_receipts")
        out = []
        if os.path.isdir(d):
            for fname in sorted(os.listdir(d)):
                if fname.endswith(".json"):
                    try:
                        with open(os.path.join(d, fname), encoding="utf-8") as f:
                            out.append(json.load(f))
                    except Exception:
                        pass
        return out
    except Exception:
        return []


def _bundle_into_stage(stage: str, modules) -> dict:
    provides = {}
    libs = os.path.join(stage, "libs")
    os.makedirs(libs, exist_ok=True)
    for top in _resolve_bundle_modules(modules):
        try:
            spec = importlib.util.find_spec(top)
        except Exception:
            spec = None
        if spec is None:
            print(f"  bundle: '{top}' not installed, skipped")
            continue
        version = _dist_version(top)
        try:
            locations = list(spec.submodule_search_locations or [])
            if locations:
                dst = os.path.join(libs, top)
                if os.path.isdir(dst):
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(locations[0], dst,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"))
                _copy_dist_info(top, libs)
            elif spec.origin and spec.origin.endswith(".py"):
                shutil.copy2(spec.origin, os.path.join(libs, top + ".py"))
            else:
                print(f"  bundle: '{top}' has no copyable source, skipped")
                continue
            provides[top] = version
            print(f"  bundle: {top} ({version})")
        except Exception as e:
            print(f"  bundle: '{top}' failed: {e}")
    return provides


def _scan_stage_libs(libs: str) -> dict:
    provides = {}
    if not os.path.isdir(libs):
        return provides
    for entry in sorted(os.listdir(libs)):
        if entry.startswith((".", "_")) or entry == "__pycache__" or entry.endswith(".dist-info"):
            continue
        full = os.path.join(libs, entry)
        if entry.endswith(".py") and os.path.isfile(full):
            provides[entry[:-3]] = "unknown"
        elif os.path.isdir(full) and os.path.isfile(os.path.join(full, "__init__.py")):
            provides[entry] = "unknown"
    return provides


def _freeze_stage(stage: str) -> dict:
    import hashlib
    files = {}
    for root, _dirs, fns in os.walk(stage):
        for fn in sorted(fns):
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, stage).replace(os.sep, "/")
            if rel == "manifest.json":
                continue
            h = hashlib.sha256()
            with open(full, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            files[rel] = h.hexdigest()
    return files


def _sign_manifest(manifest: dict, key_hex: str) -> dict:
    pub = _ed_pub(key_hex)
    manifest["signature"] = {"by": pub, "sig": _ed_sign(_pm_canonical(manifest), key_hex)}
    return manifest


ARCH_CHOICES = ["any", "current", "win_x86_64", "win_arm64",
                "linux_x86_64", "linux_arm64", "macos_x86_64", "macos_arm64"]


def list_installed_distributions() -> list:
    from importlib import metadata as _md
    out = []
    seen = set()
    for dist in _md.distributions():
        try:
            name = dist.metadata["Name"] or ""
        except Exception:
            continue
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        tops = []
        try:
            tops = _dist_top_levels(dist)
        except Exception:
            pass
        if not tops:
            tops = [name.replace("-", "_")]
        try:
            version = dist.version
        except Exception:
            version = "unknown"
        out.append({"name": name, "version": version, "modules": tops})
    return sorted(out, key=lambda d: d["name"].lower())


def create_library_pack(name: str, version: str, modules, output_dir: str = "dist",
                        arch: str = "any", python_requires: str = "",
                        engine_api: str = ">=1", description: str = "",
                        sign_key: str = "", install_dir: str | None = None,
                        download: bool = False, download_requires=None,
                        target=None, exclude=("numpy",)) -> str | None:
    if arch == "current":
        arch = current_architecture()
    if download and not download_requires:
        download_requires = [f"{m}=={v}" for m, v in _pin_installed_versions(modules)]
    excluded = set(exclude or ()) if download or download_requires else set()
    src = tempfile.mkdtemp(prefix="libpack_src_")
    try:
        meta = {"name": name, "version": version, "description": description,
                "architecture": arch, "entry": "", "module": name,
                "dependencies": ["numpy>=1"] if "numpy" in excluded else [],
                "requires": [],
                "engine_api": engine_api, "python_requires": python_requires,
                "type": "library", "provides": {},
                "exclude": list(DEFAULT_EXCLUDES)}
        with open(os.path.join(src, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        archive = build_zplugin(src, output_dir, "source", bundle=list(modules or []),
                                sign_key=sign_key, ptype="library", freeze=True,
                                download=download, download_requires=download_requires,
                                target=target, exclude=excluded)
        if archive and install_dir:
            os.makedirs(install_dir, exist_ok=True)
            dest = os.path.join(install_dir, os.path.basename(archive))
            shutil.copy2(archive, dest)
            return dest
        return archive
    finally:
        shutil.rmtree(src, ignore_errors=True)


def _pin_installed_versions(modules) -> list:
    from importlib import metadata as _md
    pinned = []
    for raw in modules or []:
        top = str(raw).strip().split(".")[0]
        if not top:
            continue
        try:
            dist = _find_distribution(top)
            dname = (dist.metadata["Name"] if dist is not None else top) or top
            ver = dist.version if dist is not None else ""
            pinned.append(f"{dname}=={ver}" if ver and ver != "unknown" else dname)
        except Exception:
            pinned.append(top)
    return pinned


def build_zplugin(plugin_src: str, output_dir: str = "dist", mode: str = "source",
                  bundle=(), sign_key: str = "", ptype: str | None = None,
                  freeze: bool = True, download: bool = False,
                  download_requires=None, target=None, exclude=()) -> str | None:
    plugin_src = os.path.abspath(plugin_src)
    if os.path.isfile(plugin_src) and plugin_src.endswith(".py"):
        raise ValueError("build_zplugin needs a plugin package directory, not a single .py file")
    if not os.path.isdir(plugin_src):
        print(f"Error: {plugin_src} not found")
        return None
    meta = load_source_manifest(plugin_src)
    name = str(meta["name"])
    version = str(meta.get("version", "0.0.1"))
    module = str(meta.get("module", name))
    ptype = ptype or str(meta.get("type", "plugin"))
    if ptype not in ("plugin", "library"):
        print(f"Unknown type: {ptype}")
        return None
    entry = str(meta.get("entry", f"{module}/__init__.py" if ptype == "plugin" else ""))
    excludes = list(meta.get("exclude", [])) + DEFAULT_EXCLUDES

    if mode not in ("source", "cython", "nuitka"):
        print(f"Unknown mode: {mode}")
        return None

    stage = tempfile.mkdtemp(prefix="zplugin_build_")
    try:
        pkg_stage = os.path.join(stage, module)
        for root, _dirs, files in os.walk(plugin_src):
            rel_root = os.path.relpath(root, plugin_src)
            for fn in files:
                rel = fn if rel_root == "." else os.path.join(rel_root, fn)
                if _excluded(rel, excludes):
                    continue
                if fn == "manifest.json" and rel_root == ".":
                    continue
                src = os.path.join(root, fn)
                dst = os.path.join(pkg_stage, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

        src_libs = os.path.join(plugin_src, "libs")
        if os.path.isdir(src_libs):
            dst_libs = os.path.join(stage, "libs")
            shutil.copytree(src_libs, dst_libs, ignore=shutil.ignore_patterns(*excludes))

        arch = str(meta.get("architecture", "any") or "any")
        if mode in ("cython", "nuitka"):
            compiler = _compile_tree_cython if mode == "cython" else _compile_tree_nuitka
            if compiler(stage, module, pkg_stage):
                arch = current_architecture()
            else:
                print(f"WARNING: {mode} compile failed, falling back to source mode.")

        bundled = _bundle_into_stage(stage, bundle) if bundle else {}
        dl_conf = dict(meta.get("download", {}) or {}) if download else {}
        reqs = list(download_requires or [])
        if download and not reqs:
            reqs = list(dl_conf.get("requires", []) or [])
        triple = dict(target or {})
        for key in ("python_version", "platform", "abi", "implementation"):
            if key not in triple and dl_conf.get(key):
                triple[key] = dl_conf[key]
        if reqs:
            dl_dir = tempfile.mkdtemp(prefix="zplugin_dl_")
            try:
                wheels = download_wheels(reqs, dl_dir,
                                         python_version=triple.get("python_version"),
                                         platform=triple.get("platform"),
                                         abi=triple.get("abi"),
                                         implementation=triple.get("implementation", "cp"))
                downloaded = vendor_wheels(wheels, os.path.join(stage, "libs"),
                                           exclude=set(exclude or ()) | set(dl_conf.get("exclude", []) or []))
            except RuntimeError as e:
                print(f"Error: {e}")
                return None
            finally:
                shutil.rmtree(dl_dir, ignore_errors=True)
        else:
            downloaded = {}
        provides = dict(meta.get("provides", {}) or {})
        for key, val in list(bundled.items()) + list(downloaded.items()):
            provides.setdefault(key, val)
        if ptype == "library":
            for key, val in _scan_stage_libs(os.path.join(stage, "libs")).items():
                provides.setdefault(key, val)

        manifest = {
            "name": name,
            "version": version,
            "description": meta.get("description", ""),
            "architecture": arch,
            "entry": entry,
            "module": module,
            "dependencies": meta.get("dependencies", []),
            "requires": meta.get("requires", []),
            "engine_api": meta.get("engine_api", ""),
            "python_requires": meta.get("python_requires", ""),
            "type": ptype,
            "provides": provides,
        }
        if freeze or sign_key:
            manifest["files"] = _freeze_stage(stage)
        if sign_key:
            if not _PM_OK:
                print("Error: signing needs core.foundation.ed25519; run from the engine root.")
                return None
            try:
                _sign_manifest(manifest, sign_key)
            except Exception as e:
                print(f"Error: signing failed: {e}")
                return None
        with open(os.path.join(stage, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        outdir = os.path.abspath(output_dir)
        os.makedirs(outdir, exist_ok=True)
        archive = os.path.join(outdir, f"{name}-{version}.zplugin")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(stage):
                for fn in sorted(files):
                    full = os.path.join(root, fn)
                    arc = os.path.relpath(full, stage).replace(os.sep, "/")
                    zf.write(full, arc)
        print(f"OK: {archive} (mode={mode}, arch={arch})")
        return archive
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _main_manage(cmd, argv) -> int:
    parser = argparse.ArgumentParser(description=f"zplugin {cmd}")
    if cmd == "install":
        parser.add_argument("archive", help="Path to .zplugin file")
    elif cmd == "uninstall":
        parser.add_argument("name", help="Installed pack name")
    parser.add_argument("--venv", default="", help="Target venv directory")
    parser.add_argument("--python", default="", help="Target venv python executable")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite files whose content differs (install only)")
    args = parser.parse_args(argv)
    if cmd == "install":
        ok, msg = install_pack(args.archive, venv=args.venv or None,
                               python_exe=args.python or None, force=args.force)
        print(("OK: " if ok else "Error: ") + msg)
        return 0 if ok else 1
    if cmd == "uninstall":
        ok, msg = uninstall_pack(args.name, venv=args.venv or None, python_exe=args.python or None)
        print(("OK: " if ok else "Error: ") + msg)
        return 0 if ok else 1
    for pack in list_installed_packs(venv=args.venv or None, python_exe=args.python or None):
        print(f"{pack.get('name', '?')} {pack.get('version', '')} ({pack.get('archive', '')})")
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("install", "uninstall", "list"):
        sys.exit(_main_manage(sys.argv[1], sys.argv[2:]))
    parser = argparse.ArgumentParser(description="Build a .zplugin distributable archive")
    parser.add_argument("plugin", nargs="?", help="Path to plugin package directory")
    parser.add_argument("--output", "-o", default="dist", help="Output directory")
    parser.add_argument("--mode", choices=["source", "cython", "nuitka"], default="source",
                        help="Build mode (default: source)")
    parser.add_argument("--lint", action="store_true", help="Validate the manifest and exit")
    parser.add_argument("--genkey", action="store_true", help="Generate an ed25519 key pair and exit")
    parser.add_argument("--bundle", default="",
                        help="Comma-separated third-party modules to vendor into libs/")
    parser.add_argument("--sign-key", default="", help="Hex private key for manifest signing")
    parser.add_argument("--type", choices=["plugin", "library"], default=None,
                        help="Override the manifest type")
    parser.add_argument("--no-freeze", action="store_true", help="Skip the file hash index")
    parser.add_argument("--download", action="store_true",
                        help="Download the dependency closure (manifest 'download' section) instead of using installed packages")
    parser.add_argument("--download-requires", default="",
                        help="Comma-separated requirements to download and vendor (implies --download)")
    parser.add_argument("--python-version", default="", help="Target python version for download (e.g. 3.13)")
    parser.add_argument("--platform", default="", help="Target platform for download (e.g. win_amd64)")
    parser.add_argument("--abi", default="", help="Target ABI for download (e.g. cp313)")
    parser.add_argument("--implementation", default="cp", help="Target implementation for download")
    parser.add_argument("--exclude-mod", action="append", default=[],
                        help="Top-level module to exclude from vendored wheels (repeatable)")
    args = parser.parse_args()
    if args.genkey:
        if not _PM_OK:
            print("Error: key generation needs core.foundation.ed25519; run from the engine root.")
            sys.exit(1)
        priv, pub = _ed_gen()
        print(f"private: {priv}")
        print(f"public:  {pub}")
        sys.exit(0)
    if not args.plugin:
        parser.print_usage()
        sys.exit(2)
    if args.lint:
        errors = lint_manifest(args.plugin)
        if errors:
            for e in errors:
                print(f"lint: {e}")
            sys.exit(1)
        print("manifest OK")
        sys.exit(0)
    try:
        dl_reqs = [m for m in args.download_requires.split(",") if m.strip()]
        triple = {k: v for k, v in
                  (("python_version", args.python_version), ("platform", args.platform),
                   ("abi", args.abi), ("implementation", args.implementation)) if v}
        result = build_zplugin(args.plugin, args.output, args.mode,
                               bundle=[m for m in args.bundle.split(",") if m.strip()],
                               sign_key=args.sign_key, ptype=args.type,
                               freeze=not args.no_freeze,
                               download=args.download or bool(dl_reqs),
                               download_requires=dl_reqs or None,
                               target=triple or None,
                               exclude=args.exclude_mod)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
