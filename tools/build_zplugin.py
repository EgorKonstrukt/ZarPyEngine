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


def build_zplugin(plugin_src: str, output_dir: str = "dist", mode: str = "source") -> str | None:
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
    entry = str(meta.get("entry", f"{module}/__init__.py"))
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

        arch = "any"
        if mode in ("cython", "nuitka"):
            compiler = _compile_tree_cython if mode == "cython" else _compile_tree_nuitka
            if compiler(stage, module, pkg_stage):
                arch = current_architecture()
            else:
                print(f"WARNING: {mode} compile failed, falling back to source mode.")

        manifest = {
            "name": name,
            "version": version,
            "description": meta.get("description", ""),
            "architecture": arch,
            "entry": entry,
            "module": module,
            "dependencies": meta.get("dependencies", []),
        }
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


def main():
    parser = argparse.ArgumentParser(description="Build a .zplugin distributable archive")
    parser.add_argument("plugin", help="Path to plugin package directory")
    parser.add_argument("--output", "-o", default="dist", help="Output directory")
    parser.add_argument("--mode", choices=["source", "cython", "nuitka"], default="source",
                        help="Build mode (default: source)")
    args = parser.parse_args()
    try:
        result = build_zplugin(args.plugin, args.output, args.mode)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
