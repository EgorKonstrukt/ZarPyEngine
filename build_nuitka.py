# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

"""
Zarin Engine вЂ” Nuitka build script.
Uses BuildSettings.json to determine which scenes and assets to include.
"""
import subprocess
import sys
import os
import json
import shutil
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if sys.platform == "win32":
    _ASSIMP_SRC = "assimp-vc143-mt.dll"
else:
    _ASSIMP_SRC = "libassimp.so.6.0.5"

parser = argparse.ArgumentParser(description="Zarin Engine Nuitka build")
parser.add_argument("--editor", action="store_true", help="Build editor (main.py) instead of player (player.py)")
parser.add_argument("--output-dir", default=str(ROOT / "build_output"), help="Output directory (default: build_output)")
parser.add_argument("--no-console", action="store_true", help="Disable console window")
parser.add_argument("--onefile", action="store_true", help="Single file build")
parser.add_argument("--strip-unused", action="store_true", default=None, help="Strip unused assets (scans scenes)")
parser.add_argument("--no-strip-unused", action="store_true", dest="no_strip", help="Include all assets")
parser.add_argument("--no-winrt", action="store_true", help="Disable Windows Runtime DLL inclusion (smaller distributable)")

_args, remaining = parser.parse_known_args()

OUTPUT_DIR = Path(_args.output_dir)
ENTRY = str(ROOT / "main.py" if _args.editor else "player.py")
BUILD_EDITOR = _args.editor
NO_CONSOLE = _args.no_console
ONEFILE = _args.onefile
CLI_STRIP = _args.strip_unused
CLI_NO_STRIP = _args.no_strip
NO_WINRT = _args.no_winrt

print("=== " + ("EDITOR BUILD" if BUILD_EDITOR else "PLAYER BUILD") + " ===")


def _resolve_physics_solver() -> str:
    """Read the active physics solver from ProjectSettings.json."""
    ps_path = ROOT / "ProjectSettings.json"
    if not ps_path.exists():
        return "culverin"
    try:
        with open(ps_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        solver = data.get("physics", {}).get("solver", "culverin")
        return solver if solver in ("pybullet", "physx", "culverin") else "culverin"
    except Exception:
        return "culverin"


def _minify_pil():
    """Temporarily reduce PIL._plugins to exclude heavy C extensions."""
    import PIL
    pil_dir = Path(PIL.__file__).parent
    src = pil_dir / "__init__.py"
    bak = pil_dir / "__init__.py.__bak__"
    if bak.exists():
        return
    original = src.read_text(encoding="utf-8")
    bak.write_text(original, encoding="utf-8")
    # Keep only plugins loaded by preinit() вЂ” drops _avif, _webp, _imagingcms, etc.
    kept = {
        "BmpImagePlugin", "GifImagePlugin", "JpegImagePlugin",
        "PpmImagePlugin", "PngImagePlugin",
    }
    new_plugins = [f'    "{p}",' for p in sorted(kept)]
    import re
    modified = re.sub(
        r'_plugins\s*=\s*\[.*?^\]',
        '_plugins = [\n' + '\n'.join(new_plugins) + '\n]',
        original, flags=re.DOTALL | re.MULTILINE
    )
    src.write_text(modified, encoding="utf-8")
    print(f"  PIL minified: {len(kept)} plugins kept (was {original.count('ImagePlugin')})")


def _restore_pil():
    import PIL
    pil_dir = Path(PIL.__file__).parent
    bak = pil_dir / "__init__.py.__bak__"
    if bak.exists():
        (pil_dir / "__init__.py").write_text(bak.read_text(encoding="utf-8"), encoding="utf-8")
        bak.unlink()
        print("  PIL restored")


def _load_build_settings() -> dict:
    """Load BuildSettings.json from project root."""
    bs_path = ROOT / "BuildSettings.json"
    if bs_path.exists():
        with open(bs_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _get_included_scenes(bs: dict) -> list[str]:
    """Get list of scene files to include from BuildSettings."""
    scenes = bs.get("scenes", [])
    scenes_dir = ROOT / "scenes"
    included = []
    for s in scenes:
        if os.path.isabs(s):
            full = Path(s)
        else:
            # Strip leading "scenes/" or "scenes\\" prefix
            s_stripped = s
            for prefix in ("scenes/", "scenes\\"):
                if s_stripped.startswith(prefix):
                    s_stripped = s_stripped[len(prefix):]
                    break
            full = scenes_dir / s_stripped
        if full.exists():
            included.append(str(full))
        else:
            print(f"  WARNING: scene not found: {full}")
    return included


def _scan_scene_assets(scene_path: str) -> set[str]:
    """Scan a scene file for referenced assets (textures, materials, meshes)."""
    PATH_FIELDS = {"mesh_path", "material_path", "clip_path", "script_path", "texture_path"}
    assets = set()
    try:
        with open(scene_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entities = data.get("entities", {})
        for eid, edata in entities.items():
            for comp in edata.get("components", []):
                for key, val in comp.items():
                    if key in PATH_FIELDS and isinstance(val, str) and val:
                        assets.add(val)
    except Exception as e:
        print(f"  WARNING: failed to scan {scene_path}: {e}")
    return assets


def _resolve_assets(assets: set[str], project_root: Path) -> set[Path]:
    """Resolve asset paths to absolute paths."""
    resolved = set()
    assets_dir = project_root / "assets"
    for a in assets:
        if os.path.isabs(a):
            p = Path(a)
            if p.exists():
                resolved.add(p)
        else:
            candidates = [
                project_root / a,
                assets_dir / a,
            ]
            for c in candidates:
                if c.exists():
                    resolved.add(c)
                    break
    return resolved


def _collect_asset_dirs(assets: set[Path], project_root: Path) -> dict[str, str]:
    """
    Group assets by directory and return as include-data-dir mappings.
    Returns {source_dir: dest_name} for Nuitka.
    """
    dirs: dict[str, set[str]] = {}
    assets_dir = project_root / "assets"
    for a in assets:
        try:
            rel = a.relative_to(assets_dir)
            parent = rel.parent
            dirs.setdefault(str(assets_dir / parent), set()).add(str(a.name))
        except ValueError:
            # Asset is outside assets/, include it directly
            dirs.setdefault(str(a.parent), set()).add(str(a.name))
    return dirs


def build():
    print("=== Zarin Engine Nuitka Build ===")
    print(f"Python: {sys.executable}")
    print(f"Root: {ROOT}")
    print()

    # Uninstall heavy optional packages before build
    # Nuitka's --exclude-module doesn't work when packages are installed
    # because --include-package follows transitive imports
    PACKAGES_TO_REMOVE = [
        "numba", "llvmlite", "llvmlite-bindings", "llvmlite-environment",

        "pygments", "pydantic", "pydantic-core", "pydantic-settings",
        "mcp", "httpx", "starlette", "cryptography",
        "pydub", "zstandard", "psutil", "anyio",
        "jinja2", "markupsafe",
        "pynput", "openal", "pyopenal", "pyogg",
    ]

    # Save which packages were actually uninstalled for reinstall
    removed = []
    for pkg in PACKAGES_TO_REMOVE:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", pkg, "-y"],
                capture_output=True, text=True
            )
            if result.returncode == 0 and "Successfully uninstalled" in result.stdout:
                removed.append(pkg)
                print(f"  Removed: {pkg}")
        except Exception:
            pass

    if removed:
        print(f"Uninstalled {len(removed)} packages for clean build\n")
    else:
        print("No optional packages to remove\n")

    # Verify numba is actually gone вЂ” abort if not
    try:
        import importlib
        importlib.import_module("numba")
        print("\nFATAL: numba is STILL installed after pip uninstall!")
        print("It will be compiled by Nuitka and bloat the build.")
        print("Try running: pip uninstall numba llvmlite -y")
        print("Then run this script again.\n")
        sys.exit(1)
    except ImportError:
        print("OK: numba not found (good)\n")

    # Load build settings
    bs = _load_build_settings()
    included_scenes = _get_included_scenes(bs)
    build_options = bs.get("build_options", {})
    strip_unused = build_options.get("strip_unused_assets", True)
    if CLI_STRIP:
        strip_unused = True
    if CLI_NO_STRIP:
        strip_unused = False

    print(f"BuildSettings loaded: {bool(bs)}")
    print(f"Scenes in BuildSettings: {bs.get('scenes', [])}")
    print(f"Included scenes: {len(included_scenes)}")
    for s in included_scenes:
        print(f"  -> {s}")

    build_plugins = bs.get("build_plugins", [])
    if not build_plugins:
        plugins_dir = ROOT / "plugins"
        if plugins_dir.is_dir():
            for f in sorted(plugins_dir.iterdir()):
                if f.suffix == ".py" and not f.stem.startswith("_"):
                    build_plugins.append(f.stem)
            user_dir = plugins_dir / "user"
            if user_dir.is_dir():
                for f in sorted(user_dir.iterdir()):
                    if f.suffix == ".py" and not f.stem.startswith("_"):
                        build_plugins.append("user." + f.stem)
        # Write augmented BuildSettings for Nuitka to copy into dist
        bs["build_plugins"] = build_plugins
        _build_bs_path = ROOT / "_build_BuildSettings.json"
        with open(_build_bs_path, "w", encoding="utf-8") as f:
            json.dump(bs, f, indent=2)
        print(f"Build plugins (auto): {build_plugins}")
    else:
        print(f"Build plugins (config): {build_plugins}")

    # Collect assets from scenes
    all_asset_refs: set[str] = set()
    for scene in included_scenes:
        refs = _scan_scene_assets(scene)
        all_asset_refs.update(refs)
        print(f"  Scene: {Path(scene).name} -> {len(refs)} asset refs")

    resolved_assets = _resolve_assets(all_asset_refs, ROOT)
    print(f"Total referenced assets: {len(resolved_assets)}")
    print(f"strip_unused_assets: {strip_unused}")
    for a in list(resolved_assets)[:10]:
        print(f"  -> {a}")
    if len(resolved_assets) > 10:
        print(f"  ... and {len(resolved_assets) - 10} more")

    # Run dependency analyzer for nofollow lists
    print("\nAnalyzing dependencies...")
    from build_analyzer import analyze
    analysis = analyze(verbose=False)
    module_nofollow = analysis["module_nofollow"]
    package_nofollow = analysis["package_nofollow"]
    print(f"Auto-excluded: {len(module_nofollow)} modules, {len(package_nofollow)} packages")
    print()

    if OUTPUT_DIR.exists():
        print(f"Cleaning old output: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    # Clean ALL Nuitka build artifacts (stale .c files from previous builds)
    for stale_dir in [ROOT / "player.build", ROOT / "main.build"]:
        if stale_dir.exists():
            print(f"Cleaning stale build dir: {stale_dir}")
            shutil.rmtree(stale_dir, ignore_errors=True)

    # Clean Nuitka cache to force fresh compilation
    nuitka_cache = Path(os.path.expanduser("~")) / ".cache" / "nuitka"
    if nuitka_cache.exists():
        print(f"Cleaning Nuitka cache: {nuitka_cache}")
        shutil.rmtree(nuitka_cache, ignore_errors=True)

    # Clean all __pycache__ dirs to avoid stale bytecode
    for pycache in ROOT.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache, ignore_errors=True)
        except Exception:
            pass
    print("Cleaned __pycache__ dirs")

    # Build NUITKA_OPTIONS dynamically
    NUITKA_OPTIONS = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--output-dir=" + str(OUTPUT_DIR),
        "--output-filename=" + ("ZarinEditor" if BUILD_EDITOR else "ZarinPlayer"),
        "--enable-plugin=pyqt6",
        "--disable-ccache",
        # Core packages (runtime)
        "--include-package=core",
        "--include-package=plugins",
        f"--include-package=physics_solvers.{_resolve_physics_solver()}_solver",
        # Data вЂ” use RELATIVE paths (Nuitka resolves relative to CWD which is ROOT)
        "--include-data-file=" + _ASSIMP_SRC + "=" + _ASSIMP_SRC,
        # Use auto-generated BuildSettings if build_plugins was empty (includes auto-discovered plugins)
        "--include-data-file=" + ("_build_BuildSettings.json" if build_plugins != bs.get("build_plugins") else "BuildSettings.json") + "=BuildSettings.json",
        # Auto-generated nofollow lists (soft exclusions)
        *[f"--nofollow-import-to={m}" for m in module_nofollow],
        *[f"--nofollow-import-to={m}" for m in package_nofollow],
        "--remove-output",
        "--clean-cache=all",
        "--warn-unusual-code",
    ]
    if ONEFILE:
        NUITKA_OPTIONS.append("--onefile")
    if NO_CONSOLE:
        NUITKA_OPTIONS.append("--disable-console")
    if NO_WINRT:
        NUITKA_OPTIONS.append("--include-windows-runtime-dlls=no")

    # Include only specified scenes
    scenes_dir = ROOT / "scenes"
    if included_scenes:
        # Create a temp scenes directory with only included scenes
        temp_scenes = ROOT / "_build_scenes"
        if temp_scenes.exists():
            shutil.rmtree(temp_scenes)
        temp_scenes.mkdir(parents=True)
        for scene in included_scenes:
            dest = temp_scenes / Path(scene).name
            shutil.copy2(scene, dest)
            print(f"  Scene copied: {scene} -> {dest}")
        NUITKA_OPTIONS.append(f"--include-data-dir=_build_scenes=scenes")
        print(f"Included {len(included_scenes)} scenes")
        print(f"  _build_scenes contents: {[str(p.name) for p in temp_scenes.iterdir()]}")
    else:
        # No BuildSettings вЂ” include all scenes (fallback)
        if scenes_dir.exists():
            NUITKA_OPTIONS.append(f"--include-data-dir=scenes=scenes")
            print(f"No BuildSettings вЂ” including ALL scenes ({len(list(scenes_dir.iterdir()))} files)")

    # Include only referenced assets (or all if strip_unused is false)
    temp_assets = ROOT / "_build_assets"
    if temp_assets.exists():
        shutil.rmtree(temp_assets)

    if strip_unused:
        # Copy only referenced assets to temp directory (even if empty)
        temp_assets.mkdir(parents=True, exist_ok=True)
        assets_dir = ROOT / "assets"
        copied = 0
        for asset_path in resolved_assets:
            try:
                rel = asset_path.relative_to(assets_dir)
                dest = temp_assets / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(asset_path, dest)
                copied += 1
                print(f"  Asset copied: {rel}")
            except ValueError:
                dest = temp_assets / asset_path.name
                shutil.copy2(asset_path, dest)
                copied += 1
                print(f"  Asset copied (flat): {asset_path.name}")
        NUITKA_OPTIONS.append(f"--include-data-dir=_build_assets=assets")
        print(f"Assets: {copied} referenced files copied (of {len(resolved_assets)} refs)")
        print(f"  _build_assets contents: {[str(p.relative_to(temp_assets)) for p in temp_assets.rglob('*') if p.is_file()]}")
    else:
        # No filtering вЂ” include everything
        if (ROOT / "assets").exists():
            NUITKA_OPTIONS.append(f"--include-data-dir=assets=assets")
            print("Assets: including ALL (no filtering)")

    # Always include materials and prefabs (small)
    if (ROOT / "materials").exists():
        NUITKA_OPTIONS.append("--include-data-dir=materials=materials")
    if (ROOT / "prefabs").exists():
        NUITKA_OPTIONS.append("--include-data-dir=prefabs=prefabs")
    # Include shaders (needed by renderer)
    if (ROOT / "editor" / "shaders").exists():
        NUITKA_OPTIONS.append("--include-data-dir=editor/shaders=editor/shaders")

    # Entry module MUST be last вЂ” Nuitka treats everything after it as positional args
    NUITKA_OPTIONS.append(ENTRY)

    print("\nRunning Nuitka with options:")
    for opt in NUITKA_OPTIONS:
        if opt.startswith("-"):
            print(f"  {opt}")
        else:
            print(f"  ENTRY: {opt}")
    print()

    _minify_pil()
    try:
        result = subprocess.run(NUITKA_OPTIONS, cwd=str(ROOT))
    finally:
        _restore_pil()

    # Cleanup temp files and directories
    temp_bs = ROOT / "_build_BuildSettings.json"
    if temp_bs.exists():
        temp_bs.unlink()
    for d in ["_build_scenes", "_build_assets"]:
        p = ROOT / d
        if p.exists():
            shutil.rmtree(p)

    if result.returncode != 0:
        print(f"\nBuild failed with code {result.returncode}")
        sys.exit(result.returncode)

    # Verify BuildSettings.json is in the dist
    dist_dirs = list(OUTPUT_DIR.glob("*.dist"))
    if dist_dirs:
        dist_dir = dist_dirs[0]
        bs_in_dist = dist_dir / "BuildSettings.json"
        print(f"\nVerifying dist: {dist_dir}")
        print(f"  BuildSettings.json: {'EXISTS' if bs_in_dist.exists() else 'MISSING!'}")
        scene_in_dist = dist_dir / "scenes"
        if scene_in_dist.exists():
            scenes_list = [p.name for p in scene_in_dist.iterdir()]
            print(f"  Scenes: {scenes_list}")
        else:
            print(f"  Scenes dir: MISSING!")
        assets_in_dist = dist_dir / "assets"
        if assets_in_dist.exists():
            assets_count = len(list(assets_in_dist.rglob("*")))
            print(f"  Assets: {assets_count} files")
        else:
            print(f"  Assets dir: MISSING!")

    print("\nBuild succeeded!")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    build()
