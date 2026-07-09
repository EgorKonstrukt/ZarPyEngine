# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import io
import json
import struct
import hashlib
import numpy as np
from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QFileDialog, QMessageBox, QDialog,
    QVBoxLayout, QFormLayout, QGroupBox,
    QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QDialogButtonBox,
)
from core.plugin_manager import PluginBase
from core.logger import Logger
from PIL import Image

BSP_LUMP_VERTICES = 3
BSP_LUMP_TEXINFO = 6
BSP_LUMP_FACES = 7
BSP_LUMP_EDGES = 12
BSP_LUMP_SURFEDGES = 13
BSP_LUMP_TEXDATA = 2
BSP_LUMP_TEXDATA_STRING_DATA = 43
BSP_LUMP_TEXDATA_STRING_TABLE = 44

_FACE_FMT = struct.Struct('<HBBiHHhHi')
_LUMP_DIR_FMT = struct.Struct('<IIII')


def _generate_texture_png(tex_name: str, size: int = 128) -> bytes:
    h = hashlib.md5(tex_name.encode()).digest()
    r_base = (h[0] + 64) % 192 + 32
    g_base = (h[1] + 64) % 192 + 32
    b_base = (h[2] + 64) % 192 + 32
    img = Image.new('RGBA', (size, size))
    pixels = []
    check = 16
    for y in range(size):
        for x in range(size):
            cx = (x // check) % 2
            cy = (y // check) % 2
            is_light = (cx ^ cy) == 0
            if is_light:
                r = min(255, r_base + 30)
                g = min(255, g_base + 30)
                b = min(255, b_base + 30)
            else:
                r = max(0, r_base - 20)
                g = max(0, g_base - 20)
                b = max(0, b_base - 20)
            pixels.append((r, g, b, 255))
    img.putdata(pixels)
    buf = io.BytesIO()
    img.save(buf, 'PNG')
    return buf.getvalue()


class BSPFile:
    def __init__(self, path: str):
        self.path = path
        self.version = 0
        self.lumps: dict[int, tuple[int, int]] = {}
        self.vertices: list[tuple[float, float, float]] = []
        self.edges: list[tuple[int, int]] = []
        self.surfedges: list[int] = []
        self.faces: list[dict] = []
        self.texinfo: list[dict] = []
        self.texdata: list[dict] = []
        self.texdata_string_data: bytes = b''
        self.texdata_string_table: list[int] = []
        self._parse()

    def _read_lump(self, data: bytes, lump_id: int) -> bytes:
        lump = self.lumps.get(lump_id)
        if not lump:
            return b''
        off, length = lump
        return data[off:off + length]

    def _parse(self):
        with open(self.path, 'rb') as f:
            data = f.read()
        if len(data) < 8 or data[:4] != b'VBSP':
            raise ValueError(f"Not a valid BSP file: {self.path}")
        self.version = struct.unpack_from('<I', data, 4)[0]
        for i in range(64):
            off = 8 + i * 16
            offset, length, ver, fourcc = _LUMP_DIR_FMT.unpack_from(data, off)
            self.lumps[i] = (offset, length)

        raw = self._read_lump(data, BSP_LUMP_VERTICES)
        self.vertices = [struct.unpack_from('<fff', raw, i * 12) for i in range(len(raw) // 12)]

        raw = self._read_lump(data, BSP_LUMP_EDGES)
        raw_len = len(raw)
        n_vert = len(self.vertices) if self.vertices else 0
        if self.version >= 20 and raw_len >= 8:
            n8 = raw_len // 8
            good_8 = 0
            for i in range(min(200, n8)):
                v0, v1 = struct.unpack_from('<ii', raw, i * 8)
                if 0 <= v0 < n_vert and 0 <= v1 < n_vert:
                    good_8 += 1
            if good_8 >= min(200, n8) * 0.75:
                self.edges = [struct.unpack_from('<ii', raw, i * 8) for i in range(n8)]
            else:
                self.edges = [struct.unpack_from('<HH', raw, i * 4) for i in range(raw_len // 4)]
        else:
            self.edges = [struct.unpack_from('<HH', raw, i * 4) for i in range(raw_len // 4)]

        raw = self._read_lump(data, BSP_LUMP_SURFEDGES)
        self.surfedges = [struct.unpack_from('<i', raw, i * 4)[0] for i in range(len(raw) // 4)]

        raw = self._read_lump(data, BSP_LUMP_FACES)
        for i in range(len(raw) // 20):
            planenum, side, onnode, firstedge, numedges, texinfo, dispinfo, surfaceflags, smoothgroups = \
                _FACE_FMT.unpack_from(raw, i * 20)
            self.faces.append(dict(planenum=planenum, side=side, onnode=onnode,
                                   firstedge=firstedge, numedges=numedges,
                                   texinfo=texinfo, dispinfo=dispinfo,
                                   surfaceflags=surfaceflags, smoothgroups=smoothgroups))

        raw = self._read_lump(data, BSP_LUMP_TEXINFO)
        self.texinfo = []
        for i in range(len(raw) // 72):
            p = i * 72
            tv0 = struct.unpack_from('<4f', raw, p)
            tv1 = struct.unpack_from('<4f', raw, p + 16)
            flags, td_idx = struct.unpack_from('<ii', raw, p + 64)
            self.texinfo.append(dict(textureVecs=(tv0, tv1), flags=flags, texdata=td_idx))

        raw = self._read_lump(data, BSP_LUMP_TEXDATA)
        self.texdata = []
        for i in range(len(raw) // 32):
            p = i * 32
            name_id, w, h, vw, vh = struct.unpack_from('<iiiii', raw, p + 12)
            self.texdata.append(dict(nameStringTableID=name_id, width=w, height=h,
                                     view_width=vw, view_height=vh))

        self.texdata_string_data = self._read_lump(data, BSP_LUMP_TEXDATA_STRING_DATA)

        raw = self._read_lump(data, BSP_LUMP_TEXDATA_STRING_TABLE)
        self.texdata_string_table = [struct.unpack_from('<i', raw, i * 4)[0] for i in range(len(raw) // 4)]

    def get_texture_name(self, texdata_index: int) -> str:
        if texdata_index < 0 or texdata_index >= len(self.texdata):
            return "unknown"
        td = self.texdata[texdata_index]
        name_id = td['nameStringTableID']
        if name_id < 0 or name_id >= len(self.texdata_string_table):
            return f"tex_{texdata_index}"
        offset = self.texdata_string_table[name_id]
        if offset >= len(self.texdata_string_data):
            return f"tex_{texdata_index}"
        end = self.texdata_string_data.find(b'\0', offset)
        if end < 0:
            end = len(self.texdata_string_data)
        name = self.texdata_string_data[offset:end].decode('ascii', errors='replace')
        clean = name.replace('/', '_').replace('\\', '_').replace('.', '_')
        clean = ''.join(c for c in clean if c.isalnum() or c == '_')
        return clean if clean else f"tex_{texdata_index}"

    def extract_mesh_groups(self, settings: dict, progress=None) -> list[dict]:
        convert_coords = settings.get('convert_coords', True)
        scale = settings.get('scale', 1.0)
        merge_by_material = settings.get('merge_by_material', True)
        center_pivot = settings.get('center_pivot', False)
        total = len(self.faces)

        if progress:
            progress(15, f"Processing {total} faces...")

        texdata_len = len(self.texdata)
        texinfo_len = len(self.texinfo)
        surfedges_len = len(self.surfedges)
        edges_len = len(self.edges)
        vertices_len = len(self.vertices)

        cached_tex_name = {}

        face_infos = []
        report_interval = max(1, total // 50)

        for fi, face in enumerate(self.faces):
            if fi % report_interval == 0 and progress:
                pct = 15 + int((fi / total) * 30)
                progress(pct, f"Reading faces: {fi}/{total}")

            if face['dispinfo'] >= 0:
                face_infos.append(None)
                continue
            ti_idx = face['texinfo']
            if ti_idx < 0 or ti_idx >= texinfo_len:
                face_infos.append(None)
                continue
            td_idx = self.texinfo[ti_idx]['texdata']

            if td_idx in cached_tex_name:
                tex_name = cached_tex_name[td_idx]
            elif td_idx < texdata_len:
                tex_name = self.get_texture_name(td_idx)
                cached_tex_name[td_idx] = tex_name
            else:
                tex_name = f"tex_{td_idx}"
                cached_tex_name[td_idx] = tex_name

            firstedge = face['firstedge']
            numedges = face['numedges']
            vi_list = []
            for i in range(numedges):
                se_idx = firstedge + i
                if se_idx < surfedges_len:
                    se = self.surfedges[se_idx]
                    edge_idx = abs(se)
                    if edge_idx < edges_len:
                        edge = self.edges[edge_idx]
                        vi = edge[1] if se >= 0 else edge[0]
                        if vi < vertices_len:
                            vi_list.append(vi)

            if len(vi_list) < 3:
                face_infos.append(None)
                continue

            verts = [self.vertices[vi] for vi in vi_list]

            if convert_coords:
                verts_zp = [(v[0] * scale, v[2] * scale, -v[1] * scale) for v in verts]
            else:
                verts_zp = [(v[0] * scale, v[1] * scale, v[2] * scale) for v in verts]

            ti = self.texinfo[ti_idx]
            s_vec = ti['textureVecs'][0]
            t_vec = ti['textureVecs'][1]
            uvs = [(v[0] * s_vec[0] + v[1] * s_vec[1] + v[2] * s_vec[2] + s_vec[3],
                    v[0] * t_vec[0] + v[1] * t_vec[1] + v[2] * t_vec[2] + t_vec[3])
                   for v in verts]

            face_infos.append(dict(tex_name=tex_name, verts=verts_zp, uvs=uvs))

        if progress:
            progress(48, "Grouping by material...")

        if merge_by_material:
            groups: dict[str, list] = {}
            for fd in face_infos:
                if fd is None:
                    continue
                name = fd['tex_name']
                if name not in groups:
                    groups[name] = []
                groups[name].append(fd)
        else:
            groups = {}
            for fi, fd in enumerate(face_infos):
                if fd is None:
                    continue
                name = f"{fd['tex_name']}_face{fi}"
                groups[name] = [fd]

        if progress:
            progress(52, f"Building {len(groups)} mesh groups...")

        result = []
        group_items = list(groups.items())
        for gi, (tex_name, face_list) in enumerate(group_items):
            if progress:
                pct = 52 + int((gi / len(group_items)) * 8)
                progress(pct, f"Building: {tex_name}")

            all_verts = []
            all_uvs = []
            all_indices = []
            base = 0

            for fd in face_list:
                n = len(fd['verts'])
                for i in range(1, n - 1):
                    all_indices.append(base)
                    all_indices.append(base + i + 1)
                    all_indices.append(base + i)
                all_verts.extend(fd['verts'])
                all_uvs.extend(fd['uvs'])
                base += n

            if len(all_verts) < 3 or len(all_indices) < 3:
                continue

            verts_arr = np.array(all_verts, dtype=np.float32)
            uvs_arr = np.array(all_uvs, dtype=np.float32)
            idx_arr = np.array(all_indices, dtype=np.uint32)

            n_verts = len(verts_arr)
            n_idx = len(idx_arr)
            normals = np.zeros((n_verts, 3), dtype=np.float32)
            for i in range(0, n_idx, 3):
                i0, i1, i2 = idx_arr[i], idx_arr[i + 1], idx_arr[i + 2]
                v0, v1, v2 = verts_arr[i0], verts_arr[i1], verts_arr[i2]
                e1 = v1 - v0
                e2 = v2 - v0
                n = np.cross(e1, e2)
                nl = np.linalg.norm(n)
                if nl > 1e-10:
                    n = n / nl
                normals[i0] += n
                normals[i1] += n
                normals[i2] += n

            nl = np.linalg.norm(normals, axis=1, keepdims=True)
            nl[nl < 1e-10] = 1.0
            normals = normals / nl

            aabb_min = verts_arr.min(axis=0)
            aabb_max = verts_arr.max(axis=0)

            centroid = verts_arr.mean(axis=0)

            if center_pivot:
                verts_arr = verts_arr - centroid

            result.append(dict(
                name=tex_name,
                vertices=verts_arr,
                uvs=uvs_arr,
                normals=normals,
                indices=idx_arr,
                aabb_min=aabb_min,
                aabb_max=aabb_max,
                centroid=centroid,
            ))

        return result


class ImportSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BSP Import Settings")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)

        geo_group = QGroupBox("Geometry", self)
        geo_layout = QFormLayout(geo_group)
        self.merge_cb = QCheckBox("Merge faces by material", self)
        self.merge_cb.setChecked(True)
        geo_layout.addRow(self.merge_cb)
        self.center_cb = QCheckBox("Center pivot on each group", self)
        self.center_cb.setChecked(True)
        geo_layout.addRow(self.center_cb)
        self.scale_sb = QDoubleSpinBox(self)
        self.scale_sb.setRange(0.001, 1000.0)
        self.scale_sb.setValue(1.0)
        self.scale_sb.setSingleStep(0.1)
        self.scale_sb.setDecimals(4)
        geo_layout.addRow("Scale factor:", self.scale_sb)
        layout.addWidget(geo_group)

        coord_group = QGroupBox("Coordinate System", self)
        coord_layout = QFormLayout(coord_group)
        self.coord_cb = QComboBox(self)
        self.coord_cb.addItems([
            "Source (Z-up) -> ZarPy (Y-up)",
            "Keep Source (Z-up)",
        ])
        coord_layout.addRow("Conversion:", self.coord_cb)
        layout.addWidget(coord_group)

        tex_group = QGroupBox("Textures", self)
        tex_layout = QFormLayout(tex_group)
        self.gen_tex_cb = QCheckBox("Generate procedural textures", self)
        self.gen_tex_cb.setChecked(True)
        tex_layout.addRow(self.gen_tex_cb)
        self.tex_size_sb = QSpinBox(self)
        self.tex_size_sb.setRange(16, 1024)
        self.tex_size_sb.setValue(128)
        self.tex_size_sb.setSingleStep(32)
        tex_layout.addRow("Texture size (px):", self.tex_size_sb)
        layout.addWidget(tex_group)

        adv_group = QGroupBox("Advanced", self)
        adv_layout = QFormLayout(adv_group)
        self.import_brush_cb = QCheckBox("Import brush entity models", self)
        adv_layout.addRow(self.import_brush_cb)
        self.import_ents_cb = QCheckBox("Import entity keyvalues", self)
        adv_layout.addRow(self.import_ents_cb)
        layout.addWidget(adv_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        layout.addWidget(buttons)

    def get_settings(self) -> dict:
        return {
            'merge_by_material': self.merge_cb.isChecked(),
            'center_pivot': self.center_cb.isChecked(),
            'scale': self.scale_sb.value(),
            'convert_coords': self.coord_cb.currentIndex() == 0,
            'generate_textures': self.gen_tex_cb.isChecked(),
            'texture_size': self.tex_size_sb.value(),
            'import_brush_entities': self.import_brush_cb.isChecked(),
            'import_entity_data': self.import_ents_cb.isChecked(),
        }


class BSPImportWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, path: str, settings: dict,
                 project_root: str, materials_dir: str):
        super().__init__()
        self._path = path
        self._settings = settings
        self._project_root = project_root
        self._materials_dir = materials_dir
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self.progress.emit(0, "Parsing BSP file...")
            bsp = BSPFile(self._path)

            if self._cancelled:
                return

            self.progress.emit(15, f"Extracting {len(bsp.faces)} faces...")
            groups = bsp.extract_mesh_groups(self._settings, self._on_sub_progress)

            if self._cancelled or not groups:
                self.progress.emit(100, "Cancelled")
                return

            gen_tex = self._settings.get('generate_textures', True)
            tex_size = self._settings.get('texture_size', 128)

            self.progress.emit(65, f"Generating {len(groups)} textures...")
            saved_textures = {}
            for i, group in enumerate(groups):
                if self._cancelled:
                    return
                tex_name = group['name']
                pct = 65 + int((i / len(groups)) * 20)
                self.progress.emit(pct, f"Generating texture: {tex_name}")

                if gen_tex:
                    png = _generate_texture_png(tex_name, tex_size)
                    tex_path = os.path.normpath(
                        os.path.join(self._materials_dir, f"{tex_name}.png"))
                    with open(tex_path, 'wb') as f:
                        f.write(png)
                    saved_textures[tex_name] = tex_path

            self.progress.emit(86, "Saving material files...")
            saved_materials = {}
            saved_materials_abs = {}
            os.makedirs(self._materials_dir, exist_ok=True)
            for i, group in enumerate(groups):
                if self._cancelled:
                    return
                tex_name = group['name']
                pct = 86 + int((i / len(groups)) * 4)
                self.progress.emit(pct, f"Saving material: {tex_name}")

                tex_path = saved_textures.get(tex_name, "")
                mat_path = os.path.normpath(
                    os.path.join(self._materials_dir, f"{tex_name}.mat"))
                if not os.path.exists(mat_path) and tex_path:
                    rel_tex = os.path.relpath(tex_path, self._project_root).replace("\\", "/")
                    mat_data = {
                        "name": tex_name,
                        "shader_path": "",
                        "properties": {
                            "albedo_texture": rel_tex,
                            "use_albedo_tex": 1,
                        }
                    }
                    with open(mat_path, 'w', encoding='utf-8') as f:
                        json.dump(mat_data, f, indent=2)

                saved_materials_abs[tex_name] = mat_path
                saved_materials[tex_name] = os.path.relpath(
                    mat_path, self._project_root).replace("\\", "/")
                group['_material_abs'] = mat_path

            self.progress.emit(90, "Preparing scene data...")
            bsp_name = os.path.splitext(os.path.basename(bsp.path))[0]

            result = dict(
                bsp_name=bsp_name,
                groups=groups,
                saved_materials=saved_materials,
                saved_materials_abs=saved_materials_abs,
                face_count=len(bsp.faces),
                center_pivot=self._settings.get('center_pivot', True),
            )
            self.progress.emit(100, "Done")
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))

    def _on_sub_progress(self, pct: int, msg: str):
        if not self._cancelled:
            self.progress.emit(pct, msg)


class BSPImportPlugin(PluginBase):
    NAME = "BSPImport"
    VERSION = "2.0.0"
    DESCRIPTION = "Import Source Engine BSP maps into the scene (async)"
    SYSTEM = False

    def __init__(self):
        super().__init__()
        self._menu_added = False
        self._thread: QThread | None = None
        self._worker: BSPImportWorker | None = None
        self._main_window = None

    def on_viewport_ready(self, viewport):
        if self._menu_added:
            return
        main_window = viewport.window()
        if not main_window:
            return
        self._main_window = main_window
        for action in main_window.menuBar().actions():
            if action.text() == "Tools":
                tools_menu = action.menu()
                sep = tools_menu.actions()
                import_act = QAction("Import BSP...", main_window)
                import_act.triggered.connect(lambda: self._on_import_bsp(main_window))
                if sep:
                    tools_menu.insertAction(sep[0], import_act)
                else:
                    tools_menu.addAction(import_act)
                self._menu_added = True
                Logger.info("[BSPImport] Added menu item to Tools")
                break

    def _on_import_bsp(self, parent_widget):
        from core.engine import Engine
        engine = Engine.instance()
        if not engine:
            return

        path, _ = QFileDialog.getOpenFileName(
            parent_widget, "Import Source Engine BSP", "",
            "BSP Files (*.bsp);;All Files (*.*)")
        if not path:
            return

        dlg = ImportSettingsDialog(parent_widget)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        settings = dlg.get_settings()

        Logger.info(f"[BSPImport] Importing: {path}")

        try:
            self._import_sync(path, settings, engine, parent_widget)
        except Exception as e:
            Logger.error(f"[BSPImport] Failed: {e}")
            print(f"[BSPImport] Failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            QMessageBox.critical(parent_widget, "BSP Import Error",
                                 f"Import failed:\n{e}")

    def _import_sync(self, path: str, settings: dict, engine, parent):
        from core.components import Transform, MeshFilter, MeshRenderer
        from core.math3d import Vec3
        from core.renderer.mesh_data import MeshData

        Logger.info("[BSPImport] Parsing BSP file...")
        bsp = BSPFile(path)
        Logger.info(f"[BSPImport] BSP v{bsp.version}, {len(bsp.faces)} faces, {len(bsp.vertices)} verts")

        merge_by_material = settings.get('merge_by_material', True)
        gen_tex = settings.get('generate_textures', True)
        tex_size = settings.get('texture_size', 128)
        center_pivot = settings.get('center_pivot', False)
        convert_coords = settings.get('convert_coords', True)
        scale = settings.get('scale', 1.0)

        materials_dir = os.path.join(engine.project_root, "assets", "materials", "bsp")
        os.makedirs(materials_dir, exist_ok=True)

        def progress_cb(pct, msg):
            pass

        Logger.info("[BSPImport] Extracting mesh groups...")
        groups = bsp.extract_mesh_groups(settings, progress_cb)

        if not groups:
            QMessageBox.information(parent, "BSP Import",
                                    "No valid mesh geometry found.")
            return

        Logger.info(f"[BSPImport] {len(groups)} material groups extracted")

        saved_textures = {}
        if gen_tex:
            Logger.info("[BSPImport] Generating textures...")
            for i, group in enumerate(groups):
                tex_name = group['name']
                png = _generate_texture_png(tex_name, tex_size)
                tex_path = os.path.normpath(
                    os.path.join(materials_dir, f"{tex_name}.png"))
                with open(tex_path, 'wb') as f:
                    f.write(png)
                saved_textures[tex_name] = tex_path

        Logger.info("[BSPImport] Saving material files...")
        for group in groups:
            tex_name = group['name']
            tex_path = saved_textures.get(tex_name, "")
            mat_path = os.path.normpath(
                os.path.join(materials_dir, f"{tex_name}.mat"))
            if not os.path.exists(mat_path) and tex_path:
                rel_tex = os.path.relpath(tex_path, engine.project_root).replace("\\", "/")
                mat_data = {
                    "name": tex_name,
                    "shader_path": "",
                    "properties": {
                        "albedo_texture": rel_tex,
                        "use_albedo_tex": 1,
                    }
                }
                with open(mat_path, 'w', encoding='utf-8') as f:
                    json.dump(mat_data, f, indent=2)
            group['_material_abs'] = mat_path

        Logger.info("[BSPImport] Building scene...")
        bsp_name = os.path.splitext(os.path.basename(path))[0]

        viewport = engine.viewport
        renderer = viewport.renderer if viewport and hasattr(viewport, 'renderer') else None
        ctx = renderer._ctx if renderer and hasattr(renderer, '_ctx') else None
        default_prog = renderer._default_prog if renderer else None
        meshes = renderer._meshes if renderer else None

        scene = engine.scene
        if not scene:
            scene = engine.new_scene("BSPMap")

        root_entity = scene.create_entity(bsp_name)
        root_entity.add_component(Transform())

        imported_count = 0

        for group in groups:
            tex_name = group['name']
            verts = group['vertices']
            normals = group['normals']
            uvs = group['uvs']
            indices = group['indices']
            centroid = group['centroid']

            if len(indices) < 3:
                continue

            mesh = MeshData()
            mesh.vertices = verts.flatten()
            mesh.normals = normals.flatten()
            mesh.uvs = uvs.flatten()
            mesh.indices = indices.astype(np.uint32)
            mesh.compute_aabb()

            if ctx and default_prog:
                try:
                    mesh.build_gl(ctx, default_prog)
                except Exception as e:
                    Logger.error(f"[BSPImport] GL build failed for {tex_name}: {e}")

            mesh_name = f"bsp_{bsp_name}_{tex_name}"

            if meshes is not None:
                meshes[mesh_name] = mesh
                meshes[f"{mesh_name}|s=1.0|cp=False|fu=False"] = mesh

            entity = scene.create_entity(tex_name)
            t = Transform()
            t.position = Vec3(centroid[0], centroid[1], centroid[2])
            entity.add_component(t)

            mf = MeshFilter()
            mf.mesh_name = mesh_name
            entity.add_component(mf)

            mr = MeshRenderer()
            mr.materials[0]["path"] = group.get('_material_abs', '')

            entity.set_parent(root_entity)
            imported_count += 1

        face_count = len(bsp.faces)
        Logger.info(
            f"[BSPImport] Imported {imported_count} groups / "
            f"{face_count} faces from {bsp_name}")

        # Refresh hierarchy panel
        if self._main_window and hasattr(self._main_window, '_hierarchy'):
            try:
                self._main_window._hierarchy.refresh()
            except Exception:
                pass

        QMessageBox.information(parent, "BSP Import",
                                f"Imported {imported_count} material groups\n"
                                f"({face_count} faces) from {bsp_name}")

        if viewport:
            QTimer.singleShot(0, viewport.update)

    def _on_import_finished(self, result: dict, engine, parent):
        print("[BSP_DEBUG] _on_import_finished start", flush=True)
        from core.components import Transform, MeshFilter, MeshRenderer
        from core.math3d import Vec3
        from core.renderer.mesh_data import MeshData

        bsp_name = result['bsp_name']
        groups = result['groups']
        face_count = result.get('face_count', 0)

        print(f"[BSP_DEBUG] groups={len(groups)}", flush=True)

        if not groups:
            self._cleanup_import()
            print("[BSP_DEBUG] no groups, cleanup done", flush=True)
            QMessageBox.information(parent, "BSP Import",
                                    "No valid mesh geometry found.")
            return

        viewport = engine.viewport
        renderer = viewport.renderer if viewport and hasattr(viewport, 'renderer') else None
        ctx = renderer._ctx if renderer and hasattr(renderer, '_ctx') else None
        default_prog = renderer._default_prog if renderer else None
        meshes = renderer._meshes if renderer else None

        scene = engine.scene
        if not scene:
            scene = engine.new_scene("BSPMap")

        root_entity = scene.create_entity(bsp_name)
        root_entity.add_component(Transform())

        imported_count = 0

        for group in groups:
            tex_name = group['name']
            verts = group['vertices']
            normals = group['normals']
            uvs = group['uvs']
            indices = group['indices']
            centroid = group['centroid']

            if len(indices) < 3:
                continue

            try:
                mesh = MeshData()
                mesh.vertices = verts.flatten()
                mesh.normals = normals.flatten()
                mesh.uvs = uvs.flatten()
                mesh.indices = indices.astype(np.uint32)
                mesh.compute_aabb()

                if ctx and default_prog:
                    try:
                        mesh.build_gl(ctx, default_prog)
                    except Exception as e:
                        Logger.error(f"[BSPImport] GL build failed for {tex_name}: {e}")

                mesh_name = f"bsp_{bsp_name}_{tex_name}"

                if meshes is not None:
                    meshes[mesh_name] = mesh
                    meshes[f"{mesh_name}|s=1.0|cp=False|fu=False"] = mesh

                entity = scene.create_entity(tex_name)
                t = Transform()
                t.position = Vec3(centroid[0], centroid[1], centroid[2])
                entity.add_component(t)

                mf = MeshFilter()
                mf.mesh_name = mesh_name
                entity.add_component(mf)

                mr = MeshRenderer()
                mr.materials[0]["path"] = group.get('_material_abs', '')

                entity.set_parent(root_entity)
                imported_count += 1
            except Exception as e:
                print(f"[BSP_DEBUG] Failed group {tex_name}: {e}", flush=True)
                Logger.error(f"[BSPImport] Failed group {tex_name}: {e}")

        self._cleanup_import()

        Logger.info(
            f"[BSPImport] Imported {imported_count} groups / "
            f"{face_count} faces from {bsp_name}")

        # Refresh hierarchy panel
        if self._main_window and hasattr(self._main_window, '_hierarchy'):
            try:
                self._main_window._hierarchy.refresh()
            except Exception:
                pass

        QMessageBox.information(parent, "BSP Import",
                                f"Imported {imported_count} material groups\n"
                                f"({face_count} faces) from {bsp_name}")

        if viewport:
            QTimer.singleShot(0, viewport.update)

    def _cleanup_import(self):
        self._thread = None
        self._worker = None

    def shutdown(self):
        if self._thread and self._thread.isRunning():
            if self._worker:
                self._worker.cancel()
            self._thread.quit()
            self._thread.wait(5000)
        self._cleanup_import()
        Logger.info("[BSPImport] Shutdown.")
