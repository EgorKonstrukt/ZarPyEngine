# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np
import moderngl
from typing import Optional, Any
from core.maths.math3d import Mat4
from core.components.rendering.renderers.text_renderer import TextRenderer, TextAlign, TextFilter

from core.assets.font_atlas import FontAtlas


class TextRendererGL:
    def __init__(self, ctx: moderngl.Context, prog: moderngl.Program):
        self._ctx = ctx
        self._prog = prog
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._max_chars: int = 4096
        self._font_atlases: dict[tuple[str, int], FontAtlas] = {}
        self._tex_cache: dict[tuple[str, int], Any] = {}
        self._verts: Optional[np.ndarray] = None
        self._geom_cache: dict[int, tuple[int, int, float, np.ndarray, int]] = {}
        self._rich_seg_info: list[dict] = []
        self._build_buffers()

    def _ensure_capacity(self, needed_chars: int):
        if needed_chars <= self._max_chars:
            return
        new_max = 1
        while new_max < needed_chars:
            new_max <<= 1
        new_verts = np.zeros(new_max * 20, dtype=np.float32)
        old_len = self._max_chars * 20
        new_verts[:old_len] = self._verts[:old_len]
        self._verts = new_verts
        new_indices = np.zeros(new_max * 6, dtype=np.uint32)
        for i in range(new_max):
            base = i * 4
            bi = i * 6
            new_indices[bi + 0] = base + 0
            new_indices[bi + 1] = base + 1
            new_indices[bi + 2] = base + 2
            new_indices[bi + 3] = base + 0
            new_indices[bi + 4] = base + 2
            new_indices[bi + 5] = base + 3
        if self._vbo:
            self._vbo.release()
        if self._ibo:
            self._ibo.release()
        if self._vao:
            self._vao.release()
        self._vbo = self._ctx.buffer(self._verts.tobytes(), dynamic=True)
        self._ibo = self._ctx.buffer(new_indices.tobytes())
        self._vao = self._ctx.vertex_array(
            self._prog,
            [(self._vbo, "3f 2f", "in_position", "in_uv")],
            self._ibo
        )
        self._max_chars = new_max

    def _build_buffers(self):
        max_verts = self._max_chars * 4
        max_indices = self._max_chars * 6
        self._verts = np.zeros(max_verts * 5, dtype=np.float32)
        indices = np.zeros(max_indices, dtype=np.uint32)
        for i in range(self._max_chars):
            base = i * 4
            indices[i * 6 + 0] = base + 0
            indices[i * 6 + 1] = base + 1
            indices[i * 6 + 2] = base + 2
            indices[i * 6 + 3] = base + 0
            indices[i * 6 + 4] = base + 2
            indices[i * 6 + 5] = base + 3
        self._vbo = self._ctx.buffer(self._verts.tobytes(), dynamic=True)
        self._ibo = self._ctx.buffer(indices.tobytes())
        self._vao = self._ctx.vertex_array(
            self._prog,
            [(self._vbo, "3f 2f", "in_position", "in_uv")],
            self._ibo
        )

    def get_or_create_atlas(self, font_path: str, base_size: int = 128) -> Optional[FontAtlas]:
        key = (font_path, base_size)
        if key in self._font_atlases:
            return self._font_atlases[key]
        if not font_path:
            return None
        try:
            atlas = FontAtlas(font_path, base_size)
            self._font_atlases[key] = atlas
            return atlas
        except Exception:
            return None

    def _get_atlas(self, font_path: str, base_size: int = 128) -> Optional[FontAtlas]:
        return self.get_or_create_atlas(font_path, base_size)

    def _ensure_texture(self, atlas: FontAtlas, filter_mode: TextFilter = TextFilter.LINEAR, anisotropy: float = 0.0) -> Any:
        _FILTER_MAP = {
            TextFilter.NEAREST: (moderngl.NEAREST, moderngl.NEAREST),
            TextFilter.LINEAR: (moderngl.LINEAR, moderngl.LINEAR),
            TextFilter.TRILINEAR: (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR),
        }
        key = (atlas.font_path, atlas.base_size)
        if key in self._tex_cache:
            tex = self._tex_cache[key]
            tex.filter = _FILTER_MAP.get(filter_mode, (moderngl.LINEAR, moderngl.LINEAR))
            if anisotropy > 0:
                tex.anisotropy = anisotropy
            return tex
        tex = self._ctx.texture((atlas.texture_width, atlas.texture_height), 4, atlas.texture.tobytes())
        tex.filter = _FILTER_MAP.get(filter_mode, (moderngl.LINEAR, moderngl.LINEAR))
        tex.repeat_x = False
        tex.repeat_y = False
        tex.build_mipmaps()
        if anisotropy > 0:
            tex.anisotropy = anisotropy
        self._tex_cache[key] = tex
        return tex

    def _build_line_quads(self, atlas: FontAtlas, text: str, scale: float, pen_x: float, pen_y: float, verts: np.ndarray, base_idx: int, italic: bool = False, z_offset: float = 0.0, bold: bool = False) -> tuple[int, float]:
        if not text:
            return 0, 0.0
        codes = np.frombuffer(text.encode('utf-32-le'), dtype=np.uint32)
        oob = codes >= atlas.max_cp
        if np.any(oob):
            codes = codes.copy()
            codes[oob] = 0
        advances = atlas._gp_advance[codes] * scale
        bw = atlas._gp_bearing_x[codes] * scale
        bh = atlas._gp_bearing_y[codes] * scale
        gw = atlas._gp_glyph_w[codes] * scale
        gh = atlas._gp_glyph_h[codes] * scale
        valid = (gw > 0) & (gh > 0) & ~oob
        vi = np.where(valid)[0]
        n = len(vi)
        total_adv = float(np.sum(advances))
        if n == 0:
            return 0, total_adv
        max_floats = len(verts)
        space_left = max_floats - base_idx
        max_n = space_left // 20
        if max_n <= 0:
            return 0, total_adv
        if n > max_n:
            n = max_n
            vi = vi[:n]
        cum = np.cumsum(advances) - advances
        ascent = atlas.ascender
        top_base = pen_y + ascent * scale
        l = pen_x + cum + bw
        r = l + gw
        t = top_base - bh
        b = t - gh
        skew = 0.25 if italic else 0.0
        sh = skew * (t - pen_y)
        uv_arr = atlas._gp_uv_bold if bold else atlas._gp_uv
        u0 = uv_arr[codes, 0]
        v0 = uv_arr[codes, 1]
        u1 = uv_arr[codes, 2]
        v1 = uv_arr[codes, 3]
        idx = vi
        e = base_idx + n * 20
        verts[base_idx + 0:e:20] = l[idx]
        verts[base_idx + 1:e:20] = b[idx]
        verts[base_idx + 2:e:20] = z_offset
        verts[base_idx + 3:e:20] = u0[idx]
        verts[base_idx + 4:e:20] = v1[idx]
        verts[base_idx + 5:e:20] = r[idx]
        verts[base_idx + 6:e:20] = b[idx]
        verts[base_idx + 7:e:20] = z_offset
        verts[base_idx + 8:e:20] = u1[idx]
        verts[base_idx + 9:e:20] = v1[idx]
        verts[base_idx + 10:e:20] = r[idx] + sh[idx]
        verts[base_idx + 11:e:20] = t[idx]
        verts[base_idx + 12:e:20] = z_offset
        verts[base_idx + 13:e:20] = u1[idx]
        verts[base_idx + 14:e:20] = v0[idx]
        verts[base_idx + 15:e:20] = l[idx] + sh[idx]
        verts[base_idx + 16:e:20] = t[idx]
        verts[base_idx + 17:e:20] = z_offset
        verts[base_idx + 18:e:20] = u0[idx]
        verts[base_idx + 19:e:20] = v0[idx]
        return n, total_adv

    def _build_effect_quads(self, atlas: FontAtlas, text: str, scale: float, pen_x: float, pen_y: float, verts: np.ndarray, base_idx: int, underline: bool, strikethrough: bool, z_offset: float = 0.0, total_w: float = 0.0) -> int:
        ascent = atlas.ascender
        descender = atlas.descender
        if total_w <= 0:
            return 0
        count = 0
        if underline:
            line_y = pen_y - descender * scale * 0.5
            thickness = scale * 4.0
            b = base_idx + count * 20
            verts[b + 0] = pen_x
            verts[b + 1] = line_y - thickness
            verts[b + 2] = z_offset
            verts[b + 3] = 0.0
            verts[b + 4] = 0.0
            verts[b + 5] = pen_x + total_w
            verts[b + 6] = line_y - thickness
            verts[b + 7] = z_offset
            verts[b + 8] = 0.0
            verts[b + 9] = 0.0
            verts[b + 10] = pen_x + total_w
            verts[b + 11] = line_y + thickness
            verts[b + 12] = z_offset
            verts[b + 13] = 0.0
            verts[b + 14] = 0.0
            verts[b + 15] = pen_x
            verts[b + 16] = line_y + thickness
            verts[b + 17] = z_offset
            verts[b + 18] = 0.0
            verts[b + 19] = 0.0
            count += 1
        if strikethrough:
            line_y = pen_y + ascent * scale * 0.35
            thickness = scale * 3.0
            b = base_idx + count * 20
            verts[b + 0] = pen_x
            verts[b + 1] = line_y - thickness
            verts[b + 2] = z_offset
            verts[b + 3] = 0.0
            verts[b + 4] = 0.0
            verts[b + 5] = pen_x + total_w
            verts[b + 6] = line_y - thickness
            verts[b + 7] = z_offset
            verts[b + 8] = 0.0
            verts[b + 9] = 0.0
            verts[b + 10] = pen_x + total_w
            verts[b + 11] = line_y + thickness
            verts[b + 12] = z_offset
            verts[b + 13] = 0.0
            verts[b + 14] = 0.0
            verts[b + 15] = pen_x
            verts[b + 16] = line_y + thickness
            verts[b + 17] = z_offset
            verts[b + 18] = 0.0
            verts[b + 19] = 0.0
            count += 1
        return count

    def _render_quads(self, vi: int, color: list[float], tex: Any, write_depth: bool, solid: bool, start_v: int = 0, clip_alpha: float = 0.01):
        prog = self._prog
        if vi == 0:
            return
        if "u_color" in prog:
            prog["u_color"].write(np.array(color, dtype=np.float32).tobytes())
        if "u_solid" in prog:
            prog["u_solid"].value = 1.0 if solid else 0.0
        if "u_clip_alpha" in prog:
            prog["u_clip_alpha"].value = clip_alpha
        tex.use(0)
        if "u_texture" in prog:
            prog["u_texture"].value = 0
        if write_depth:
            self._ctx.depth_mask = True
        else:
            self._ctx.depth_mask = False
        self._vao.render(moderngl.TRIANGLES, vertices=vi * 6, first=start_v * 6)

    def _geom_hash(self, tr: TextRenderer, atlas: FontAtlas) -> int:
        props = (
            tr.text, tr.font_path, tr.font_size, tuple(tr.color),
            tr.font_world_space, tr.billboard, tr.alignment, tr.line_spacing,
            tr.italic, tr.underline, tr.strikethrough,
            tr.shadow, tuple(tr.shadow_offset), tuple(tuple(tr.shadow_color)),
            tr.use_3d, tr.extrusion_depth, tr.extrusion_layers,
            tuple(tuple(tr.extrusion_color)), tr.atlas_resolution, tr.bold,
            tr.use_rich_text, atlas.font_path, atlas.base_size,
        )
        return hash(props)

    def _build_text_verts(self, tr: TextRenderer, atlas: FontAtlas) -> tuple[int, int, float]:
        self._rich_seg_info.clear()
        inv_lh = atlas._inv_lh()
        scale = float(tr.font_size) * inv_lh * 0.01
        line_h = atlas.line_height * scale * tr.line_spacing
        self._verts[:] = 0.0
        verts = self._verts
        vi = 0
        evi = 0
        if tr.use_rich_text and tr._rich_segments:
            line_pieces: list[list[tuple[str, list[float], bool, bool, bool, bool]]] = [[]]
            for seg in tr._rich_segments:
                seg_lines = seg.text.split('\n')
                for i, piece in enumerate(seg_lines):
                    if i > 0:
                        line_pieces.append([])
                    if piece:
                        if not line_pieces:
                            line_pieces.append([])
                        line_pieces[-1].append((piece, seg.color, seg.bold, seg.italic, seg.underline, seg.strikethrough))
            if not line_pieces or not line_pieces[0]:
                return 0, 0, scale
            line_advs = []
            pen_y = 0.0
            vi = 0
            for pieces in line_pieces:
                pen_x = 0.0
                for piece_text, color, bold, italic, ul, st in pieces:
                    c, pw = self._build_line_quads(atlas, piece_text, scale, pen_x, pen_y, verts, vi * 20, italic, 0.0, bold)
                    if c > 0:
                        self._rich_seg_info.append({
                            'vi_start': vi, 'vi_count': c,
                            'evi_start': 0, 'evi_count': 0,
                            'color': color, 'bold': bold, 'italic': italic,
                            'underline': ul, 'strikethrough': st,
                            'width': pw,
                        })
                        vi += c
                        pen_x += pw
                line_advs.append(pen_x)
                pen_y -= line_h
            max_w = max(line_advs) if line_advs else 0.0
            seg_i = 0
            for line_idx, pieces in enumerate(line_pieces):
                if tr.alignment == TextAlign.LEFT:
                    off_x = 0.0
                elif tr.alignment == TextAlign.CENTER:
                    off_x = (max_w - line_advs[line_idx]) * 0.5
                elif tr.alignment == TextAlign.RIGHT:
                    off_x = max_w - line_advs[line_idx]
                else:
                    off_x = 0.0
                if off_x != 0.0:
                    for _ in pieces:
                        seg = self._rich_seg_info[seg_i]
                        s = seg['vi_start'] * 20
                        e = (seg['vi_start'] + seg['vi_count']) * 20
                        verts[s:e:5] -= off_x
                        seg_i += 1
                else:
                    seg_i += len(pieces)
            if vi > 0:
                xs = verts[0:vi * 20:5]
                ys = verts[1:vi * 20:5]
                x_mid = (float(np.min(xs)) + float(np.max(xs))) * 0.5
                y_mid = (float(np.min(ys)) + float(np.max(ys))) * 0.5
                verts[0:vi * 20:5] -= x_mid
                verts[1:vi * 20:5] -= y_mid
            pen_y = 0.0
            seg_effect_i = 0
            for line_idx, pieces in enumerate(line_pieces):
                for piece_text, color, bold, italic, ul, st in pieces:
                    if ul or st:
                        while (seg_effect_i < len(self._rich_seg_info) and
                               not (self._rich_seg_info[seg_effect_i]['underline'] or self._rich_seg_info[seg_effect_i]['strikethrough'])):
                            seg_effect_i += 1
                        if seg_effect_i < len(self._rich_seg_info):
                            seg_w = self._rich_seg_info[seg_effect_i]['width']
                            ec = self._build_effect_quads(atlas, piece_text, scale, 0.0, pen_y, verts, (vi + evi) * 20, ul, st, 0.0, seg_w)
                            self._rich_seg_info[seg_effect_i]['evi_start'] = evi
                            self._rich_seg_info[seg_effect_i]['evi_count'] = ec
                            seg_effect_i += 1
                            evi += ec
                pen_y -= line_h
            if evi > 0:
                es = vi * 20
                ee = (vi + evi) * 20
                xs = verts[es:ee:5]
                ys = verts[es + 1:ee:5]
                x_mid = (float(np.min(xs)) + float(np.max(xs))) * 0.5
                y_mid = (float(np.min(ys)) + float(np.max(ys))) * 0.5
                verts[es:ee:5] -= x_mid
                verts[es + 1:ee:5] -= y_mid
            return vi, evi, scale
        lines = tr.text.split("\n")
        line_start_idxs = [0]
        line_advs = []
        pen_y = 0.0
        for line in lines:
            c, adv = self._build_line_quads(atlas, line, scale, 0.0, pen_y, verts, vi * 20, tr.italic, 0.0, tr.bold)
            vi += c
            line_start_idxs.append(vi)
            line_advs.append(adv)
            pen_y -= line_h
        max_w = max(line_advs) if line_advs else 0.0
        for i in range(len(lines)):
            if tr.alignment == TextAlign.LEFT:
                off_x = 0.0
            elif tr.alignment == TextAlign.CENTER:
                off_x = (max_w - line_advs[i]) * 0.5
            elif tr.alignment == TextAlign.RIGHT:
                off_x = max_w - line_advs[i]
            else:
                off_x = 0.0
            if off_x != 0.0:
                s = line_start_idxs[i] * 20
                e = line_start_idxs[i + 1] * 20
                verts[s:e:5] -= off_x
        if vi > 0:
            xs = verts[0:vi * 20:5]
            ys = verts[1:vi * 20:5]
            x_mid = (float(np.min(xs)) + float(np.max(xs))) * 0.5
            y_mid = (float(np.min(ys)) + float(np.max(ys))) * 0.5
            verts[0:vi * 20:5] -= x_mid
            verts[1:vi * 20:5] -= y_mid
        need_effects = tr.underline or tr.strikethrough
        if need_effects:
            pen_y = 0.0
            e_start_idxs = [0]
            for i, line in enumerate(lines):
                evi += self._build_effect_quads(atlas, line, scale, 0.0, pen_y, verts, (vi + evi) * 20, tr.underline, tr.strikethrough, 0.0, line_advs[i])
                e_start_idxs.append(evi)
                pen_y -= line_h
            for i in range(len(lines)):
                if tr.alignment == TextAlign.LEFT:
                    off_x = 0.0
                elif tr.alignment == TextAlign.CENTER:
                    off_x = (max_w - line_advs[i]) * 0.5
                elif tr.alignment == TextAlign.RIGHT:
                    off_x = max_w - line_advs[i]
                else:
                    off_x = 0.0
                if off_x != 0.0:
                    s = vi * 20 + e_start_idxs[i] * 20
                    e = vi * 20 + e_start_idxs[i + 1] * 20
                    verts[s:e:5] -= off_x
            if evi > 0:
                es = vi * 20
                ee = (vi + evi) * 20
                xs = verts[es:ee:5]
                ys = verts[es + 1:ee:5]
                x_mid = (float(np.min(xs)) + float(np.max(xs))) * 0.5
                y_mid = (float(np.min(ys)) + float(np.max(ys))) * 0.5
                verts[es:ee:5] -= x_mid
                verts[es + 1:ee:5] -= y_mid
        return vi, evi, scale

    _OUTLINE_DIRS = [
        (1.0, 0.0), (0.9239, 0.3827), (0.7071, 0.7071), (0.3827, 0.9239),
        (0.0, 1.0), (-0.3827, 0.9239), (-0.7071, 0.7071), (-0.9239, 0.3827),
        (-1.0, 0.0), (-0.9239, -0.3827), (-0.7071, -0.7071), (-0.3827, -0.9239),
        (0.0, -1.0), (0.3827, -0.9239), (0.7071, -0.7071), (0.9239, -0.3827),
    ]

    _GLOW_RINGS = [
        (1.0, 0.5),
        (1.5, 0.3),
        (2.5, 0.15),
    ]

    def render(self, scene, view_mat: Mat4, proj_mat: Mat4, viewport_w: int, viewport_h: int, world_space_only: bool | None = None):
        if not self._prog or not self._vao:
            return
        alive_ids = {e.id for e in scene.get_entities_with_component(TextRenderer)}
        stale = [eid for eid in self._geom_cache if eid not in alive_ids]
        for eid in stale:
            del self._geom_cache[eid]
        prog = self._prog
        view_f32 = view_mat.to_f32()
        proj_f32 = proj_mat.to_f32()
        zero3 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        if "u_view" in prog:
            prog["u_view"].write(view_f32.tobytes())
        if "u_proj" in prog:
            prog["u_proj"].write(proj_f32.tobytes())
        if "u_viewport_size" in prog:
            prog["u_viewport_size"].write(np.array([float(viewport_w), float(viewport_h)], dtype=np.float32).tobytes())
        for ent in scene.get_entities_with_component(TextRenderer):
            if not ent.active:
                continue
            tr = ent.get_component(TextRenderer)
            if not tr or not tr.enabled or not tr.text:
                continue
            if world_space_only is not None:
                if world_space_only and not tr.font_world_space:
                    continue
                if not world_space_only and tr.font_world_space:
                    continue
            t = ent.transform
            if not t:
                continue
            atlas = self._get_atlas(tr.font_path, tr.atlas_resolution)
            if atlas is None:
                continue
            tex = self._ensure_texture(atlas, tr.filter_mode, tr.anisotropy)
            self._ensure_capacity(len(tr.text))
            model_f32 = t.world_matrix.to_f32()
            if "u_model" in prog:
                prog["u_model"].write(model_f32.tobytes())
            if "u_billboard" in prog:
                prog["u_billboard"].value = 1.0 if tr.billboard else 0.0
            if "u_screen_space" in prog:
                prog["u_screen_space"].value = 0.0 if tr.font_world_space else 1.0
            gh = self._geom_hash(tr, atlas)
            cached = self._geom_cache.get(ent.id)
            if cached is not None and cached[4] == gh:
                vi, evi, scale, vdata, _, seg_info = cached
                self._verts[:len(vdata)] = vdata
                self._rich_seg_info = seg_info
            else:
                vi, evi, scale = self._build_text_verts(tr, atlas)
                self._geom_cache[ent.id] = (vi, evi, scale, self._verts[:(vi+evi)*20].copy(), gh, self._rich_seg_info.copy())
            if vi == 0 and evi == 0:
                continue
            total_verts = (vi + evi) * 4
            self._vbo.write(self._verts[:total_verts * 5].tobytes())
            ev = vi
            is_3d = tr.use_3d and tr.extrusion_layers > 0 and tr.extrusion_depth > 0
            has_shadow = tr.shadow and list(tr.shadow_color)[3] > 0
            has_outline = tr.outline and not is_3d and list(tr.outline_color)[3] > 0
            has_glow = tr.glow and not is_3d and list(tr.glow_color)[3] > 0
            has_3d = is_3d
            clip_main = 0.05 if (has_outline or has_glow) else 0.01
            if tr.font_world_space:
                self._ctx.enable(moderngl.DEPTH_TEST)
            else:
                self._ctx.disable(moderngl.DEPTH_TEST)
            if has_shadow:
                self._ctx.disable(moderngl.DEPTH_TEST)
                sx, sy = tr.shadow_offset[0], tr.shadow_offset[1]
                if "u_offset" in prog:
                    prog["u_offset"].write(np.array([-sx, -sy, 0.0], dtype=np.float32).tobytes())
                if vi > 0:
                    self._render_quads(vi, list(tr.shadow_color), tex, False, False, 0)
                if evi > 0:
                    self._render_quads(evi, list(tr.shadow_color), tex, False, True, ev)
                if "u_offset" in prog:
                    prog["u_offset"].write(zero3.tobytes())
                if tr.font_world_space:
                    self._ctx.enable(moderngl.DEPTH_TEST)
            write_depth = False
            if has_glow or has_outline:
                self._ctx.disable(moderngl.DEPTH_TEST)
            if has_glow:
                gc = list(tr.glow_color)
                base_intensity = gc[3] * tr.glow_intensity
                for radius, alpha_scale in self._GLOW_RINGS:
                    r = tr.glow_size * radius
                    a = base_intensity * alpha_scale
                    if a < 0.005:
                        continue
                    gc[3] = a
                    if "u_offset" in prog:
                        for dx, dy in self._OUTLINE_DIRS:
                            prog["u_offset"].write(np.array([dx * r, dy * r, 0.0], dtype=np.float32).tobytes())
                            if vi > 0:
                                self._render_quads(vi, gc, tex, write_depth, False, 0)
                            if evi > 0:
                                self._render_quads(evi, gc, tex, write_depth, False, ev)
                        prog["u_offset"].write(zero3.tobytes())
            if has_outline:
                oc = list(tr.outline_color)
                ow = tr.outline_width
                if "u_offset" in prog:
                    for dx, dy in self._OUTLINE_DIRS:
                        prog["u_offset"].write(np.array([dx * ow, dy * ow, 0.0], dtype=np.float32).tobytes())
                        if vi > 0:
                            self._render_quads(vi, oc, tex, write_depth, False, 0)
                        if evi > 0:
                            self._render_quads(evi, oc, tex, write_depth, False, ev)
                    prog["u_offset"].write(zero3.tobytes())
            if has_glow or has_outline:
                if tr.font_world_space:
                    self._ctx.enable(moderngl.DEPTH_TEST)
            if has_3d:
                layer_step = tr.extrusion_depth / max(tr.extrusion_layers, 1)
                if "u_offset" in prog:
                    for layer in range(tr.extrusion_layers, 0, -1):
                        z_off = layer * layer_step
                        t_factor = 0.3 + 0.7 * (1.0 - layer / max(tr.extrusion_layers, 1))
                        ecolor = [
                            tr.extrusion_color[0] * t_factor,
                            tr.extrusion_color[1] * t_factor,
                            tr.extrusion_color[2] * t_factor,
                            tr.color[3],
                        ]
                        prog["u_offset"].write(np.array([0.0, 0.0, -z_off], dtype=np.float32).tobytes())
                        if vi > 0:
                            self._render_quads(vi, ecolor, tex, write_depth, False, 0)
                        if evi > 0:
                            self._render_quads(evi, ecolor, tex, write_depth, True, ev)
                    prog["u_offset"].write(zero3.tobytes())
                if vi > 0:
                    self._render_quads(vi, list(tr.color), tex, write_depth, False, 0, clip_main)
                if evi > 0:
                    self._render_quads(evi, list(tr.color), tex, write_depth, True, ev)
            elif self._rich_seg_info:
                for seg in self._rich_seg_info:
                    seg_color = list(seg['color'])
                    seg_vc = seg['vi_count']
                    if seg_vc > 0:
                        self._render_quads(seg_vc, seg_color, tex, write_depth, False, seg['vi_start'], clip_main)
                    if seg['underline'] or seg['strikethrough']:
                        self._render_quads(seg['evi_count'], seg_color, tex, write_depth, True, ev + seg['evi_start'])
            else:
                if vi > 0:
                    self._render_quads(vi, list(tr.color), tex, write_depth, False, 0, clip_main)
                if evi > 0:
                    self._render_quads(evi, list(tr.color), tex, write_depth, True, ev)
        self._ctx.depth_mask = True
        self._ctx.enable(moderngl.CULL_FACE)

    def release(self):
        for tex in self._tex_cache.values():
            try:
                tex.release()
            except Exception:
                pass
        if self._vao:
            self._vao.release()
        if self._vbo:
            self._vbo.release()
        if self._ibo:
            self._ibo.release()
