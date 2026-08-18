from __future__ import annotations

import os
import numpy as np
import moderngl
from typing import Optional

from core.shaders.compute_shader import compile_compute_shader


class BVHDebugRenderer:
    def __init__(self):
        self._prog: Optional[moderngl.ComputeShader] = None
        self._line_prog: Optional[moderngl.Program] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._counter_buf: Optional[moderngl.Buffer] = None
        self._depth_buf: Optional[moderngl.Buffer] = None
        self._bvh_buf: Optional[moderngl.Buffer] = None
        self._ctx_id = 0

    def render(self, ctx: moderngl.Context, bvh, world_matrix, vp_mat,
               max_depth: int = 4):
        if not bvh or not bvh.nodes or len(bvh.nodes) == 0:
            return
        ctx_id = id(ctx)
        if self._ctx_id != ctx_id:
            self._release(ctx)
            self._ctx_id = ctx_id
        nn = len(bvh.nodes)
        flat = np.asarray(bvh.flatten_for_gpu(), dtype=np.float32)
        nn = flat.shape[0]
        if nn == 0:
            return
        if not self._ensure_progs(ctx):
            return
        self._ensure_buffers(ctx, nn, flat, bvh)
        self._counter_buf.write(np.zeros(1, dtype=np.uint32).tobytes())
        self._depth_buf.bind_to_storage_buffer(6)
        self._vbo.bind_to_storage_buffer(7)
        self._counter_buf.bind_to_storage_buffer(8)
        self._bvh_buf.bind_to_storage_buffer(0)
        try:
            self._prog["u_bvh_offset"] = 0
            self._prog["u_node_count"] = nn
            self._prog["u_max_depth"] = max_depth
            self._prog["u_vp_matrix"].write(vp_mat.to_f32().tobytes())
            self._prog["u_world_matrix"].write(world_matrix.to_f32().tobytes())
        except KeyError:
            return
        groups = (nn + 255) // 256
        self._prog.run(group_x=groups, group_y=1, group_z=1)
        ctx.memory_barrier(moderngl.ALL_BARRIER_BITS)
        counter_data = self._counter_buf.read()
        line_vert_count = int(np.frombuffer(counter_data, dtype=np.uint32)[0])
        if line_vert_count == 0:
            return
        old_cull = bool(ctx.cull_face)
        ctx.disable(moderngl.CULL_FACE)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._vao.render(moderngl.LINES, vertices=line_vert_count)
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST)
        if old_cull:
            ctx.enable(moderngl.CULL_FACE)

    def _ensure_progs(self, ctx: moderngl.Context) -> bool:
        if self._prog is None:
            path = os.path.abspath("core/shaders/BVHDebug.compute")
            if not os.path.exists(path):
                return False
            try:
                with open(path) as f:
                    src = f.read()
                glsl_start = src.find("GLSLPROGRAM")
                glsl_end = src.find("ENDGLSL", glsl_start)
                if glsl_start < 0 or glsl_end < 0:
                    return False
                source = src[glsl_start + len("GLSLPROGRAM"):glsl_end].strip()
                self._prog = compile_compute_shader(ctx, source, path)
                if self._prog is None:
                    return False
            except Exception:
                return False
        if self._line_prog is None:
            try:
                self._line_prog = ctx.program(
                    vertex_shader="""
                    #version 460 core
                    in vec4 in_position;
                    in vec4 in_color;
                    out vec4 v_color;
                    void main() {
                        gl_Position = in_position;
                        v_color = in_color;
                    }
                    """,
                    fragment_shader="""
                    #version 460 core
                    in vec4 v_color;
                    out vec4 frag_color;
                    void main() {
                        frag_color = v_color;
                    }
                    """,
                )
            except Exception:
                return False
        return True

    def _ensure_buffers(self, ctx: moderngl.Context, nn: int, flat_np: np.ndarray, bvh):
        if self._counter_buf is None:
            self._counter_buf = ctx.buffer(reserve=4)
        depth_np = np.array(bvh.node_depths, dtype=np.int32)
        if self._depth_buf is not None:
            self._depth_buf.release()
        self._depth_buf = ctx.buffer(depth_np.tobytes())
        needed_vbo = nn * 24 * 32
        if self._vbo is None or self._vbo.size < needed_vbo:
            if self._vbo is not None:
                self._vbo.release()
            self._vbo = ctx.buffer(reserve=max(needed_vbo, 4096))
        if self._vao is None:
            self._vao = ctx.vertex_array(
                self._line_prog,
                [(self._vbo, "4f 4f", "in_position", "in_color")],
            )
        if self._bvh_buf is not None:
            self._bvh_buf.release()
        self._bvh_buf = ctx.buffer(flat_np.tobytes())

    def _release(self, ctx: moderngl.Context):
        for buf in [self._vbo, self._counter_buf, self._depth_buf, self._bvh_buf]:
            if buf:
                try:
                    buf.release()
                except Exception:
                    pass
        self._vbo = self._counter_buf = self._depth_buf = self._bvh_buf = None
        if self._vao:
            try:
                self._vao.release()
            except Exception:
                pass
            self._vao = None
        if self._prog:
            try:
                self._prog.release()
            except Exception:
                pass
            self._prog = None
        if self._line_prog:
            try:
                self._line_prog.release()
            except Exception:
                pass
            self._line_prog = None
