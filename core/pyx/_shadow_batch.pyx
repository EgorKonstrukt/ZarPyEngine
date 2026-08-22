# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, tan, floor, ceil, cos, sin, fabs

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef np.float32_t FLOAT32_t

cdef extern from "math.h":
    double fmax(double, double) nogil

cdef inline double _max3(double a, double b, double c) nogil:
    if a > b:
        return a if a > c else c
    return b if b > c else c

cdef inline double _min3(double a, double b, double c) nogil:
    if a < b:
        return a if a < c else c
    return b if b < c else c


def compute_frustum_corners(float near_z, float far_z, float cam_fov, float aspect,
                            np.ndarray[DTYPE_t, ndim=2] inv_view):
    cdef float tan_half_fov = tan(cam_fov * 0.017453292519943295 * 0.5)
    cdef float half_h, half_w
    cdef float view_pt[4]
    cdef float world_pt[4]
    cdef float w_inv
    cdef int x_sign, y_sign, zi, idx
    cdef list corners = []
    cdef DTYPE_t iv00 = inv_view[0, 0], iv01 = inv_view[0, 1], iv02 = inv_view[0, 2], iv03 = inv_view[0, 3]
    cdef DTYPE_t iv10 = inv_view[1, 0], iv11 = inv_view[1, 1], iv12 = inv_view[1, 2], iv13 = inv_view[1, 3]
    cdef DTYPE_t iv20 = inv_view[2, 0], iv21 = inv_view[2, 1], iv22 = inv_view[2, 2], iv23 = inv_view[2, 3]
    cdef DTYPE_t iv30 = inv_view[3, 0], iv31 = inv_view[3, 1], iv32 = inv_view[3, 2], iv33 = inv_view[3, 3]
    cdef double z_vals[2]
    z_vals[0] = <double>near_z
    z_vals[1] = <double>far_z
    cdef double res[3]
    for zi in range(2):
        half_h = tan_half_fov * <float>z_vals[zi]
        half_w = half_h * aspect
        for y_sign in (-1, 1):
            for x_sign in (-1, 1):
                view_pt[0] = x_sign * half_w
                view_pt[1] = y_sign * half_h
                view_pt[2] = -<float>z_vals[zi]
                view_pt[3] = 1.0
                world_pt[0] = view_pt[0] * iv00 + view_pt[1] * iv10 + view_pt[2] * iv20 + view_pt[3] * iv30
                world_pt[1] = view_pt[0] * iv01 + view_pt[1] * iv11 + view_pt[2] * iv21 + view_pt[3] * iv31
                world_pt[2] = view_pt[0] * iv02 + view_pt[1] * iv12 + view_pt[2] * iv22 + view_pt[3] * iv32
                world_pt[3] = view_pt[0] * iv03 + view_pt[1] * iv13 + view_pt[2] * iv23 + view_pt[3] * iv33
                w_inv = 1.0 / world_pt[3] if world_pt[3] != 0.0 else 0.0
                res[0] = world_pt[0] * w_inv
                res[1] = world_pt[1] * w_inv
                res[2] = world_pt[2] * w_inv
                corners.append(np.array([res[0], res[1], res[2]], dtype=np.float64))
    return corners


def build_directional_cascade(float light_dir_x, float light_dir_y, float light_dir_z,
                               list corners, float depth_span, int shadow_resolution):
    cdef int n = len(corners)
    cdef double cx = 0.0, cy = 0.0, cz = 0.0
    cdef int i
    cdef double dx, dy, dz, r2, radius2 = 0.0
    cdef double radius
    for i in range(n):
        c_arr = corners[i]
        cx += c_arr[0]
        cy += c_arr[1]
        cz += c_arr[2]
    cx /= n
    cy /= n
    cz /= n
    for i in range(n):
        c_arr = corners[i]
        dx = c_arr[0] - cx
        dy = c_arr[1] - cy
        dz = c_arr[2] - cz
        r2 = dx * dx + dy * dy + dz * dz
        if r2 > radius2:
            radius2 = r2
    radius = sqrt(radius2)
    if radius < 0.25:
        radius = 0.25
    radius = ceil(radius * 16.0) / 16.0
    cdef double light_x = cx - light_dir_x * max(radius * 2.0, depth_span + 10.0)
    cdef double light_y = cy - light_dir_y * max(radius * 2.0, depth_span + 10.0)
    cdef double light_z = cz - light_dir_z * max(radius * 2.0, depth_span + 10.0)
    cdef double up_x = 0.0, up_y = 1.0, up_z = 0.0
    cdef double dot_val = light_dir_x * up_x + light_dir_y * up_y + light_dir_z * up_z
    if fabs(dot_val) > 0.999:
        up_x = 0.0; up_y = 0.0; up_z = 1.0
    cdef double eye_x = light_x, eye_y = light_y, eye_z = light_z
    cdef double center_x = cx, center_y = cy, center_z = cz
    cdef double fx = center_x - eye_x, fy = center_y - eye_y, fz = center_z - eye_z
    cdef double fl = sqrt(fx * fx + fy * fy + fz * fz)
    if fl < 1e-10:
        fl = 1e-10
    fx /= fl; fy /= fl; fz /= fl
    cdef double sx = fy * up_z - fz * up_y
    cdef double sy = fz * up_x - fx * up_z
    cdef double sz = fx * up_y - fy * up_x
    cdef double sl = sqrt(sx * sx + sy * sy + sz * sz)
    if sl < 1e-10:
        sx = 1.0; sy = 0.0; sz = 0.0; sl = 1.0
    sx /= sl; sy /= sl; sz /= sl
    cdef double ux = fy * sz - fz * sy
    cdef double uy = fz * sx - fx * sz
    cdef double uz = fx * sy - fy * sx
    cdef np.ndarray[DTYPE_t, ndim=2] view = np.eye(4, dtype=np.float64)
    view[0, 0] = sx; view[0, 1] = sy; view[0, 2] = sz; view[0, 3] = -(sx * eye_x + sy * eye_y + sz * eye_z)
    view[1, 0] = ux; view[1, 1] = uy; view[1, 2] = uz; view[1, 3] = -(ux * eye_x + uy * eye_y + uz * eye_z)
    view[2, 0] = -fx; view[2, 1] = -fy; view[2, 2] = -fz; view[2, 3] = (fx * eye_x + fy * eye_y + fz * eye_z)
    view[3, 0] = 0.0; view[3, 1] = 0.0; view[3, 2] = 0.0; view[3, 3] = 1.0
    cdef double m00 = view[0,0], m01 = view[0,1], m02 = view[0,2], m03 = view[0,3]
    cdef double m10 = view[1,0], m11 = view[1,1], m12 = view[1,2], m13 = view[1,3]
    cdef double m20 = view[2,0], m21 = view[2,1], m22 = view[2,2], m23 = view[2,3]
    cdef double m30 = view[3,0], m31 = view[3,1], m32 = view[3,2], m33 = view[3,3]
    cdef double cl_x = cx * m00 + cy * m10 + cz * m20 + m30
    cdef double cl_y = cx * m01 + cy * m11 + cz * m21 + m31
    cdef double texel_size = (radius * 2.0) / max(1, shadow_resolution)
    cdef double cx_l = floor(cl_x / texel_size) * texel_size
    cdef double cy_l = floor(cl_y / texel_size) * texel_size
    cdef double left = cx_l - radius
    cdef double right = cx_l + radius
    cdef double bottom = cy_l - radius
    cdef double top = cy_l + radius
    cdef double min_z = 1e300, max_z = -1e300
    cdef double px, py, pz, pw
    for i in range(n):
        c_arr = corners[i]
        px = c_arr[0] * m00 + c_arr[1] * m10 + c_arr[2] * m20 + m30
        py = c_arr[0] * m01 + c_arr[1] * m11 + c_arr[2] * m21 + m31
        pz = c_arr[0] * m02 + c_arr[1] * m12 + c_arr[2] * m22 + m32
        pw = c_arr[0] * m03 + c_arr[1] * m13 + c_arr[2] * m23 + m33
        pz = pz / pw if pw != 0.0 else 0.0
        if pz < min_z:
            min_z = pz
        if pz > max_z:
            max_z = pz
    cdef double z_margin = max(depth_span * 0.45, 6.0)
    cdef double n_val = max(-max_z - z_margin, 0.01)
    cdef double f_val = max(-min_z + z_margin, n_val + 0.01)
    cdef np.ndarray[DTYPE_t, ndim=2] proj = np.eye(4, dtype=np.float64)
    proj[0, 0] = 2.0 / (right - left)
    proj[0, 3] = -(right + left) / (right - left)
    proj[1, 1] = 2.0 / (top - bottom)
    proj[1, 3] = -(top + bottom) / (top - bottom)
    proj[2, 2] = 2.0 / (f_val - n_val)
    proj[2, 3] = -(f_val + n_val) / (f_val - n_val)
    cdef np.ndarray[DTYPE_t, ndim=2] vp = view @ proj
    return vp


def compute_cascade_distances(float cam_near, float cam_far, float shadow_distance):
    cdef float near_z = max(cam_near, 0.01)
    cdef float far_z = max(near_z + 0.1, min(cam_far, shadow_distance))
    cdef float span = far_z - near_z
    if span <= 0.1:
        return [far_z, far_z, far_z, far_z]
    cdef float l0 = near_z * pow(far_z / near_z if near_z>0 else 1, 0.25) if near_z>0 else near_z + span*0.25
    cdef float l1 = near_z * pow(far_z / near_z if near_z>0 else 1, 0.5) if near_z>0 else near_z + span*0.5
    cdef float l2 = near_z * pow(far_z / near_z if near_z>0 else 1, 0.75) if near_z>0 else near_z + span*0.75
    cdef float u0 = near_z + span*0.25
    cdef float u1 = near_z + span*0.5
    cdef float u2 = near_z + span*0.75
    cdef float lam = 0.85
    cdef float s0 = lam*l0 + (1-lam)*u0
    cdef float s1 = lam*l1 + (1-lam)*u1
    cdef float s2 = lam*l2 + (1-lam)*u2
    return [s0, s1, s2, far_z]


def pack_cascade_matrices_f32(list light_space_matrices, np.ndarray[FLOAT32_t, ndim=3] out_buf):
    cdef int ci, r, c
    cdef object vp
    cdef object d
    for ci in range(4):
        vp = light_space_matrices[ci]
        d = vp._d if hasattr(vp, '_d') else vp
        for r in range(4):
            for c in range(4):
                out_buf[ci, r, c] = <FLOAT32_t>d[r, c]


def pack_cascade_splits_f32(list splits, np.ndarray[FLOAT32_t, ndim=1] out_buf):
    out_buf[0] = <FLOAT32_t>splits[0]
    out_buf[1] = <FLOAT32_t>splits[1]
    out_buf[2] = <FLOAT32_t>splits[2]
    out_buf[3] = <FLOAT32_t>splits[3]


def pack_point_vps_f32(list point_light_vps, np.ndarray[FLOAT32_t, ndim=3] out_buf):
    cdef int fi, r, c
    cdef object vp
    cdef object d
    for fi in range(6):
        vp = point_light_vps[fi]
        d = vp._d if hasattr(vp, '_d') else vp
        for r in range(4):
            for c in range(4):
                out_buf[fi, r, c] = <FLOAT32_t>d[r, c]


def build_shadow_groups(list renderable_shadow):
    cdef dict groups = {}
    cdef int i
    cdef object mesh, tr
    cdef unsigned long long mesh_id
    for i in range(len(renderable_shadow)):
        mesh, tr = renderable_shadow[i]
        mesh_id = <unsigned long long><void *>mesh
        if mesh_id in groups:
            groups[mesh_id].append((mesh, tr))
        else:
            groups[mesh_id] = [(mesh, tr)]
    return groups


def face_cull_point_shadow(float frag_x, float frag_y, float frag_z,
                           float lp_x, float lp_y, float lp_z):
    cdef float dx = frag_x - lp_x
    cdef float dy = frag_y - lp_y
    cdef float dz = frag_z - lp_z
    cdef float ax = fabs(dx)
    cdef float ay = fabs(dy)
    cdef float az = fabs(dz)
    cdef int face
    if ax >= ay and ax >= az:
        face = 0 if dx >= 0 else 1
    elif ay >= az:
        face = 2 if dy >= 0 else 3
    else:
        face = 4 if dz >= 0 else 5
    return face


def cull_shadow_groups_for_vp(dict groups, np.ndarray[FLOAT32_t, ndim=2] vp_f32):
    cdef np.ndarray[FLOAT32_t, ndim=2] planes = np.empty((6, 4), dtype=np.float32)
    cdef int r, c, i, pi
    cdef FLOAT32_t px, py, pz, dist, radius, sx, sy, sz, cx, cy, cz
    cdef FLOAT32_t p0, p1, p2, p3, norm
    cdef FLOAT32_t min_dist
    cdef FLOAT32_t c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33
    c00 = vp_f32[0, 0]; c01 = vp_f32[1, 0]; c02 = vp_f32[2, 0]; c03 = vp_f32[3, 0]
    c10 = vp_f32[0, 1]; c11 = vp_f32[1, 1]; c12 = vp_f32[2, 1]; c13 = vp_f32[3, 1]
    c20 = vp_f32[0, 2]; c21 = vp_f32[1, 2]; c22 = vp_f32[2, 2]; c23 = vp_f32[3, 2]
    c30 = vp_f32[0, 3]; c31 = vp_f32[1, 3]; c32 = vp_f32[2, 3]; c33 = vp_f32[3, 3]
    for pi in range(6):
        if pi == 0:
            p0 = c30+c00; p1 = c31+c01; p2 = c32+c02; p3 = c33+c03
        elif pi == 1:
            p0 = c30-c00; p1 = c31-c01; p2 = c32-c02; p3 = c33-c03
        elif pi == 2:
            p0 = c30+c10; p1 = c31+c11; p2 = c32+c12; p3 = c33+c13
        elif pi == 3:
            p0 = c30-c10; p1 = c31-c11; p2 = c32-c12; p3 = c33-c13
        elif pi == 4:
            p0 = c30+c20; p1 = c31+c21; p2 = c32+c22; p3 = c33+c23
        else:
            p0 = c30-c20; p1 = c31-c21; p2 = c32-c22; p3 = c33-c23
        norm = sqrt(p0*p0 + p1*p1 + p2*p2)
        if norm < 1e-10:
            norm = 1.0
        planes[pi, 0] = p0 / norm
        planes[pi, 1] = p1 / norm
        planes[pi, 2] = p2 / norm
        planes[pi, 3] = p3 / norm
    result = {}
    for mesh_id, group in groups.items():
        visible = []
        for mesh, tr in group:
            wm = tr.world_matrix._d
            cx = <FLOAT32_t>wm[3, 0]
            cy = <FLOAT32_t>wm[3, 1]
            cz = <FLOAT32_t>wm[3, 2]
            sx = sqrt(<FLOAT32_t>(wm[0,0]*wm[0,0] + wm[1,0]*wm[1,0] + wm[2,0]*wm[2,0]))
            sy = sqrt(<FLOAT32_t>(wm[0,1]*wm[0,1] + wm[1,1]*wm[1,1] + wm[2,1]*wm[2,1]))
            sz = sqrt(<FLOAT32_t>(wm[0,2]*wm[0,2] + wm[1,2]*wm[1,2] + wm[2,2]*wm[2,2]))
            radius = sx
            if sy > radius: radius = sy
            if sz > radius: radius = sz
            radius = radius * <FLOAT32_t>getattr(mesh, 'bounding_radius', 10.0)
            min_dist = 1e30
            for pi in range(6):
                dist = planes[pi, 0] * cx + planes[pi, 1] * cy + planes[pi, 2] * cz + planes[pi, 3]
                if dist < min_dist:
                    min_dist = dist
                if dist < -radius:
                    break
            else:
                visible.append((mesh, tr))
        if visible:
            result[mesh_id] = visible
    return result


def pack_model_matrices_f32(list model_matrices, FLOAT32_t[::1] out):
    cdef int n = len(model_matrices)
    cdef int idx = 0
    cdef int r, c
    cdef object mat
    cdef object d
    for i in range(n):
        mat = model_matrices[i]
        d = mat._d if hasattr(mat, '_d') else mat
        for r in range(4):
            for c in range(4):
                out[idx] = <FLOAT32_t>d[r, c]
                idx += 1


def compute_frustum_corners_out(float near_z, float far_z, float cam_fov, float aspect,
                                np.ndarray[DTYPE_t, ndim=2] inv_view,
                                np.ndarray[DTYPE_t, ndim=2] out_corners):
    cdef float tan_half_fov = tan(cam_fov * 0.017453292519943295 * 0.5)
    cdef float half_h, half_w
    cdef float view_pt[4]
    cdef float world_pt[4]
    cdef float w_inv
    cdef int x_sign, y_sign, zi, idx = 0
    cdef DTYPE_t iv00 = inv_view[0,0], iv01 = inv_view[0,1], iv02 = inv_view[0,2], iv03 = inv_view[0,3]
    cdef DTYPE_t iv10 = inv_view[1,0], iv11 = inv_view[1,1], iv12 = inv_view[1,2], iv13 = inv_view[1,3]
    cdef DTYPE_t iv20 = inv_view[2,0], iv21 = inv_view[2,1], iv22 = inv_view[2,2], iv23 = inv_view[2,3]
    cdef DTYPE_t iv30 = inv_view[3,0], iv31 = inv_view[3,1], iv32 = inv_view[3,2], iv33 = inv_view[3,3]
    cdef double z_vals[2]
    z_vals[0] = <double>near_z
    z_vals[1] = <double>far_z
    for zi in range(2):
        half_h = tan_half_fov * <float>z_vals[zi]
        half_w = half_h * aspect
        for y_sign in (-1, 1):
            for x_sign in (-1, 1):
                view_pt[0] = x_sign * half_w
                view_pt[1] = y_sign * half_h
                view_pt[2] = -<float>z_vals[zi]
                view_pt[3] = 1.0
                world_pt[0] = view_pt[0]*iv00 + view_pt[1]*iv10 + view_pt[2]*iv20 + view_pt[3]*iv30
                world_pt[1] = view_pt[0]*iv01 + view_pt[1]*iv11 + view_pt[2]*iv21 + view_pt[3]*iv31
                world_pt[2] = view_pt[0]*iv02 + view_pt[1]*iv12 + view_pt[2]*iv22 + view_pt[3]*iv32
                world_pt[3] = view_pt[0]*iv03 + view_pt[1]*iv13 + view_pt[2]*iv23 + view_pt[3]*iv33
                w_inv = 1.0 / world_pt[3] if world_pt[3] != 0.0 else 0.0
                out_corners[idx, 0] = world_pt[0] * w_inv
                out_corners[idx, 1] = world_pt[1] * w_inv
                out_corners[idx, 2] = world_pt[2] * w_inv
                idx += 1


def build_directional_cascade_fast(float ld_x, float ld_y, float ld_z,
                                   np.ndarray[DTYPE_t, ndim=2] corners,
                                   float depth_span, int shadow_res,
                                   np.ndarray[DTYPE_t, ndim=2] out_vp):
    cdef int n = corners.shape[0]
    cdef double cx = 0.0, cy = 0.0, cz = 0.0
    cdef int i, j
    cdef double dx, dy, dz, r2, radius2 = 0.0
    for i in range(n):
        cx += corners[i, 0]
        cy += corners[i, 1]
        cz += corners[i, 2]
    cx /= n
    cy /= n
    cz /= n
    for i in range(n):
        dx = corners[i, 0] - cx
        dy = corners[i, 1] - cy
        dz = corners[i, 2] - cz
        r2 = dx * dx + dy * dy + dz * dz
        if r2 > radius2:
            radius2 = r2
    cdef double radius = sqrt(radius2)
    if radius < 0.25:
        radius = 0.25
    radius = ceil(radius * 16.0) / 16.0
    cdef double ext = max(radius * 2.0, depth_span + 10.0)
    cdef double eye_x = cx - ld_x * ext
    cdef double eye_y = cy - ld_y * ext
    cdef double eye_z = cz - ld_z * ext
    cdef double up_x = 0.0, up_y = 1.0, up_z = 0.0
    cdef double dot_val = ld_x * up_x + ld_y * up_y + ld_z * up_z
    if fabs(dot_val) > 0.999:
        up_x = 0.0
        up_y = 0.0
        up_z = 1.0
    cdef double fx = cx - eye_x, fy = cy - eye_y, fz = cz - eye_z
    cdef double fl = sqrt(fx * fx + fy * fy + fz * fz)
    if fl < 1e-10:
        fl = 1e-10
    fx /= fl
    fy /= fl
    fz /= fl
    cdef double sx = fy * up_z - fz * up_y
    cdef double sy = fz * up_x - fx * up_z
    cdef double sz = fx * up_y - fy * up_x
    cdef double sl = sqrt(sx * sx + sy * sy + sz * sz)
    if sl < 1e-10:
        sx = 1.0
        sy = 0.0
        sz = 0.0
        sl = 1.0
    sx /= sl
    sy /= sl
    sz /= sl
    cdef double ux = sy * fz - sz * fy
    cdef double uy = sz * fx - sx * fz
    cdef double uz = sx * fy - sy * fx
    cdef double v03 = -(sx * eye_x + sy * eye_y + sz * eye_z)
    cdef double v13 = -(ux * eye_x + uy * eye_y + uz * eye_z)
    cdef double v23 = fx * eye_x + fy * eye_y + fz * eye_z
    cdef double cl_x = cx * sx + cy * sy + cz * sz + v03
    cdef double cl_y = cx * ux + cy * uy + cz * uz + v13
    cdef double texel_size = (radius * 2.0) / max(1, shadow_res)
    cdef double cx_l = floor(cl_x / texel_size) * texel_size
    cdef double cy_l = floor(cl_y / texel_size) * texel_size
    cdef double left = cx_l - radius
    cdef double right = cx_l + radius
    cdef double bottom = cy_l - radius
    cdef double top = cy_l + radius
    cdef double min_z = 1e300, max_z = -1e300
    cdef double pz
    for i in range(n):
        pz = corners[i, 0] * (-fx) + corners[i, 1] * (-fy) + corners[i, 2] * (-fz) + v23
        if pz < min_z:
            min_z = pz
        if pz > max_z:
            max_z = pz
    cdef double z_margin = max(depth_span * 0.45, 6.0)
    cdef double n_val = max(-max_z - z_margin, 0.01)
    cdef double f_val = max(-min_z + z_margin, n_val + 0.01)
    cdef double inv_rl = 1.0 / (right - left)
    cdef double inv_tb = 1.0 / (top - bottom)
    cdef double inv_fn = 1.0 / (f_val - n_val)
    cdef double sum_rl = right + left
    cdef double sum_tb = top + bottom
    cdef double sum_fn = f_val + n_val
    cdef double p00 = 2.0 * inv_rl
    cdef double p11 = 2.0 * inv_tb
    cdef double p22 = -2.0 * inv_fn
    cdef double p30 = -sum_rl * inv_rl
    cdef double p31 = -sum_tb * inv_tb
    cdef double p32 = -sum_fn * inv_fn
    out_vp[0, 0] = sx * p00
    out_vp[0, 1] = ux * p11
    out_vp[0, 2] = -fx * p22
    out_vp[0, 3] = 0.0
    out_vp[1, 0] = sy * p00
    out_vp[1, 1] = uy * p11
    out_vp[1, 2] = -fy * p22
    out_vp[1, 3] = 0.0
    out_vp[2, 0] = sz * p00
    out_vp[2, 1] = uz * p11
    out_vp[2, 2] = -fz * p22
    out_vp[2, 3] = 0.0
    out_vp[3, 0] = v03 * p00 + p30
    out_vp[3, 1] = v13 * p11 + p31
    out_vp[3, 2] = v23 * p22 + p32
    out_vp[3, 3] = 1.0


def frustum_cull_shadow_groups(dict groups, np.ndarray[FLOAT32_t, ndim=2] vp_f32):
    cdef FLOAT32_t c00 = vp_f32[0,0], c01 = vp_f32[1,0], c02 = vp_f32[2,0], c03 = vp_f32[3,0]
    cdef FLOAT32_t c10 = vp_f32[0,1], c11 = vp_f32[1,1], c12 = vp_f32[2,1], c13 = vp_f32[3,1]
    cdef FLOAT32_t c20 = vp_f32[0,2], c21 = vp_f32[1,2], c22 = vp_f32[2,2], c23 = vp_f32[3,2]
    cdef FLOAT32_t c30 = vp_f32[0,3], c31 = vp_f32[1,3], c32 = vp_f32[2,3], c33 = vp_f32[3,3]
    cdef FLOAT32_t p0, p1, p2, p3, norm
    cdef FLOAT32_t plane00, plane01, plane02, plane03
    cdef FLOAT32_t plane10, plane11, plane12, plane13
    cdef FLOAT32_t plane20, plane21, plane22, plane23
    cdef FLOAT32_t plane30, plane31, plane32, plane33
    cdef FLOAT32_t plane40, plane41, plane42, plane43
    cdef FLOAT32_t plane50, plane51, plane52, plane53

    p0 = c30+c00; p1 = c31+c01; p2 = c32+c02; p3 = c33+c03
    norm = sqrt(p0*p0 + p1*p1 + p2*p2)
    if norm < 1e-10: norm = 1.0
    plane00 = p0/norm; plane01 = p1/norm; plane02 = p2/norm; plane03 = p3/norm

    p0 = c30-c00; p1 = c31-c01; p2 = c32-c02; p3 = c33-c03
    norm = sqrt(p0*p0 + p1*p1 + p2*p2)
    if norm < 1e-10: norm = 1.0
    plane10 = p0/norm; plane11 = p1/norm; plane12 = p2/norm; plane13 = p3/norm

    p0 = c30+c10; p1 = c31+c11; p2 = c32+c12; p3 = c33+c13
    norm = sqrt(p0*p0 + p1*p1 + p2*p2)
    if norm < 1e-10: norm = 1.0
    plane20 = p0/norm; plane21 = p1/norm; plane22 = p2/norm; plane23 = p3/norm

    p0 = c30-c10; p1 = c31-c11; p2 = c32-c12; p3 = c33-c13
    norm = sqrt(p0*p0 + p1*p1 + p2*p2)
    if norm < 1e-10: norm = 1.0
    plane30 = p0/norm; plane31 = p1/norm; plane32 = p2/norm; plane33 = p3/norm

    p0 = c30+c20; p1 = c31+c21; p2 = c32+c22; p3 = c33+c23
    norm = sqrt(p0*p0 + p1*p1 + p2*p2)
    if norm < 1e-10: norm = 1.0
    plane40 = p0/norm; plane41 = p1/norm; plane42 = p2/norm; plane43 = p3/norm

    p0 = c30-c20; p1 = c31-c21; p2 = c32-c22; p3 = c33-c23
    norm = sqrt(p0*p0 + p1*p1 + p2*p2)
    if norm < 1e-10: norm = 1.0
    plane50 = p0/norm; plane51 = p1/norm; plane52 = p2/norm; plane53 = p3/norm

    cdef dict result = {}
    cdef FLOAT32_t cx, cy, cz, sx, sy, sz, radius, dist
    cdef unsigned long long mesh_id
    cdef list visible
    cdef object mesh, tr, wm, group

    for mesh_id, group in groups.items():
        visible = []
        for mesh, tr in group:
            wm = tr.world_matrix._d
            cx = <FLOAT32_t>wm[3, 0]
            cy = <FLOAT32_t>wm[3, 1]
            cz = <FLOAT32_t>wm[3, 2]
            sx = sqrt(<FLOAT32_t>(wm[0,0]*wm[0,0] + wm[1,0]*wm[1,0] + wm[2,0]*wm[2,0]))
            sy = sqrt(<FLOAT32_t>(wm[0,1]*wm[0,1] + wm[1,1]*wm[1,1] + wm[2,1]*wm[2,1]))
            sz = sqrt(<FLOAT32_t>(wm[0,2]*wm[0,2] + wm[1,2]*wm[1,2] + wm[2,2]*wm[2,2]))
            radius = sx
            if sy > radius: radius = sy
            if sz > radius: radius = sz
            radius = radius * <FLOAT32_t>getattr(mesh, 'bounding_radius', 10.0)

            dist = plane00*cx + plane01*cy + plane02*cz + plane03
            if dist < -radius: continue
            dist = plane10*cx + plane11*cy + plane12*cz + plane13
            if dist < -radius: continue
            dist = plane20*cx + plane21*cy + plane22*cz + plane23
            if dist < -radius: continue
            dist = plane30*cx + plane31*cy + plane32*cz + plane33
            if dist < -radius: continue
            dist = plane40*cx + plane41*cy + plane42*cz + plane43
            if dist < -radius: continue
            dist = plane50*cx + plane51*cy + plane52*cz + plane53
            if dist < -radius: continue
            visible.append((mesh, tr))
        if visible:
            result[mesh_id] = visible
    return result


def pack_cascade_vps_transposed(np.ndarray[DTYPE_t, ndim=3] source,
                                np.ndarray[FLOAT32_t, ndim=3] out_col_major):
    cdef int ci, r, c
    for ci in range(4):
        for r in range(4):
            for c in range(4):
                out_col_major[ci, c, r] = <FLOAT32_t>source[ci, r, c]


def pack_cascade_vps_for_gpu(list matrices, np.ndarray[FLOAT32_t, ndim=3] out_col_major):
    cdef int ci, r, c
    cdef object d
    for ci in range(4):
        d = matrices[ci]._d if hasattr(matrices[ci], '_d') else matrices[ci]
        for r in range(4):
            for c in range(4):
                out_col_major[ci, c, r] = <FLOAT32_t>d[r, c]
