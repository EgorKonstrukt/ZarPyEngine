# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef np.float32_t F32_t


cdef inline void _mat4_mul_flat(const DTYPE_t* l, const DTYPE_t* r,
                                 DTYPE_t* o) noexcept nogil:
    o[0]  = l[0]*r[0]  + l[1]*r[4]  + l[2]*r[8]  + l[3]*r[12]
    o[1]  = l[0]*r[1]  + l[1]*r[5]  + l[2]*r[9]  + l[3]*r[13]
    o[2]  = l[0]*r[2]  + l[1]*r[6]  + l[2]*r[10] + l[3]*r[14]
    o[3]  = l[0]*r[3]  + l[1]*r[7]  + l[2]*r[11] + l[3]*r[15]
    o[4]  = l[4]*r[0]  + l[5]*r[4]  + l[6]*r[8]  + l[7]*r[12]
    o[5]  = l[4]*r[1]  + l[5]*r[5]  + l[6]*r[9]  + l[7]*r[13]
    o[6]  = l[4]*r[2]  + l[5]*r[6]  + l[6]*r[10] + l[7]*r[14]
    o[7]  = l[4]*r[3]  + l[5]*r[7]  + l[6]*r[11] + l[7]*r[15]
    o[8]  = l[8]*r[0]  + l[9]*r[4]  + l[10]*r[8] + l[11]*r[12]
    o[9]  = l[8]*r[1]  + l[9]*r[5]  + l[10]*r[9] + l[11]*r[13]
    o[10] = l[8]*r[2]  + l[9]*r[6]  + l[10]*r[10]+ l[11]*r[14]
    o[11] = l[8]*r[3]  + l[9]*r[7]  + l[10]*r[11]+ l[11]*r[15]
    o[12] = l[12]*r[0] + l[13]*r[4] + l[14]*r[8] + l[15]*r[12]
    o[13] = l[12]*r[1] + l[13]*r[5] + l[14]*r[9] + l[15]*r[13]
    o[14] = l[12]*r[2] + l[13]*r[6] + l[14]*r[10]+ l[15]*r[14]
    o[15] = l[12]*r[3] + l[13]*r[7] + l[14]*r[11]+ l[15]*r[15]


cdef inline void _mat4_inv_flat(const DTYPE_t* m, DTYPE_t* inv) noexcept nogil:
    cdef DTYPE_t m00=m[0], m01=m[1], m02=m[2], m03=m[3]
    cdef DTYPE_t m10=m[4], m11=m[5], m12=m[6], m13=m[7]
    cdef DTYPE_t m20=m[8], m21=m[9], m22=m[10], m23=m[11]
    cdef DTYPE_t m30=m[12], m31=m[13], m32=m[14], m33=m[15]
    cdef DTYPE_t t00 = m11*m22*m33 - m11*m23*m32 - m12*m21*m33 + m12*m23*m31 + m13*m21*m32 - m13*m22*m31
    cdef DTYPE_t t10 = -m10*m22*m33 + m10*m23*m32 + m12*m20*m33 - m12*m23*m30 - m13*m20*m32 + m13*m22*m30
    cdef DTYPE_t t20 = m10*m21*m33 - m10*m23*m31 - m11*m20*m33 + m11*m23*m30 + m13*m20*m31 - m13*m21*m30
    cdef DTYPE_t t30 = -m10*m21*m32 + m10*m22*m31 + m11*m20*m32 - m11*m22*m30 - m12*m20*m31 + m12*m21*m30
    cdef DTYPE_t det = m00*t00 + m01*t10 + m02*t20 + m03*t30
    if det == 0.0:
        inv[0]=1; inv[1]=0; inv[2]=0; inv[3]=0
        inv[4]=0; inv[5]=1; inv[6]=0; inv[7]=0
        inv[8]=0; inv[9]=0; inv[10]=1; inv[11]=0
        inv[12]=0; inv[13]=0; inv[14]=0; inv[15]=1
        return
    cdef DTYPE_t idet = 1.0 / det
    inv[0] = t00*idet
    inv[4] = t10*idet
    inv[8] = t20*idet
    inv[12] = t30*idet
    inv[1] = (-m01*m22*m33 + m01*m23*m32 + m02*m21*m33 - m02*m23*m31 - m03*m21*m32 + m03*m22*m31)*idet
    inv[5] = ( m00*m22*m33 - m00*m23*m32 - m02*m20*m33 + m02*m23*m30 + m03*m20*m32 - m03*m22*m30)*idet
    inv[9] = (-m00*m21*m33 + m00*m23*m31 + m01*m20*m33 - m01*m23*m30 - m03*m20*m31 + m03*m21*m30)*idet
    inv[13] = ( m00*m21*m32 - m00*m22*m31 - m01*m20*m32 + m01*m22*m30 + m02*m20*m31 - m02*m21*m30)*idet
    inv[2] = ( m01*m12*m33 - m01*m13*m32 - m02*m11*m33 + m02*m13*m31 + m03*m11*m32 - m03*m12*m31)*idet
    inv[6] = (-m00*m12*m33 + m00*m13*m32 + m02*m10*m33 - m02*m13*m30 - m03*m10*m32 + m03*m12*m30)*idet
    inv[10] = ( m00*m11*m33 - m00*m13*m31 - m01*m10*m33 + m01*m13*m30 + m03*m10*m31 - m03*m11*m30)*idet
    inv[14] = (-m00*m11*m32 + m00*m12*m31 + m01*m10*m32 - m01*m12*m30 - m02*m10*m31 + m02*m11*m30)*idet
    inv[3] = (-m01*m12*m23 + m01*m13*m22 + m02*m11*m23 - m02*m13*m21 - m03*m11*m22 + m03*m12*m21)*idet
    inv[7] = ( m00*m12*m23 - m00*m13*m22 - m02*m10*m23 + m02*m13*m20 + m03*m10*m22 - m03*m12*m20)*idet
    inv[11] = (-m00*m11*m23 + m00*m13*m21 + m01*m10*m23 - m01*m13*m20 - m03*m10*m21 + m03*m11*m20)*idet
    inv[15] = ( m00*m11*m22 - m00*m12*m21 - m01*m10*m22 + m01*m12*m20 + m02*m10*m21 - m02*m11*m20)*idet


def compute_skinning_buffer_cy(
    list bone_offsets,
    list bone_ids,
    scene,
    object renderer_world,
    object inv_cache,
    long long inv_cache_key,
):
    cdef int n = len(bone_offsets)
    if n == 0:
        return np.empty((0, 16), dtype=np.float32), 0, inv_cache, inv_cache_key

    cdef np.ndarray[DTYPE_t, ndim=2] wm_d = renderer_world._d
    cdef np.ndarray[DTYPE_t, ndim=2] wm2_d
    cdef np.ndarray[DTYPE_t, ndim=2] inv_cache_np
    cdef DTYPE_t inv_d[16]
    cdef DTYPE_t rel[16]
    cdef DTYPE_t result[16]
    cdef DTYPE_t off_buf[16]
    cdef int j, row, num_ids, col
    cdef long long wm_id

    wm_id = <long long>(<void*>(&wm_d[0, 0]))
    if wm_id != inv_cache_key or inv_cache is None:
        _mat4_inv_flat(&wm_d[0, 0], inv_d)
        inv_cache_np = np.empty((4, 4), dtype=DTYPE)
        for row in range(4):
            for col in range(4):
                inv_cache_np[row, col] = inv_d[row * 4 + col]
        inv_cache_key = wm_id
        inv_cache = inv_cache_np
    else:
        inv_cache_np = inv_cache
        for row in range(16):
            inv_d[row] = inv_cache_np[row // 4, row % 4]

    cdef np.ndarray[F32_t, ndim=2] flat = np.zeros((n, 16), dtype=np.float32)
    cdef np.ndarray[F32_t, ndim=2] off_f32
    num_ids = len(bone_ids) if bone_ids else 0

    for j in range(n):
        if scene is not None and j < num_ids and bone_ids[j]:
            ent = scene.get_entity(bone_ids[j])
            if ent is not None:
                tr = ent.transform
                if tr is not None:
                    wm2_d = tr.world_matrix._d
                    _mat4_mul_flat(&wm2_d[0, 0], inv_d, rel)
                    off_f32 = bone_offsets[j]
                    for row in range(16):
                        off_buf[row] = <DTYPE_t>off_f32[row // 4, row % 4]
                    _mat4_mul_flat(off_buf, rel, result)
                    for row in range(16):
                        flat[j, row] = <F32_t>result[row]
                    continue
        flat[j, 0] = 1.0
        flat[j, 5] = 1.0
        flat[j, 10] = 1.0
        flat[j, 15] = 1.0

    return flat, n, inv_cache, inv_cache_key


def batch_normal_matrices_cy(list entries, dict cache):
    cdef int n = len(entries)
    cdef int i, eid
    cdef DTYPE_t m00, m01, m02, m10, m11, m12, m20, m21, m22
    cdef DTYPE_t n0, n1, n2
    cdef np.ndarray[F32_t, ndim=2] nm

    for i in range(n):
        entry = entries[i]
        ent = entry[0]
        eid = ent._id
        if eid in cache:
            continue
        wm = entry[4]
        _d = wm._d
        m00 = _d[0, 0]; m01 = _d[0, 1]; m02 = _d[0, 2]
        m10 = _d[1, 0]; m11 = _d[1, 1]; m12 = _d[1, 2]
        m20 = _d[2, 0]; m21 = _d[2, 1]; m22 = _d[2, 2]

        n0 = sqrt(m00 * m00 + m10 * m10 + m20 * m20)
        if n0 < 1e-10: n0 = 1e-10
        m00 /= n0; m10 /= n0; m20 /= n0

        n1 = sqrt(m01 * m01 + m11 * m11 + m21 * m21)
        if n1 < 1e-10: n1 = 1e-10
        m01 /= n1; m11 /= n1; m21 /= n1

        n2 = sqrt(m02 * m02 + m12 * m12 + m22 * m22)
        if n2 < 1e-10: n2 = 1e-10
        m02 /= n2; m12 /= n2; m22 /= n2

        nm = np.empty((3, 3), dtype=np.float32)
        nm[0, 0] = <F32_t>m00; nm[0, 1] = <F32_t>m10; nm[0, 2] = <F32_t>m20
        nm[1, 0] = <F32_t>m01; nm[1, 1] = <F32_t>m11; nm[1, 2] = <F32_t>m21
        nm[2, 0] = <F32_t>m02; nm[2, 1] = <F32_t>m12; nm[2, 2] = <F32_t>m22
        cache[eid] = nm


def batch_wm_to_f32(list entries):
    cdef int n = len(entries)
    cdef int i, j, k
    cdef list results = [None] * n
    cdef object entry, wm
    cdef np.ndarray[DTYPE_t, ndim=2] _d
    cdef np.ndarray[F32_t, ndim=1] f32

    for i in range(n):
        entry = entries[i]
        wm = entry[4]
        _d = wm._d
        f32 = np.empty(16, dtype=np.float32)
        for j in range(4):
            for k in range(4):
                f32[j * 4 + k] = <F32_t>_d[j, k]
        results[i] = f32

    return results
