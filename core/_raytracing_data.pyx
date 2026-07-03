# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, embedsignature=False
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

cdef inline void cross3(const float ax, const float ay, const float az,
                         const float bx, const float by, const float bz,
                         float* cx, float* cy, float* cz) noexcept nogil:
    cx[0] = ay * bz - az * by
    cy[0] = az * bx - ax * bz
    cz[0] = ax * by - ay * bx

cdef inline void normalize3(float* x, float* y, float* z) noexcept nogil:
    cdef float lensq = x[0]*x[0] + y[0]*y[0] + z[0]*z[0]
    cdef float inv_len
    if lensq > 1e-30:
        inv_len = 1.0 / sqrt(lensq)
        x[0] *= inv_len; y[0] *= inv_len; z[0] *= inv_len

def compute_face_normals_float(object verts3, object idxs):
    cdef int n_tris = idxs.shape[0]
    cdef int i, i0, i1, i2
    cdef float nx, ny, nz
    cdef np.ndarray v = np.ascontiguousarray(verts3, dtype=np.float32)
    cdef np.ndarray idx = np.ascontiguousarray(idxs, dtype=np.intp)
    cdef np.ndarray out = np.empty((n_tris * 3, 3), dtype=np.float32)
    cdef float[:, ::1] vv = v.reshape(n_tris * 3, 3)
    cdef float[:, ::1] oo = out.reshape(n_tris * 3, 3)

    for i in range(n_tris):
        i0 = idx[i, 0]; i1 = idx[i, 1]; i2 = idx[i, 2]
        cross3(vv[i1,0]-vv[i0,0], vv[i1,1]-vv[i0,1], vv[i1,2]-vv[i0,2],
               vv[i2,0]-vv[i0,0], vv[i2,1]-vv[i0,1], vv[i2,2]-vv[i0,2], &nx, &ny, &nz)
        normalize3(&nx, &ny, &nz)
        oo[i*3, 0] = nx; oo[i*3, 1] = ny; oo[i*3, 2] = nz
        oo[i*3+1, 0] = nx; oo[i*3+1, 1] = ny; oo[i*3+1, 2] = nz
        oo[i*3+2, 0] = nx; oo[i*3+2, 1] = ny; oo[i*3+2, 2] = nz
    return out
