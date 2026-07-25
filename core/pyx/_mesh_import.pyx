# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
# cython: language_level=3
from libc.stdint cimport uint32_t, uintptr_t
from libc.string cimport memcpy
import numpy as np
cimport numpy as np

DTYPE_F32 = np.float32
DTYPE_U32 = np.uint32
ctypedef np.float32_t F32_t
ctypedef np.uint32_t U32_t


def extract_faces(uintptr_t faces_ptr, int nf):
    cdef:
        int j, idx = 0
        uint32_t n_idx
        uintptr_t ptr_val
        uint32_t *idx_ptr
        char *face_base = <char *>faces_ptr
        int face_size = 16
        np.ndarray[U32_t, ndim=1] out = np.empty(nf * 3, dtype=np.uint32)

    with nogil:
        for j in range(nf):
            n_idx = (<uint32_t *>(face_base + j * face_size))[0]
            ptr_val = (<uintptr_t *>(face_base + j * face_size + 8))[0]
            if n_idx >= 3 and ptr_val != 0:
                idx_ptr = <uint32_t *>ptr_val
                out[idx] = idx_ptr[0]
                out[idx + 1] = idx_ptr[1]
                out[idx + 2] = idx_ptr[2]
            idx += 3
    return out


def smooth_normals(np.ndarray[F32_t, ndim=2] verts,
                   np.ndarray[U32_t, ndim=1] indices):
    cdef:
        int n_verts = verts.shape[0]
        int n_tris = indices.shape[0] // 3
        int i, i0, i1, i2
        float f0x, f0y, f0z, f1x, f1y, f1z
        float nx, ny, nz, len
        np.ndarray[F32_t, ndim=2] normals = np.zeros((n_verts, 3), dtype=np.float32)

    with nogil:
        for i in range(n_tris):
            i0 = indices[i * 3]
            i1 = indices[i * 3 + 1]
            i2 = indices[i * 3 + 2]
            f0x = verts[i1, 0] - verts[i0, 0]
            f0y = verts[i1, 1] - verts[i0, 1]
            f0z = verts[i1, 2] - verts[i0, 2]
            f1x = verts[i2, 0] - verts[i0, 0]
            f1y = verts[i2, 1] - verts[i0, 1]
            f1z = verts[i2, 2] - verts[i0, 2]
            nx = f0y * f1z - f0z * f1y
            ny = f0z * f1x - f0x * f1z
            nz = f0x * f1y - f0y * f1x
            len = (nx * nx + ny * ny + nz * nz)
            if len > 1e-20:
                len = 1.0 / (len ** 0.5)
                nx *= len
                ny *= len
                nz *= len
            normals[i0, 0] += nx
            normals[i0, 1] += ny
            normals[i0, 2] += nz
            normals[i1, 0] += nx
            normals[i1, 1] += ny
            normals[i1, 2] += nz
            normals[i2, 0] += nx
            normals[i2, 1] += ny
            normals[i2, 2] += nz

    cdef np.ndarray[F32_t, ndim=2] out = np.empty_like(normals)
    cdef float nl
    with nogil:
        for i in range(n_verts):
            nl = normals[i, 0] * normals[i, 0] + normals[i, 1] * normals[i, 1] + normals[i, 2] * normals[i, 2]
            if nl > 1e-20:
                nl = 1.0 / (nl ** 0.5)
                out[i, 0] = normals[i, 0] * nl
                out[i, 1] = normals[i, 1] * nl
                out[i, 2] = normals[i, 2] * nl
            else:
                out[i, 0] = 0.0
                out[i, 1] = 1.0
                out[i, 2] = 0.0
    return out


def apply_zup_to_yup(np.ndarray[F32_t, ndim=1] data):
    cdef:
        int n = data.shape[0] // 3
        int i
        float x, y, z
        np.ndarray[F32_t, ndim=1] out = np.empty_like(data)

    with nogil:
        for i in range(n):
            x = data[i * 3]
            y = data[i * 3 + 1]
            z = data[i * 3 + 2]
            out[i * 3] = x
            out[i * 3 + 1] = z
            out[i * 3 + 2] = -y
    return out
