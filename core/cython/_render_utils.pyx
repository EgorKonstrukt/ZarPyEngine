# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def compute_bounding_spheres(list matrices, np.ndarray[DTYPE_t, ndim=1] bounding_radii):
    """Build (n,4) float32 sphere data for GPU culling.

    Each sphere: x, y, z = center, w = radius.
    """
    cdef int n = len(matrices)
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)

    cdef np.ndarray[np.float32_t, ndim=2] spheres = np.empty((n, 4), dtype=np.float32)
    cdef int i
    cdef object wm, _d

    for i in range(n):
        wm = matrices[i]
        _d = wm._d
        spheres[i, 0] = <np.float32_t>_d[3, 0]
        spheres[i, 1] = <np.float32_t>_d[3, 1]
        spheres[i, 2] = <np.float32_t>_d[3, 2]
        spheres[i, 3] = <np.float32_t>bounding_radii[i]

    return spheres
