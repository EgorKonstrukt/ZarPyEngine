# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def extract_transform_soa(list transforms):
    cdef int n = len(transforms)
    if n == 0:
        return (np.zeros(0, dtype=DTYPE), np.zeros(0, dtype=DTYPE), np.zeros(0, dtype=DTYPE),
                np.zeros(0, dtype=DTYPE), np.zeros(0, dtype=DTYPE), np.zeros(0, dtype=DTYPE),
                np.zeros(0, dtype=DTYPE), np.zeros(0, dtype=DTYPE), np.zeros(0, dtype=DTYPE),
                np.zeros(0, dtype=DTYPE), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32))

    cdef np.ndarray[DTYPE_t, ndim=1] pos_x = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] pos_y = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] pos_z = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] rot_x = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] rot_y = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] rot_z = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] rot_w = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] sc_x = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] sc_y = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] sc_z = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[np.int32_t, ndim=1] has_parent = np.zeros(n, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] parent_idx = np.full(n, -1, dtype=np.int32)

    cdef int i
    cdef object t, p, q, s, e, parent_e

    for i in range(n):
        t = transforms[i]
        p = t._local_pos
        q = t._local_rot
        s = t._local_scale
        pos_x[i] = p.x; pos_y[i] = p.y; pos_z[i] = p.z
        rot_x[i] = q.x; rot_y[i] = q.y; rot_z[i] = q.z; rot_w[i] = q.w
        sc_x[i] = s.x; sc_y[i] = s.y; sc_z[i] = s.z

    return (pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w,
            sc_x, sc_y, sc_z, has_parent, parent_idx)

def resolve_parent_indices(list transforms, dict id_to_idx):
    cdef int n = len(transforms)
    cdef np.ndarray[np.int32_t, ndim=1] has_parent = np.zeros(n, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] parent_idx = np.full(n, -1, dtype=np.int32)
    cdef int i
    cdef object t, e, parent_e
    cdef str parent_id

    for i in range(n):
        t = transforms[i]
        e = t._entity
        if e is not None:
            parent_e = e.parent
            if parent_e is not None:
                has_parent[i] = 1
                parent_id = parent_e.id
                if parent_id in id_to_idx:
                    parent_idx[i] = id_to_idx[parent_id]

    return has_parent, parent_idx

def write_world_matrices(list transforms, np.ndarray[DTYPE_t, ndim=3] world_mats):
    cdef int n = len(transforms)
    cdef int i
    cdef object t, wm_class

    for i in range(n):
        t = transforms[i]
        t._world_matrix = t._world_matrix.__class__(world_mats[i])
        t._world_target = None
        t._dirty = False

def collect_dirty_transforms(list all_dirty):
    cdef int n = len(all_dirty)
    if n == 0:
        return all_dirty

    cdef dict depth_cache = {}
    cdef list result = []
    cdef object t
    cdef int depth

    for i in range(n):
        t = all_dirty[i]
        depth = _get_depth(t, depth_cache)
        result.append((depth, i, t))

    result.sort(key=lambda x: x[0])
    return [t for _, _, t in result]

cdef inline int _get_depth(object transform, dict cache):
    cdef int d = 0
    cdef object t = transform
    cdef str tid
    while t is not None:
        tid = id(t)
        if tid in cache:
            return d + cache[tid]
        d += 1
        if t._entity is not None and t._entity.parent is not None:
            t = t._entity.parent.transform
        else:
            break
    cache[id(transform)] = d
    return d
