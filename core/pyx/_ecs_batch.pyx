# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef inline void _mat4_mul(const DTYPE_t a[4][4], const DTYPE_t[:, :] b,
                            DTYPE_t out[4][4]) noexcept nogil:
    cdef int i
    cdef DTYPE_t a0, a1, a2, a3
    for i in range(4):
        a0 = a[i][0]; a1 = a[i][1]; a2 = a[i][2]; a3 = a[i][3]
        out[i][0] = a0*b[0,0] + a1*b[1,0] + a2*b[2,0] + a3*b[3,0]
        out[i][1] = a0*b[0,1] + a1*b[1,1] + a2*b[2,1] + a3*b[3,1]
        out[i][2] = a0*b[0,2] + a1*b[1,2] + a2*b[2,2] + a3*b[3,2]
        out[i][3] = a0*b[0,3] + a1*b[1,3] + a2*b[2,3] + a3*b[3,3]

cdef inline void _local_matrix(
    DTYPE_t tx, DTYPE_t ty, DTYPE_t tz,
    DTYPE_t rx, DTYPE_t ry, DTYPE_t rz, DTYPE_t rw,
    DTYPE_t sx, DTYPE_t sy, DTYPE_t sz,
    DTYPE_t out[4][4],
) noexcept nogil:
    cdef DTYPE_t rm[4][4], temp[4][4]
    cdef int i
    cdef DTYPE_t a0, a1, a2, a3
    cdef DTYPE_t x2 = rx + rx, y2 = ry + ry, z2 = rz + rz
    cdef DTYPE_t xx = rx * x2, xy = rx * y2, xz = rx * z2
    cdef DTYPE_t yy = ry * y2, yz = ry * z2, zz = rz * z2
    cdef DTYPE_t wx = rw * x2, wy = rw * y2, wz = rw * z2

    rm[0][0] = 1.0 - (yy + zz); rm[0][1] = xy + wz;      rm[0][2] = xz - wy;  rm[0][3] = 0.0
    rm[1][0] = xy - wz;         rm[1][1] = 1.0 - (xx + zz); rm[1][2] = yz + wx;  rm[1][3] = 0.0
    rm[2][0] = xz + wy;         rm[2][1] = yz - wx;         rm[2][2] = 1.0 - (xx + yy); rm[2][3] = 0.0
    rm[3][0] = 0.0; rm[3][1] = 0.0; rm[3][2] = 0.0; rm[3][3] = 1.0

    temp[0][0] = sx*rm[0][0]; temp[0][1] = sx*rm[0][1]; temp[0][2] = sx*rm[0][2]; temp[0][3] = 0.0
    temp[1][0] = sy*rm[1][0]; temp[1][1] = sy*rm[1][1]; temp[1][2] = sy*rm[1][2]; temp[1][3] = 0.0
    temp[2][0] = sz*rm[2][0]; temp[2][1] = sz*rm[2][1]; temp[2][2] = sz*rm[2][2]; temp[2][3] = 0.0
    temp[3][0] = 0.0; temp[3][1] = 0.0; temp[3][2] = 0.0; temp[3][3] = 1.0

    for i in range(4):
        a0 = temp[i][0]; a1 = temp[i][1]; a2 = temp[i][2]; a3 = temp[i][3]
        out[i][0] = a0 + a3*tx
        out[i][1] = a1 + a3*ty
        out[i][2] = a2 + a3*tz
        out[i][3] = a3

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

def extract_parent_outside(list transforms, np.ndarray[np.int32_t, ndim=1] has_parent,
                           np.ndarray[np.int32_t, ndim=1] parent_idx):
    cdef int n = len(transforms)
    cdef np.ndarray[DTYPE_t, ndim=3] parent_outside = np.zeros((n, 4, 4), dtype=DTYPE)
    cdef int i
    cdef object t, e, parent_e, pt, wm
    cdef np.ndarray[DTYPE_t, ndim=2] wm_arr

    for i in range(n):
        if has_parent[i] == 1 and parent_idx[i] < 0:
            t = transforms[i]
            e = t._entity
            if e is not None:
                parent_e = e.parent
                if parent_e is not None:
                    pt = parent_e.transform
                    if pt is not None:
                        wm = pt._world_matrix
                        wm_arr = wm._d
                        parent_outside[i, 0, 0] = wm_arr[0, 0]
                        parent_outside[i, 0, 1] = wm_arr[0, 1]
                        parent_outside[i, 0, 2] = wm_arr[0, 2]
                        parent_outside[i, 0, 3] = wm_arr[0, 3]
                        parent_outside[i, 1, 0] = wm_arr[1, 0]
                        parent_outside[i, 1, 1] = wm_arr[1, 1]
                        parent_outside[i, 1, 2] = wm_arr[1, 2]
                        parent_outside[i, 1, 3] = wm_arr[1, 3]
                        parent_outside[i, 2, 0] = wm_arr[2, 0]
                        parent_outside[i, 2, 1] = wm_arr[2, 1]
                        parent_outside[i, 2, 2] = wm_arr[2, 2]
                        parent_outside[i, 2, 3] = wm_arr[2, 3]
                        parent_outside[i, 3, 0] = wm_arr[3, 0]
                        parent_outside[i, 3, 1] = wm_arr[3, 1]
                        parent_outside[i, 3, 2] = wm_arr[3, 2]
                        parent_outside[i, 3, 3] = wm_arr[3, 3]
    return parent_outside

def batch_update_from_transforms(list transforms):
    cdef int n = len(transforms)
    if n == 0:
        return

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
    cdef np.ndarray[DTYPE_t, ndim=3] parent_outside = np.zeros((n, 4, 4), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] world_mats = np.empty((n, 4, 4), dtype=DTYPE)

    cdef dict id_to_idx = {}
    cdef int i, pi, j, k
    cdef object t, e, parent_e, pt, p, q, s, wm, wm_arr
    cdef str parent_id
    cdef DTYPE_t local[4][4], result[4][4]

    for i in range(n):
        t = transforms[i]
        e = t._entity
        if e is not None:
            id_to_idx[e.id] = i

    for i in range(n):
        t = transforms[i]
        p = t._local_pos
        q = t._local_rot
        s = t._local_scale
        pos_x[i] = p.x; pos_y[i] = p.y; pos_z[i] = p.z
        rot_x[i] = q.x; rot_y[i] = q.y; rot_z[i] = q.z; rot_w[i] = q.w
        sc_x[i] = s.x; sc_y[i] = s.y; sc_z[i] = s.z

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

    for i in range(n):
        if has_parent[i] == 1 and parent_idx[i] < 0:
            t = transforms[i]
            e = t._entity
            if e is not None:
                parent_e = e.parent
                if parent_e is not None:
                    pt = parent_e.transform
                    if pt is not None:
                        wm = pt._world_matrix
                        wm_arr = wm._d
                        for j in range(4):
                            for k in range(4):
                                parent_outside[i, j, k] = wm_arr[j, k]

    for i in range(n):
        _local_matrix(
            pos_x[i], pos_y[i], pos_z[i],
            rot_x[i], rot_y[i], rot_z[i], rot_w[i],
            sc_x[i], sc_y[i], sc_z[i],
            local,
        )
        if not has_parent[i]:
            for j in range(4):
                for k in range(4):
                    world_mats[i, j, k] = local[j][k]
        elif parent_idx[i] >= 0:
            pi = parent_idx[i]
            _mat4_mul(local, world_mats[pi], result)
            for j in range(4):
                for k in range(4):
                    world_mats[i, j, k] = result[j][k]
        else:
            _mat4_mul(local, parent_outside[i], result)
            for j in range(4):
                for k in range(4):
                    world_mats[i, j, k] = result[j][k]

    for i in range(n):
        t = transforms[i]
        t._world_matrix = t._world_matrix.__class__(world_mats[i])
        t._world_target = None
        t._dirty = False

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
