# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef inline DTYPE_t _hermite_interpolate(
    DTYPE_t t, DTYPE_t k0_val, DTYPE_t k1_val,
    DTYPE_t m0, DTYPE_t m1,
) noexcept nogil:
    cdef DTYPE_t t2 = t * t
    cdef DTYPE_t t3 = t2 * t
    return ((2.0 * t3 - 3.0 * t2 + 1.0) * k0_val +
            (t3 - 2.0 * t2 + t) * m0 +
            (-2.0 * t3 + 3.0 * t2) * k1_val +
            (t3 - t2) * m1)

def evaluate_curve_batch(
    np.ndarray[DTYPE_t, ndim=1] times,
    np.ndarray[DTYPE_t, ndim=1] key_times,
    np.ndarray[DTYPE_t, ndim=1] key_values,
    np.ndarray[DTYPE_t, ndim=1] key_in_tangents,
    np.ndarray[DTYPE_t, ndim=1] key_out_tangents,
    np.ndarray[np.int32_t, ndim=1] key_tangent_modes,
    str pre_wrap,
    str post_wrap,
):
    cdef int nk = key_times.shape[0]
    cdef int nt = times.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] out = np.empty(nt, dtype=DTYPE)
    cdef int i, j, idx
    cdef DTYPE_t time, t, dt, m0, m1
    cdef int mode

    if nk == 0:
        for i in range(nt):
            out[i] = 0.0
        return out

    if nk == 1:
        for i in range(nt):
            out[i] = key_values[0]
        return out

    for i in range(nt):
        time = times[i]

        if time <= key_times[0]:
            if pre_wrap == "loop":
                t = key_times[nk - 1] - key_times[0]
                if t > 1e-10:
                    time = key_times[0] + ((time - key_times[0]) % t + t) % t
            elif pre_wrap == "ping_pong":
                t = key_times[nk - 1] - key_times[0]
                if t > 1e-10:
                    cycle = ((time - key_times[0]) / t)
                    if <int>cycle % 2 != 0:
                        time = key_times[nk - 1] - ((time - key_times[0]) % t)
                    else:
                        time = key_times[0] + ((time - key_times[0]) % t)
            else:
                out[i] = key_values[0]
                continue

        if time >= key_times[nk - 1]:
            if post_wrap == "loop":
                t = key_times[nk - 1] - key_times[0]
                if t > 1e-10:
                    time = key_times[0] + ((time - key_times[0]) % t + t) % t
            elif post_wrap == "ping_pong":
                t = key_times[nk - 1] - key_times[0]
                if t > 1e-10:
                    cycle = ((time - key_times[0]) / t)
                    if <int>cycle % 2 != 0:
                        time = key_times[nk - 1] - ((time - key_times[0]) % t)
                    else:
                        time = key_times[0] + ((time - key_times[0]) % t)
            else:
                out[i] = key_values[nk - 1]
                continue

        idx = 0
        for j in range(nk - 1):
            if key_times[j] <= time <= key_times[j + 1]:
                idx = j
                break

        dt = key_times[idx + 1] - key_times[idx]
        if dt < 1e-10:
            out[i] = key_values[idx]
            continue

        t = (time - key_times[idx]) / dt
        mode = key_tangent_modes[idx]

        if mode == 2:
            out[i] = key_values[idx]
        elif mode == 1:
            out[i] = key_values[idx] + (key_values[idx + 1] - key_values[idx]) * t
        else:
            m0 = key_out_tangents[idx] * dt
            m1 = key_in_tangents[idx + 1] * dt
            out[i] = _hermite_interpolate(t, key_values[idx], key_values[idx + 1], m0, m1)

    return out

def evaluate_curve_single(
    DTYPE_t time,
    np.ndarray[DTYPE_t, ndim=1] key_times,
    np.ndarray[DTYPE_t, ndim=1] key_values,
    np.ndarray[DTYPE_t, ndim=1] key_in_tangents,
    np.ndarray[DTYPE_t, ndim=1] key_out_tangents,
    np.ndarray[np.int32_t, ndim=1] key_tangent_modes,
    str wrap_mode,
):
    cdef int nk = key_times.shape[0]
    cdef int j, idx
    cdef DTYPE_t t, dt, m0, m1
    cdef int mode

    if nk == 0:
        return 0.0
    if nk == 1:
        return key_values[0]

    if time <= key_times[0]:
        if wrap_mode == "loop":
            dt = key_times[nk - 1] - key_times[0]
            if dt > 1e-10:
                time = key_times[0] + ((time - key_times[0]) % dt + dt) % dt
        else:
            return key_values[0]

    if time >= key_times[nk - 1]:
        if wrap_mode == "loop":
            dt = key_times[nk - 1] - key_times[0]
            if dt > 1e-10:
                time = key_times[0] + ((time - key_times[0]) % dt + dt) % dt
        else:
            return key_values[nk - 1]

    idx = 0
    for j in range(nk - 1):
        if key_times[j] <= time <= key_times[j + 1]:
            idx = j
            break

    dt = key_times[idx + 1] - key_times[idx]
    if dt < 1e-10:
        return key_values[idx]

    t = (time - key_times[idx]) / dt
    mode = key_tangent_modes[idx]

    if mode == 2:
        return key_values[idx]
    elif mode == 1:
        return key_values[idx] + (key_values[idx + 1] - key_values[idx]) * t
    else:
        m0 = key_out_tangents[idx] * dt
        m1 = key_in_tangents[idx + 1] * dt
        return _hermite_interpolate(t, key_values[idx], key_values[idx + 1], m0, m1)

def evaluate_multi_curve_batch(
    np.ndarray[DTYPE_t, ndim=1] times,
    list curves_data,
):
    cdef int nt = times.shape[0]
    cdef int nc = len(curves_data)
    cdef np.ndarray[DTYPE_t, ndim=2] out = np.empty((nt, nc), dtype=DTYPE)
    cdef int ci, i
    for ci in range(nc):
        cd = curves_data[ci]
        result = evaluate_curve_batch(
            times,
            cd[0], cd[1], cd[2], cd[3], cd[4],
            cd[5], cd[6],
        )
        for i in range(nt):
            out[i, ci] = result[i]
    return out
