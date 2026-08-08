# cython: language_level=3
import numpy as np
cimport numpy as cnp
from libc.math cimport cos, sin, log10, sqrt, pow, M_PI

cnp.import_array()

cdef double _PI = 3.141592653589793

cdef class FFTTable:
    cdef int n
    cdef float[::1] w_re
    cdef float[::1] w_im
    cdef int[::1] rev

    def __cinit__(self, int n):
        self.n = n
        cdef cnp.ndarray[cnp.float32_t, ndim=1] wre = np.empty(n // 2, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] wim = np.empty(n // 2, dtype=np.float32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] rv = np.empty(n, dtype=np.int32)
        cdef int i, j, m
        for i in range(n // 2):
            wre[i] = cos(-2.0 * _PI * i / n)
            wim[i] = sin(-2.0 * _PI * i / n)
        j = 0
        for i in range(n):
            rv[i] = j
            m = n >> 1
            while j >= m and m:
                j -= m
                m >>= 1
            j += m
        self.w_re = wre
        self.w_im = wim
        self.rev = rv


cdef dict _tables = {}


cdef FFTTable _get_table(int n):
    cdef FFTTable t = _tables.get(n)
    if t is None:
        t = FFTTable(n)
        _tables[n] = t
    return t


cdef void _fft(float[::1] re, float[::1] im, FFTTable tab) noexcept nogil:
    cdef int n = tab.n
    cdef int i, j, L, half, step, k, j1, j2
    cdef float wr, wi, u, v
    for i in range(n):
        j = tab.rev[i]
        if j > i:
            u = re[i]
            re[i] = re[j]
            re[j] = u
            v = im[i]
            im[i] = im[j]
            im[j] = v
    L = 2
    while L <= n:
        half = L >> 1
        step = n // L
        i = 0
        while i < n:
            k = 0
            while k < half:
                j1 = i + k
                j2 = j1 + half
                wr = tab.w_re[k * step]
                wi = tab.w_im[k * step]
                u = re[j2] * wr - im[j2] * wi
                v = re[j2] * wi + im[j2] * wr
                re[j2] = re[j1] - u
                im[j2] = im[j1] - v
                re[j1] = re[j1] + u
                im[j1] = im[j1] + v
                k += 1
            i += L
        L <<= 1


def push_samples(float[::1] ring, int ring_size, int head,
                 float[::1] pcm, int start, int count, float gain) -> int:
    cdef int n = pcm.shape[0]
    cdef int i, idx
    cdef float v
    if count <= 0 or gain == 0.0:
        return head
    if start < 0:
        start = 0
    if start >= n:
        return head
    if start + count > n:
        count = n - start
    if count > ring_size:
        count = ring_size
    for i in range(count):
        idx = head + i
        if idx >= ring_size:
            idx -= ring_size
        v = pcm[start + i] * gain
        ring[idx] = v
    head += count
    if head >= ring_size:
        head -= ring_size
    return head


def mix_add(float[::1] dst, int dst_start, float[::1] src, int src_start, int count, float gain):
    cdef int n = src.shape[0]
    cdef int i
    if count <= 0 or gain == 0.0:
        return
    if src_start < 0:
        src_start = 0
    if src_start >= n:
        return
    if src_start + count > n:
        count = n - src_start
    for i in range(count):
        dst[dst_start + i] += src[src_start + i] * gain


def extract_last(float[::1] ring, int ring_size, int head,
                 float[::1] out):
    cdef int n = out.shape[0]
    cdef int i, idx, start
    if n <= 0:
        return
    if n > ring_size:
        n = ring_size
    start = head - n
    if start < 0:
        start += ring_size
    for i in range(n):
        idx = start + i
        if idx >= ring_size:
            idx -= ring_size
        out[i] = ring[idx]


def analyze_spectrum(float[::1] ring, int ring_size, int head,
                     int fft_size,
                     float[::1] spec_out, int spec_bins,
                     float sample_rate) -> float:
    cdef int n = fft_size
    cdef FFTTable tab = _get_table(n)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] re_np = np.zeros(n, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] im_np = np.zeros(n, dtype=np.float32)
    cdef float[::1] re = re_np
    cdef float[::1] im = im_np
    cdef int i, idx, start, half, b0, b1, b
    cdef double w, df, lmin, lmax, span, f0, f1
    cdef double acc, avg, db, mag, norm, max_db
    if n <= 0 or spec_bins <= 0:
        return -120.0
    if ring_size <= 0:
        ring_size = ring.shape[0]
    start = head - n
    if start < 0:
        start += ring_size
    for i in range(n):
        idx = start + i
        if idx >= ring_size:
            idx -= ring_size
        re[i] = ring[idx]
    for i in range(n):
        w = 0.5 - 0.5 * cos(2.0 * _PI * i / n)
        re[i] = re[i] * w
    _fft(re, im, tab)
    half = n // 2
    df = sample_rate / n
    lmin = log10(20.0)
    lmax = log10(sample_rate * 0.5)
    if lmax <= lmin:
        lmax = lmin + 1.0
    span = lmax - lmin
    norm = 1.0 / half
    max_db = -120.0
    for i in range(spec_bins):
        f0 = pow(10.0, lmin + span * i / spec_bins)
        f1 = pow(10.0, lmin + span * (i + 1) / spec_bins)
        b0 = <int>(f0 / df)
        b1 = <int>(f1 / df)
        if b0 < 1:
            b0 = 1
        if b1 > half:
            b1 = half
        if b1 >= b0:
            acc = 0.0
            for b in range(b0, b1 + 1):
                mag = sqrt(re[b] * re[b] + im[b] * im[b]) * norm
                acc += mag
            avg = acc / (b1 - b0 + 1)
            db = 20.0 * log10(avg) if avg > 1e-7 else -120.0
        else:
            db = -120.0
        spec_out[i] = db
        if db > max_db:
            max_db = db
    return max_db
