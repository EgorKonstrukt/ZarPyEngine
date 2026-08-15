# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

import numpy as np

try:
    from core import _audio_dsp_cy as _fast
    _FAST = True
except Exception:
    _FAST = False


def push_samples(ring, ring_size, head, pcm, start, count, gain):
    if _FAST:
        return int(_fast.push_samples(ring, ring_size, head, pcm, start, count, gain))
    n = len(pcm)
    if count <= 0 or gain == 0.0:
        return head
    if start < 0:
        start = 0
    if start >= n:
        return head
    if start + count > n:
        count = n - start
    count = min(count, ring_size)
    idxs = np.arange(head, head + count)
    idxs %= ring_size
    ring[idxs] = pcm[start:start + count] * gain
    head += count
    head %= ring_size
    return head


def extract_last(ring, ring_size, head, out):
    if _FAST:
        _fast.extract_last(ring, ring_size, head, out)
        return
    n = min(len(out), ring_size)
    idxs = (np.arange(head, head + n) % ring_size)
    out[:n] = ring[idxs]


def mix_add(dst, dst_start, src, src_start, count, gain):
    if _FAST:
        _fast.mix_add(dst, dst_start, src, src_start, count, gain)
        return
    n = len(src)
    if count <= 0 or gain == 0.0:
        return
    if src_start < 0:
        src_start = 0
    if src_start >= n:
        return
    if src_start + count > n:
        count = n - src_start
    dst[dst_start:dst_start + count] += src[src_start:src_start + count] * gain


def analyze_spectrum(ring, ring_size, head, fft_size, spec_out, spec_bins, sample_rate):
    if _FAST:
        return float(_fast.analyze_spectrum(ring, ring_size, head, fft_size, spec_out, spec_bins, sample_rate))
    n = fft_size
    idxs = (np.arange(head - n, head) % ring_size)
    x = ring[idxs].astype(np.float64)
    win = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)
    x = x * win
    mag = np.abs(np.fft.rfft(x)) * (2.0 / n)
    half = n // 2
    df = sample_rate / n
    lmin = np.log10(20.0)
    lmax = np.log10(sample_rate * 0.5)
    if lmax <= lmin:
        lmax = lmin + 1.0
    freqs = np.logspace(lmin, lmax, spec_bins + 1)
    bins = []
    for i in range(spec_bins):
        b0 = max(1, int(freqs[i] / df))
        b1 = min(half, int(freqs[i + 1] / df))
        if b1 >= b0:
            avg = mag[b0:b1 + 1].mean()
            db = 20.0 * np.log10(avg) if avg > 1e-7 else -120.0
        else:
            db = -120.0
        bins.append(db)
    spec_out[:] = np.asarray(bins, dtype=np.float32)
    return float(max(bins)) if bins else -120.0
