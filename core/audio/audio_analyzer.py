# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import ctypes
import numpy as np

from core._audio_dsp import push_samples, mix_add, extract_last, analyze_spectrum

_AL_SOURCE_STATE = 0x1010
_AL_PLAYING = 0x1012
_AL_SAMPLE_OFFSET = 0x1025

RING_SIZE = 65536
WAVE_N = 4096
SPEC_BINS = 96
FFT_SIZE = 4096
SCOPE_N = 4096
_DECAY = 0.9
_DECAY_IDLE = 0.82
_ATTACK = 0.6
_RELEASE = 0.15
_FLOOR_DB = -70.0


class AudioAnalyzer:
    _instance = None

    def __init__(self):
        AudioAnalyzer._instance = self
        self._ring = np.zeros(RING_SIZE, dtype=np.float32)
        self._head = 0
        self._src_state = {}
        self._wave = np.zeros(WAVE_N, dtype=np.float32)
        self._spec_db = np.zeros(SPEC_BINS, dtype=np.float32)
        self._spec_prev = np.full(SPEC_BINS, -120.0, dtype=np.float32)
        self._spec = np.zeros(SPEC_BINS, dtype=np.float32)
        self._spec_hold = np.zeros(SPEC_BINS, dtype=np.float32)
        self._spec_hold_db = np.full(SPEC_BINS, -120.0, dtype=np.float32)
        self._hold_decay = 0.8
        self._spec_top_db = 0.0
        self._spec_floor_db = -60.0
        self._level = 0.0
        self._peak = 1e-4
        self._rms = 0.0
        self._peak_freq = 0.0
        self._centroid = 0.0
        self._sample_rate = 48000
        self._active = 0
        self._has_signal = False
        self._scope_l = np.zeros(RING_SIZE, dtype=np.float32)
        self._scope_r = np.zeros(RING_SIZE, dtype=np.float32)
        self._scope_head = 0
        self._dsp = None

    @classmethod
    def instance(cls):
        return cls._instance

    def reset(self):
        self._ring *= 0.0
        self._head = 0
        self._src_state = {}
        self._level = 0.0
        self._peak = 1e-4
        self._spec_prev[:] = -120.0
        self._spec[:] = 0.0
        self._spec_hold[:] = 0.0
        self._spec_hold_db[:] = -120.0
        self._has_signal = False
        self._active = 0
        self._scope_l *= 0.0
        self._scope_r *= 0.0
        self._scope_head = 0

    @property
    def wave(self):
        return self._wave

    @property
    def scope(self):
        n = min(SCOPE_N, RING_SIZE)
        idx = (np.arange(self._scope_head - n, self._scope_head) % RING_SIZE)
        l = np.ascontiguousarray(self._scope_l[idx])
        r = np.ascontiguousarray(self._scope_r[idx])
        return np.ascontiguousarray(np.stack([l, r], axis=1), dtype=np.float32)

    @property
    def spec(self):
        return self._spec

    @property
    def spec_hold(self):
        return self._spec_hold

    @property
    def hold_decay(self):
        return self._hold_decay

    @hold_decay.setter
    def hold_decay(self, value: float):
        self._hold_decay = float(value)

    @property
    def spec_top_db(self):
        return self._spec_top_db

    @property
    def spec_floor_db(self):
        return self._spec_floor_db

    @property
    def level(self):
        return self._level

    @property
    def level_db(self):
        import math
        return 20.0 * math.log10(max(self._peak, 1e-4))

    @property
    def rms(self):
        return self._rms

    @property
    def rms_db(self):
        import math
        return 20.0 * math.log10(max(self._rms, 1e-4))

    @property
    def peak_freq(self):
        return self._peak_freq

    @property
    def centroid(self):
        return self._centroid

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def active(self):
        return self._active

    @property
    def has_signal(self):
        return self._has_signal

    @staticmethod
    def _source_playing(src) -> bool:
        from core.audio.audio_system import al
        if al is None:
            return False
        try:
            st = ctypes.c_int()
            al.alGetSourcei(src, _AL_SOURCE_STATE, ctypes.byref(st))
            return st.value == _AL_PLAYING
        except Exception:
            return False

    @staticmethod
    def _sample_offset(src) -> int:
        from core.audio.audio_system import al
        if al is None:
            return 0
        try:
            st = ctypes.c_int()
            al.alGetSourcei(src, _AL_SAMPLE_OFFSET, ctypes.byref(st))
            return st.value
        except Exception:
            return 0

    def _add_source(self, pcm, last, off, gain, looping, frame, frame_len):
        n = len(pcm)
        if n <= 0 or last >= n:
            return
        off = min(off, n)
        count = off - last
        if count <= 0:
            return
        if not looping and last + count > n:
            count = n - last
        if count > frame_len:
            count = frame_len
        head = last
        remaining = count
        while remaining > 0:
            avail = n - head
            if avail <= 0:
                if not looping:
                    break
                head = 0
                avail = n
            take = min(avail, remaining)
            mix_add(frame, 0, pcm, head, take, gain)
            head += take
            remaining -= take
            if head >= n and looping:
                head = 0

    def _add_source_stereo(self, pcm, last, off, gain, looping, frame, frame_len):
        n = len(pcm)
        if n <= 0 or last >= n:
            return
        off = min(off, n)
        count = off - last
        if count <= 0:
            return
        if not looping and last + count > n:
            count = n - last
        if count > frame_len:
            count = frame_len
        head = last
        remaining = count
        while remaining > 0:
            avail = n - head
            if avail <= 0:
                if not looping:
                    break
                head = 0
                avail = n
            take = min(avail, remaining)
            mix_add(frame[0], 0, np.ascontiguousarray(pcm[head:head + take, 0]), 0, take, gain)
            mix_add(frame[1], 0, np.ascontiguousarray(pcm[head:head + take, 1]), 0, take, gain)
            head += take
            remaining -= take
            if head >= n and looping:
                head = 0

    def update(self):
        from core.audio.audio_system import AudioSourceManager
        mgr = AudioSourceManager.instance()
        playing = []
        if mgr is not None:
            for src in mgr.get_active_source_ids():
                if not self._source_playing(src):
                    continue
                clip = mgr._source_clips.get(src)
                if clip is None:
                    continue
                pcm = clip.ensure_pcm()
                if pcm is None:
                    continue
                info = mgr._source_info.get(src) or {}
                playing.append((src, clip, pcm, info))
        active_ids = {s[0] for s in playing}
        self._active = len(playing)
        frame_len = 0
        rates = []
        for src, clip, pcm, info in playing:
            off = self._sample_offset(src)
            rate = getattr(clip, "_vis_rate", 0) or clip.sample_rate or 48000
            prev = self._src_state.get(src)
            if prev is None:
                self._src_state[src] = (off, rate)
                continue
            last = prev[0]
            delta = off - last
            if delta < 0:
                delta = off if info.get("looping", False) else 0
            if delta > frame_len:
                frame_len = delta
            rates.append(rate)
        self._src_state = {k: v for k, v in self._src_state.items() if k in active_ids}
        if frame_len > RING_SIZE:
            frame_len = RING_SIZE
        if frame_len > 0:
            frame = np.zeros(frame_len, dtype=np.float32)
            frame_st = (np.zeros(frame_len, dtype=np.float32), np.zeros(frame_len, dtype=np.float32))
            for src, clip, pcm, info in playing:
                off = self._sample_offset(src)
                prev = self._src_state.get(src)
                if prev is None:
                    continue
                last = prev[0]
                gain = float((mgr._source_gains.get(src, 1.0) if mgr is not None else 1.0) or 1.0)
                gain = min(1.0, max(0.0, gain))
                self._add_source(pcm, last, off, gain, info.get("looping", False), frame, frame_len)
                spcm = clip.ensure_stereo_pcm()
                if spcm is not None:
                    self._add_source_stereo(spcm, last, off, gain, info.get("looping", False), frame_st, frame_len)
                self._src_state[src] = (off, getattr(clip, "_vis_rate", 0) or clip.sample_rate or 48000)
            self._head = push_samples(self._ring, RING_SIZE, self._head, frame, 0, frame_len, 1.0)
            new_scope_head = push_samples(self._scope_l, RING_SIZE, self._scope_head, frame_st[0], 0, frame_len, 1.0)
            push_samples(self._scope_r, RING_SIZE, self._scope_head, frame_st[1], 0, frame_len, 1.0)
            self._scope_head = new_scope_head
        elif len(active_ids):
            self._ring *= _DECAY
            self._scope_l *= _DECAY
            self._scope_r *= _DECAY
        else:
            self._ring *= _DECAY_IDLE
            self._scope_l *= _DECAY_IDLE
            self._scope_r *= _DECAY_IDLE
        if rates:
            self._sample_rate = max(rates)
        energy = float(np.abs(self._ring).max())
        if frame_len > 0:
            self._has_signal = True
        elif energy < 1e-4:
            self._has_signal = False
        self._analyze()

    def _analyze(self):
        extract_last(self._ring, RING_SIZE, self._head, self._wave)
        max_db = analyze_spectrum(self._ring, RING_SIZE, self._head, FFT_SIZE,
                                  self._spec_db, SPEC_BINS, float(self._sample_rate))
        a = _ATTACK
        r = _RELEASE
        cur = self._spec_db
        prev = self._spec_prev
        gate = np.where(cur >= prev, a, r)
        smooth = prev * (1.0 - gate) + cur * gate
        self._spec_prev = smooth
        smooth = np.clip(smooth, -120.0, max_db)
        top = max(float(smooth.max()), -18.0)
        floor = top - 60.0
        span = max(top - floor, 1e-3)
        self._spec_top_db = top
        self._spec_floor_db = floor
        self._spec[:] = np.clip((smooth - floor) / span, 0.0, 1.0)
        hold_db = np.maximum(smooth, self._spec_hold_db - self._hold_decay)
        self._spec_hold_db[:] = hold_db
        self._spec_hold[:] = np.clip((hold_db - floor) / span, 0.0, 1.0)
        cur_peak = float(np.abs(self._wave).max())
        self._peak = max(self._peak * 0.94, max(cur_peak, 1e-4))
        self._level = min(1.0, self._peak)
        self._rms = float(np.sqrt(np.mean(np.square(self._wave))))
        sr = max(float(self._sample_rate), 1.0)
        lmin = np.log10(20.0)
        lmax = np.log10(sr * 0.5)
        if lmax <= lmin:
            lmax = lmin + 1.0
        n = SPEC_BINS
        edges = np.logspace(lmin, lmax, n + 1)
        cents = np.sqrt(edges[:-1] * edges[1:])
        db = np.asarray(self._spec_db, dtype=np.float64)
        dbn = np.where(np.isfinite(db), db, -120.0)
        arg = int(np.argmax(dbn))
        self._peak_freq = float(cents[arg]) if n > 0 else 0.0
        w = np.maximum(dbn + 120.0, 0.0)
        if float(w.sum()) > 1e-6:
            self._centroid = float(np.sum(cents * w) / np.sum(w))
        else:
            self._centroid = 0.0


def get_analyzer():
    a = AudioAnalyzer.instance()
    if a is None:
        a = AudioAnalyzer()
    return a
