# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import math
import os
import shutil

import numpy as np

_NTSC_SR = 8287.0
_AMIGA_CLOCK = 3546894.6
_RENDER_VERSION = 3

_MOD_PERIODS = {
    856, 808, 762, 720, 678, 640, 604, 570, 538, 508, 480, 453,
    428, 404, 381, 360, 339, 320, 302, 285, 269, 254, 240, 226,
    214, 202, 190, 180, 169, 160, 151, 143, 135, 127, 120, 113,
}

_NOTE_NAMES = ["C-", "C#", "D-", "D#", "E-", "F-", "F#", "G-", "G#", "A-", "A#", "B-"]


def _note_to_midi(note_str: str) -> int | None:
    s = str(note_str).strip().upper()
    if not s or s == "OFF":
        return None
    if len(s) < 3:
        return None
    try:
        name = s[:2]
        octave = int(s[2:])
        idx = _NOTE_NAMES.index(name)
    except (ValueError, IndexError):
        return None
    return (octave + 1) * 12 + idx


def _parse_effect(effect: str) -> tuple[str, int] | None:
    if not effect:
        return None
    cmd = effect[0]
    arg = 0
    try:
        arg = int(effect[1:], 16)
    except ValueError:
        return None
    return cmd, arg


class _Voice:
    __slots__ = ("active", "sample", "wave", "loop_start", "loop_end", "loop_mode",
                 "pos", "vol", "pan", "freq", "base_period", "arp_base",
                 "porta_target", "porta_speed", "vib_pos", "vib_depth", "vib_speed",
                 "vol_up", "vol_down", "offset", "delay_tick", "cut_tick",
                 "retrig_every", "retrig_count", "finetune", "last_volslide",
                 "pitch_mode", "base_freq")

    def __init__(self):
        self.active = False
        self.sample = None
        self.wave = np.zeros(1, dtype=np.float32)
        self.loop_start = 0
        self.loop_end = 0
        self.loop_mode = 0
        self.pos = 0.0
        self.vol = 0.0
        self.pan = 128
        self.freq = 110.0
        self.base_period = 0
        self.arp_base = 0
        self.porta_target = 0
        self.porta_speed = 0
        self.vib_pos = 0
        self.vib_depth = 0
        self.vib_speed = 0
        self.vol_up = 0
        self.vol_down = 0
        self.offset = 0
        self.delay_tick = -1
        self.cut_tick = -1
        self.retrig_every = 0
        self.retrig_count = 0
        self.finetune = 0
        self.last_volslide = 0
        self.pitch_mode = "mod"
        self.base_freq = 0.0


class TrackerSoftwareRenderer:
    METER_BLOCK_SEC = 0.1

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.voices: list[_Voice] = []
        self.n_channels = 0
        self.channel_gains: list[float] = []
        self.master_gain: float = 1.0
        self._arange_cache: dict[int, np.ndarray] = {}
        self.playback_map: list[tuple] = []
        self.meter_peaks: list[list[float]] = []
        self.meter_block_sec: float = self.METER_BLOCK_SEC
        self._meter_block: list[float] = []
        self._meter_blocks: list[list[float]] = []
        self._meter_cursor: int = 0

    @staticmethod
    def db_to_gain(db: float) -> float:
        if db <= -60.0:
            return 0.0
        return 10.0 ** (db / 20.0)

    def set_channel_db(self, ch: int, db: float):
        while len(self.channel_gains) <= ch:
            self.channel_gains.append(1.0)
        self.channel_gains[ch] = self.db_to_gain(db)

    def get_meter_data(self) -> tuple[float, list[list[float]]]:
        return self.meter_block_sec, self.meter_peaks

    def _voice_frequency(self, v: _Voice, arp_semi: int, tick_cycle: int) -> float:
        if v.pitch_mode == "mod":
            period = v.base_period
            period = max(40.0, min(856.0, period))
            f = _AMIGA_CLOCK / period
            if tick_cycle == 1 and arp_semi:
                f *= 2.0 ** (arp_semi / 12.0)
            elif tick_cycle == 2 and arp_semi:
                f *= 2.0 ** (arp_semi / 12.0)
            return f
        base = v.base_freq
        if tick_cycle == 1 and arp_semi:
            base *= 2.0 ** (arp_semi / 12.0)
        elif tick_cycle == 2 and arp_semi:
            base *= 2.0 ** (arp_semi / 12.0)
        return base

    def _base_freq_from_song(self, song, fmt: str, instrument_idx: int, midi: int) -> float:
        if midi is None:
            midi = 60
        if fmt == "s3m":
            c2spd = 8363.0
            try:
                if instrument_idx > 0 and instrument_idx <= len(song.samples):
                    c2spd = float(getattr(song.samples[instrument_idx - 1], "c2spd", 8363) or 8363)
            except Exception:
                pass
            return c2spd * 2.0 ** ((midi - 60) / 12.0)
        rel = 0
        fine = 0
        try:
            inst = song.instruments[instrument_idx - 1] if instrument_idx >= 1 and instrument_idx <= len(
                song.instruments) else None
            if inst is not None and inst.samples:
                smp = inst.samples[0]
                rel = int(getattr(smp, "relative_note", 0) or 0)
                fine = int(getattr(inst, "vibrato_depth", 0) or 0)
                if hasattr(smp, "finetune"):
                    fine = int(getattr(smp, "finetune", 0) or 0)
        except Exception:
            pass
        ft2_midi = midi - 12
        return 8363.0 * 2.0 ** ((ft2_midi - 60 + rel) / 12.0 + fine / (128.0 * 12.0))

    def _song_format(self, song) -> str:
        return getattr(song, "file_extension", "mod")

    def _reset(self, song):
        fmt = self._song_format(song)
        try:
            self.n_channels = int(song.n_channels)
        except Exception:
            self.n_channels = getattr(song, "patterns", None) and song.patterns and song.patterns[0].n_channels or 4
        while len(self.voices) < self.n_channels:
            self.voices.append(_Voice())
        for v in self.voices:
            v.active = False
        self._meter_block = [0.0] * self.n_channels
        self._meter_blocks = []
        self._meter_cursor = 0
        self.meter_peaks = []
        self.meter_block_sec = self.METER_BLOCK_SEC
        self.playback_map = []

    def get_playback_map(self) -> list[tuple]:
        return list(self.playback_map)

    def render(self, song, max_seconds: float | None = None,
               progress=None, should_cancel=None) -> np.ndarray | None:
        try:
            initial_bpm = int(getattr(song, "initial_bpm", None) or getattr(song, "bpm", 125) or 125)
        except Exception:
            initial_bpm = 125
        try:
            speed = int(getattr(song, "initial_speed", None) or getattr(song, "speed", 6) or 6)
        except Exception:
            speed = 6
        if speed < 1:
            speed = 6
        if initial_bpm < 32:
            initial_bpm = 125
        self._reset(song)
        sequence = list(getattr(song, "pattern_seq", []))
        patterns = list(getattr(song, "patterns", []))
        sr = self.sample_rate
        total = int(max_seconds * sr) if max_seconds else None
        total_sec = float(max_seconds) if max_seconds else None
        if total_sec is None:
            try:
                info = (getattr(song, "get_song_info", lambda: {})() or {})
                total_sec = float(info.get("duration_seconds") or 0) or None
            except Exception:
                total_sec = None
        left = []
        right = []
        elapsed_frames = 0
        bpm = initial_bpm
        next_bpm = bpm
        seq_idx = 0
        pattern_loop_start_row = None
        pattern_loop_counts: dict[int, int] = {}
        back_jumps = 0
        song_end = False
        while seq_idx < len(sequence) and not song_end:
            pat_uniq = sequence[seq_idx]
            if pat_uniq < 0 or pat_uniq >= len(patterns):
                seq_idx += 1
                continue
            pat = patterns[pat_uniq]
            n_rows = int(pat.n_rows)
            row = 0
            row_speed = speed
            while row < n_rows:
                block, row_speed = self._render_row(
                    song, pat, row, seq_idx, speed, bpm, elapsed_frames, total)
                if block is not None and len(block[0]) > 0:
                    left.append(block[0])
                    right.append(block[1])
                    elapsed_frames += len(block[0])
                    self.playback_map.append((elapsed_frames / sr, seq_idx, row))
                    if progress is not None:
                        progress(elapsed_frames / sr, total_sec)
                    if total is not None and elapsed_frames >= total:
                        return self._assemble(left, right)
                if should_cancel is not None and should_cancel():
                    return None
                new_speed = self._row_speed_change(song, pat, row)
                if new_speed:
                    speed = new_speed
                if self._bpm_change(song, pat, row):
                    nb = self._bpm_change(song, pat, row)
                    if 32 <= nb <= 255:
                        next_bpm = nb
                bpm = next_bpm
                continue_row = self._loop_directive(song, pat, row, pattern_loop_start_row, pattern_loop_counts)
                if continue_row is not None:
                    pattern_loop_start_row, row = continue_row
                    continue
                if self._jump_directive(song, pat, row, is_jump=True):
                    target = self._jump_target(song, pat, row, seq_idx, sequence)
                    row = self._pattern_break_row(song, pat, row)
                    if target <= seq_idx:
                        back_jumps += 1
                        if back_jumps > 2:
                            song_end = True
                            break
                    seq_idx = min(max(int(target), 0), len(sequence))
                    pattern_loop_start_row = None
                    pattern_loop_counts = {}
                    break
                if self._break_directive(song, pat, row):
                    seq_idx += 1
                    row = self._pattern_break_row(song, pat, row)
                    pattern_loop_start_row = None
                    pattern_loop_counts = {}
                    break
                row += 1
            else:
                pattern_loop_start_row = None
                pattern_loop_counts = {}
                seq_idx += 1
            if total is not None and elapsed_frames >= total:
                break
        return self._assemble(left, right)

    def _row_speed_change(self, song, pat, row) -> int:
        fmt = self._song_format(song)
        for ch in range(pat.n_channels):
            cell = pat.data[ch][row]
            efx = _parse_effect(getattr(cell, "effect", ""))
            if not efx:
                continue
            if fmt == "s3m":
                if efx[0] == "A" and 1 <= efx[1] <= 255:
                    return efx[1]
            elif efx[0] == "F" and 1 <= efx[1] <= 31:
                return efx[1]
        return 0

    def _bpm_change(self, song, pat, row) -> int:
        fmt = self._song_format(song)
        for ch in range(pat.n_channels):
            cell = pat.data[ch][row]
            efx = _parse_effect(getattr(cell, "effect", ""))
            if not efx:
                continue
            if fmt == "s3m":
                if efx[0] == "T" and efx[1] >= 32:
                    return efx[1]
            elif efx[0] == "F" and efx[1] >= 32:
                return efx[1]
        return 0

    def _loop_directive(self, song, pat, row, loop_start, loop_counts) -> tuple | None:
        fmt = self._song_format(song)
        for ch in range(pat.n_channels):
            cell = pat.data[ch][row]
            efx = _parse_effect(getattr(cell, "effect", ""))
            if not efx:
                continue
            if fmt == "s3m":
                if efx[0] != "S" or (efx[1] >> 4) != 0xB:
                    continue
            elif efx[0] != "E" or (efx[1] >> 4) != 0xE:
                continue
            x = efx[1] & 0x0F
            if x == 0:
                return (row, row + 1)
            if loop_start is not None:
                counts = loop_counts.get(ch, 0) + 1
                if counts < x:
                    loop_counts[ch] = counts
                    return (loop_start, loop_start)
                loop_counts[ch] = 0
        return None

    def _jump_directive(self, song, pat, row, is_jump: bool) -> bool:
        for ch in range(pat.n_channels):
            cell = pat.data[ch][row]
            efx = _parse_effect(getattr(cell, "effect", ""))
            if efx and efx[0] == "B":
                return True
        return False

    def _jump_target(self, song, pat, row, seq_idx, sequence) -> int:
        for ch in range(pat.n_channels):
            cell = pat.data[ch][row]
            efx = _parse_effect(getattr(cell, "effect", ""))
            if efx and efx[0] == "B":
                return efx[1]
        return seq_idx + 1

    def _break_directive(self, song, pat, row) -> bool:
        fmt = self._song_format(song)
        want = "C" if fmt == "s3m" else "D"
        for ch in range(pat.n_channels):
            cell = pat.data[ch][row]
            efx = _parse_effect(getattr(cell, "effect", ""))
            if efx and efx[0] == want:
                return True
        return False

    def _pattern_break_row(self, song, pat, row) -> int:
        fmt = self._song_format(song)
        want = "C" if fmt == "s3m" else "D"
        n_rows = int(getattr(pat, "n_rows", 64) or 64)
        for ch in range(pat.n_channels):
            cell = pat.data[ch][row]
            efx = _parse_effect(getattr(cell, "effect", ""))
            if efx and efx[0] == want:
                arg = efx[1]
                target = (arg >> 4) * 10 + (arg & 0x0F)
                return max(0, min(int(target), n_rows))
        return 0

    def _render_row(self, song, pat, row_idx, seq_idx, speed, bpm, elapsed_frames, total) -> tuple:
        efxs = []
        fmt = self._song_format(song)
        for ch in range(pat.n_channels):
            cell = pat.data[ch][row_idx]
            period = getattr(cell, "period", "") or ""
            inst = int(getattr(cell, "instrument_idx", 0) or 0)
            effect = getattr(cell, "effect", "") or ""
            v = self.voices[ch]
            efx = _parse_effect(effect)
            efxs.append((ch, efx))
            delay = -1
            cut = -1
            if fmt != "s3m" and efx and efx[0] == "E":
                sub = efx[1] >> 4
                x = efx[1] & 0x0F
                if sub == 0xC:
                    cut = x
                elif sub == 0xD:
                    delay = x
            v.cut_tick = cut
            v.delay_tick = delay
            porta = False
            if fmt != "s3m" and efx and efx[0] in ("3", "5"):
                porta = True
            if period and period.upper() != "OFF" and not porta:
                self._start_voice(song, v, period, inst, ch)
            elif period and period.upper() == "OFF":
                v.active = False
            elif inst and not period:
                smp = self._sample_for_instrument(song, inst)
                if smp is not None:
                    v.vol = self._sample_volume(smp) / 64.0
            self._apply_volume_column(cell, v)
            self._apply_row_effect(song, v, ch, efx, inst)
        tick_frames = int(round(self.sample_rate * 2.5 / bpm))
        if tick_frames < 1:
            tick_frames = 1
        left_chunks = []
        right_chunks = []
        for tick in range(speed):
            for ch in range(pat.n_channels):
                v = self.voices[ch]
                _, efx = efxs[ch]
                self._tick_effect(song, v, efx, tick, ch)
                if v.delay_tick == tick:
                    cell = pat.data[ch][row_idx]
                    period = getattr(cell, "period", "") or ""
                    inst = int(getattr(cell, "instrument_idx", 0) or 0)
                    if period and period.upper() != "OFF":
                        self._start_voice(song, v, period, inst, ch)
                if v.cut_tick == tick:
                    v.vol = 0.0
            l, r = self._mix_tick(tick_frames)
            left_chunks.append(l)
            right_chunks.append(r)
        l_all = np.concatenate(left_chunks) if left_chunks else np.zeros(1, dtype=np.float32)
        r_all = np.concatenate(right_chunks) if right_chunks else np.zeros(1, dtype=np.float32)
        return ((l_all, r_all), speed)

    def _sample_for_instrument(self, song, inst_idx: int):
        fmt = self._song_format(song)
        try:
            if fmt == "mod":
                if 1 <= inst_idx <= len(song.samples):
                    return song.samples[inst_idx - 1]
            elif fmt == "s3m":
                if 1 <= inst_idx <= len(song.samples):
                    return song.samples[inst_idx - 1]
            elif fmt == "xm":
                if 1 <= inst_idx <= len(song.instruments):
                    inst = song.instruments[inst_idx - 1]
                    if inst.samples:
                        return inst.samples[0]
        except Exception:
            return None
        return None

    def _sample_volume(self, smp) -> int:
        return int(getattr(smp, "volume", 64) or 64)

    def _sample_wave(self, smp) -> np.ndarray:
        try:
            raw = smp.waveform
            if getattr(smp, "is_16bit", False):
                arr = np.asarray(raw, dtype=np.float32) / 32768.0
            else:
                arr = np.asarray(raw, dtype=np.float32) / 128.0
        except Exception:
            arr = np.zeros(1, dtype=np.float32)
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _sample_loop(self, smp) -> tuple[int, int, int]:
        start = int(getattr(smp, "repeat_point", 0) or 0)
        length = int(getattr(smp, "repeat_len", 0) or 0)
        mode = 1
        loop_type = getattr(smp, "loop_type", None)
        if loop_type is not None:
            mode = int(loop_type)
        n = int(getattr(smp, "length", 0) or len(getattr(smp, "waveform", [])))
        end = min(start + length, n) if length > 1 else n
        if length <= 1 or start >= n:
            start = n
            end = n
            mode = 0
        return start, end, mode

    def _start_voice(self, song, v: _Voice, period: str, inst: int, ch: int, retrig: bool = False):
        smp = self._sample_for_instrument(song, inst)
        if smp is None:
            if period:
                try:
                    fl = self._finetune_for_note(song, inst, period)
                    v.base_period = self._period_for_str(song, period)
                    v.finetune = fl
                    v.porta_target = v.base_period
                    v.freq = self._freq_for_base(v)
                except Exception:
                    pass
            return
        v.sample = smp
        v.wave = self._sample_wave(smp)
        v.loop_start, v.loop_end, v.loop_mode = self._sample_loop(smp)
        pos = float(v.offset)
        if pos >= len(v.wave):
            pos = float(len(v.wave) - 1)
        v.pos = max(0.0, pos)
        v.vol = max(0.0, min(1.0, self._sample_volume(smp) / 64.0))
        v.pan = self._pan_for_sample(smp)
        v.active = True
        v.offset = 0
        fmt = self._song_format(song)
        if fmt == "mod":
            try:
                v.base_period = song.note_to_period(period)
            except Exception:
                v.base_period = 428
            v.pitch_mode = "mod"
            v.finetune = int(getattr(smp, "finetune", 0) or 0)
            v.porta_target = v.base_period
            v.freq = self._freq_for_base(v)
        else:
            midi = _note_to_midi(period)
            v.base_freq = self._base_freq_from_song(song, fmt, inst, midi)
            v.pitch_mode = "xm" if fmt == "xm" else "s3m"
            v.base_period = 0
            v.freq = v.base_freq

    def _period_for_str(self, song, period: str) -> int:
        try:
            return song.note_to_period(period)
        except Exception:
            return 428

    def _finetune_for_note(self, song, inst, period) -> int:
        smp = self._sample_for_instrument(song, inst)
        if smp is not None:
            return int(getattr(smp, "finetune", 0) or 0)
        return 0

    def _freq_for_base(self, v: _Voice) -> float:
        p = v.base_period
        p = max(40.0, min(856.0, p))
        base = _AMIGA_CLOCK / p
        ft = v.finetune
        if ft:
            base *= 2.0 ** (ft / 96.0)
        return base

    def _pan_for_sample(self, smp) -> int:
        pan = getattr(smp, "panning", None)
        if pan is not None:
            return max(0, min(255, int(pan)))
        return 128

    def _apply_row_effect(self, song, v: _Voice, ch: int, efx, inst: int):
        if not efx:
            return
        cmd, arg = efx
        if self._song_format(song) == "s3m" and cmd in ("A", "C", "E"):
            return
        if cmd == "0":
            v.arp_base = v.base_period or 0
        elif cmd == "1":
            v.porta_speed = arg
        elif cmd == "2":
            v.porta_speed = arg
        elif cmd == "3":
            v.porta_speed = arg
        elif cmd == "4":
            v.vib_speed = (arg >> 4) or 1
            v.vib_depth = arg & 0x0F
        elif cmd == "5":
            v.porta_speed = arg >> 4
            v.vol_down = arg & 0x0F
            v.vol_up = arg >> 4
        elif cmd == "6":
            v.vib_speed = (arg >> 4) or 1
            v.vib_depth = arg & 0x0F
            v.vol_down = arg & 0x0F
            v.vol_up = arg >> 4
        elif cmd == "8":
            v.pan = max(0, min(255, arg * 17))
        elif cmd == "9":
            v.offset = arg * 256
            if arg == 0:
                v.offset = 0
        elif cmd == "A":
            v.vol_up = arg >> 4
            v.vol_down = arg & 0x0F
            v.last_volslide = arg
        elif cmd == "C":
            v.vol = max(0.0, min(1.0, arg / 64.0))
        elif cmd == "E":
            sub = arg >> 4
            x = arg & 0x0F
            if sub == 0x1:
                p = v.base_period if v.pitch_mode == "mod" else 0
                if p:
                    v.base_period = max(40, p - x)
                    v.freq = self._freq_for_base(v)
                    v.porta_target = max(40, p - x)
            elif sub == 0x2:
                p = v.base_period if v.pitch_mode == "mod" else 0
                if p:
                    v.base_period = min(856, p + x)
                    v.freq = self._freq_for_base(v)
                    v.porta_target = min(856, p + x)
            elif sub == 0x9:
                v.retrig_every = x
                v.retrig_count = 0
            elif sub == 0xA:
                pass

    def _apply_volume_column(self, cell, v: _Voice):
        vol = getattr(cell, "volume", None)
        if vol is not None and vol >= 0:
            v.vol = max(0.0, min(1.0, vol / 64.0))
        cmd = getattr(cell, "vol_cmd", "") or ""
        if not cmd:
            return
        try:
            val = int(getattr(cell, "vol_val", -1))
        except (TypeError, ValueError):
            return
        if val < 0:
            return
        if cmd == "v":
            v.vol = max(0.0, min(1.0, val / 64.0))
        elif cmd == "p":
            v.pan = max(0, min(255, int(val * 4.25)))
        elif cmd == "d":
            v.vol_down = val
            v.vol_up = 0
        elif cmd == "c":
            v.vol_up = val
            v.vol_down = 0

    def _tick_effect(self, song, v: _Voice, efx, tick: int, ch: int):
        if not v.active:
            return
        if not efx:
            return
        cmd, arg = efx
        if self._song_format(song) == "s3m" and cmd in ("A", "C", "E"):
            return
        if cmd == "0":
            x = (arg >> 4) or 0
            y = arg & 0x0F
            cycle = tick % 3
            if cycle == 1 and x:
                self._apply_semi_shift(v, x)
            elif cycle == 2 and y:
                self._apply_semi_shift(v, y)
            else:
                if v.pitch_mode == "mod":
                    v.freq = self._freq_for_base(v)
                else:
                    v.freq = v.base_freq
        elif cmd == "1":
            if v.base_period:
                v.base_period = max(40, v.base_period - arg)
                v.freq = self._freq_for_base(v)
        elif cmd == "2":
            if v.base_period:
                v.base_period = min(856, v.base_period + arg)
                v.freq = self._freq_for_base(v)
        elif cmd == "3":
            if v.base_period and v.porta_target:
                diff = v.porta_target - v.base_period
                step = v.porta_speed or arg
                if diff > 0:
                    v.base_period = min(v.base_period + step, v.porta_target)
                else:
                    v.base_period = max(v.base_period - step, v.porta_target)
                v.freq = self._freq_for_base(v)
        elif cmd == "4":
            if v.base_period:
                vib = self._vibrato(tick, v)
                v.freq = self._freq_for_base(v)
                if vib:
                    p = v.base_period + vib * (v.vib_depth or 0)
                    p = max(40, min(856, p))
                    v.freq = _AMIGA_CLOCK / p
        elif cmd == "A":
            if v.vol_up:
                v.vol = min(1.0, v.vol + v.vol_up * (1.0 / 64.0))
            if v.vol_down:
                v.vol = max(0.0, v.vol - v.vol_down * (1.0 / 64.0))
        elif cmd == "E":
            sub = arg >> 4
            x = arg & 0x0F
            if sub == 0x9 and x:
                v.retrig_count += 1
                if v.retrig_count >= x:
                    v.retrig_count = 0
                    v.pos = 0.0

    def _apply_semi_shift(self, v: _Voice, semis: int):
        if v.pitch_mode == "mod" and v.base_period:
            p = v.base_period / (2.0 ** (semis / 12.0))
            v.freq = _AMIGA_CLOCK / max(40, min(856, p))
        else:
            v.freq = v.base_freq * (2.0 ** (semis / 12.0))

    def _vibrato(self, tick: int, v: _Voice) -> float:
        phase = (tick * v.vib_speed) * (math.pi * 2.0 / 64.0)
        return math.sin(phase)

    def _mix_one(self, v: _Voice, n: int) -> np.ndarray:
        if not v.active or len(v.wave) == 0 or v.vol <= 0.0:
            return np.zeros(n, dtype=np.float32)
        freq = v.freq
        if freq <= 0:
            return np.zeros(n, dtype=np.float32)
        ar = self._arange_cache.get(n)
        if ar is None:
            ar = np.arange(n, dtype=np.float64)
            self._arange_cache[n] = ar
        t = v.pos + freq / self.sample_rate * ar
        wave = v.wave
        nlen = len(wave)
        loop_start = v.loop_start
        loop_end = v.loop_end
        mode = v.loop_mode
        if mode == 0 or loop_start >= loop_end:
            idx = np.clip(t, 0, nlen - 1)
        else:
            span = loop_end - loop_start
            rel = np.mod(t - loop_start, span)
            tmod = loop_start + rel
            head = np.clip(t, 0, loop_start)
            idx = np.where(t < loop_start, head, tmod)
            if mode == 2:
                tri = np.minimum(rel, span - rel)
                idx = np.where(t < loop_start, head, loop_start + tri)
            idx = np.clip(idx, 0, nlen - 1.0)
        i0 = np.floor(idx).astype(np.int64)
        frac = idx - i0
        i1 = np.minimum(i0 + 1, nlen - 1)
        return (wave[i0] * (1.0 - frac) + wave[i1] * frac).astype(np.float32) * v.vol

    def _mix_tick(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        n = max(1, n)
        left = np.zeros(n, dtype=np.float32)
        right = np.zeros(n, dtype=np.float32)
        gains = self.channel_gains
        for ch, v in enumerate(self.voices):
            if not v.active:
                continue
            mono = self._mix_one(v, n)
            g = gains[ch] if ch < len(gains) else 1.0
            if g != 1.0:
                mono = mono * g
            if len(self._meter_block) <= ch:
                self._meter_block.extend([0.0] * (ch + 1 - len(self._meter_block)))
            try:
                peak = float(np.abs(mono).max()) if len(mono) else 0.0
            except Exception:
                peak = 0.0
            if peak > self._meter_block[ch]:
                self._meter_block[ch] = peak
            pan = v.pan
            lg = np.sqrt(max(0.0, (255.0 - pan) / 255.0))
            rg = np.sqrt(max(0.0, pan / 255.0))
            left += mono * lg
            right += mono * rg
            v.pos += v.freq * n / self.sample_rate
            if v.pos >= len(v.wave):
                if v.loop_mode != 0 and v.loop_start < v.loop_end:
                    v.pos = v.loop_start + ((v.pos - v.loop_start) % (v.loop_end - v.loop_start))
                else:
                    v.active = False
        self._meter_advance(n)
        return left, right

    def _meter_advance(self, n: int):
        block_frames = max(1, int(self.sample_rate * self.METER_BLOCK_SEC))
        self._meter_cursor += n
        while self._meter_cursor >= block_frames:
            self._meter_cursor -= block_frames
            self._meter_blocks.append(list(self._meter_block))
            self._meter_block = [0.0] * max(1, self.n_channels)

    def _flush_meter(self):
        if any(self._meter_block):
            self._meter_blocks.append(list(self._meter_block))
        self._meter_block = [0.0] * max(1, self.n_channels)
        self._meter_cursor = 0
        self.meter_peaks = self._meter_blocks
        self.meter_block_sec = self.METER_BLOCK_SEC

    def _assemble(self, left: list, right: list) -> np.ndarray:
        self._flush_meter()
        if not left:
            return np.zeros((0, 2), dtype=np.int16)
        l_all = np.concatenate(left)
        r_all = np.concatenate(right)
        n = min(len(l_all), len(r_all))
        if self.master_gain != 1.0:
            l_all = l_all * self.master_gain
            r_all = r_all * self.master_gain
        peak = float(max(np.abs(l_all[:n]).max(), np.abs(r_all[:n]).max()))
        if peak > 1.0:
            l_all = l_all / peak
            r_all = r_all / peak
        l16 = np.clip(l_all[:n], -1.0, 1.0)
        r16 = np.clip(r_all[:n], -1.0, 1.0)
        stereo = np.empty((n, 2), dtype=np.float32)
        stereo[:, 0] = l16
        stereo[:, 1] = r16
        return (stereo * 32767.0).astype(np.int16)


def _sample_for_instrument_mode(song, inst_idx: int):
    fmt = getattr(song, "file_extension", "mod")
    try:
        if fmt == "xm":
            if 1 <= inst_idx <= len(song.instruments):
                inst = song.instruments[inst_idx - 1]
                if inst.samples:
                    return inst.samples[0]
        elif 1 <= inst_idx <= len(song.samples):
            return song.samples[inst_idx - 1]
    except Exception:
        return None
    return None


def render_to_wav_file(pcm: np.ndarray, path: str, sample_rate: int = 44100):
    import wave
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        interleaved = np.zeros(pcm.shape[0] * 2, dtype=np.int16)
        interleaved[0::2] = pcm[:, 0]
        interleaved[1::2] = pcm[:, 1]
        wf.writeframes(interleaved.tobytes())


def find_external_renderer() -> str | None:
    for tool in ("openmpt123", "ffmpeg"):
        if shutil.which(tool):
            return tool
    return None


class TrackerRenderService:
    def __init__(self, cache_dir: str, sample_rate: int = 44100):
        self.cache_dir = cache_dir
        self.sample_rate = sample_rate
        os.makedirs(cache_dir, exist_ok=True)

    def _file_key(self, path: str) -> str:
        import hashlib
        try:
            with open(path, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()[:16]
        except Exception:
            h = os.path.basename(path)
        return f"{h}_{self.sample_rate}_v{_RENDER_VERSION}"

    def render_wav(self, module_path: str, quality: str = "software") -> str | None:
        cache_path = os.path.join(self.cache_dir, self._file_key(module_path) + ".wav")
        if os.path.isfile(cache_path):
            return cache_path
        song = load_module(module_path)
        if song is None:
            return None
        pcm = None
        try:
            from nodmod.song import Song
        except Exception:
            pass
        if quality == "external":
            tool = find_external_renderer()
            if tool is not None:
                try:
                    tmp_wav = cache_path + ".tmp.wav"
                    try:
                        song.render(tmp_wav, verbose=False, channels=2)
                        if os.path.isfile(tmp_wav):
                            shutil.move(tmp_wav, cache_path)
                            return cache_path
                    except Exception:
                        pass
                except Exception:
                    pass
        try:
            pcm = TrackerSoftwareRenderer(self.sample_rate).render(song)
        except Exception as e:
            from core.foundation.logger import Logger
            Logger.error(f"Tracker render failed: {e}", e)
            return None
        if pcm is None or len(pcm) == 0:
            return None
        render_to_wav_file(pcm, cache_path, self.sample_rate)
        return cache_path


def load_module(path: str):
    if not path or not os.path.isfile(path):
        return None
    try:
        from nodmod.loader import load_song
        return load_song(path, verbose=False)
    except Exception as e:
        from core.foundation.logger import Logger
        Logger.error(f"Failed to load tracker module '{path}': {e}", e)
        return None