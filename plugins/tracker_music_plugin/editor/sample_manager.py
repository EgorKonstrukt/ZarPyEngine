# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import array
import os
import wave

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def _fmt(song) -> str:
    return str(getattr(song, "file_extension", "mod") or "mod").lower().strip(".")


def _wave_to_float(smp) -> np.ndarray:
    try:
        if getattr(smp, "is_16bit", False):
            return np.asarray(smp.waveform, dtype=np.float32) / 32768.0
        return np.asarray(smp.waveform, dtype=np.float32) / 128.0
    except Exception:
        return np.zeros(0, dtype=np.float32)


def _float_to_array(floats: np.ndarray, is_16bit: bool) -> array.array:
    data = np.clip(np.nan_to_num(floats, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    if is_16bit:
        return array.array("h", np.clip(np.round(data * 32767.0), -32768, 32767).astype(np.int16))
    return array.array("b", np.clip(np.round(data * 127.0), -128, 127).astype(np.int8))


def _natural_rate(song, smp) -> int:
    f = _fmt(song)
    if f == "s3m":
        return max(1, int(getattr(smp, "c2spd", 8363) or 8363))
    if f == "xm":
        try:
            rel = int(getattr(smp, "relative_note", 0) or 0)
            fine = int(getattr(smp, "finetune", 0) or 0)
            return max(1, int(8363.0 * 2.0 ** (rel / 12.0 + fine / (128.0 * 12.0))))
        except Exception:
            return 8363
    try:
        return max(1, int(song._get_effective_sample_rate(smp, "C-5")))
    except Exception:
        return 8287


def _resample(x: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    if in_rate == out_rate or len(x) == 0:
        return x
    n_out = max(1, round(len(x) * out_rate / in_rate))
    pos = np.linspace(0, len(x) - 1, num=n_out)
    i0 = pos.astype(np.int64)
    i1 = np.clip(i0 + 1, 0, len(x) - 1)
    frac = (pos - i0).astype(np.float32)
    return (x[i0] * (1.0 - frac) + x[i1] * frac).astype(np.float32)


def _write_wav(path: str, floats: np.ndarray, rate: int) -> None:
    pcm = _float_to_array(floats, True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(max(1, int(rate)))
        wf.writeframes(pcm.tobytes())


def _read_wav(path: str):
    with wave.open(path, "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        rate = max(1, wf.getframerate())
        frames = wf.readframes(wf.getnframes())
    if not frames or sw == 0:
        return np.zeros(0, dtype=np.float32), rate
    if sw == 1:
        a = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        data = (a - 128.0) / 128.0
    elif sw == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).astype(np.uint32)
        vals = raw[0::3] | (raw[1::3] << 8) | (raw[2::3] << 16)
        vals = np.where(vals >= 0x800000, vals - 0x1000000, vals)
        data = vals.astype(np.float32) / 8388608.0
    else:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    if nch > 1:
        data = data[: (len(data) // nch) * nch].reshape(-1, nch).mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), rate


class SampleManagerWidget(QWidget):
    samplesChanged = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._song = None
        self._rows: list[dict] = []
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(3)
        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_select)
        v.addWidget(self._list, 1)
        self._info = QLabel("-")
        self._info.setWordWrap(True)
        v.addWidget(self._info)
        row1 = QHBoxLayout()
        self._export_btn = QPushButton("Export WAV")
        self._export_btn.clicked.connect(self._export)
        self._import_btn = QPushButton("Import WAV")
        self._import_btn.clicked.connect(self._import)
        row1.addWidget(self._export_btn)
        row1.addWidget(self._import_btn)
        v.addLayout(row1)
        row2 = QHBoxLayout()
        self._norm_btn = QPushButton("Norm")
        self._norm_btn.setToolTip("Normalize to peak 95%")
        self._norm_btn.clicked.connect(lambda: self._edit(self._normalize, "Normalize"))
        self._rev_btn = QPushButton("Rev")
        self._rev_btn.setToolTip("Reverse")
        self._rev_btn.clicked.connect(lambda: self._edit(self._reverse, "Reverse"))
        self._trim_btn = QPushButton("Trim")
        self._trim_btn.setToolTip("Trim leading/trailing silence")
        self._trim_btn.clicked.connect(lambda: self._edit(self._trim, "Trim"))
        self._gain_btn = QPushButton("Amp")
        self._gain_btn.setToolTip("Amplify x2")
        self._gain_btn.clicked.connect(lambda: self._edit(self._amplify, "Amplify"))
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setToolTip("Clear the sample")
        self._clear_btn.clicked.connect(self._clear)
        for b in (self._norm_btn, self._rev_btn, self._trim_btn, self._gain_btn, self._clear_btn):
            row2.addWidget(b)
        v.addLayout(row2)
        self._update_buttons()

    def load_song(self, song) -> None:
        self._song = song
        self._list.blockSignals(True)
        self._list.clear()
        self._rows = []
        if song is not None:
            self._populate(song, _fmt(song))
        self._list.blockSignals(False)
        if self._rows:
            self._list.setCurrentRow(0)
        self._update_info()

    def _populate(self, song, f: str) -> None:
        self._rows = []
        if f == "xm":
            instruments = getattr(song, "instruments", []) or []
            for inst_idx, inst in enumerate(instruments, start=1):
                name = str(getattr(inst, "name", "") or "").strip()
                samples = getattr(inst, "samples", []) or []
                if not samples:
                    self._rows.append({"inst": inst_idx, "sample": 0,
                                       "label": f"Inst {inst_idx}: {name} (no samples)"})
                    continue
                for si, smp in enumerate(samples, start=1):
                    self._rows.append({"inst": inst_idx, "sample": si, "smp": smp})
        else:
            samples = getattr(song, "samples", []) or []
            for i, smp in enumerate(samples, start=1):
                self._rows.append({"inst": 0, "sample": i, "smp": smp})
        for row in self._rows:
            item = QListWidgetItem(self._label(song, row))
            item.setData(Qt.ItemDataRole.UserRole, (row["inst"], row["sample"]))
            self._list.addItem(item)
            if row.get("smp") is None or len(getattr(row.get("smp"), "waveform", [])) == 0:
                item.setForeground(Qt.GlobalColor.gray)

    def _label(self, song, row: dict) -> str:
        smp = row.get("smp")
        idx = row["sample"]
        f = _fmt(song)
        name = str(getattr(smp, "name", "") or "").strip() if smp is not None else ""
        n = len(getattr(smp, "waveform", [])) if smp is not None else 0
        bits = ""
        if smp is not None and f == "xm":
            bits = "16b" if getattr(smp, "is_16bit", False) else "8b"
        if n == 0:
            return f"{idx}: empty"
        return f"{idx}: {name[:24] or 'unnamed'} ({n}{' ' + bits if bits else ''})"

    def _selected_row(self) -> dict | None:
        cur = self._list.currentItem()
        if cur is None or self._song is None:
            return None
        key = cur.data(Qt.ItemDataRole.UserRole)
        for row in self._rows:
            if (row["inst"], row["sample"]) == key:
                return row
        return None

    def _get_sample(self, row: dict):
        if self._song is None:
            return None
        if _fmt(self._song) == "xm":
            inst = self._song.instruments[row["inst"] - 1] if 1 <= row["inst"] <= len(self._song.instruments) else None
            if inst is None or row["sample"] <= 0 or row["sample"] > len(inst.samples):
                return None
            return inst.samples[row["sample"] - 1]
        if 1 <= row["sample"] <= len(self._song.samples):
            return self._song.samples[row["sample"] - 1]
        return None

    def _on_select(self, *_args):
        self._update_buttons()
        self._update_info()

    def _update_buttons(self) -> None:
        has = self._has_data()
        self._export_btn.setEnabled(has)
        self._import_btn.setEnabled(self._song is not None)
        for b in (self._norm_btn, self._rev_btn, self._trim_btn, self._gain_btn):
            b.setEnabled(has)
        self._clear_btn.setEnabled(has)

    def _has_data(self) -> bool:
        row = self._selected_row()
        if row is None:
            return False
        smp = self._get_sample(row)
        return smp is not None and len(getattr(smp, "waveform", [])) > 0

    def _update_info(self) -> None:
        row = self._selected_row()
        if self._song is None or row is None:
            self._info.setText("-")
            return
        smp = self._get_sample(row)
        idx = row["sample"]
        inst = row["inst"]
        if smp is None or len(getattr(smp, "waveform", [])) == 0:
            self._info.setText(f"Slot {idx} (empty)")
            return
        n = len(smp.waveform)
        rate = _natural_rate(self._song, smp)
        dur = n / float(rate)
        vol = int(getattr(smp, "volume", 64) or 64)
        parts = [f"{idx}: {n} samples", f"{rate} Hz", f"{dur:.2f}s", f"vol {vol}"]
        loop = getattr(smp, "repeat_len", 0) or 0
        if loop > 0:
            parts.append(f"loop {getattr(smp, 'repeat_point', 0)}+{loop}")
        if inst:
            parts.insert(0, f"Inst {inst}")
        self._info.setText("  |  ".join(parts))

    def _refresh_label(self, row: dict) -> None:
        cur = self._list.currentItem()
        if cur is not None and cur.data(Qt.ItemDataRole.UserRole) == (row["inst"], row["sample"]):
            cur.setText(self._label(self._song, row))
            cur.setForeground(Qt.GlobalColor.black)
        self._update_info()

    def _apply(self, row: dict, floats: np.ndarray, keep_rate: bool = True) -> None:
        if self._song is None:
            return
        smp = self._get_sample(row)
        if smp is None:
            return
        f = _fmt(self._song)
        is16 = bool(getattr(smp, "is_16bit", False)) if f == "xm" else False
        arr = _float_to_array(floats, is16)
        if f == "mod":
            self._song.set_sample_pcm_i8(row["sample"], arr, reset_meta=False)
            smp = self._song.get_sample(row["sample"])
            smp.repeat_point = 0
            smp.repeat_len = 0
            smp.sanitize_loop()
            try:
                self._song._update_n_actual_samples()
            except Exception:
                pass
        elif f == "s3m":
            smp.waveform = arr
            if not keep_rate:
                smp.c2spd = _natural_rate(self._song, smp)
            smp.repeat_point = 0
            smp.repeat_len = 0
            smp.sanitize_loop()
            try:
                self._song._update_n_actual_samples()
            except Exception:
                pass
        else:
            smp.waveform = arr
            smp.is_16bit = is16
            smp.loop_type = 0
            smp.repeat_point = 0
            smp.repeat_len = 0
            smp.sanitize_loop()
        self._refresh_label(row)
        self.samplesChanged.emit()

    def _current_floats(self) -> np.ndarray | None:
        row = self._selected_row()
        if row is None:
            return None
        smp = self._get_sample(row)
        if smp is None or len(getattr(smp, "waveform", [])) == 0:
            return None
        return _wave_to_float(smp)

    def _export(self) -> None:
        row = self._selected_row()
        floats = self._current_floats()
        if row is None or floats is None or self._song is None:
            return
        smp = self._get_sample(row)
        rate = _natural_rate(self._song, smp)
        name_lines = str(getattr(smp, "name", "") or "").strip().splitlines() if smp is not None else []
        name = name_lines[0] if name_lines else ""
        base = "".join(c if c.isalnum() or c in "_- " else "_" for c in name).strip() or f"sample_{row['sample']}"
        default = f"{base}.wav"
        path, _ = QFileDialog.getSaveFileName(self, "Export Sample WAV", default, "WAV (*.wav)")
        if not path:
            return
        try:
            _write_wav(path, floats, rate)
            self._info.setText(f"Exported to {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, "Sample", f"Export failed: {e}")

    def _import(self) -> None:
        if self._song is None:
            return
        row = self._selected_row()
        if row is None or row.get("smp") is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import Sample WAV", "", "WAV (*.wav);;All files (*)")
        if not path:
            return
        try:
            floats, sr = _read_wav(path)
            if len(floats) == 0:
                QMessageBox.warning(self, "Sample", "The selected file contains no audio data.")
                return
            target = _natural_rate(self._song, self._get_sample(row))
            if sr != target:
                floats = _resample(floats, sr, target)
            self._apply(row, floats, keep_rate=False)
            self._info.setText(f"Imported {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, "Sample", f"Import failed: {e}")

    def _clear(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        self._apply(row, np.zeros(0, dtype=np.float32))

    def _edit(self, fn, title: str) -> None:
        row = self._selected_row()
        floats = self._current_floats()
        if row is None or floats is None:
            return
        try:
            out = fn(floats)
            self._apply(row, out)
        except Exception as e:
            QMessageBox.warning(self, "Sample", f"{title} failed: {e}")

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(x)) if len(x) else 0.0
        if peak <= 1e-9 or np.isnan(peak):
            return x
        return (x / peak * 0.95).astype(np.float32)

    @staticmethod
    def _reverse(x: np.ndarray) -> np.ndarray:
        return np.flip(x).copy()

    @staticmethod
    def _trim(x: np.ndarray) -> np.ndarray:
        floor = 1.0 / 128.0
        mask = np.abs(x) > floor
        if not np.any(mask):
            return np.zeros(0, dtype=np.float32)
        idx = np.flatnonzero(mask)
        return x[idx[0]:idx[-1] + 1].copy()

    @staticmethod
    def _amplify(x: np.ndarray) -> np.ndarray:
        return (x * 2.0).astype(np.float32)