# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import bisect
import hashlib
import json
import os
import shutil
import threading
import time

import numpy as np
from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.foundation.plugin_manager import _zarin_user_dir
from ..renderer import (_RENDER_VERSION, TrackerSoftwareRenderer, load_module,
                        render_to_wav_file)
from .instrument_list import InstrumentListWidget
from .mixer_widget import MixerWidget
from .module_info import ModuleInfoWidget
from .pattern_editor import PatternEditorWidget
from .sample_manager import SampleManagerWidget

_METER_MAX_BLOCKS = 2048


def _downsample_peaks(peaks: list, max_blocks: int) -> list:
    try:
        n = len(peaks)
        if n <= max_blocks or n == 0:
            return [list(b) for b in peaks]
        stride = n / max_blocks
        out = []
        for i in range(max_blocks):
            lo = int(i * stride)
            hi = max(lo + 1, int((i + 1) * stride))
            cols = list(zip(*peaks[lo:hi]))
            out.append([max(c) for c in cols] if cols else [])
        return out
    except Exception:
        return []


def _write_peaks_sidecar(wav_path: str, block_sec: float, peaks: list, playmap: list = ()):
    try:
        with open(wav_path + ".peaks.json", "w", encoding="utf-8") as f:
            json.dump({"block_sec": block_sec,
                       "peaks": [[round(float(v), 3) for v in b] for b in peaks],
                       "playmap": [[round(float(e), 4), int(s), int(r)] for e, s, r in playmap]}, f)
    except Exception:
        pass


def _load_peaks_sidecar(wav_path: str):
    try:
        with open(wav_path + ".peaks.json", "r", encoding="utf-8") as f:
            d = json.load(f)
        block_sec = float(d.get("block_sec") or 0.1)
        peaks = [[float(v) for v in b] for b in (d.get("peaks") or [])]
        playmap = [(float(e), int(s), int(r)) for e, s, r in (d.get("playmap") or [])]
        return block_sec, peaks, playmap
    except Exception:
        return None


class _PreviewJob(QObject):
    progress = pyqtSignal(float, object)
    done = pyqtSignal(int, str, float, float, object, object)
    failed = pyqtSignal(int, str)

    def __init__(self, job_id: int, kind: str, path: str, out_path: str,
                 sample_rate: int, muted: set, gains_db: list,
                 bake_master_gain: float | None):
        super().__init__()
        self._job_id = job_id
        self._kind = kind
        self._path = path
        self._out_path = out_path
        self._sample_rate = sample_rate
        self._muted = set(muted)
        self._gains_db = list(gains_db)
        self._bake_master_gain = bake_master_gain
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        try:
            if self._kind == "save":
                song = load_module(self._path)
                if song is None:
                    self.failed.emit(self._job_id, "Failed to load module.")
                    return
                song.save(self._out_path, verbose=False)
                self.done.emit(self._job_id, self._out_path, 0.0, 0.1, [], [])
                return
            if os.path.isfile(self._out_path):
                self._emit_cached()
                return
            song = load_module(self._path)
            if song is None:
                self.failed.emit(self._job_id, "Failed to load module.")
                return
            for ch in self._muted:
                try:
                    song.mute_channel(ch)
                except Exception:
                    pass
            renderer = TrackerSoftwareRenderer(self._sample_rate)
            for ch, db in enumerate(self._gains_db):
                try:
                    renderer.set_channel_db(ch, float(db))
                except Exception:
                    pass
            last_emit = [0.0]

            def prog(elapsed: float, total):
                now = time.perf_counter()
                if now - last_emit[0] >= 0.15:
                    last_emit[0] = now
                    self.progress.emit(float(elapsed), total)

            pcm = renderer.render(song, progress=prog, should_cancel=self._cancel.is_set)
            if pcm is None:
                self.failed.emit(self._job_id, "cancelled")
                return
            if self._bake_master_gain is not None and self._bake_master_gain != 1.0:
                try:
                    pcm = (pcm.astype(np.float32) * float(self._bake_master_gain)).clip(
                        -32768.0, 32767.0).astype(np.int16)
                except Exception:
                    pass
            render_to_wav_file(pcm, self._out_path, self._sample_rate)
            block_sec, peaks = renderer.get_meter_data()
            peaks = _downsample_peaks(peaks, _METER_MAX_BLOCKS)
            try:
                playmap = renderer.get_playback_map()
            except Exception:
                playmap = []
            _write_peaks_sidecar(self._out_path, block_sec, peaks, playmap)
            duration = len(pcm) / float(self._sample_rate)
            self.done.emit(self._job_id, self._out_path, duration, block_sec, peaks, playmap)
        except Exception as e:
            try:
                self.failed.emit(self._job_id, str(e))
            except Exception:
                pass

    def _emit_cached(self):
        try:
            import wave
            duration = 0.0
            block_sec = TrackerSoftwareRenderer.METER_BLOCK_SEC
            peaks: list = []
            playmap: list = []
            try:
                with wave.open(self._out_path, "rb") as wf:
                    duration = wf.getnframes() / float(wf.getframerate() or self._sample_rate)
            except Exception:
                pass
            try:
                sidecar = _load_peaks_sidecar(self._out_path)
                if sidecar is not None:
                    block_sec, peaks, playmap = sidecar
            except Exception:
                pass
            self.done.emit(self._job_id, self._out_path, duration, block_sec, peaks, playmap)
        except Exception as e:
            self.failed.emit(self._job_id, str(e))


class TrackerEditorWidget(QWidget):
    SAMPLE_RATES = (22050, 44100, 48000)

    def __init__(self, engine, plugin):
        super().__init__()
        self._engine = engine
        self._plugin = plugin
        self._song = None
        self._song_path = ""
        self._pattern_seq_index = 0
        self._edit_rev = 0
        self._source_id = 0
        self._playing = False
        self._play_start = 0.0
        self._paused_elapsed = 0.0
        self._thread: QThread | None = None
        self._worker: _PreviewJob | None = None
        self._job_id = 0
        self._job_kind = "preview"
        self._job_dest = ""
        self._pending_autoplay = False
        self._preview_wav = ""
        self._preview_duration = 0.0
        self._meter_block_sec = TrackerSoftwareRenderer.METER_BLOCK_SEC
        self._meter_peaks: list = []
        self._playback_map: list = []
        self._playback_ends: list = []
        self._timeline: list = []
        self._timeline_ends: list = []
        self._build_ui()
        self._rerender_timer = QTimer(self)
        self._rerender_timer.setSingleShot(True)
        self._rerender_timer.setInterval(400)
        self._rerender_timer.timeout.connect(self._on_rerender_timeout)
        self._ticker = QTimer(self)
        self._ticker.timeout.connect(self._on_tick)
        self._ticker.start(50)
        self._pattern_editor.patternChanged.connect(self._on_pattern_edited)
        self._module_info.metadataEdited.connect(self._on_metadata_edited)
        self._save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self._save_shortcut.activated.connect(self._save_module)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        top = QHBoxLayout()
        self._open_btn = QPushButton("Open Module...")
        self._open_btn.clicked.connect(self._open_module)
        top.addWidget(self._open_btn)

        self._save_btn = QPushButton("Save")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_module)
        top.addWidget(self._save_btn)

        self._seq_combo = QComboBox()
        self._seq_combo.setMinimumWidth(160)
        self._seq_combo.currentIndexChanged.connect(self._on_seq_changed)
        top.addWidget(self._seq_combo)

        self._seq_ins_btn = QPushButton("Ins")
        self._seq_ins_btn.setToolTip("Insert a copy of the current pattern before this position")
        self._seq_ins_btn.setEnabled(False)
        self._seq_ins_btn.clicked.connect(self._seq_insert)
        top.addWidget(self._seq_ins_btn)

        self._seq_add_btn = QPushButton("Add")
        self._seq_add_btn.setToolTip("Append a copy of the current pattern to the sequence")
        self._seq_add_btn.setEnabled(False)
        self._seq_add_btn.clicked.connect(self._seq_add)
        top.addWidget(self._seq_add_btn)

        self._seq_del_btn = QPushButton("Del")
        self._seq_del_btn.setToolTip("Remove this sequence position")
        self._seq_del_btn.setEnabled(False)
        self._seq_del_btn.clicked.connect(self._seq_remove)
        top.addWidget(self._seq_del_btn)

        self._seq_up_btn = QPushButton("Up")
        self._seq_up_btn.setToolTip("Move this position earlier")
        self._seq_up_btn.setEnabled(False)
        self._seq_up_btn.clicked.connect(lambda: self._seq_move(-1))
        top.addWidget(self._seq_up_btn)

        self._seq_down_btn = QPushButton("Down")
        self._seq_down_btn.setToolTip("Move this position later")
        self._seq_down_btn.setEnabled(False)
        self._seq_down_btn.clicked.connect(lambda: self._seq_move(1))
        top.addWidget(self._seq_down_btn)

        self._play_btn = QPushButton("Play")
        self._play_btn.clicked.connect(self._toggle_play)
        top.addWidget(self._play_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._stop)
        top.addWidget(self._stop_btn)

        self._export_btn = QPushButton("Export WAV...")
        self._export_btn.clicked.connect(self._export_wav)
        top.addWidget(self._export_btn)

        self._export_mod_btn = QPushButton("Export Module...")
        self._export_mod_btn.clicked.connect(self._export_module)
        top.addWidget(self._export_mod_btn)

        self._time_lbl = QLabel("0:00.00")
        top.addWidget(self._time_lbl)
        top.addStretch()

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setMaximumWidth(160)
        self._progress.setVisible(False)
        top.addWidget(self._progress)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._cancel_job)
        top.addWidget(self._cancel_btn)
        root.addLayout(top)

        self._pattern_editor = PatternEditorWidget()
        self._pattern_editor.setEnabled(False)

        edit_bar = QHBoxLayout()
        self._edit_undo_btn = QPushButton("Undo")
        self._edit_undo_btn.setEnabled(False)
        self._edit_undo_btn.clicked.connect(self._pattern_editor.undo)
        edit_bar.addWidget(self._edit_undo_btn)
        self._edit_redo_btn = QPushButton("Redo")
        self._edit_redo_btn.setEnabled(False)
        self._edit_redo_btn.clicked.connect(self._pattern_editor.redo)
        edit_bar.addWidget(self._edit_redo_btn)
        edit_bar.addSpacing(8)
        self._edit_ins_btn = QPushButton("Insert Row")
        self._edit_ins_btn.setEnabled(False)
        self._edit_ins_btn.clicked.connect(self._pattern_editor.insert_row)
        edit_bar.addWidget(self._edit_ins_btn)
        self._edit_del_btn = QPushButton("Delete Row")
        self._edit_del_btn.setEnabled(False)
        self._edit_del_btn.clicked.connect(self._pattern_editor.delete_row)
        edit_bar.addWidget(self._edit_del_btn)
        self._edit_clear_btn = QPushButton("Clear Row")
        self._edit_clear_btn.setEnabled(False)
        self._edit_clear_btn.clicked.connect(self._pattern_editor.clear_row)
        edit_bar.addWidget(self._edit_clear_btn)
        edit_bar.addSpacing(8)
        self._edit_down_btn = QPushButton("T -")
        self._edit_down_btn.setToolTip("Transpose selection down")
        self._edit_down_btn.setEnabled(False)
        self._edit_down_btn.clicked.connect(lambda: self._pattern_editor.transpose_rows(-1))
        edit_bar.addWidget(self._edit_down_btn)
        self._edit_up_btn = QPushButton("T +")
        self._edit_up_btn.setToolTip("Transpose selection up")
        self._edit_up_btn.setEnabled(False)
        self._edit_up_btn.clicked.connect(lambda: self._pattern_editor.transpose_rows(1))
        edit_bar.addWidget(self._edit_up_btn)
        edit_bar.addSpacing(8)
        self._edit_copy_btn = QPushButton("Copy")
        self._edit_copy_btn.setEnabled(False)
        self._edit_copy_btn.clicked.connect(self._pattern_editor.copy_row)
        edit_bar.addWidget(self._edit_copy_btn)
        self._edit_cut_btn = QPushButton("Cut")
        self._edit_cut_btn.setEnabled(False)
        self._edit_cut_btn.clicked.connect(self._pattern_editor.cut_row)
        edit_bar.addWidget(self._edit_cut_btn)
        self._edit_paste_btn = QPushButton("Paste")
        self._edit_paste_btn.setEnabled(False)
        self._edit_paste_btn.clicked.connect(self._pattern_editor.paste_row)
        edit_bar.addWidget(self._edit_paste_btn)
        edit_bar.addStretch()
        root.addLayout(edit_bar)

        self._module_lbl = QLabel("No module loaded")
        root.addWidget(self._module_lbl)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._pattern_editor)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)

        mixer_group = QGroupBox("Mixer")
        mixer_layout = QVBoxLayout(mixer_group)
        mixer_layout.setContentsMargins(4, 4, 4, 4)
        self._mixer = MixerWidget()
        self._mixer.muteChanged.connect(self._on_mute_changed)
        self._mixer.gainChanged.connect(self._on_gain_changed)
        self._mixer.masterChanged.connect(self._on_master_changed)
        self._mixer.soloChanged.connect(self._on_solo_changed)
        mixer_layout.addWidget(self._mixer)
        mixer_group.setMaximumHeight(300)
        side_layout.addWidget(mixer_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        inst_group = QGroupBox("Instruments")
        inst_layout = QVBoxLayout(inst_group)
        self._instruments = InstrumentListWidget()
        inst_layout.addWidget(self._instruments)
        scroll_layout.addWidget(inst_group)

        sample_group = QGroupBox("Samples")
        sample_layout = QVBoxLayout(sample_group)
        self._sample_manager = SampleManagerWidget()
        self._sample_manager.samplesChanged.connect(self._on_samples_changed)
        sample_layout.addWidget(self._sample_manager)
        scroll_layout.addWidget(sample_group)

        info_group = QGroupBox("Module Info")
        info_layout = QVBoxLayout(info_group)
        self._module_info = ModuleInfoWidget()
        info_layout.addWidget(self._module_info)
        scroll_layout.addWidget(info_group)

        settings_group = QGroupBox("Music Settings")
        settings_layout = QVBoxLayout(settings_group)
        rate_row = QHBoxLayout()
        rate_row.addWidget(QLabel("Sample rate:"))
        self._rate_combo = QComboBox()
        self._rate_combo.addItems([str(r) for r in self.SAMPLE_RATES])
        self._rate_combo.currentTextChanged.connect(self._on_settings_changed)
        rate_row.addWidget(self._rate_combo, 1)
        settings_layout.addLayout(rate_row)
        qual_row = QHBoxLayout()
        qual_row.addWidget(QLabel("Quality:"))
        self._quality_combo = QComboBox()
        self._quality_combo.addItems(["software", "external"])
        self._quality_combo.currentTextChanged.connect(self._on_settings_changed)
        qual_row.addWidget(self._quality_combo, 1)
        settings_layout.addLayout(qual_row)
        self._clear_cache_btn = QPushButton("Clear Preview Cache")
        self._clear_cache_btn.clicked.connect(self._clear_cache)
        settings_layout.addWidget(self._clear_cache_btn)
        scroll_layout.addWidget(settings_group)
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        side_layout.addWidget(scroll, 1)

        splitter.addWidget(side)
        splitter.setStretchFactor(0, 3)
        splitter.setSizes([700, 260])
        root.addWidget(splitter, 1)
        self._load_settings_into_ui()

    def _load_settings_into_ui(self):
        try:
            rate = int(self._plugin.get_config("tracker_sample_rate", 44100))
        except Exception:
            rate = 44100
        if rate not in self.SAMPLE_RATES:
            rate = 44100
        self._rate_combo.blockSignals(True)
        self._rate_combo.setCurrentText(str(rate))
        self._rate_combo.blockSignals(False)
        quality = str(self._plugin.get_config("tracker_quality", "software"))
        if quality not in ("software", "external"):
            quality = "software"
        self._quality_combo.blockSignals(True)
        self._quality_combo.setCurrentText(quality)
        self._quality_combo.blockSignals(False)
        try:
            master_db = float(self._plugin.get_config("tracker_master_db", 0.0))
        except Exception:
            master_db = 0.0
        self._mixer.set_master_db(master_db)

    def _sample_rate(self) -> int:
        try:
            rate = int(self._plugin.get_config("tracker_sample_rate", 44100))
        except Exception:
            rate = 44100
        return rate if rate in self.SAMPLE_RATES else 44100

    def _quality(self) -> str:
        quality = str(self._plugin.get_config("tracker_quality", "software"))
        return quality if quality in ("software", "external") else "software"

    def _master_gain(self) -> float:
        try:
            return TrackerSoftwareRenderer.db_to_gain(self._mixer.master_db())
        except Exception:
            return 1.0

    def _cache_dir(self) -> str:
        return _zarin_user_dir("tracker_cache")

    def _open_module(self):
        last_dir = self._plugin.get_config("last_tracker_dir", "")
        start = last_dir if last_dir and os.path.isdir(last_dir) else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Module", start,
            "Tracker modules (*.mod *.xm *.s3m *.it);;All files (*)",
        )
        if not path:
            return
        self._plugin.set_config("last_tracker_dir", os.path.dirname(path))
        self._load_module(path)

    def _load_module(self, path: str):
        song = load_module(path)
        if song is None:
            QMessageBox.warning(self, "Tracker", f"Failed to load module:\n{path}")
            return
        self._stop()
        self._song_path = path
        self._song = song
        self._paused_elapsed = 0.0
        self._preview_wav = ""
        self._preview_duration = 0.0
        self._meter_peaks = []
        self._playback_map = []
        self._playback_ends = []
        self._timeline = []
        self._timeline_ends = []
        self._pattern_seq_index = 0
        self._refresh_sequence_ui(0)
        self._refresh_pattern()
        self._instruments.load_song(song)
        self._module_info.load_song(song)
        self._sample_manager.load_song(song)
        try:
            self._mixer.set_channels(int(song.n_channels))
        except Exception:
            self._mixer.set_channels(0)
        try:
            saved_master = float(self._plugin.get_config("tracker_master_db", 0.0))
        except Exception:
            saved_master = 0.0
        self._mixer.set_master_db(saved_master)
        try:
            info = song.get_song_info() or {}
            self._preview_duration = float(info.get("duration_seconds") or 0.0)
        except Exception:
            pass
        title = getattr(song, "songname", "") or ""
        self._module_lbl.setText(f"{os.path.basename(path)}   -   {title}")
        self._pattern_editor.setEnabled(True)
        self._save_btn.setEnabled(True)
        self._seq_ins_btn.setEnabled(True)
        self._seq_add_btn.setEnabled(True)
        self._seq_del_btn.setEnabled(True)
        self._seq_up_btn.setEnabled(True)
        self._seq_down_btn.setEnabled(True)
        self._update_edit_state()
        self._start_render(autoplay=False)

    def _refresh_sequence_ui(self, select: int | None = None) -> None:
        idx = select if select is not None else self._pattern_seq_index
        self._seq_combo.blockSignals(True)
        self._seq_combo.clear()
        for i, pat in enumerate(self._song.pattern_seq or []):
            self._seq_combo.addItem(f"Position {i}  -  Pattern {pat}")
        self._seq_combo.blockSignals(False)
        if self._seq_combo.count():
            idx = max(0, min(idx, self._seq_combo.count() - 1))
            self._seq_combo.setCurrentIndex(idx)
        self._pattern_seq_index = self._seq_combo.currentIndex()
        self._rebuild_timeline()

    def _rebuild_timeline(self) -> None:
        try:
            self._timeline = [(float(v.end_sec), int(v.sequence_idx), int(v.row))
                              for v in self._song.iter_playback_rows()]
            self._timeline_ends = [t[0] for t in self._timeline]
        except Exception:
            self._timeline = []
            self._timeline_ends = []
        try:
            info = self._song.get_song_info() or {}
            self._preview_duration = float(info.get("duration_seconds") or 0.0)
        except Exception:
            pass

    def _refresh_pattern(self):
        if self._song is None:
            return
        idx = self._pattern_seq_index
        seq = self._song.pattern_seq or []
        if 0 <= idx < len(seq):
            pat_idx = seq[idx]
            patterns = getattr(self._song, "patterns", []) or []
            if 0 <= pat_idx < len(patterns):
                self._pattern_editor.load_pattern(patterns[pat_idx])
                return
        self._pattern_editor.load_pattern(None)

    def _on_seq_changed(self, index: int):
        self._pattern_seq_index = max(0, index)
        self._refresh_pattern()

    def _seq_insert(self):
        if self._song is None:
            return
        idx = self._pattern_seq_index
        if 0 <= idx < len(self._song.pattern_seq):
            self._song.insert_pattern(idx, after=False)
            self._refresh_sequence_ui(idx + 1)
            self._refresh_pattern()
            self._mark_edited()

    def _seq_add(self):
        if self._song is None:
            return
        idx = self._pattern_seq_index
        if 0 <= idx < len(self._song.pattern_seq):
            self._song.duplicate_pattern(idx)
            self._refresh_sequence_ui(len(self._song.pattern_seq) - 1)
            self._refresh_pattern()
            self._mark_edited()

    def _seq_remove(self):
        if self._song is None:
            return
        idx = self._pattern_seq_index
        if 0 <= idx < len(self._song.pattern_seq):
            self._song.remove_pattern(idx)
            self._refresh_sequence_ui(min(idx, len(self._song.pattern_seq) - 1))
            self._refresh_pattern()
            self._mark_edited()

    def _seq_move(self, delta: int):
        if self._song is None:
            return
        idx = self._pattern_seq_index
        n = len(self._song.pattern_seq)
        if not n or not (0 <= idx < n):
            return
        j = idx + delta
        if not (0 <= j < n):
            return
        seq = list(self._song.pattern_seq)
        seq[idx], seq[j] = seq[j], seq[idx]
        self._song.set_sequence(seq)
        self._refresh_sequence_ui(j)
        self._refresh_pattern()
        self._mark_edited()

    def _save_module(self):
        if self._song is None or not self._song_path:
            return
        try:
            self._stop()
            self._song.save(self._song_path, verbose=False)
            self._time_lbl.setText(f"Saved {os.path.basename(self._song_path)}")
            self._start_render(autoplay=False)
        except Exception as e:
            QMessageBox.warning(self, "Tracker", f"Save failed: {e}")

    def _mark_edited(self):
        self._edit_rev += 1
        self._schedule_rerender()

    def _on_pattern_edited(self):
        self._update_edit_state()
        self._mark_edited()

    def _on_samples_changed(self):
        self._mark_edited()

    def _on_metadata_edited(self, key: str, value: str):
        try:
            title = str(getattr(self._song, "songname", "") or "")
            if not title:
                title = os.path.basename(self._song_path or "module")
            self._module_lbl.setText(f"{os.path.basename(self._song_path)}   -   {title}")
        except Exception:
            pass
        self._schedule_rerender()

    def _update_edit_state(self):
        if self._song is None:
            self._edit_undo_btn.setEnabled(False)
            self._edit_redo_btn.setEnabled(False)
            self._edit_ins_btn.setEnabled(False)
            self._edit_del_btn.setEnabled(False)
            self._edit_clear_btn.setEnabled(False)
            self._edit_down_btn.setEnabled(False)
            self._edit_up_btn.setEnabled(False)
            self._edit_copy_btn.setEnabled(False)
            self._edit_cut_btn.setEnabled(False)
            self._edit_paste_btn.setEnabled(False)
            return
        self._edit_undo_btn.setEnabled(self._pattern_editor.can_undo())
        self._edit_redo_btn.setEnabled(self._pattern_editor.can_redo())
        self._edit_ins_btn.setEnabled(True)
        self._edit_del_btn.setEnabled(True)
        self._edit_clear_btn.setEnabled(True)
        self._edit_down_btn.setEnabled(True)
        self._edit_up_btn.setEnabled(True)
        self._edit_copy_btn.setEnabled(True)
        self._edit_cut_btn.setEnabled(True)
        self._edit_paste_btn.setEnabled(True)

    def _on_mute_changed(self, ch: int, muted: bool):
        self._schedule_rerender()

    def _on_solo_changed(self):
        self._schedule_rerender()

    def _on_gain_changed(self, ch: int, db: float):
        self._schedule_rerender()

    def _on_master_changed(self, db: float):
        try:
            self._plugin.set_config("tracker_master_db", float(db))
        except Exception:
            pass
        if self._playing and self._source_id:
            mgr = TrackerEditorWidget._audio_mgr()
            if mgr:
                try:
                    mgr.update_source(self._source_id, self._master_gain(), 1.0, (0.0, 0.0, 0.0))
                except Exception:
                    pass

    def _on_settings_changed(self, _text: str = ""):
        try:
            self._plugin.set_config("tracker_sample_rate", int(self._rate_combo.currentText()))
            self._plugin.set_config("tracker_quality", str(self._quality_combo.currentText()))
        except Exception:
            pass
        if self._song is not None:
            self._schedule_rerender()

    def _clear_cache(self):
        try:
            cache = self._cache_dir()
            removed = 0
            for fn in os.listdir(cache):
                if fn.startswith("prev_") and (fn.endswith(".wav") or fn.endswith(".peaks.json")):
                    try:
                        os.remove(os.path.join(cache, fn))
                        removed += 1
                    except Exception:
                        pass
            QMessageBox.information(self, "Tracker", f"Preview cache cleared ({removed} files).")
        except Exception as e:
            QMessageBox.warning(self, "Tracker", f"Failed to clear cache: {e}")

    def _schedule_rerender(self):
        if self._song is None:
            return
        self._rerender_timer.stop()
        self._rerender_timer.start()

    def _on_rerender_timeout(self):
        if self._song is None:
            return
        resume = self._playing or self._pending_autoplay
        self._start_render(autoplay=resume)

    def _preview_cache_path(self, bake_master_gain: float | None) -> str:
        h = hashlib.sha1()
        try:
            st = os.stat(self._song_path)
            h.update(f"{self._song_path}|{st.st_mtime}|{st.st_size}".encode())
        except Exception:
            h.update(self._song_path.encode("utf-8", "ignore"))
        gains = self._mixer.all_gains_db()
        muted = sorted(self._mixer.muted_channels())
        h.update(f"|{_RENDER_VERSION}|{self._sample_rate()}|{self._quality()}|{muted}|{gains}|{bake_master_gain}|rev{self._edit_rev}".encode())
        return os.path.join(self._cache_dir(), f"prev_{h.hexdigest()[:16]}.wav")

    def _busy(self) -> bool:
        return self._worker is not None

    def _start_render(self, autoplay: bool, kind: str = "preview", dest: str = "",
                      bake_master: bool = False):
        if self._song is None or not self._song_path:
            return
        self._cancel_worker()
        self._job_id += 1
        self._job_kind = kind
        self._job_dest = dest
        self._pending_autoplay = autoplay
        bake_gain = self._master_gain() if bake_master else None
        out_path = dest if kind == "save" else self._preview_cache_path(bake_gain)
        if kind != "save" and os.path.isfile(out_path):
            self._on_cached_preview(out_path, bake_gain)
            if autoplay:
                self._play()
            return
        worker = _PreviewJob(
            self._job_id, kind, self._song_path, out_path,
            self._sample_rate(), self._mixer.muted_channels(),
            self._mixer.all_gains_db(), bake_gain,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_job_progress)
        worker.done.connect(self._on_job_done)
        worker.failed.connect(self._on_job_failed)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._cancel_btn.setVisible(True)
        thread.start()

    def _on_cached_preview(self, out_path: str, bake_gain):
        self._preview_wav = out_path
        self._meter_peaks = []
        self._meter_block_sec = TrackerSoftwareRenderer.METER_BLOCK_SEC
        self._playback_map = []
        self._playback_ends = []
        try:
            sidecar = _load_peaks_sidecar(out_path)
            if sidecar is not None:
                self._meter_block_sec, self._meter_peaks, self._playback_map = sidecar
                self._playback_ends = [e for e, _s, _r in self._playback_map]
        except Exception:
            pass
        try:
            import wave
            with wave.open(out_path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate() or self._sample_rate()
                self._preview_duration = frames / float(rate)
        except Exception:
            pass

    def _cancel_worker(self):
        worker = self._worker
        self._worker = None
        self._thread = None
        if worker is not None:
            try:
                worker.cancel()
            except Exception:
                pass
        self._progress.setVisible(False)
        self._cancel_btn.setVisible(False)

    def _cancel_job(self):
        self._pending_autoplay = False
        self._cancel_worker()

    def _on_job_progress(self, elapsed: float, total):
        try:
            if total:
                pct = max(0, min(100, int(elapsed * 100.0 / max(1e-6, float(total)))))
                self._progress.setRange(0, 100)
                self._progress.setValue(pct)
            else:
                self._progress.setRange(0, 0)
        except Exception:
            pass

    def _on_job_done(self, job_id: int, out_path: str, duration: float,
                     block_sec: float, peaks, playmap):
        if job_id != self._job_id:
            return
        kind = self._job_kind
        self._cancel_worker()
        if kind == "save":
            QMessageBox.information(self, "Tracker", f"Module exported to\n{out_path}")
            return
        self._preview_wav = out_path
        if duration > 0:
            self._preview_duration = duration
        try:
            self._meter_block_sec = float(block_sec) or TrackerSoftwareRenderer.METER_BLOCK_SEC
        except Exception:
            self._meter_block_sec = TrackerSoftwareRenderer.METER_BLOCK_SEC
        try:
            self._meter_peaks = [list(b) for b in (peaks or [])]
        except Exception:
            self._meter_peaks = []
        try:
            self._playback_map = [(float(e), int(s), int(r)) for e, s, r in (playmap or [])]
            self._playback_ends = [e for e, _s, _r in self._playback_map]
        except Exception:
            self._playback_map = []
            self._playback_ends = []
        if kind == "export":
            try:
                shutil.copyfile(out_path, self._job_dest)
                self._time_lbl.setText(f"Exported to {os.path.basename(self._job_dest)}")
            except Exception as e:
                QMessageBox.warning(self, "Tracker", f"Export failed: {e}")
            return
        if self._pending_autoplay:
            self._pending_autoplay = False
            self._stop_audio()
            self._play()

    def _on_job_failed(self, job_id: int, message: str):
        if job_id != self._job_id:
            return
        self._cancel_worker()
        self._pending_autoplay = False
        if message != "cancelled":
            QMessageBox.warning(self, "Tracker", f"Render failed: {message}")

    def _toggle_play(self):
        if self._playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        if self._song is None:
            return
        if self._source_id:
            mgr = TrackerEditorWidget._audio_mgr()
            if mgr:
                mgr.resume(self._source_id)
                self._playing = True
                self._play_start = time.perf_counter() - self._paused_elapsed
                self._play_btn.setText("Pause")
            return
        if self._busy():
            self._pending_autoplay = True
            return
        if not self._preview_wav or not os.path.isfile(self._preview_wav):
            self._start_render(autoplay=True)
            return
        mgr = TrackerEditorWidget._audio_mgr()
        if mgr is None:
            QMessageBox.information(self, "Tracker", "Audio engine is not available for preview.")
            return
        src = mgr.play(clip_path=self._preview_wav, loop=False,
                       volume=self._master_gain(), spatial_blend=0.0)
        if src is None:
            QMessageBox.information(self, "Tracker", "Failed to start audio preview.")
            return
        self._source_id = src
        self._playing = True
        self._paused_elapsed = 0.0
        self._play_start = time.perf_counter()
        self._play_btn.setText("Pause")

    def _pause(self):
        mgr = TrackerEditorWidget._audio_mgr()
        if mgr and self._source_id:
            mgr.pause(self._source_id)
            self._paused_elapsed = time.perf_counter() - self._play_start
        self._playing = False
        self._play_btn.setText("Play")

    def _stop_audio(self):
        if self._source_id:
            mgr = TrackerEditorWidget._audio_mgr()
            if mgr:
                try:
                    mgr.stop(self._source_id)
                except Exception:
                    pass
            self._source_id = 0
        self._playing = False
        self._paused_elapsed = 0.0
        self._play_btn.setText("Play")
        try:
            self._mixer.reset_levels()
        except Exception:
            pass
        self._pattern_editor.clear_rows()
        self._time_lbl.setText("0:00.00")

    def _stop(self):
        self._pending_autoplay = False
        self._rerender_timer.stop()
        self._cancel_worker()
        self._stop_audio()

    @staticmethod
    def _audio_mgr():
        from core.audio.audio_system import AudioSourceManager
        return AudioSourceManager.instance()

    def _song_duration(self) -> float:
        try:
            if self._preview_duration > 0:
                return float(self._preview_duration)
        except Exception:
            pass
        return 0.0

    def _on_tick(self):
        if not self._playing or not self._song:
            return
        elapsed = time.perf_counter() - self._play_start
        dur = self._song_duration()
        if dur > 0 and elapsed > dur + 0.4:
            self._stop_audio()
            return
        seq_idx, row = self._resolve_position(elapsed)
        if (seq_idx is not None and self._seq_combo.count() > 0
                and 0 <= seq_idx < self._seq_combo.count()
                and seq_idx != self._pattern_seq_index):
            self._seq_combo.setCurrentIndex(seq_idx)
        idx = self._pattern_seq_index
        seq = self._song.pattern_seq or []
        if 0 <= idx < len(seq):
            pat_idx = seq[idx]
            patterns = getattr(self._song, "patterns", []) or []
            if 0 <= pat_idx < len(patterns):
                nrows = int(getattr(patterns[pat_idx], "n_rows", 0) or 0)
                if nrows > 0:
                    self._pattern_editor.highlight_row(row % nrows)
        try:
            if self._meter_peaks and self._meter_block_sec > 0:
                mi = int(elapsed / self._meter_block_sec)
                if 0 <= mi < len(self._meter_peaks):
                    block = self._meter_peaks[mi]
                    self._mixer.set_levels(block)
                    peak = 0.0
                    for v in block:
                        try:
                            if float(v) > peak:
                                peak = float(v)
                        except Exception:
                            pass
                    self._mixer.set_master_level(peak * self._master_gain())
        except Exception:
            pass
        m = int(elapsed) // 60
        s = int(elapsed) % 60
        cs = int((elapsed % 1.0) * 100)
        self._time_lbl.setText(f"{m}:{s:02d}.{cs:02d}")

    @staticmethod
    def _lookup_map(entries, ends, elapsed: float):
        try:
            if entries and ends:
                i = bisect.bisect_left(ends, elapsed)
                if i >= len(entries):
                    i = len(entries) - 1
                _e, s, r = entries[i]
                return int(s), int(r)
        except Exception:
            pass
        return None

    def _resolve_position(self, elapsed: float):
        pos = self._lookup_map(self._playback_map, self._playback_ends, elapsed)
        if pos is not None:
            return pos
        pos = self._lookup_map(self._timeline, self._timeline_ends, elapsed)
        if pos is not None:
            return pos
        return None, self._estimate_row(elapsed)

    def _resolve_row(self, elapsed: float) -> int:
        return int(self._resolve_position(elapsed)[1])

    def _estimate_row(self, elapsed: float) -> int:
        try:
            info = self._song.get_song_info()
            bpm = int(info.get("bpm") or 125)
            speed = int(info.get("speed") or 6)
        except Exception:
            bpm, speed = 125, 6
        try:
            row_dur = (2.5 / max(32, bpm)) * max(1, speed)
        except Exception:
            row_dur = 0.12
        return int(elapsed / max(1e-6, row_dur))

    def _export_wav(self):
        if self._song is None:
            return
        if self._busy():
            QMessageBox.information(self, "Tracker", "Please wait for the current render to finish.")
            return
        default = os.path.splitext(os.path.basename(self._song_path))[0] + ".wav"
        path, _ = QFileDialog.getSaveFileName(self, "Export WAV", default, "WAV (*.wav)")
        if not path:
            return
        self._start_render(autoplay=False, kind="export", dest=path, bake_master=True)

    def _export_module(self):
        if self._song is None or not self._song_path:
            return
        if self._busy():
            QMessageBox.information(self, "Tracker", "Please wait for the current render to finish.")
            return
        try:
            ext = str(getattr(self._song, "file_extension", "") or "").lower().strip(".")
        except Exception:
            ext = ""
        if not ext:
            ext = os.path.splitext(self._song_path)[1].lower().strip(".") or "mod"
        default = os.path.splitext(os.path.basename(self._song_path))[0] + f".{ext}"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Module", default, f"{ext.upper()} (*.{ext})")
        if not path:
            return
        self._start_render(autoplay=False, kind="save", dest=path)
