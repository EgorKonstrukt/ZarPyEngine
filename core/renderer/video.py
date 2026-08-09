# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import json
import struct
import subprocess
import threading
import tempfile
import time
import numpy as np
import moderngl
from typing import Optional, Any
from core.engine.engine import Engine
from core.maths.math3d import Mat4
from core.components.rendering.renderers.video_renderer import VideoRenderer



class VideoPlayer:
    def __init__(self, ctx: moderngl.Context, video_path: str, loop: bool = True,
                 volume: float = 1.0, offset: float = 0.0, audio_source_entity_id: str = ""):
        self._ctx = ctx
        self._video_path = video_path
        self._loop = loop
        self._volume = volume
        self._offset = offset
        self._audio_source_entity_id = audio_source_entity_id
        self._texture: Optional[moderngl.Texture] = None
        self._width: int = 0
        self._height: int = 0
        self._fps: float = 30.0
        self._duration: float = 0.0
        self._frame_count: int = 0
        self._playing: bool = False
        self._finished: bool = False
        self._paused: bool = False
        self._elapsed: float = 0.0
        self._frame_duration: float = 0.0
        self._last_frame_index: int = -1
        self._frame_ready: bool = False
        self._frame_data: Optional[bytes] = None
        self._frame_lock: threading.Lock = threading.Lock()
        self._decode_thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None
        self._audio_temp: Optional[str] = None
        self._audio_source_id: int = 0
        self._audio_start_time: float = 0.0
        self._audio_available: bool = False
        self._disposed: bool = False

    def _probe(self) -> bool:
        if not os.path.exists(self._video_path):
            return False
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", self._video_path],
                capture_output=True, timeout=10
            )
            info = json.loads(result.stdout)
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "video":
                    self._width = int(stream.get("width", 0))
                    self._height = int(stream.get("height", 0))
                    avg_fps = stream.get("avg_frame_rate", "30/1")
                    if "/" in avg_fps:
                        num, den = avg_fps.split("/")
                        self._fps = float(num) / max(float(den), 1.0)
                    else:
                        self._fps = float(avg_fps)
                    self._duration = float(stream.get("duration", 0) or info.get("format", {}).get("duration", 0))
                    nb_frames = stream.get("nb_frames")
                    if nb_frames:
                        self._frame_count = int(nb_frames)
                    else:
                        self._frame_count = int(self._duration * self._fps)
                    self._frame_duration = 1.0 / max(self._fps, 0.001)
                    return self._width > 0 and self._height > 0
            return False
        except Exception:
            return False

    def _extract_audio(self):
        try:
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            cmd = ["ffmpeg", "-y"]
            if self._offset > 0:
                cmd.extend(["-ss", str(self._offset)])
            cmd.extend(["-i", self._video_path, "-vn", "-acodec", "pcm_s16le",
                        "-ar", "44100", "-ac", "2", "-f", "wav", path])
            subprocess.run(cmd, capture_output=True, timeout=30)
            if os.path.exists(path) and os.path.getsize(path) > 44:
                self._audio_temp = path
                self._audio_available = True
            else:
                self._cleanup_audio(path)
        except Exception:
            pass

    def _cleanup_audio(self, path: Optional[str] = None):
        p = path or self._audio_temp
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    def _get_audio_config(self) -> dict:
        cfg = {
            "volume": self._volume,
            "loop": self._loop,
            "spatial_blend": 0.0,
            "min_distance": 1.0,
            "max_distance": 50.0,
            "volume_rolloff": [[0, 1], [1, 0]],
            "offset": self._offset,
        }
        if self._audio_source_entity_id:
            eng = Engine.instance()
            scene = eng.scene if eng else None
            if scene:
                entity = scene.get_entity(self._audio_source_entity_id)
                if entity:
                    from core.components.audio.audio_source import AudioSource
                    audio_src = entity.get_component(AudioSource)
                    if audio_src:
                        cfg.update({
                            "volume": self._volume * audio_src.volume,
                            "loop": self._loop,
                            "spatial_blend": audio_src.spatial_blend,
                            "min_distance": audio_src.min_distance,
                            "max_distance": audio_src.max_distance,
                            "volume_rolloff": audio_src.volume_rolloff or [[0, 1], [1, 0]],
                            "offset": self._offset,
                        })
        return cfg

    def _play_audio(self):
        if not self._audio_available or not self._audio_temp:
            return
        try:
            from core.audio.audio_system import AudioSourceManager
            mgr = AudioSourceManager.instance()
            if not mgr:
                return
            cfg = self._get_audio_config()
            src_id = mgr.play(
                clip_path=self._audio_temp,
                loop=cfg["loop"],
                volume=cfg["volume"],
                spatial_blend=cfg["spatial_blend"],
                min_distance=cfg["min_distance"],
                max_distance=cfg["max_distance"],
                volume_rolloff=cfg["volume_rolloff"],
                offset=cfg["offset"],
            )
            if src_id:
                self._audio_source_id = src_id
                self._audio_start_time = time.perf_counter()
        except Exception:
            self._audio_available = False

    def _stop_audio(self):
        if self._audio_source_id:
            try:
                from core.audio.audio_system import AudioSourceManager
                mgr = AudioSourceManager.instance()
                if mgr:
                    mgr.stop(self._audio_source_id)
            except Exception:
                pass
            self._audio_source_id = 0
        if self._audio_temp:
            self._cleanup_audio()
            self._audio_temp = None
        self._audio_available = False

    def _decode_loop(self):
        try:
            cmd = [
                "ffmpeg", "-y",
            ]
            if self._offset > 0:
                cmd.extend(["-ss", str(self._offset)])
            cmd.extend([
                "-i", self._video_path,
                "-f", "rawvideo",
                "-pix_fmt", "rgba",
                "-an",
                "-vsync", "drop",
                "-",
            ])
            self._process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8
            )
            frame_size = self._width * self._height * 4
            raw_stdout = self._process.stdout
            if not raw_stdout:
                return

            while not self._disposed:
                if self._paused:
                    time.sleep(0.01)
                    continue
                data = raw_stdout.read(frame_size)
                if not data or len(data) < frame_size:
                    if self._loop and not self._disposed:
                        self._process.terminate()
                        self._process.wait()
                        self._process = subprocess.Popen(
                            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8
                        )
                        raw_stdout = self._process.stdout
                        self._elapsed = 0.0
                        self._audio_start_time = time.perf_counter()
                        continue
                    self._finished = True
                    break
                with self._frame_lock:
                    self._frame_data = data
                    self._frame_ready = True
                target_sleep = self._frame_duration
                elapsed_in_frame = time.perf_counter()
                sleep_time = target_sleep - (elapsed_in_frame % target_sleep)
                if sleep_time > 0:
                    time.sleep(sleep_time * 0.9)
        except Exception:
            pass
        finally:
            if self._process:
                try:
                    self._process.terminate()
                except Exception:
                    pass
                self._process = None

    def play(self):
        if not self._probe():
            return
        self._playing = True
        self._finished = False
        self._extract_audio()
        if self._audio_available:
            self._play_audio()
        self._decode_thread = threading.Thread(target=self._decode_loop, daemon=True)
        self._decode_thread.start()

    def pause(self):
        self._paused = True
        if self._audio_source_id:
            try:
                from core.audio.audio_system import AudioSourceManager
                mgr = AudioSourceManager.instance()
                if mgr:
                    mgr.pause(self._audio_source_id)
            except Exception:
                pass

    def resume(self):
        self._paused = False
        if self._audio_source_id:
            try:
                from core.audio.audio_system import AudioSourceManager
                mgr = AudioSourceManager.instance()
                if mgr:
                    mgr.resume(self._audio_source_id)
            except Exception:
                pass

    def stop(self):
        self._playing = False
        self._finished = True
        self._paused = False
        self._elapsed = 0.0
        self._frame_data = None
        self._frame_ready = False
        self._last_frame_index = -1
        self._stop_audio()
        if self._process:
            try:
                self._process.terminate()
            except Exception:
                pass
            self._process = None

    def update_texture(self):
        if not self._playing or self._paused:
            return
        frame_data = None
        with self._frame_lock:
            if self._frame_ready:
                frame_data = self._frame_data
                self._frame_ready = False
        if frame_data is not None:
            arr = np.frombuffer(frame_data, dtype=np.uint8).reshape(self._height, self._width, 4)
            arr = arr[::-1, :, :]
            flipped = arr.tobytes()
            if self._texture is None:
                self._texture = self._ctx.texture((self._width, self._height), 4, dtype='f1')
                self._texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self._texture.swizzle = 'BGRA' if self._ctx.info.get('GL_BGRA') else 'RGBA'
            self._texture.write(flipped)

    def get_texture(self) -> Optional[moderngl.Texture]:
        return self._texture

    def is_playing(self) -> bool:
        return self._playing and not self._finished

    def is_finished(self) -> bool:
        return self._finished

    def set_loop(self, loop: bool):
        self._loop = loop

    def set_volume(self, volume: float):
        self._volume = volume
        if self._audio_source_id:
            cfg = self._get_audio_config()
            try:
                from core.audio.audio_system import AudioSourceManager
                mgr = AudioSourceManager.instance()
                if mgr:
                    mgr.update_source(self._audio_source_id, cfg["volume"], 1.0, (0, 0, 0), cfg["spatial_blend"])
            except Exception:
                pass

    def set_offset(self, offset: float):
        self._offset = offset

    def seek(self, time_seconds: float):
        self.stop()
        self._elapsed = time_seconds
        self.play()

    def dispose(self):
        self._disposed = True
        self.stop()
        if self._texture:
            try:
                self._texture.release()
            except Exception:
                pass
            self._texture = None
        if self._audio_temp:
            self._cleanup_audio()
            self._audio_temp = None


class VideoRendererGL:
    def __init__(self, ctx: moderngl.Context, prog: moderngl.Program):
        self._ctx = ctx
        self._prog = prog
        self._vbo: Optional[moderngl.Buffer] = None
        self._ibo: Optional[moderngl.Buffer] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._players: dict[str, VideoPlayer] = {}
        self._build_buffers()

    def _build_buffers(self):
        quad = np.array([
            -0.5, -0.5, 0.0, 0.0, 0.0,
             0.5, -0.5, 0.0, 1.0, 0.0,
             0.5,  0.5, 0.0, 1.0, 1.0,
            -0.5,  0.5, 0.0, 0.0, 1.0,
        ], dtype=np.float32)
        idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        self._vbo = self._ctx.buffer(quad.tobytes())
        self._ibo = self._ctx.buffer(idx.tobytes())
        self._vao = self._ctx.vertex_array(
            self._prog,
            [(self._vbo, "3f 2f", "in_position", "in_uv")],
            self._ibo
        )

    def _player_key(self, entity_id: str, video_path: str) -> str:
        return f"{entity_id}|{video_path}"

    def ensure_player(self, entity_id: str, video_path: str,
                       loop: bool, volume: float, offset: float = 0.0,
                       audio_source_entity_id: str = "") -> VideoPlayer:
        key = self._player_key(entity_id, video_path)
        player = self._players.get(key)
        if player is None:
            player = VideoPlayer(self._ctx, video_path, loop, volume, offset, audio_source_entity_id)
            self._players[key] = player
        return player

    def remove_player(self, entity_id: str, video_path: str):
        key = self._player_key(entity_id, video_path)
        player = self._players.pop(key, None)
        if player:
            player.dispose()

    def _dispose_all(self):
        for player in self._players.values():
            player.dispose()
        self._players.clear()

    def _remove_stale_players(self, active_keys: set[str]):
        stale = [k for k in self._players if k not in active_keys]
        for k in stale:
            p = self._players.pop(k, None)
            if p:
                p.dispose()

    def render_snapshot(self, video_items: list, view_mat: Mat4, proj_mat: Mat4):
        if not self._prog or not self._vao:
            return
        eng = Engine.instance()
        is_play = eng is not None and eng.play_mode
        if not is_play:
            self._dispose_all()
            return
        if not video_items:
            self._dispose_all()
            return
        active_keys: set[str] = set()
        prog = self._prog
        view_f32 = view_mat.to_f32()
        proj_f32 = proj_mat.to_f32()
        if "u_view" in prog:
            prog["u_view"].write(view_f32.tobytes())
        if "u_proj" in prog:
            prog["u_proj"].write(proj_f32.tobytes())
        self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.depth_mask = True
        for item in video_items:
            player_key = self._player_key(item.entity_id, item.video_path)
            active_keys.add(player_key)
            player = self._players.get(player_key)
            if player is None:
                player = self.ensure_player(
                    item.entity_id, item.video_path,
                    item.loop, item.volume, item.offset,
                    item.audio_source_entity_id
                )
                active_keys.add(player_key)
            if not player.is_playing() and not player.is_finished():
                player.play()
            player.set_loop(item.loop)
            player.set_volume(item.volume)
            player.set_offset(item.offset)
            player.update_texture()
            tex = player.get_texture()
            if tex is None:
                continue
            model_f32 = item.world_matrix.to_f32()
            if "u_model" in prog:
                prog["u_model"].write(model_f32.tobytes())
            if "u_color" in prog:
                prog["u_color"].write(np.array(item.color, dtype=np.float32).tobytes())
            if "u_flip" in prog:
                prog["u_flip"].write(np.array(
                    [1.0 if item.flip_x else 0.0, 1.0 if item.flip_y else 0.0],
                    dtype=np.float32
                ).tobytes())
            if "u_alpha_cutoff" in prog:
                prog["u_alpha_cutoff"].value = 0.01
            tex.use(0)
            if "u_texture" in prog:
                prog["u_texture"].value = 0
            self._vao.render(moderngl.TRIANGLES)
        self._ctx.enable(moderngl.CULL_FACE)
        self._remove_stale_players(active_keys)

    def render(self, scene, view_mat: Mat4, proj_mat: Mat4):
        if not self._prog or not self._vao:
            return
        eng = Engine.instance()
        is_play = eng is not None and eng.play_mode
        if not is_play:
            self._dispose_all()
            return
        prog = self._prog
        view_f32 = view_mat.to_f32()
        proj_f32 = proj_mat.to_f32()
        if "u_view" in prog:
            prog["u_view"].write(view_f32.tobytes())
        if "u_proj" in prog:
            prog["u_proj"].write(proj_f32.tobytes())
        self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.depth_mask = True
        for ent in scene.get_entities_with_component(VideoRenderer):
            if not ent.active:
                continue
            vr = ent.get_component(VideoRenderer)
            if not vr or not vr.enabled:
                continue
            tr = ent.transform
            if not tr:
                continue
            player_key = self._player_key(ent._id, vr.video_path)
            active_keys.add(player_key)
            if not vr.video_path:
                continue
            player = self._players.get(player_key)
            if player is None:
                player = self.ensure_player(ent._id, vr.video_path, vr.loop, vr.volume, vr.offset, vr.audio_source_entity_id)
                active_keys.add(player_key)
            if not player.is_playing() and not player.is_finished():
                player.play()
            player.set_loop(vr.loop)
            player.set_volume(vr.volume)
            player.set_offset(vr.offset)
            player.update_texture()
            tex = player.get_texture()
            if tex is None:
                continue
            model_f32 = tr.world_matrix.to_f32()
            if "u_model" in prog:
                prog["u_model"].write(model_f32.tobytes())
            color = vr.color
            if "u_color" in prog:
                prog["u_color"].write(np.array(color, dtype=np.float32).tobytes())
            if "u_flip" in prog:
                prog["u_flip"].write(np.array(
                    [1.0 if vr.flip_x else 0.0, 1.0 if vr.flip_y else 0.0],
                    dtype=np.float32
                ).tobytes())
            if "u_alpha_cutoff" in prog:
                prog["u_alpha_cutoff"].value = 0.01
            tex.use(0)
            if "u_texture" in prog:
                prog["u_texture"].value = 0
            self._vao.render(moderngl.TRIANGLES)
        self._ctx.enable(moderngl.CULL_FACE)
        self._remove_stale_players(active_keys)

    def release(self):
        for player in self._players.values():
            player.dispose()
        self._players.clear()
        if self._vao:
            self._vao.release()
        if self._vbo:
            self._vbo.release()
        if self._ibo:
            self._ibo.release()
