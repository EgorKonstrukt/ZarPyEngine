from __future__ import annotations
import os
from typing import Optional, NamedTuple

try:
    import miniaudio
    _miniaudio_available = True
except Exception:
    _miniaudio_available = False

AUDIO_EXTENSIONS = (".wav", ".wave", ".mp3", ".flac", ".ogg", ".vorbis")


class DecodedAudio(NamedTuple):
    sample_rate: int
    channels: int
    pcm_int16: bytes


def miniaudio_available() -> bool:
    return _miniaudio_available


def _read_file_bytes(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None


def _decode_bytes(data: bytes, nchannels: int) -> Optional[DecodedAudio]:
    if not _miniaudio_available or not data:
        return None
    try:
        decoded = miniaudio.decode(data, output_format=miniaudio.SampleFormat.SIGNED16, nchannels=nchannels)
        return DecodedAudio(
            sample_rate=decoded.sample_rate,
            channels=decoded.nchannels,
            pcm_int16=decoded.samples.tobytes(),
        )
    except Exception:
        return None


def decode_audio(path: str) -> Optional[DecodedAudio]:
    if not _miniaudio_available:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext not in AUDIO_EXTENSIONS:
        return None
    data = _read_file_bytes(path)
    return _decode_bytes(data, 1)


def decode_audio_stereo(path: str) -> Optional[DecodedAudio]:
    if not _miniaudio_available:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext not in AUDIO_EXTENSIONS:
        return None
    data = _read_file_bytes(path)
    return _decode_bytes(data, 2)


def decode_audio_channels(path: str, nchannels: int) -> Optional[DecodedAudio]:
    if not _miniaudio_available:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext not in AUDIO_EXTENSIONS:
        return None
    data = _read_file_bytes(path)
    return _decode_bytes(data, nchannels)


def _detect_openal_format(num_channels: int, sample_width: int) -> int:
    try:
        import openal as al
    except Exception:
        return 0
    if num_channels == 1:
        if sample_width == 2:
            return al.AL_FORMAT_MONO16
    elif num_channels == 2:
        if sample_width == 2:
            return al.AL_FORMAT_STEREO16
    return 0
