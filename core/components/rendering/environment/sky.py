# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import os
import numpy as np
import moderngl
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
from core.maths.math3d import Mat4, Vec3, Quat, FLOAT_TYPE
from core.components.lighting.light import Light, LightType
from core.components.rendering.environment.sky_ibl import get_sky_ibl, release_sky_ibl_cache, get_procedural_sky_ibl
from core.components.rendering.environment.atmosphere import Atmosphere

_ENV_TEX_CACHE: dict[str, tuple[float, object]] = {}


def _decode_rgbe_scanline_rle(raw: bytes, pos: int, width: int):
    channels = [np.empty(width, dtype=np.uint8) for _ in range(4)]
    for ch in channels:
        filled = 0
        while filled < width:
            t = raw[pos]
            pos += 1
            if t > 128:
                cnt = t - 128
                val = raw[pos]
                pos += 1
                ch[filled:filled + cnt] = val
            else:
                cnt = t
                ch[filled:filled + cnt] = np.frombuffer(raw[pos:pos + cnt], dtype=np.uint8)
                pos += cnt
            filled += cnt
    return channels, pos


def _decode_rgbe(data: bytes):
    line_end = data.index(b"\n")
    if not data[:line_end].strip().startswith(b"#?"):
        raise ValueError("not radiance hdr")
    pos = line_end + 1
    while True:
        line_end = data.index(b"\n", pos)
        line = data[pos:line_end].strip()
        pos = line_end + 1
        if not line:
            break
        if line.lower().startswith(b"format=") and b"32-bit_rle_rgbe" not in line.lower():
            raise ValueError("unsupported hdr format")
    res_end = data.index(b"\n", pos)
    parts = data[pos:res_end].strip().split()
    pos = res_end + 1
    if len(parts) != 4:
        raise ValueError("bad resolution line")
    if parts[0] == b"-Y":
        height = int(parts[1])
        width = int(parts[3])
        flip = False
    elif parts[0] == b"+Y":
        height = int(parts[1])
        width = int(parts[3])
        flip = True
    else:
        raise ValueError("bad resolution line")
    out = np.empty((height, width, 3), dtype=np.float32)
    for y in range(height):
        a = data[pos]
        b = data[pos + 1]
        if a == 2 and b == 2:
            w = (data[pos + 2] << 8) | data[pos + 3]
            if w != width:
                raise ValueError("scanline width mismatch")
            pos += 4
            chans, pos = _decode_rgbe_scanline_rle(data, pos, width)
            rr, gg, bb, ee = chans
        elif a == 2 and b < 128:
            w = b
            if w != width:
                raise ValueError("scanline width mismatch")
            pos += 2
            rr = np.empty(width, dtype=np.uint8)
            gg = np.empty(width, dtype=np.uint8)
            bb = np.empty(width, dtype=np.uint8)
            ee = np.empty(width, dtype=np.uint8)
            filled = 0
            while filled < width:
                t = data[pos]
                pos += 1
                if t > 128:
                    cnt = t - 128
                    val = data[pos]
                    pos += 1
                    rr[filled:filled + cnt] = val
                    gg[filled:filled + cnt] = val
                    bb[filled:filled + cnt] = val
                    ee[filled:filled + cnt] = val
                else:
                    cnt = t
                    seg = np.frombuffer(data[pos:pos + cnt * 4], dtype=np.uint8).reshape(cnt, 4)
                    pos += cnt * 4
                    rr[filled:filled + cnt] = seg[:, 0]
                    gg[filled:filled + cnt] = seg[:, 1]
                    bb[filled:filled + cnt] = seg[:, 2]
                    ee[filled:filled + cnt] = seg[:, 3]
                filled += cnt
        else:
            pixels = np.frombuffer(
                bytes([a, b]) + data[pos:pos + width * 4 - 2], dtype=np.uint8
            ).reshape(width, 4)
            pos += width * 4 - 2
            rr, gg, bb, ee = pixels[:, 0], pixels[:, 1], pixels[:, 2], pixels[:, 3]
        scale = np.exp2((ee.astype(np.int32) - 136).astype(np.float32))
        row = np.empty((width, 3), dtype=np.float32)
        row[:, 0] = rr.astype(np.float32) * scale
        row[:, 1] = gg.astype(np.float32) * scale
        row[:, 2] = bb.astype(np.float32) * scale
        out[y if not flip else height - 1 - y] = row
    return out


def _load_env_float(path: str):
    ext = os.path.splitext(path)[1].lower()
    img = None
    try:
        import imageio.v2 as iio
        img = iio.imread(path)
    except Exception:
        img = None
    if img is None and ext == ".hdr":
        try:
            with open(path, "rb") as f:
                img = _decode_rgbe(f.read())
        except Exception:
            img = None
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    arr = arr[:, :, :3]
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return np.ascontiguousarray(arr)


def _get_env_texture(ctx: moderngl.Context, path: str):
    if not path:
        return None
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return None
    mtime = os.path.getmtime(abs_path)
    cached = _ENV_TEX_CACHE.get(abs_path)
    if cached is not None:
        cm, tex = cached
        if abs(mtime - cm) < 0.001:
            return tex
        if tex is not None:
            try:
                tex.release()
            except Exception:
                pass
    arr = _load_env_float(abs_path)
    if arr is None:
        _ENV_TEX_CACHE[abs_path] = (mtime, None)
        return None
    h, w = arr.shape[:2]
    tex = ctx.texture((w, h), 3, arr.tobytes(), dtype="f4")
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = True
    tex.repeat_y = True
    _ENV_TEX_CACHE[abs_path] = (mtime, tex)
    return tex


_MOON_TEX_CACHE: dict[str, tuple[float, object]] = {}
_WHITE_TEX: object = None


def _load_moon_float(path: str):
    try:
        import imageio.v2 as iio
        img = iio.imread(path)
    except Exception:
        return None
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    if arr.shape[2] == 3:
        ones = np.ones((arr.shape[0], arr.shape[1], 1), dtype=arr.dtype)
        arr = np.concatenate([arr, ones], axis=2)
    else:
        arr = arr[:, :, :4]
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return np.ascontiguousarray(np.clip(arr, 0.0, 1.0))


def _get_moon_texture(ctx: moderngl.Context, path: str):
    if not path:
        return None
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return None
    mtime = os.path.getmtime(abs_path)
    cached = _MOON_TEX_CACHE.get(abs_path)
    if cached is not None:
        cm, tex = cached
        if abs(mtime - cm) < 0.001:
            return tex
        if tex is not None:
            try:
                tex.release()
            except Exception:
                pass
    arr = _load_moon_float(abs_path)
    if arr is None:
        _MOON_TEX_CACHE[abs_path] = (mtime, None)
        return None
    h, w = arr.shape[:2]
    tex = ctx.texture((w, h), 4, arr.tobytes(), dtype="f4")
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = False
    tex.repeat_y = False
    _MOON_TEX_CACHE[abs_path] = (mtime, tex)
    return tex


def _get_white_tex(ctx: moderngl.Context):
    global _WHITE_TEX
    if _WHITE_TEX is None:
        _WHITE_TEX = ctx.texture((1, 1), 3, b"\xff\xff\xff")
    return _WHITE_TEX


_DEFAULT_YEAR = 2024
_DEFAULT_MONTH = 6
_DEFAULT_DAY = 21
_DEFAULT_HOUR = 12
_DEFAULT_MINUTE = 0
_DEFAULT_SECOND = 0
_DEFAULT_LATITUDE = 55.75
_DEFAULT_LONGITUDE = 37.62
_DEFAULT_UTC_OFFSET = 3.0

_J2000_JD = 2451545.0
_SYNODIC_MONTH = 29.530588853
_NEW_MOON_J2000_JD = 2451550.258


def _julian_day(year: int, month: int, day: int) -> float:
    y = year
    m = month
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + day + b - 1524.5


def _sun_ecliptic_longitude(d: float) -> float:
    g = (357.5291 + 0.98560028 * d) % 360.0
    q = (280.459 + 0.98564736 * d) % 360.0
    return q + 1.915 * math.sin(math.radians(g)) + 0.020 * math.sin(math.radians(2.0 * g))


def _obliquity(d: float) -> float:
    return 23.439 - 0.0000004 * d


def _equatorial(lon_deg: float, lat_deg: float, ecl_deg: float):
    ln = math.radians(lon_deg)
    bt = math.radians(lat_deg)
    ep = math.radians(ecl_deg)
    ra = math.atan2(math.sin(ln) * math.cos(ep) - math.tan(bt) * math.sin(ep), math.cos(ln))
    dec = math.asin(math.sin(bt) * math.cos(ep) + math.cos(bt) * math.sin(ep) * math.sin(ln))
    return math.degrees(ra) % 360.0, math.degrees(dec)


def _solar_equatorial(d: float):
    ecl = _obliquity(d)
    return _equatorial(_sun_ecliptic_longitude(d), 0.0, ecl)


def _moon_ecliptic(d: float):
    lp = (218.316 + 13.176396 * d) % 360.0
    mp = (134.963 + 13.064993 * d) % 360.0
    ms = (357.5291 + 0.98560028 * d) % 360.0
    dm = (297.850 + 12.190749 * d) % 360.0
    f = (93.272 + 13.229350 * d) % 360.0
    lon = lp + 6.289 * math.sin(math.radians(mp)) \
          - 1.274 * math.sin(math.radians(2.0 * dm - mp)) \
          + 0.658 * math.sin(math.radians(2.0 * dm)) \
          - 0.186 * math.sin(math.radians(ms)) \
          - 0.060 * math.sin(math.radians(2.0 * mp - 2.0 * dm))
    lat = 5.128 * math.sin(math.radians(f)) \
          + 0.280 * math.sin(math.radians(mp + f)) \
          + 0.277 * math.sin(math.radians(mp - f)) \
          - 0.017 * math.sin(math.radians(2.0 * dm - f))
    return lon % 360.0, lat


def _moon_equatorial(d: float):
    lon, lat = _moon_ecliptic(d)
    return _equatorial(lon, lat, _obliquity(d))


def _local_sidereal_hours(d: float, longitude_deg: float) -> float:
    gmst = (18.697374558 + 24.06570982441908 * d) % 24.0
    return (gmst + longitude_deg / 15.0) % 24.0


def _alt_az_from_equatorial(ra_deg: float, dec_deg: float, lst_hours: float, latitude_deg: float):
    h = math.radians((lst_hours - ra_deg / 15.0) * 15.0)
    lat = math.radians(latitude_deg)
    dec = math.radians(dec_deg)
    sin_alt = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(h)
    alt = math.asin(max(-1.0, min(1.0, sin_alt)))
    cos_az = (math.sin(dec) - math.sin(alt) * math.sin(lat)) / (math.cos(alt) * math.cos(lat) + 1e-9)
    cos_az = max(-1.0, min(1.0, cos_az))
    az = math.degrees(math.acos(cos_az))
    if math.sin(h) > 0.0:
        az = 360.0 - az
    return math.degrees(alt), az


def _dir_from_alt_az(alt_deg: float, az_deg: float) -> Vec3:
    a = math.radians(alt_deg)
    z = math.radians(az_deg)
    return Vec3(math.sin(z) * math.cos(a), math.sin(a), -math.cos(z) * math.cos(a))


def _sep_deg(a: Vec3, b: Vec3) -> float:
    dot = max(-1.0, min(1.0, a.x * b.x + a.y * b.y + a.z * b.z))
    return math.degrees(math.acos(dot))


def _moon_phase_from_days(d: float) -> float:
    age = (d - _NEW_MOON_J2000_JD) % _SYNODIC_MONTH
    return age / _SYNODIC_MONTH


def _sun_moon_separation(d: float) -> float:
    lon_m, lat_m = _moon_ecliptic(d)
    lon_s = _sun_ecliptic_longitude(d)
    lat_s = 0.0
    cos_sep = (math.sin(math.radians(lat_m)) * math.sin(math.radians(lat_s))
               + math.cos(math.radians(lat_m)) * math.cos(math.radians(lat_s))
               * math.cos(math.radians(lon_m - lon_s)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_sep))))


def _new_moon_offset(d: float) -> float:
    r = (_moon_ecliptic(d)[0] - _sun_ecliptic_longitude(d)) % 360.0
    return ((r - 180.0) % 360.0) - 180.0


def _refine_new_moon(t0: float, t1: float) -> float:
    for _ in range(24):
        tm = (t0 + t1) * 0.5
        if _new_moon_offset(tm) < 0.0:
            t0 = tm
        else:
            t1 = tm
    return (t0 + t1) * 0.5


_ECLIPSE_SEARCH_DAYS = 365 * 5
_ECLIPSE_SAMPLE_STEP = 0.25
_ECLIPSE_MAX_SEPARATION = 0.9


def nearest_solar_eclipse(d0: float):
    best = None
    prev_t = None
    prev_r = None
    t = d0 - _ECLIPSE_SEARCH_DAYS
    end = d0 + _ECLIPSE_SEARCH_DAYS
    while t <= end:
        r = _new_moon_offset(t)
        if prev_r is not None and prev_r <= 0.0 < r:
            nm = _refine_new_moon(prev_t, t)
            sep = _sun_moon_separation(nm)
            if sep < _ECLIPSE_MAX_SEPARATION:
                if best is None or abs(nm - d0) < abs(best[0] - d0):
                    best = (nm, sep)
        prev_t = t
        prev_r = r
        t += _ECLIPSE_SAMPLE_STEP
    return best


def _julian_day_to_ymd(jd: float):
    jd = jd + 0.5
    z = int(jd)
    f = jd - z
    if z < 2299161:
        a = z
    else:
        alpha = int((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - int(alpha / 4)
    b = a + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)
    day = b - d - int(30.6001 * e) + f
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    return int(year), int(month), day


def release_env_cache():
    for _m, tex in _ENV_TEX_CACHE.values():
        if tex is not None:
            try:
                tex.release()
            except Exception:
                pass
    _ENV_TEX_CACHE.clear()
    for _m, tex in _MOON_TEX_CACHE.values():
        if tex is not None:
            try:
                tex.release()
            except Exception:
                pass
    _MOON_TEX_CACHE.clear()
    global _WHITE_TEX
    if _WHITE_TEX is not None:
        try:
            _WHITE_TEX.release()
        except Exception:
            pass
        _WHITE_TEX = None
    release_sky_ibl_cache()


@ComponentRegistry.register
class Sky(Component):
    _icon = "Sky.png"
    _registry: list = []

    def on_awake(self):
        if self not in self._registry:
            self._registry.append(self)

    def on_destroy(self):
        if self in self._registry:
            self._registry.remove(self)

    def on_disable(self):
        if self in self._registry:
            self._registry.remove(self)

    def on_enable(self):
        if self not in self._registry:
            self._registry.append(self)

    @staticmethod
    def _eclipse_coverage(sun_r_deg: float, moon_r_deg: float, sep_deg: float) -> float:
        import math as _m
        if sep_deg <= 0.0:
            return 1.0 if moon_r_deg >= sun_r_deg else (moon_r_deg / sun_r_deg) ** 2
        R = _m.radians(sun_r_deg)
        r = _m.radians(moon_r_deg)
        d = _m.radians(sep_deg)
        if d >= R + r:
            return 0.0
        if d <= r - R:
            return 1.0
        if d <= R - r:
            return (r / R) ** 2
        cos1 = (d * d + R * R - r * r) / (2.0 * d * R)
        cos1 = max(-1.0, min(1.0, cos1))
        a1 = _m.acos(cos1) * R * R
        cos2 = (d * d + r * r - R * R) / (2.0 * d * r)
        cos2 = max(-1.0, min(1.0, cos2))
        a2 = _m.acos(cos2) * r * r
        p = (R + r + d) * 0.5
        tri = 2.0 * _m.sqrt(max(0.0, (p - R) * (p - r) * (p - d) * p))
        return (a1 + a2 - tri) / (_m.pi * R * R)

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("", "Procedural Night Sky", FieldType.HEADER),
            InspectorField("night_sky_enabled", "Night Sky", FieldType.BOOL),
            InspectorField("night_exposure", "Night Exposure", FieldType.SLIDER, min_val=0.0, max_val=3.0, step=0.1, decimals=1),
            InspectorField("", "Stars", FieldType.HEADER),
            InspectorField("star_enabled", "Stars", FieldType.BOOL),
            InspectorField("star_density", "Density", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=2),
            InspectorField("star_intensity", "Intensity", FieldType.SLIDER, min_val=0.0, max_val=5.0, step=0.1, decimals=1),
            InspectorField("star_scale", "Scale", FieldType.SLIDER, min_val=20.0, max_val=200.0, step=1.0, decimals=0),
            InspectorField("star_twinkle", "Twinkle", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.05, decimals=2),
            InspectorField("star_seed", "Seed", FieldType.SLIDER, min_val=0.0, max_val=100.0, step=0.1, decimals=1),
            InspectorField("star_color", "Tint", FieldType.COLOR),
            InspectorField("", "Milky Way", FieldType.HEADER),
            InspectorField("milky_way_enabled", "Milky Way", FieldType.BOOL),
            InspectorField("milky_way_intensity", "Intensity", FieldType.SLIDER, min_val=0.0, max_val=3.0, step=0.1, decimals=1),
            InspectorField("milky_way_pole", "Band Pole", FieldType.VEC3),
            InspectorField("", "Time Controller", FieldType.HEADER),
            InspectorField("year", "Year", FieldType.INT, min_val=1, max_val=9999, step=1),
            InspectorField("month", "Month", FieldType.INT, min_val=1, max_val=12, step=1, on_set="_on_time_field_set"),
            InspectorField("day", "Day", FieldType.INT, min_val=1, max_val=31, step=1, on_set="_on_time_field_set"),
            InspectorField("hour", "Hour", FieldType.INT, min_val=0, max_val=23, step=1, on_set="_on_time_field_set"),
            InspectorField("minute", "Minute", FieldType.INT, min_val=0, max_val=59, step=1, on_set="_on_time_field_set"),
            InspectorField("second", "Second", FieldType.INT, min_val=0, max_val=59, step=1, on_set="_on_time_field_set"),
            InspectorField("_btn_nearest_solar_eclipse", "Nearest Solar Eclipse", FieldType.BUTTON),
            InspectorField("latitude", "Latitude (deg)", FieldType.SLIDER, min_val=-90.0, max_val=90.0, step=0.1, decimals=2),
            InspectorField("longitude", "Longitude (deg)", FieldType.SLIDER, min_val=-180.0, max_val=180.0, step=0.1, decimals=2),
            InspectorField("utc_offset", "UTC Offset (h)", FieldType.SLIDER, min_val=-12.0, max_val=14.0, step=0.5, decimals=1),
            InspectorField("", "Sun Light", FieldType.HEADER),
            InspectorField("sun_light_entity_id", "Sun Light", FieldType.GAMEOBJECT),
            InspectorField("", "Moon", FieldType.HEADER),
            InspectorField("moon_enabled", "Moon", FieldType.BOOL),
            InspectorField("moon_size", "Angular Radius (deg)", FieldType.SLIDER, min_val=0.05, max_val=1.5, step=0.01, decimals=2),
            InspectorField("moon_intensity", "Intensity", FieldType.SLIDER, min_val=0.0, max_val=5.0, step=0.1, decimals=1),
            InspectorField("moon_texture_path", "Texture", FieldType.RESOURCE_PATH, file_filter="Images (*.png *.jpg *.jpeg *.tga *.bmp)"),
        ]

    def __init__(self):
        super().__init__()
        self.material_path: str = "core/shaders/Sky.shader"
        self.environment_path: str = ""
        self.night_sky_enabled: bool = True
        self.night_exposure: float = 1.0
        self.star_enabled: bool = True
        self.star_density: float = 0.45
        self.star_intensity: float = 1.0
        self.star_scale: float = 80.0
        self.star_twinkle: float = 0.5
        self.star_seed: float = 1.0
        self.star_color: list[float] = [0.9, 0.93, 1.0]
        self.milky_way_enabled: bool = True
        self.milky_way_intensity: float = 0.6
        self.milky_way_pole: Vec3 = Vec3(0.4, 0.3, 0.85)
        self.year: int = _DEFAULT_YEAR
        self.month: int = _DEFAULT_MONTH
        self.day: int = _DEFAULT_DAY
        self.hour: int = _DEFAULT_HOUR
        self.minute: int = _DEFAULT_MINUTE
        self.second: int = _DEFAULT_SECOND
        self.latitude: float = _DEFAULT_LATITUDE
        self.longitude: float = _DEFAULT_LONGITUDE
        self.utc_offset: float = _DEFAULT_UTC_OFFSET
        self.sun_light_entity_id: str = ""
        self.moon_enabled: bool = True
        self.moon_size: float = 0.27
        self.moon_intensity: float = 1.0
        self.moon_texture_path: str = "core/textures/moon.tga"
        self._sky_ibl = None
        self._time_cache_key = None
        self._sun_dir: Vec3 = Vec3(0.0, 0.3, 1.0)
        self._moon_dir: Vec3 = Vec3(0.25, 0.6, 0.75)
        self._moon_phase: float = 1.0
        self._day_seconds: float = 0.0
        self._sim_seconds: float = 0.0
        self._eclipse_darkness: float = 0.0
        self._star_pole: Vec3 = Vec3(0.0, 1.0, 0.0)
        self._star_rotation: float = 0.0

    def set_time(self, year=None, month=None, day=None,
                 hour=None, minute=None, second=None):
        if year is not None:
            self.year = int(year)
        if month is not None:
            self.month = int(month)
        if day is not None:
            self.day = int(day)
        if hour is not None:
            self.hour = int(hour)
        if minute is not None:
            self.minute = int(minute)
        if second is not None:
            self.second = int(second)
        self._invalidate_time()

    def _days_in_month(self, y: int, m: int) -> int:
        if m == 2:
            leap = (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)
            return 29 if leap else 28
        return 31 if m in (1, 3, 5, 7, 8, 10, 12) else 30

    def _normalize_days(self):
        while self.day > self._days_in_month(self.year, self.month):
            self.day -= self._days_in_month(self.year, self.month)
            self.month += 1
            if self.month > 12:
                self.month = 1
                self.year += 1
        while self.day < 1:
            self.month -= 1
            if self.month < 1:
                self.month = 12
                self.year -= 1
            self.day += self._days_in_month(self.year, self.month)

    def _carry_days(self, n: int):
        if n == 0:
            return
        self.day += n
        self._normalize_days()

    def _carry_hours(self, n: int):
        if n == 0:
            return
        total = self.hour + n
        carry, self.hour = divmod(total, 24)
        self._carry_days(carry)

    def _carry_minutes(self, n: int):
        if n == 0:
            return
        total = self.minute + n
        carry, self.minute = divmod(total, 60)
        self._carry_hours(carry)

    def _on_time_field_set(self, name: str, value: int) -> bool:
        v = int(value)
        before = self._time_fields_key()
        if name == "second":
            carry, self.second = divmod(v, 60)
            self._carry_minutes(carry)
        elif name == "minute":
            carry, self.minute = divmod(v, 60)
            self._carry_hours(carry)
        elif name == "hour":
            carry, self.hour = divmod(v, 24)
            self._carry_days(carry)
        elif name == "day":
            self.day = v
            self._normalize_days()
        elif name == "month":
            carry, self.month = divmod(v - 1, 12)
            self.month += 1
            self.year += carry
        else:
            return False
        self._invalidate_time()
        return self._time_fields_key() != before

    def set_sun_light(self, entity) -> bool:
        if entity is None or entity.transform is None:
            return False
        self.sun_light_entity_id = entity.id
        return True

    def _btn_nearest_solar_eclipse(self):
        jd = _julian_day(self.year, self.month, self.day)
        civil = self.hour + self.minute / 60.0 + self.second / 3600.0
        utc = civil - self.utc_offset
        d0 = jd - _J2000_JD + utc / 24.0
        best = nearest_solar_eclipse(d0)
        if best is None:
            return
        jd = best[0] + _J2000_JD
        moon_ra, moon_dec = _moon_equatorial(best[0])
        gmst = (18.697374558 + 24.06570982441908 * best[0]) % 24.0
        sub_lon = ((moon_ra - 15.0 * gmst + 180.0) % 360.0) - 180.0
        self.latitude = max(-90.0, min(90.0, moon_dec))
        self.longitude = sub_lon
        self.utc_offset = float(int(round(self.longitude / 15.0)))
        local_hours = ((jd + 0.5) % 1.0) * 24.0 + self.utc_offset
        day_shift = math.floor(local_hours / 24.0)
        local_hours -= day_shift * 24.0
        y, m, day = _julian_day_to_ymd(jd + day_shift)
        h = int(local_hours)
        mi = int((local_hours - h) * 60.0)
        s = int(round(((local_hours - h) * 60.0 - mi) * 60.0))
        if s == 60:
            s = 0
            mi += 1
        if mi == 60:
            mi = 0
            h += 1
        self.set_time(year=y, month=m, day=int(day), hour=h, minute=mi, second=s)

    def _invalidate_time(self):
        self._time_cache_key = None

    def _time_fields_key(self):
        return (self.year, self.month, self.day, self.hour, self.minute,
                self.second, self.latitude, self.longitude, self.utc_offset)

    def _update_time_cache(self):
        key = self._time_fields_key()
        if key == self._time_cache_key:
            return
        self._time_cache_key = key
        jd = _julian_day(self.year, self.month, self.day)
        civil = self.hour + self.minute / 60.0 + self.second / 3600.0
        utc = civil - self.utc_offset
        d = jd - _J2000_JD + utc / 24.0
        self._day_seconds = civil * 3600.0
        self._sim_seconds = (jd - _J2000_JD) * 86400.0 + utc * 3600.0
        lst = _local_sidereal_hours(d, self.longitude)
        self._star_pole = _dir_from_alt_az(self.latitude, 0.0)
        self._star_rotation = math.radians(lst * 15.0)
        sun_ra, sun_dec = _solar_equatorial(d)
        sal, saz = _alt_az_from_equatorial(sun_ra, sun_dec, lst, self.latitude)
        self._sun_dir = _dir_from_alt_az(sal, saz)
        moon_ra, moon_dec = _moon_equatorial(d)
        mal, maz = _alt_az_from_equatorial(moon_ra, moon_dec, lst, self.latitude)
        self._moon_dir = _dir_from_alt_az(mal, maz)
        self._moon_phase = _moon_phase_from_days(d)
        sun_r_deg = 0.27
        try:
            atmos = next((a for a in Atmosphere._registry
                          if a.enabled and a.entity and a.entity.active), None)
            if atmos is not None:
                sun_r_deg = float(getattr(atmos, "_sun_angular_radius", 0.27))
        except Exception:
            pass
        sep = _sep_deg(self._sun_dir, self._moon_dir)
        self._eclipse_darkness = self._eclipse_coverage(sun_r_deg, self.moon_size, sep)

    @property
    def sun_direction(self) -> Vec3:
        self._update_time_cache()
        return self._sun_dir

    @property
    def moon_direction(self) -> Vec3:
        self._update_time_cache()
        return self._moon_dir

    @property
    def moon_phase(self) -> float:
        self._update_time_cache()
        return self._moon_phase

    @property
    def day_seconds(self) -> float:
        self._update_time_cache()
        return self._day_seconds

    @property
    def sim_seconds(self) -> float:
        self._update_time_cache()
        return self._sim_seconds

    @property
    def star_pole(self) -> Vec3:
        self._update_time_cache()
        return self._star_pole

    @property
    def star_rotation(self) -> float:
        self._update_time_cache()
        return self._star_rotation

    @property
    def eclipse_darkness(self) -> float:
        self._update_time_cache()
        return self._eclipse_darkness

    def get_sun_light(self):
        scene = self._entity._scene if self._entity else None
        ent = None
        if self.sun_light_entity_id and scene:
            ent = scene.get_entity(self.sun_light_entity_id)
        if ent is not None:
            l = ent.get_component(Light)
            t = ent.transform
            if l and l.enabled and t and l.light_type == LightType.DIRECTIONAL:
                return (l, t)
        if scene:
            for e in scene.get_entities_with_component(Light):
                l = e.get_component(Light)
                t = e.transform
                if l and l.enabled and t and l.light_type == LightType.DIRECTIONAL:
                    return (l, t)
        return None

    def _sync_sun_light(self):
        if not self.enabled:
            return
        self._update_time_cache()
        sl = self.get_sun_light()
        if sl is None:
            return
        _l, tr = sl
        f = self._sun_dir * -1.0
        r = f.cross(Vec3.up())
        rl = r.length()
        if rl < 1e-6:
            r = f.cross(Vec3.right())
            rl = r.length()
        r = r * (1.0 / rl)
        u = r.cross(f)
        m3 = np.eye(3, dtype=FLOAT_TYPE)
        m3[0] = [r.x, r.y, r.z]
        m3[1] = [u.x, u.y, u.z]
        m3[2] = [-f.x, -f.y, -f.z]
        tr.local_rotation = Quat._from_rotation_matrix3(m3.T)

    def on_update(self, dt: float):
        self._sync_sun_light()

    def _night_settings_key(self) -> tuple:
        self._update_time_cache()
        return (
            self.night_sky_enabled, self.night_exposure,
            self.star_enabled, self.star_density, self.star_intensity,
            self.star_scale, self.star_twinkle, self.star_seed,
            tuple(self.star_color),
            tuple(self.star_pole), round(self.star_rotation, 6),
            self.milky_way_enabled, self.milky_way_intensity,
            tuple(self.milky_way_pole),
            self.moon_enabled, tuple(self._moon_dir),
            self.moon_size, self.moon_intensity,
            self._moon_phase, self.moon_texture_path,
        )

    def _ibl_settings_key(self, atmos) -> tuple:
        key = self._night_settings_key()
        if atmos is not None and getattr(atmos, "enabled", False):
            key = key + (
                atmos._intensity, atmos._sun_intensity, atmos._resolution_scale,
                atmos._ozone_factor, atmos._aerosol_scale,
                atmos._sun_angular_radius, atmos._sun_limb_darkening,
                atmos._sun_convergence, atmos._color_temperature,
            )
        return key

    def _apply_night_sky(self, prog, ctx=None):
        self._sync_sun_light()
        def f(name, val):
            if name in prog:
                prog[name].value = float(val)

        def v3(name, val):
            if name in prog:
                if isinstance(val, Vec3):
                    val = [val.x, val.y, val.z]
                prog[name].write(np.array(val, dtype=np.float32).tobytes())

        f("_NightSkyEnabled", 1.0 if self.night_sky_enabled else 0.0)
        f("_NightExposure", self.night_exposure)
        f("_StarEnabled", 1.0 if self.star_enabled else 0.0)
        f("_StarDensity", self.star_density)
        f("_StarIntensity", self.star_intensity)
        f("_StarScale", self.star_scale)
        f("_StarTwinkle", self.star_twinkle)
        f("_StarSeed", self.star_seed)
        v3("_StarColor", self.star_color)
        v3("_StarPole", self.star_pole)
        f("_StarRotation", self.star_rotation)
        f("_MilkyWayEnabled", 1.0 if self.milky_way_enabled else 0.0)
        f("_MilkyWayIntensity", self.milky_way_intensity)
        v3("_MilkyWayPole", self.milky_way_pole)
        f("_MoonEnabled", 1.0 if self.moon_enabled else 0.0)
        v3("_MoonDirection", self._moon_dir)
        f("_MoonSize", self.moon_size)
        f("_MoonIntensity", self.moon_intensity)
        f("_MoonPhase", self._moon_phase)
        if ctx is not None and "u_moon_tex" in prog:
            moon_tex = _get_moon_texture(ctx, self.moon_texture_path)
            if moon_tex is None:
                moon_tex = _get_white_tex(ctx)
            moon_tex.use(4)
            prog["u_moon_tex"].value = 4
            prog["u_use_moon_tex"].value = 1.0 if self.moon_texture_path else 0.0
        if "u_time" in prog:
            prog["u_time"].value = self._day_seconds

    def render_sky(self, ctx, shaders, view_mat, proj_mat, dir_light, cube_mesh):
        prog = shaders.get_or_compile(self.material_path) if shaders else None
        if not prog:
            return
        self._apply_night_sky(prog, ctx)

        def set_sun_disk_defaults(p):
            if "_SunAngularRadius" in p:
                p["_SunAngularRadius"].value = 0.27
            if "_SunLimbDarkening" in p:
                p["_SunLimbDarkening"].value = 0.7
            if "_SunConvergence" in p:
                p["_SunConvergence"].value = 0.5

        sl = self.get_sun_light()
        if sl is not None:
            dir_light = sl
        sun_world = None
        sun_color = None
        sun_intensity = None
        if dir_light:
            dl, dt = dir_light
            sky_color, sky_intensity = Light.shader_radiance(dl, dt)
            if "_SunDirection" in prog:
                sun_dir = -dt.forward
                prog["_SunDirection"].write(np.array([sun_dir.x, sun_dir.y, sun_dir.z], dtype=np.float32).tobytes())
            if "_SunColor" in prog:
                prog["_SunColor"].write(np.array(sky_color, dtype=np.float32).tobytes())
            if "_SunIntensity" in prog:
                prog["_SunIntensity"].value = sky_intensity
            set_sun_disk_defaults(prog)
            sun_world = (-dt.forward.x, -dt.forward.y, -dt.forward.z)
            sun_color = sky_color
            sun_intensity = sky_intensity
        else:
            if "_SunDirection" in prog:
                prog["_SunDirection"].write(np.array([0.0, -0.3, -1.0], dtype=np.float32).tobytes())
            if "_SunColor" in prog:
                prog["_SunColor"].write(np.array([1.0, 0.95, 0.85], dtype=np.float32).tobytes())
            if "_SunIntensity" in prog:
                prog["_SunIntensity"].value = 1.0
            set_sun_disk_defaults(prog)
            sun_world = (0.0, -0.3, -1.0)
            sun_color = [1.0, 0.95, 0.85]
            sun_intensity = 1.0
        atmos = next((a for a in Atmosphere._registry
                      if a.enabled and a.entity and a.entity.active), None)
        if atmos is not None and "u_use_atmosphere" in prog:
            if atmos.ensure_luts(ctx, sun_world, sun_color, sun_intensity):
                atmos.bind_sky(prog)
            else:
                prog["u_use_atmosphere"].value = 0
        elif "u_use_atmosphere" in prog:
            prog["u_use_atmosphere"].value = 0
        env_tex = _get_env_texture(ctx, self.environment_path) if self.environment_path else None
        if self.environment_path:
            self._sky_ibl = get_sky_ibl(ctx, self.environment_path, env_tex)
        else:
            sun_dir = None
            sun_color = None
            sun_intensity = None
            if dir_light:
                dl, dt = dir_light
                sky_c, sky_i = Light.shader_radiance(dl, dt)
                sun_dir = (-dt.forward.x, -dt.forward.y, -dt.forward.z)
                sun_color = sky_c
                sun_intensity = sky_i
            self._sky_ibl = get_procedural_sky_ibl(ctx, prog, self.material_path,
                                                   sun_dir, sun_color, sun_intensity,
                                                   settings_key=self._ibl_settings_key(atmos))
        if "u_env_tex" in prog:
            if env_tex is not None:
                env_tex.use(0)
                prog["u_env_tex"].value = 0
                if "u_use_env" in prog:
                    prog["u_use_env"].value = 1
            elif "u_use_env" in prog:
                prog["u_use_env"].value = 0
        sky_view = np.eye(4, dtype=np.float64)
        sky_view[:3, :3] = view_mat._d[:3, :3].copy()
        sky_mat4 = Mat4(sky_view)
        mvp = sky_mat4 * proj_mat
        if "u_mvp" in prog:
            prog["u_mvp"].write(mvp.to_f32().tobytes())
        ctx.disable(moderngl.CULL_FACE)
        ctx.disable(moderngl.DEPTH_TEST)
        cube_mesh.render(prog)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)

    def serialize(self) -> dict:
        d = super().serialize()
        d["material_path"] = self.material_path
        d["environment_path"] = self.environment_path
        d["night_sky_enabled"] = self.night_sky_enabled
        d["night_exposure"] = self.night_exposure
        d["star_enabled"] = self.star_enabled
        d["star_density"] = self.star_density
        d["star_intensity"] = self.star_intensity
        d["star_scale"] = self.star_scale
        d["star_twinkle"] = self.star_twinkle
        d["star_seed"] = self.star_seed
        d["star_color"] = self.star_color
        d["milky_way_enabled"] = self.milky_way_enabled
        d["milky_way_intensity"] = self.milky_way_intensity
        d["milky_way_pole"] = self.milky_way_pole.to_list()
        d["year"] = self.year
        d["month"] = self.month
        d["day"] = self.day
        d["hour"] = self.hour
        d["minute"] = self.minute
        d["second"] = self.second
        d["latitude"] = self.latitude
        d["longitude"] = self.longitude
        d["utc_offset"] = self.utc_offset
        d["sun_light_entity_id"] = self.sun_light_entity_id
        d["moon_enabled"] = self.moon_enabled
        d["moon_size"] = self.moon_size
        d["moon_intensity"] = self.moon_intensity
        d["moon_texture_path"] = self.moon_texture_path
        return d

    @classmethod
    def deserialize(cls, data: dict) -> Sky:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.material_path = data.get("material_path", "core/shaders/Sky.shader")
        c.environment_path = data.get("environment_path", "")
        c.night_sky_enabled = data.get("night_sky_enabled", True)
        c.night_exposure = data.get("night_exposure", 1.0)
        c.star_enabled = data.get("star_enabled", True)
        c.star_density = data.get("star_density", 0.45)
        c.star_intensity = data.get("star_intensity", 1.0)
        c.star_scale = data.get("star_scale", 80.0)
        c.star_twinkle = data.get("star_twinkle", 0.5)
        c.star_seed = data.get("star_seed", 1.0)
        c.star_color = data.get("star_color", [0.9, 0.93, 1.0])
        c.milky_way_enabled = data.get("milky_way_enabled", True)
        c.milky_way_intensity = data.get("milky_way_intensity", 0.6)
        c.milky_way_pole = Vec3(*data.get("milky_way_pole", [0.4, 0.3, 0.85]))
        c.year = data.get("year", _DEFAULT_YEAR)
        c.month = data.get("month", _DEFAULT_MONTH)
        c.day = data.get("day", _DEFAULT_DAY)
        c.hour = data.get("hour", _DEFAULT_HOUR)
        c.minute = data.get("minute", _DEFAULT_MINUTE)
        c.second = data.get("second", _DEFAULT_SECOND)
        c.latitude = data.get("latitude", _DEFAULT_LATITUDE)
        c.longitude = data.get("longitude", _DEFAULT_LONGITUDE)
        c.utc_offset = data.get("utc_offset", _DEFAULT_UTC_OFFSET)
        c.sun_light_entity_id = data.get("sun_light_entity_id", "")
        c.moon_enabled = data.get("moon_enabled", True)
        c.moon_size = data.get("moon_size", 0.27)
        c.moon_intensity = data.get("moon_intensity", 1.0)
        c.moon_texture_path = data.get("moon_texture_path", "core/textures/moon.tga")
        return c
