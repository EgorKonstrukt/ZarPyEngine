# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import math
import os

from core.components.animation.animation_clip import (
    AnimationClip,
    AnimationEvent,
    RawCurve,
)
from core.foundation.curve import Curve, CurveKey, TangentMode

TRANSFORM_CLASS_ID = 4

_VECTOR_BLOCKS = {
    "m_PositionCurves": "Transform/local_position",
    "m_ScaleCurves": "Transform/local_scale",
    "m_EulerCurves": "Transform/local_euler_angles",
}

_FLOAT_ATTR_MAP = {
    "m_LocalPosition": "Transform/local_position",
    "localPosition": "Transform/local_position",
    "m_LocalEulerAnglesRaw": "Transform/local_euler_angles",
    "localEulerAnglesRaw": "Transform/local_euler_angles",
    "m_LocalEulerAngles": "Transform/local_euler_angles",
    "localEulerAngles": "Transform/local_euler_angles",
    "m_LocalScale": "Transform/local_scale",
    "localScale": "Transform/local_scale",
    "m_LocalRotation": "Transform/local_rotation",
    "localRotation": "Transform/local_rotation",
}


def is_unity_yaml(text_or_path: str) -> bool:
    header = text_or_path
    if os.path.exists(text_or_path):
        with open(text_or_path, encoding="utf-8", errors="replace") as f:
            header = f.read(4096)
    head = header.lstrip()
    return (
        head.startswith("%YAML")
        or head.startswith("%TAG")
        or head.startswith("--- !u!")
    )


def _tangent_mode(in_slope: float, out_slope: float) -> TangentMode:
    if not math.isfinite(in_slope) or not math.isfinite(out_slope):
        return TangentMode.CONSTANT
    if abs(in_slope) < 1e-9 and abs(out_slope) < 1e-9:
        return TangentMode.SMOOTH
    return TangentMode.FREE


def _curve_from_unity_keys(key_data: list, channel: str) -> Curve:
    curve = Curve()
    for kd in key_data:
        value = float(_channel(kd.get("value"), channel))
        in_slope = float(_channel(kd.get("inSlope"), channel))
        out_slope = float(_channel(kd.get("outSlope"), channel))
        curve.keys.append(
            CurveKey(
                time=float(kd.get("time")),
                value=value,
                in_tangent=in_slope,
                out_tangent=out_slope,
                tangent_mode=_tangent_mode(in_slope, out_slope),
            )
        )
    curve.keys.sort(key=lambda x: x.time)
    curve._auto_smooth()
    return curve


def _channel(d, channel: str):
    if d is None:
        return 0.0
    if isinstance(d, dict):
        return d.get(channel, 0.0)
    if isinstance(d, (list, tuple)) and len(d) == 3:
        idx = {"x": 0, "y": 1, "z": 2}[channel]
        return d[idx]
    return d


def _vector_curve_to_leaves(curve_def: dict, property_base: str) -> dict:
    return {
        chan: _curve_from_unity_keys(curve_def, chan)
        for chan in ("x", "y", "z")
    }


def _import_vector_block(doc: dict, key: str, property_base: str) -> dict:
    result: dict = {}
    items = doc.get(key) or []
    if not isinstance(items, list):
        return result
    for item in items:
        bone_path = item.get("path", "") or ""
        curve_def = ((item.get("curve") or {}).get("m_Curve")) or []
        leaves = _vector_curve_to_leaves(curve_def, property_base)
        for chan, curve in leaves.items():
            result[(bone_path, f"{property_base}.{chan}")] = curve
    return result


def _import_rotation_curves(doc: dict) -> dict:
    result: dict = {}
    items = doc.get("m_RotationCurves") or []
    if not isinstance(items, list):
        return result
    for item in items:
        bone_path = item.get("path", "") or ""
        from core.foundation.curve import QuaternionCurve

        qcurve = QuaternionCurve()
        curve_def = ((item.get("curve") or {}).get("m_Curve")) or []
        for kd in curve_def:
            value = kd.get("value") or {}
            qcurve.add_key(float(kd.get("time")), (value["x"], value["y"], value["z"], value["w"]))
        result[bone_path] = qcurve
    return result


def _import_float_curve_map(doc: dict) -> dict:
    mapped: dict = {}
    raw: dict = {}
    for key in ("m_FloatCurves", "m_EditorCurves"):
        items = doc.get(key) or []
        if not isinstance(items, list):
            continue
        for item in items:
            bone_path = item.get("path", "") or ""
            attribute = item.get("attribute", "") or ""
            class_id = int(item.get("classID") or 0)
            curve_def = ((item.get("curve") or {}).get("m_Curve")) or []
            curve = _curve_from_unity_keys(curve_def, "value")
            for prefix, prop_base in _FLOAT_ATTR_MAP.items():
                if attribute == prefix or attribute.startswith(prefix + "."):
                    chan = attribute[len(prefix):].lstrip(".")
                    if chan in ("x", "y", "z"):
                        if class_id == TRANSFORM_CLASS_ID or not bone_path:
                            mapped[(bone_path, f"{prop_base}.{chan}")] = curve
                            break
                    elif chan == "w":
                        break  # rotation handled by rotation curves
                    else:
                        break
            else:
                raw[f"{bone_path}/{attribute}"] = RawCurve(
                    attribute=attribute,
                    bone_path=bone_path,
                    class_id=class_id,
                    curve=curve,
                )
    return mapped, raw


def _import_events(doc: dict) -> list:
    events: list = []
    items = doc.get("m_Events") or []
    if not isinstance(items, list):
        return events
    for item in items:
        if not isinstance(item, dict):
            continue
        event = AnimationEvent(
            time=float(item.get("time", 0.0)),
            function_name=str(item.get("functionName", "") or ""),
            string_parameter=str(item.get("data", "") or ""),
            float_parameter=float(item.get("floatParameter", 0.0) or 0.0),
            int_parameter=int(item.get("intParameter", 0) or 0),
        )
        events.append(event)
    events.sort(key=lambda e: e.time)
    return events


def import_anim(source) -> AnimationClip:
    from core.components.animation.unity.yaml_util import parse_unity_documents

    text = source
    if os.path.exists(source):
        with open(source, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    docs = parse_unity_documents(text)
    clip_doc = None
    for doc in docs:
        if doc["class_id"] == 74 or doc["name"] == "AnimationClip":
            clip_doc = doc["data"]
            break
    if clip_doc is None:
        raise ValueError("no AnimationClip document found in source")

    name = str(clip_doc.get("m_Name") or "Clip")
    settings = clip_doc.get("m_AnimationClipSettings") or {}
    loop = bool(settings.get("m_LoopTime"))
    stop_time = float(settings.get("m_StopTime") or 0.0)
    sample_rate = int(clip_doc.get("m_SampleRate") or 60)

    clip = AnimationClip(name=name, length=max(stop_time, 1.0))
    clip.loop = loop
    clip.sample_rate = sample_rate

    by_key: dict = {}
    primary_keys = {"m_PositionCurves", "m_ScaleCurves", "m_EulerCurves"}
    for key, prop_base in _VECTOR_BLOCKS.items():
        kind = "position" if key == "m_PositionCurves" else (
            "scale" if key == "m_ScaleCurves" else "euler")
        for (bone_path, prop), curve in _import_vector_block(clip_doc, key, prop_base).items():
            by_key[(bone_path, prop)] = (curve, kind)

    float_mapped, raw = _import_float_curve_map(clip_doc)

    # primary vector blocks win over per-channel editor curves
    covered: set = set()
    present_bases = [k for k in primary_keys if clip_doc.get(k)]
    for key in present_bases:
        for (bone_path, prop) in _import_vector_block(
            clip_doc, key, _VECTOR_BLOCKS[key]
        ):
            covered.add(prop)

    for (bone_path, prop), curve in float_mapped.items():
        if prop in covered or (bone_path, prop) in by_key:
            continue
        kind = "position" if ".local_position." in prop else \
            "euler" if ".local_euler_angles." in prop else "scale"
        by_key[(bone_path, prop)] = (curve, kind)

    legacy_view: dict = {}
    for (bone_path, prop), (curve, kind) in by_key.items():
        legacy_view[prop] = curve
        clip.bone_curves[(bone_path, prop)] = curve
        clip.curve_targets[prop] = bone_path
        clip.annotations[prop] = {
            "bone_path": bone_path,
            "attribute": prop.rsplit("/", 1)[-1],
            "class_id": TRANSFORM_CLASS_ID,
            "raw": False,
        }
    clip.curves = legacy_view

    for bone_path, qcurve in _import_rotation_curves(clip_doc).items():
        prop = "Transform/local_rotation"
        clip.rotation_curves[prop] = qcurve
        clip.bone_rotation_curves[(bone_path, prop)] = qcurve
        clip.curve_targets[prop] = bone_path
        clip.annotations[prop] = {
            "bone_path": bone_path,
            "attribute": "m_LocalRotation",
            "class_id": TRANSFORM_CLASS_ID,
            "raw": False,
        }

    for raw_key, rc in raw.items():
        clip.raw_curves[raw_key] = rc

    max_time = 0.0
    for curve in clip.curves.values():
        if curve.keys:
            max_time = max(max_time, curve.keys[-1].time)
    for curve in clip.rotation_curves.values():
        if curve.keys:
            max_time = max(max_time, curve.keys[-1].time)
    clip.length = max(clip.length, max_time + (1.0 / max(sample_rate, 1)))

    clip.events = _import_events(clip_doc)
    return clip