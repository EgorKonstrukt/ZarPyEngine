# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import os
from typing import Callable

from core.components.animation.animator_controller import (
    AnimatorCondition,
    AnimatorConditionMode,
    AnimatorController,
    AnimatorControllerLayer,
    AnimatorParameter,
    AnimatorParameterType,
    AnimatorState,
    AnimatorTransition,
)

_CONDITION_MODE = {
    1: AnimatorConditionMode.IF,
    2: AnimatorConditionMode.IF_NOT,
    3: AnimatorConditionMode.GREATER,
    4: AnimatorConditionMode.LESS,
    5: AnimatorConditionMode.NOT_EQUAL,
    6: AnimatorConditionMode.EQUALS,
}

_PARAM_TYPE = {
    1: AnimatorParameterType.FLOAT,
    2: AnimatorParameterType.INT,
    4: AnimatorParameterType.BOOL,
    9: AnimatorParameterType.TRIGGER,
}

_CONTROLLER_CLASS = 91
_STATE_CLASS = 1102
_SM_CLASS = 1107
_TRANSITION_CLASS = 1101

_META_CACHE: dict = {}


def _scan_meta(root_dir: str) -> dict:
    key = os.path.normpath(root_dir)
    if key in _META_CACHE:
        return _META_CACHE[key]
    guid_map: dict = {}
    for folder, _dirs, files in os.walk(root_dir):
        for fn in files:
            if not fn.lower().endswith(".anim.meta"):
                continue
            meta_path = os.path.join(folder, fn)
            try:
                with open(meta_path, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("guid:"):
                    anim_file = os.path.join(folder, fn[: -len(".meta")])
                    guid_map[line.split(":", 1)[1].strip()] = anim_file
                    break
    _META_CACHE[key] = guid_map
    return guid_map


def _resolve_motion(data: dict, root_dir: str) -> str:
    if not isinstance(data, dict):
        return ""
    guid = str(data.get("guid", "") or "")
    if not guid:
        return ""
    if root_dir:
        return _scan_meta(root_dir).get(guid, "")
    return guid


def _build_transition(td: dict, target_name: str) -> AnimatorTransition:
    conditions = []
    for c in td.get("m_Conditions", []) or []:
        mode = _CONDITION_MODE.get(int(c.get("m_ConditionMode") or 0), AnimatorConditionMode.IF)
        conditions.append(
            AnimatorCondition(
                parameter=str(c.get("m_ConditionEvent") or ""),
                mode=mode,
                threshold=float(c.get("m_EventTreshold") or 0.0),
            )
        )
    return AnimatorTransition(
        destination_state=target_name,
        conditions=conditions,
        has_exit_time=bool(td.get("m_HasExitTime") or 0),
        exit_time=float(td.get("m_ExitTime") or 0.0),
        transition_duration=float(td.get("m_TransitionDuration") or 0.25),
        has_fixed_duration=bool(td.get("m_HasFixedDuration") or 1),
        offset=float(td.get("m_TransitionOffset") or 0.0),
        mute=bool(td.get("m_Mute") or 0),
        solo=bool(td.get("m_Solo") or 0),
    )


def import_controller(
    source,
    resolve_clip: Callable[[str], str] | None = None,
    root_dir: str = "",
) -> AnimatorController:
    from core.components.animation.unity.yaml_util import parse_unity_documents

    text = source
    if os.path.exists(source):
        text_path = source
        with open(text_path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        if not root_dir:
            root_dir = os.path.dirname(os.path.abspath(text_path))
    if root_dir and os.path.isabs(root_dir):
        root_dir = os.path.normpath(root_dir)

    docs = parse_unity_documents(text)
    state_docs: dict = {}
    transition_docs: dict = {}
    sm_docs: dict = {}
    controller_doc = None
    for doc in docs:
        if doc["class_id"] == _STATE_CLASS:
            state_docs[str(doc["file_id"])] = doc["data"]
        elif doc["class_id"] == _TRANSITION_CLASS:
            transition_docs[str(doc["file_id"])] = doc["data"]
        elif doc["class_id"] == _SM_CLASS:
            sm_docs[str(doc["file_id"])] = doc["data"]
        elif doc["class_id"] == _CONTROLLER_CLASS or doc["name"] == "AnimatorController":
            controller_doc = doc["data"]

    if controller_doc is None:
        raise ValueError("no AnimatorController document found in source")

    ctrl = AnimatorController(
        name=str(controller_doc.get("m_Name") or "New Controller")
    )

    for p in controller_doc.get("m_AnimatorParameters", []) or []:
        ptype = _PARAM_TYPE.get(int(p.get("m_Type") or 0), AnimatorParameterType.FLOAT)
        ctrl.add_parameter(
            AnimatorParameter(
                name=str(p.get("m_Name") or "NewParam"),
                param_type=ptype,
                default_float=float(p.get("m_DefaultFloat") or 0.0),
                default_int=int(p.get("m_DefaultInt") or 0),
                default_bool=bool(p.get("m_DefaultBool") or 0),
            )
        )

    state_objs: dict = {}
    name_by_id: dict = {}
    layers: list = []
    for li in controller_doc.get("m_AnimatorLayers", []) or []:
        sm_file_id = str((li.get("m_StateMachine") or {}).get("fileID") or "")
        sm_doc = sm_docs.get(sm_file_id)
        if not sm_doc:
            continue
        layer = AnimatorControllerLayer(
            name=str(li.get("m_Name") or "Base Layer"),
            weight=float(li.get("m_DefaultWeight") or 1.0),
            blending_mode="override" if not int(li.get("m_BlendingMode") or 0) else "additive",
        )
        for cs in sm_doc.get("m_ChildStates", []) or []:
            state_file_id = str((cs.get("m_State") or {}).get("fileID") or "")
            sd = state_docs.get(state_file_id)
            if not sd:
                continue
            motion_guid = str((sd.get("m_Motion") or {}).get("guid", "") or "")
            clip_path = _resolve_motion(sd.get("m_Motion"), root_dir)
            if not clip_path and motion_guid and resolve_clip is not None:
                clip_path = resolve_clip(motion_guid)
            pos = sd.get("m_Position") or {}
            state = AnimatorState(
                name=str(sd.get("m_Name") or "New State"),
                clip_path=clip_path,
                speed=float(sd.get("m_Speed") or 1.0),
                tag=str(sd.get("m_Tag") or ""),
                x=float(pos.get("x") or 0.0),
                y=float(pos.get("y") or 0.0),
            )
            state_objs[state_file_id] = state
            name_by_id[state_file_id] = state.name
            layer.states.append(state)
        default_id = str((sm_doc.get("m_DefaultState") or {}).get("fileID") or "")
        if default_id and default_id in state_objs:
            layer.default_state = state_objs[default_id].name
        ctrl.layers = [layer]
        layers = [layer]

    for state_file_id, sd in state_docs.items():
        state = state_objs.get(state_file_id)
        if state is None:
            continue
        for tid_ref in sd.get("m_Transitions", []) or []:
            tid = str((tid_ref or {}).get("fileID") or "")
            td = transition_docs.get(tid)
            if not td:
                continue
            dst_id = str((td.get("m_DstState") or {}).get("fileID") or "")
            target_name = name_by_id.get(dst_id, dst_id)
            state.transitions.append(_build_transition(td, target_name))

    for sm_file_id, sm_doc in sm_docs.items():
        if sm_file_id not in {str((li.get("m_StateMachine") or {}).get("fileID") or "")
                              for li in controller_doc.get("m_AnimatorLayers", []) or []}:
            continue
        for tid_ref in sm_doc.get("m_AnyStateTransitions", []) or []:
            tid = str((tid_ref or {}).get("fileID") or "")
            td = transition_docs.get(tid)
            if not td:
                continue
            dst_id = str((td.get("m_DstState") or {}).get("fileID") or "")
            target_name = name_by_id.get(dst_id, dst_id)
            tr = _build_transition(td, target_name)
            for state in state_objs.values():
                state.transitions.append(tr)

    for layer in layers:
        if not layer.default_state and layer.states:
            layer.default_state = layer.states[0].name
    return ctrl


def clear_meta_cache():
    _META_CACHE.clear()