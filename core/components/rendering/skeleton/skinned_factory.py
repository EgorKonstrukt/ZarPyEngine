# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.

from core.components.rendering.skeleton.armature import Armature
from core.components.rendering.renderers.skinned_mesh_renderer import SkinnedMeshRenderer


def create_skinned_mesh_entity(scene, path: str, name: str, mesh_path: str, world_pos=None):
    from core.components import Transform
    from core.assets.asset_importer import load_mesh

    try:
        import_data = load_mesh(path)
    except Exception:
        import_data = None

    if import_data is None or not getattr(import_data, "has_skeleton", False):
        return None

    return _build_skinned_entity(scene, path, name, mesh_path, world_pos, import_data)


def create_skinned_mesh_entity_async(scene, path: str, name: str, mesh_path: str,
                                     world_pos=None, callback=None):
    if callback:
        callback(None, False)


def _build_skinned_entity(scene, path, name, mesh_path, world_pos, import_data):
    from core.components import Transform

    ent = scene.create_entity(name)
    tr = Transform()
    if world_pos is not None:
        tr.local_position = world_pos
    try:
        from core.assets.asset_importer import _read_mesh_import
        _scale = float(_read_mesh_import(path).get("scale", 1.0))
        if _scale and _scale != 1.0:
            from core.math.math3d import Vec3
            tr.local_scale = Vec3(_scale, _scale, _scale)
    except Exception:
        pass
    ent.add_component(tr)

    smr = SkinnedMeshRenderer()
    smr.mesh_name = name
    smr.mesh_path = mesh_path
    ent.add_component(smr)

    arm = Armature()
    arm.setup(import_data)
    ent.add_component(arm)

    arm.create_bone_entities(scene, ent)
    return ent
