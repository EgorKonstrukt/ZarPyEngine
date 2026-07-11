# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import Component, ComponentRegistry
from core.math.math3d import Vec3


@ComponentRegistry.register
class RemoteCollaborator(Component):
    _icon = "RemoteCollaborator.png"
    _gizmo_icon_color = (60, 180, 255)
    _gizmo_icon_label = "R"

    def __init__(self):
        super().__init__()
        self.peer_id: str = ""
        self.peer_name: str = ""
        self.color: list[float] = [0.2, 0.6, 1.0]

    def gizmo_lines(self):
        t = self.transform
        if not t:
            return []
        pos = t.position
        col = self.color + [1.0]
        size = 0.3
        lines = []
        lines.append((pos + Vec3(-size, -size, 0), pos + Vec3(size, -size, 0), col))
        lines.append((pos + Vec3(size, -size, 0), pos + Vec3(size, size, 0), col))
        lines.append((pos + Vec3(size, size, 0), pos + Vec3(-size, size, 0), col))
        lines.append((pos + Vec3(-size, size, 0), pos + Vec3(-size, -size, 0), col))
        lines.append((pos, pos + Vec3(0, 0, -size * 2), col))
        return lines

    def gizmo_meshes(self):
        return []

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({"peer_id": self.peer_id, "peer_name": self.peer_name, "color": self.color})
        return d

    @classmethod
    def deserialize(cls, data: dict) -> RemoteCollaborator:
        rc = cls()
        rc.enabled = data.get("enabled", True)
        rc.peer_id = data.get("peer_id", "")
        rc.peer_name = data.get("peer_name", "")
        rc.color = data.get("color", [0.2, 0.6, 1.0])
        return rc
