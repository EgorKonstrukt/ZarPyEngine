from __future__ import annotations
import time
import numpy as np
from core.ecs.ecs import ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
from core.math.math3d import Vec3
from core.components.rendering.effects.object_effect import ObjectEffect


@ComponentRegistry.register
class TreeWindEffect(ObjectEffect):
    _gizmo_icon_label = "TW"
    fx_uniform_defaults = {"_WindInfluence": 1.0}

    @classmethod
    def fx_geometry_shader(cls) -> "str | None":
        try:
            from core.renderer.mesh_data import read_shader
            return read_shader("Tree.gshader")
        except Exception:
            return None

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("wind_influence", "Wind Influence", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("wind_direction", "Wind Direction", FieldType.VEC3, min_val=-1.0, max_val=1.0, step=0.05, decimals=3),
            InspectorField("wind_speed", "Wind Speed", FieldType.FLOAT, min_val=0.0, max_val=20.0, step=0.1, decimals=2),
            InspectorField("wind_strength", "Wind Strength", FieldType.FLOAT, min_val=0.0, max_val=3.0, step=0.01, decimals=3),
            InspectorField("leaf_flutter_speed", "Leaf Flutter Speed", FieldType.FLOAT, min_val=0.0, max_val=30.0, step=0.1, decimals=2),
            InspectorField("leaf_flutter_amount", "Leaf Flutter Amount", FieldType.FLOAT, min_val=0.0, max_val=0.3, step=0.005, decimals=4),
            InspectorField("turbulence_scale", "Turbulence Scale", FieldType.FLOAT, min_val=0.0, max_val=5.0, step=0.1, decimals=2),
            InspectorField("turbulence_amount", "Turbulence Amount", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
        ]

    def __init__(self):
        super().__init__()
        self.double_sided: bool = True
        self.wind_influence: float = 1.0
        self.wind_direction: Vec3 = Vec3(1.0, 0.0, 0.5)
        self.wind_speed: float = 2.0
        self.wind_strength: float = 0.3
        self.leaf_flutter_speed: float = 8.0
        self.leaf_flutter_amount: float = 0.04
        self.turbulence_scale: float = 0.5
        self.turbulence_amount: float = 0.3
        self._dir_buf = np.zeros(3, dtype=np.float32)

    def _apply(self, prog):
        if not self.enabled or self.wind_influence <= 0.0:
            self._set(prog, "_WindInfluence", 0.0)
            return
        d = self.wind_direction
        length = (d.x * d.x + d.y * d.y + d.z * d.z) ** 0.5
        if length > 1e-6:
            inv = 1.0 / length
            self._dir_buf[0] = d.x * inv
            self._dir_buf[1] = d.y * inv
            self._dir_buf[2] = d.z * inv
        else:
            self._dir_buf[0] = 1.0
            self._dir_buf[1] = 0.0
            self._dir_buf[2] = 0.0
        self._set(prog, "_WindInfluence", float(self.wind_influence))
        self._set_vec_bytes(prog, "_WindDir", self._dir_buf)
        self._set(prog, "_WindSpeed", float(self.wind_speed))
        self._set(prog, "_WindStrength", float(self.wind_strength))
        self._set(prog, "_LeafFlutterSpeed", float(self.leaf_flutter_speed))
        self._set(prog, "_LeafFlutterAmount", float(self.leaf_flutter_amount))
        self._set(prog, "_TurbulenceScale", float(self.turbulence_scale))
        self._set(prog, "_TurbulenceAmount", float(self.turbulence_amount))

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "double_sided": self.double_sided,
            "wind_influence": self.wind_influence,
            "wind_direction": [self.wind_direction.x, self.wind_direction.y, self.wind_direction.z],
            "wind_speed": self.wind_speed,
            "wind_strength": self.wind_strength,
            "leaf_flutter_speed": self.leaf_flutter_speed,
            "leaf_flutter_amount": self.leaf_flutter_amount,
            "turbulence_scale": self.turbulence_scale,
            "turbulence_amount": self.turbulence_amount,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> TreeWindEffect:
        fx = cls()
        fx.enabled = data.get("enabled", True)
        fx.double_sided = data.get("double_sided", True)
        fx.wind_influence = data.get("wind_influence", 1.0)
        fd = data.get("wind_direction", [1.0, 0.0, 0.5])
        fx.wind_direction = Vec3(*fd[:3])
        fx.wind_speed = data.get("wind_speed", 2.0)
        fx.wind_strength = data.get("wind_strength", 0.3)
        fx.leaf_flutter_speed = data.get("leaf_flutter_speed", 8.0)
        fx.leaf_flutter_amount = data.get("leaf_flutter_amount", 0.04)
        fx.turbulence_scale = data.get("turbulence_scale", 0.5)
        fx.turbulence_amount = data.get("turbulence_amount", 0.3)
        return fx
