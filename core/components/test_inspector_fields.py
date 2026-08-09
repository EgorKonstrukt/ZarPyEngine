from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField, ListElementField
from core.maths.math3d import Vec2, Vec3, Vec4
from core.foundation.curve import Curve


@ComponentRegistry.register
class TestInspectorFields(Component):
    _icon = "Script.png"
    _allow_multiple = False

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("header_basic", "Basic Types", FieldType.HEADER),
            InspectorField("float_field", "Float", FieldType.FLOAT, min_val=-10.0, max_val=100.0, step=0.5, decimals=3),
            InspectorField("int_field", "Int", FieldType.INT, min_val=0, max_val=255),
            InspectorField("bool_field", "Bool", FieldType.BOOL),
            InspectorField("string_field", "String", FieldType.STRING),
            InspectorField("textarea_field", "Text Area", FieldType.TEXTAREA),
            InspectorField("enum_field", "Enum", FieldType.ENUM, enum_options=["Option A", "Option B", "Option C"]),

            InspectorField("header_numeric", "Numeric Sliders", FieldType.HEADER),
            InspectorField("slider_field", "Slider", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01),
            InspectorField("int_slider_field", "Int Slider", FieldType.INT_SLIDER, min_val=0, max_val=100, step=1),
            InspectorField("layer_field", "Layer", FieldType.LAYER),
            InspectorField("layer_mask_field", "Layer Mask", FieldType.LAYER_MASK),

            InspectorField("header_vector", "Vector Types", FieldType.HEADER),
            InspectorField("vec2_field", "Vec2", FieldType.VEC2),
            InspectorField("vec3_field", "Vec3", FieldType.VEC3),
            InspectorField("vec4_field", "Vec4", FieldType.VEC4),
            InspectorField("vec2_slider_field", "Vec2 Slider", FieldType.VEC2_SLIDER, min_val=-5.0, max_val=5.0),
            InspectorField("vec3_slider_field", "Vec3 Slider", FieldType.VEC3_SLIDER, min_val=-1.0, max_val=1.0),

            InspectorField("header_resource", "Resources & References", FieldType.HEADER),
            InspectorField("gameobject_field", "GameObject", FieldType.GAMEOBJECT),
            InspectorField("resource_path_field", "Resource Path", FieldType.RESOURCE_PATH, file_filter="Text Files (*.txt)"),
            InspectorField("resource_field", "Resource", FieldType.RESOURCE, file_filter="All Files (*)"),
            InspectorField("asset_field", "Asset", FieldType.ASSET, resource_type="mesh"),

            InspectorField("header_visual", "Visual Types", FieldType.HEADER),
            InspectorField("color_field", "Color", FieldType.COLOR),
            InspectorField("curve_field", "Curve", FieldType.CURVE),
            InspectorField("gradient_field", "Gradient", FieldType.GRADIENT),
            InspectorField("keybinding_field", "Keybinding", FieldType.KEYBINDING),
            InspectorField("anchor_field", "Anchor", FieldType.ANCHOR),

            InspectorField("header_action", "Actions", FieldType.HEADER),
            InspectorField("button_field", "Test Action", FieldType.BUTTON),

            InspectorField("header_list", "List Container", FieldType.HEADER),
            InspectorField("list_field", "List", FieldType.LIST,
                element_fields=[
                    ListElementField("name", "Name", FieldType.STRING),
                    ListElementField("weight", "Weight", FieldType.FLOAT, 0.0, 1.0, 0.1, 2),
                    ListElementField("enabled", "Enabled", FieldType.BOOL),
                    ListElementField("color", "Color", FieldType.COLOR),
                ],
            ),
        ]

    def __init__(self):
        super().__init__()
        self.float_field: float = 42.5
        self.int_field: int = 7
        self.bool_field: bool = True
        self.string_field: str = "Hello World"
        self.textarea_field: str = "Line 1\nLine 2\nLine 3"
        self.enum_field: str = "Option A"

        self.slider_field: float = 0.45
        self.int_slider_field: int = 50
        self.layer_field: int = 0
        self.layer_mask_field: int = 0xFFFF

        self.vec2_field: Vec2 = Vec2(1.5, 2.5)
        self.vec3_field: Vec3 = Vec3(3, 4, 5)
        self.vec4_field: Vec4 = Vec4(0.5, 0.25, 0.75, 1.0)
        self.vec2_slider_field: Vec2 = Vec2(0.0, 2.5)
        self.vec3_slider_field: Vec3 = Vec3(-0.5, 0.0, 0.5)

        self.gameobject_field: str = ""
        self.resource_path_field: str = ""
        self.resource_field: str = ""
        self.asset_field: str = ""

        self.color_field: list = [1.0, 0.2, 0.1]
        self.curve_field = Curve()
        self.gradient_field = [
            (0.0, [1.0, 0.0, 0.0, 1.0]),
            (0.5, [0.0, 1.0, 0.0, 1.0]),
            (1.0, [0.0, 0.0, 1.0, 1.0]),
        ]
        self.keybinding_field: str = "Space"
        self.anchor_field: int = 4  # ANCHOR_CENTER

        self.list_field: list = []

    def button_field(self):
        print("TestInspectorFields button clicked")
