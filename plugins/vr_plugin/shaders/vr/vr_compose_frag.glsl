#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_eye_left;
uniform sampler2D u_eye_right;
uniform vec2 u_resolution;
void main() {
    float hx = 0.5;
    if (v_uv.x < hx)
        fragColor = texture(u_eye_left,  vec2(v_uv.x / hx, v_uv.y));
    else
        fragColor = texture(u_eye_right, vec2((v_uv.x - hx) / hx, v_uv.y));
}
