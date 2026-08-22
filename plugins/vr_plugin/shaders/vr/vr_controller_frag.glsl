#version 330 core
in vec3 v_normal;
uniform vec3 u_color;
uniform float u_grip;
out vec4 fragColor;
void main() {
    vec3 light = normalize(vec3(0.5, 1.0, 0.7));
    float diff = max(dot(v_normal, light), 0.0);
    vec3 ambient = u_color * 0.3;
    vec3 col = u_color * diff + ambient;
    col = mix(col, vec3(1.0, 0.5, 0.1), u_grip * 0.6);
    fragColor = vec4(col, 1.0);
}
