#version 330 core
in vec3 in_pos;
in vec3 in_normal;
uniform mat4 u_mvp;
uniform mat3 u_normal_mat;
out vec3 v_normal;
void main() {
    v_normal = normalize(u_normal_mat * in_normal);
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
