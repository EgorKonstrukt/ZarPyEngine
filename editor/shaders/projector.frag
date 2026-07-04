#version 460 core
uniform sampler2D u_depth_tex;
uniform mat4 u_inv_vp;
uniform mat4 u_pj_0_vp;
uniform mat4 u_pj_1_vp;
uniform vec3 u_pj_0_pos;
uniform vec3 u_pj_1_pos;
uniform vec3 u_pj_0_dir;
uniform vec3 u_pj_1_dir;
uniform vec3 u_pj_0_color;
uniform vec3 u_pj_1_color;
uniform float u_pj_0_intensity;
uniform float u_pj_1_intensity;
uniform float u_pj_0_range;
uniform float u_pj_1_range;
uniform float u_pj_0_spot_angle;
uniform float u_pj_1_spot_angle;
uniform float u_pj_0_has_tex;
uniform float u_pj_1_has_tex;
uniform sampler2D u_pj_0_tex;
uniform sampler2D u_pj_1_tex;
uniform int u_projector_count;
in vec2 v_uv;
out vec4 frag_color;
vec3 process_projector(mat4 vp, vec3 pos, vec3 dir, vec3 color, float intensity, float range, float spot_angle, float has_tex, sampler2D tex, vec3 wpos) {
    vec4 proj_space = vp * vec4(wpos, 1.0);
    vec3 proj_coords = proj_space.xyz / proj_space.w;
    proj_coords = proj_coords * 0.5 + 0.5;
    if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
        proj_coords.y < 0.0 || proj_coords.y > 1.0 ||
        proj_coords.z <= 0.0 || proj_coords.z >= 1.0) return vec3(0.0);
    vec4 tex_color = vec4(1.0);
    if (has_tex > 0.5) tex_color = texture(tex, proj_coords.xy);
    vec3 to_light = pos - wpos;
    float dist = length(to_light);
    vec3 light_dir = normalize(to_light);
    float att = clamp(1.0 - dist / range, 0.0, 1.0);
    att *= att;
    float theta = dot(light_dir, normalize(-dir));
    float outer = cos(radians(spot_angle));
    att *= smoothstep(outer * 0.95, outer, theta);
    return color * tex_color.rgb * intensity * att;
}
void main() {
    float depth = texture(u_depth_tex, v_uv).r;
    if (depth >= 1.0) discard;
    vec4 clip = vec4(v_uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 world = u_inv_vp * clip;
    world /= world.w;
    vec3 wpos = world.xyz;
    vec3 result = vec3(0.0);
    if (u_projector_count >= 1) result += process_projector(u_pj_0_vp, u_pj_0_pos, u_pj_0_dir, u_pj_0_color, u_pj_0_intensity, u_pj_0_range, u_pj_0_spot_angle, u_pj_0_has_tex, u_pj_0_tex, wpos);
    if (u_projector_count >= 2) result += process_projector(u_pj_1_vp, u_pj_1_pos, u_pj_1_dir, u_pj_1_color, u_pj_1_intensity, u_pj_1_range, u_pj_1_spot_angle, u_pj_1_has_tex, u_pj_1_tex, wpos);
    frag_color = vec4(result, 1.0);
}
