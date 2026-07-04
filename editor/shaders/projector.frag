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
uniform float u_pj_0_flip_y;
uniform float u_pj_1_flip_y;
uniform float u_pj_0_flip_x;
uniform float u_pj_1_flip_x;
uniform sampler2D u_pj_0_tex;
uniform sampler2D u_pj_1_tex;
uniform int u_projector_count;
uniform sampler2D u_pj_0_shadow_map;
uniform sampler2D u_pj_1_shadow_map;
uniform mat4 u_pj_0_shadow_vp;
uniform mat4 u_pj_1_shadow_vp;
uniform float u_shadow_bias;
in vec2 v_uv;
out vec4 frag_color;
float sample_projector_shadow(sampler2D shadow_map, vec3 proj_coords) {
    float current_depth = proj_coords.z - u_shadow_bias;
    float result = 0.0;
    vec2 texel_size = 1.0 / vec2(textureSize(shadow_map, 0));
    float radius = 0.75;
    float weight_sum = 0.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float weight = 1.0;
            if (x == 0) weight += 1.0;
            if (y == 0) weight += 1.0;
            float pcf_depth = texture(shadow_map, proj_coords.xy + vec2(x, y) * texel_size * radius).r;
            result += (current_depth > pcf_depth ? 1.0 : 0.0) * weight;
            weight_sum += weight;
        }
    }
    float lit = 1.0 - result / weight_sum;
    return smoothstep(0.12, 0.88, lit);
}
float compute_projector_shadow(mat4 shadow_vp, sampler2D shadow_map, vec3 wpos) {
    vec4 light_space_pos = shadow_vp * vec4(wpos, 1.0);
    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    proj_coords = proj_coords * 0.5 + 0.5;
    if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
        proj_coords.y < 0.0 || proj_coords.y > 1.0 ||
        proj_coords.z < 0.0 || proj_coords.z > 1.0) return 1.0;
    return sample_projector_shadow(shadow_map, proj_coords);
}
vec3 process_projector(mat4 vp, vec3 pos, vec3 dir, vec3 color, float intensity, float range, float spot_angle, float has_tex, float flip_y, float flip_x, sampler2D tex, mat4 shadow_vp, sampler2D shadow_map, vec3 wpos) {
    vec4 proj_space = vp * vec4(wpos, 1.0);
    vec3 proj_coords = proj_space.xyz / proj_space.w;
    proj_coords = proj_coords * 0.5 + 0.5;
    if (flip_y > 0.5) proj_coords.y = 1.0 - proj_coords.y;
    if (flip_x > 0.5) proj_coords.x = 1.0 - proj_coords.x;
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
    float shadow = compute_projector_shadow(shadow_vp, shadow_map, wpos);
    return color * tex_color.rgb * intensity * att * shadow;
}
void main() {
    float depth = texture(u_depth_tex, v_uv).r;
    if (depth >= 1.0) discard;
    vec4 clip = vec4(v_uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 world = u_inv_vp * clip;
    world /= world.w;
    vec3 wpos = world.xyz;
    vec3 result = vec3(0.0);
    if (u_projector_count >= 1) result += process_projector(u_pj_0_vp, u_pj_0_pos, u_pj_0_dir, u_pj_0_color, u_pj_0_intensity, u_pj_0_range, u_pj_0_spot_angle, u_pj_0_has_tex, u_pj_0_flip_y, u_pj_0_flip_x, u_pj_0_tex, u_pj_0_shadow_vp, u_pj_0_shadow_map, wpos);
    if (u_projector_count >= 2) result += process_projector(u_pj_1_vp, u_pj_1_pos, u_pj_1_dir, u_pj_1_color, u_pj_1_intensity, u_pj_1_range, u_pj_1_spot_angle, u_pj_1_has_tex, u_pj_1_flip_y, u_pj_1_flip_x, u_pj_1_tex, u_pj_1_shadow_vp, u_pj_1_shadow_map, wpos);
    frag_color = vec4(result, 1.0);
}
