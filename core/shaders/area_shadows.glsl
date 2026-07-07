// Area Light Shadow Functions (PCSS)
// Injected via // @SHADOW_INCLUDE marker
// Requires: u_area_shadow_map, u_area_light_vp, u_area_light_size,
//           u_area_light_fov_scale, u_area_light_near_far,
//           u_area_shadow_light_index, u_area_shadow_bias, v_world_pos

#ifndef _AREA_SHADOWS_GLSL
#define _AREA_SHADOWS_GLSL

#ifndef POISSON_SAMPLES
#define POISSON_SAMPLES 16
#endif

const vec2 _area_poisson_disk[POISSON_SAMPLES] = vec2[](
    vec2(0.079575, 0.578682), vec2(-0.882423, 0.073115),
    vec2(-0.362586, 0.584344), vec2(0.509705, -0.620710),
    vec2(-0.123670, -0.953501), vec2(0.860160, 0.299207),
    vec2(0.691263, -0.507288), vec2(-0.453053, -0.725332),
    vec2(-0.725407, 0.390592), vec2(0.251858, 0.158784),
    vec2(0.744786, -0.187990), vec2(-0.293813, -0.178377),
    vec2(-0.659467, -0.024936), vec2(0.192856, 0.880197),
    vec2(-0.087864, -0.142065), vec2(0.218582, -0.643362)
);

float _area_hash2(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float _area_pcss_blocker_search(sampler2D shadow_map, vec3 proj_coords, float z_ndc, float bias, float search_step) {
    float a = _area_hash2(gl_FragCoord.xy) * 6.2831853;
    float ca = cos(a);
    float sa = sin(a);
    float blocker_sum = 0.0;
    float blocker_count = 0.0;
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            vec2 off = vec2(float(x), float(y));
            vec2 rot = vec2(off.x * ca - off.y * sa, off.x * sa + off.y * ca);
            vec2 uv = proj_coords.xy + rot * search_step;
            float d = texture(shadow_map, uv).r;
            if (d < z_ndc - bias) {
                blocker_sum += d;
                blocker_count += 1.0;
            }
        }
    }
    if (blocker_count < 1.0) return -1.0;
    return blocker_sum / blocker_count;
}

#ifdef PBR_AREA_SHADOWS
float area_pcss(sampler2D shadow_map, vec3 proj_coords, float z_view) {
    vec2 texel_size = 1.0 / vec2(textureSize(shadow_map, 0));
    float z_ndc = proj_coords.z;
    float near_z = u_area_light_near_far.x;
    float far_z = u_area_light_near_far.y;
    float bias = max(u_area_shadow_bias, 0.001 + 0.0002 * z_view);
    float proj_light_size = u_area_light_size * u_area_light_fov_scale / max(z_view, 0.001);
    float search_step = max(proj_light_size * 0.25, texel_size.x * 2.0);
    float avg_blocker_ndc = _area_pcss_blocker_search(shadow_map, proj_coords, z_ndc, bias, search_step);
    if (avg_blocker_ndc < 0.0) return 1.0;
    float avg_blocker_z = 2.0 * near_z * far_z / max(far_z + near_z - (avg_blocker_ndc * 2.0 - 1.0) * (far_z - near_z), 0.001);
    float penumbra_world = u_area_light_size * (z_view - avg_blocker_z) / max(avg_blocker_z, 0.001);
    float proj_penumbra = penumbra_world * u_area_light_fov_scale / max(z_view, 0.001);
    proj_penumbra = max(proj_penumbra, texel_size.x);
    float a2 = _area_hash2(gl_FragCoord.xy + vec2(0.5, 0.5)) * 6.2831853;
    float ca2 = cos(a2);
    float sa2 = sin(a2);
    float pcf_radius = max(proj_penumbra * 0.5, texel_size.x * 2.0);
    float result = 0.0;
    for (int s = 0; s < POISSON_SAMPLES; s++) {
        vec2 off = _area_poisson_disk[s];
        vec2 rot = vec2(off.x * ca2 - off.y * sa2, off.x * sa2 + off.y * ca2);
        vec2 uv = proj_coords.xy + rot * pcf_radius;
        float d = texture(shadow_map, uv).r;
        result += (z_ndc - bias > d ? 1.0 : 0.0);
    }
    float shadow = 1.0 - result / float(POISSON_SAMPLES);
    return smoothstep(0.05, 0.65, shadow);
}
#else
float area_pcss(sampler2D shadow_map, vec3 proj_coords, float z_view) {
    vec2 texel_size = 1.0 / vec2(textureSize(shadow_map, 0));
    float z_ndc = proj_coords.z;
    float near_z = u_area_light_near_far.x;
    float far_z = u_area_light_near_far.y;
    float bias = max(u_area_shadow_bias, 0.001 + 0.0002 * z_view);
    float proj_light_size = u_area_light_size * u_area_light_fov_scale / max(z_view, 0.001);
    float search_step = max(proj_light_size * 0.25, texel_size.x * 2.0);
    float avg_blocker_ndc = _area_pcss_blocker_search(shadow_map, proj_coords, z_ndc, bias, search_step);
    if (avg_blocker_ndc < 0.0) return 1.0;
    float avg_blocker_z = 2.0 * near_z * far_z / max(far_z + near_z - (avg_blocker_ndc * 2.0 - 1.0) * (far_z - near_z), 0.001);
    float penumbra_world = u_area_light_size * (z_view - avg_blocker_z) / max(avg_blocker_z, 0.001);
    float proj_penumbra = penumbra_world * u_area_light_fov_scale / max(z_view, 0.001);
    proj_penumbra = max(proj_penumbra, texel_size.x);
    float filter_texels = proj_penumbra / texel_size.x;
    int k = clamp(int(filter_texels * 0.3), 2, 9);
    float pcf_step = proj_penumbra / max(float(k + k), 1.0);
    float a = _area_hash2(gl_FragCoord.xy) * 6.2831853;
    float ca = cos(a);
    float sa = sin(a);
    float result = 0.0;
    float wsum = 0.0;
    for (int x = -k; x <= k; x++) {
        for (int y = -k; y <= k; y++) {
            float w = 1.0;
            if (x == 0) w += 1.0;
            if (y == 0) w += 1.0;
            vec2 off = vec2(float(x), float(y));
            vec2 rot = vec2(off.x * ca - off.y * sa, off.x * sa + off.y * ca);
            vec2 uv = proj_coords.xy + rot * pcf_step;
            float d = texture(shadow_map, uv).r;
            result += (z_ndc - bias > d ? 1.0 : 0.0) * w;
            wsum += w;
        }
    }
    float shadow = 1.0 - result / max(wsum, 1.0);
    return smoothstep(0.05, 0.65, shadow);
}
#endif

float compute_area_shadow() {
    if (u_area_shadow_light_index < 0) return 1.0;
    vec4 light_space_pos = u_area_light_vp * vec4(v_world_pos, 1.0);
    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    proj_coords = proj_coords * 0.5 + 0.5;
    if (any(lessThan(proj_coords, vec3(0.0))) || any(greaterThan(proj_coords, vec3(1.0)))) return 1.0;
    float z_ndc = proj_coords.z * 2.0 - 1.0;
    float near_z = u_area_light_near_far.x;
    float far_z = u_area_light_near_far.y;
    float z_view = 2.0 * near_z * far_z / max(far_z + near_z - z_ndc * (far_z - near_z), 0.001);
    return area_pcss(u_area_shadow_map, proj_coords, z_view);
}

#endif
