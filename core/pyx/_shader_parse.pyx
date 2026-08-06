# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from cpython.unicode cimport PyUnicode_AsUTF8, PyUnicode_FromStringAndSize


cdef inline int _find_closing_brace(const char *text, int start, int length) noexcept nogil:
    cdef int depth = 0
    cdef int i = start
    cdef char c
    while i < length:
        c = text[i]
        if c == b'{':
            depth += 1
        elif c == b'}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


cdef inline int _find_marker(const char *text, const char *marker, int marker_len, int start, int length) noexcept nogil:
    cdef int i = start
    cdef int j
    cdef bint found
    while i <= length - marker_len:
        found = True
        for j in range(marker_len):
            if text[i + j] != marker[j]:
                found = False
                break
        if found:
            return i
        i += 1
    return -1


cdef inline int _skip_ws_left(const char *text, int pos) noexcept nogil:
    while pos > 0 and (text[pos - 1] == b' ' or text[pos - 1] == b'\t' or
                        text[pos - 1] == b'\n' or text[pos - 1] == b'\r'):
        pos -= 1
    return pos


cdef inline int _skip_ws_right(const char *text, int pos, int length) noexcept nogil:
    while pos < length and (text[pos] == b' ' or text[pos] == b'\t' or
                             text[pos] == b'\n' or text[pos] == b'\r'):
        pos += 1
    return pos


def py_find_closing_brace(text: str, start: int) -> int:
    cdef const char *data = PyUnicode_AsUTF8(text)
    cdef int length = len(text)
    return _find_closing_brace(data, start, length)


def extract_glsl_from_shader(text: str):
    cdef const char *data = PyUnicode_AsUTF8(text)
    cdef int length = len(text)
    cdef const char *glsl_marker = b"GLSLPROGRAM"
    cdef int glsl_marker_len = 11
    cdef const char *end_marker = b"ENDGLSL"
    cdef int end_marker_len = 7
    cdef const char *frag_marker = b"// @FRAGMENT"
    cdef int frag_marker_len = 12
    cdef const char *geom_marker = b"// @GEOMETRY"
    cdef int geom_marker_len = 12
    cdef int glsl_start, glsl_end, body_start, body_end
    cdef int frag_pos, geom_pos, vert_end, rest_start, frag_pos2, geom_end
    cdef str vert_src, geom_src, frag_src

    glsl_start = _find_marker(data, glsl_marker, glsl_marker_len, 0, length)
    if glsl_start < 0:
        return None
    glsl_start += glsl_marker_len
    glsl_end = _find_marker(data, end_marker, end_marker_len, glsl_start, length)
    if glsl_end < 0:
        return None
    body_start = _skip_ws_right(data, glsl_start, glsl_end)
    body_end = _skip_ws_left(data, glsl_end)
    frag_pos = _find_marker(data, frag_marker, frag_marker_len, body_start, body_end)
    geom_pos = _find_marker(data, geom_marker, geom_marker_len, body_start, body_end)

    if geom_pos >= 0 and (frag_pos < 0 or geom_pos < frag_pos):
        vert_end = _skip_ws_left(data, geom_pos)
        vert_src = PyUnicode_FromStringAndSize(data + body_start, vert_end - body_start)
        rest_start = geom_pos + geom_marker_len
        frag_pos2 = _find_marker(data, frag_marker, frag_marker_len, rest_start, body_end)
        if frag_pos2 >= 0:
            geom_end = _skip_ws_left(data, frag_pos2)
            geom_src = PyUnicode_FromStringAndSize(data + rest_start, geom_end - rest_start)
            frag_src = PyUnicode_FromStringAndSize(
                data + frag_pos2 + frag_marker_len,
                body_end - frag_pos2 - frag_marker_len,
            )
        else:
            geom_src = PyUnicode_FromStringAndSize(data + rest_start, body_end - rest_start)
            frag_src = ""
        return (vert_src.strip(), geom_src.strip(), frag_src.strip())

    if frag_pos < 0:
        return None
    vert_end = _skip_ws_left(data, frag_pos)
    vert_src = PyUnicode_FromStringAndSize(data + body_start, vert_end - body_start)
    frag_src = PyUnicode_FromStringAndSize(
        data + frag_pos + frag_marker_len,
        body_end - frag_pos - frag_marker_len,
    )
    return (vert_src.strip(), "", frag_src.strip())


cdef inline void _parse_pass_glsl(const char *data, int length, int pass_brace, int pass_end, list out):
    cdef const char *glsl_marker = b"GLSLPROGRAM"
    cdef int glsl_marker_len = 11
    cdef const char *end_marker = b"ENDGLSL"
    cdef int end_marker_len = 7
    cdef const char *frag_marker = b"// @FRAGMENT"
    cdef int frag_marker_len = 12
    cdef const char *geom_marker = b"// @GEOMETRY"
    cdef int geom_marker_len = 12
    cdef int gs = _find_marker(data, glsl_marker, glsl_marker_len, pass_brace + 1, pass_end)
    if gs < 0:
        return
    gs += glsl_marker_len
    cdef int ge = _find_marker(data, end_marker, end_marker_len, gs, pass_end)
    if ge < 0:
        return
    cdef int bs = _skip_ws_right(data, gs, ge)
    cdef int be = _skip_ws_left(data, ge)
    cdef int fp = _find_marker(data, frag_marker, frag_marker_len, bs, be)
    cdef int gp = _find_marker(data, geom_marker, geom_marker_len, bs, be)
    cdef int vert_end, rs, fp2, geom_end
    cdef str vert_s, geom_s, frag_s
    if gp >= 0 and (fp < 0 or gp < fp):
        vert_end = _skip_ws_left(data, gp)
        vert_s = PyUnicode_FromStringAndSize(data + bs, vert_end - bs).strip()
        rs = gp + geom_marker_len
        fp2 = _find_marker(data, frag_marker, frag_marker_len, rs, be)
        if fp2 >= 0:
            geom_end = _skip_ws_left(data, fp2)
            geom_s = PyUnicode_FromStringAndSize(data + rs, geom_end - rs).strip()
            frag_s = PyUnicode_FromStringAndSize(data + fp2 + frag_marker_len, be - fp2 - frag_marker_len).strip()
        else:
            geom_s = PyUnicode_FromStringAndSize(data + rs, be - rs).strip()
            frag_s = ""
        out.append((vert_s, geom_s, frag_s))
    elif fp >= 0:
        vert_end = _skip_ws_left(data, fp)
        vert_s = PyUnicode_FromStringAndSize(data + bs, vert_end - bs).strip()
        frag_s = PyUnicode_FromStringAndSize(data + fp + frag_marker_len, be - fp - frag_marker_len).strip()
        out.append((vert_s, "", frag_s))


cdef list _parse_properties(const char *data, int pb, int pe):
    cdef list properties = []
    cdef int pos = pb + 1
    cdef int line_start, line_end, paren, close_paren, eq_pos, default_start, k
    cdef int di, dlen, num_start, dq1, dq2
    cdef str prop_name, prop_type_raw, prop_type, type_args, default_str, tl, pdname
    cdef double range_min, range_max
    cdef object default_value
    cdef list nums
    cdef list rp

    while pos < pe:
        line_start = pos
        while pos < pe and data[pos] != b'\n':
            pos += 1
        line_end = pos
        pos += 1
        while line_start < line_end and data[line_start] in (b' ', b'\t', b'\r'):
            line_start += 1
        while line_end > line_start and data[line_end - 1] in (b' ', b'\t', b'\r'):
            line_end -= 1
        if line_end <= line_start:
            continue
        if data[line_start] in (b'{', b'}'):
            continue
        if data[line_start] == b'[':
            while line_start < line_end and data[line_start] != b']':
                line_start += 1
            if line_start < line_end:
                line_start += 1
            while line_start < line_end and data[line_start] in (b' ', b'\t'):
                line_start += 1
        if line_start >= line_end:
            continue
        paren = -1
        for k in range(line_start, line_end):
            if data[k] == b'(':
                paren = k
                break
        if paren < 0:
            continue
        k = paren - 1
        while k > line_start and data[k] not in (b' ', b'\t'):
            k -= 1
        if k <= line_start:
            continue
        prop_name = bytes(data[line_start:k]).decode('utf-8').strip()
        prop_type_raw = bytes(data[k + 1:paren]).decode('utf-8').strip()
        close_paren = paren + 1
        while close_paren < line_end and data[close_paren] != b')':
            close_paren += 1
        type_args = bytes(data[paren + 1:close_paren]).decode('utf-8').strip() if close_paren > paren + 1 else ""
        eq_pos = close_paren + 1
        while eq_pos < line_end and data[eq_pos] != b'=':
            eq_pos += 1
        default_start = eq_pos + 1 if eq_pos < line_end else line_end
        while default_start < line_end and data[default_start] in (b' ', b'\t'):
            default_start += 1
        default_str = bytes(data[default_start:line_end]).decode('utf-8').strip()

        prop_type = prop_type_raw
        range_min = 0.0
        range_max = 1.0
        default_value = default_str
        tl = prop_type_raw.lower()

        if tl.startswith('range') and type_args:
            prop_type = "Range"
            rp = type_args.split(',')
            if len(rp) >= 2:
                try:
                    range_min = float(rp[0].strip())
                    range_max = float(rp[1].strip())
                except ValueError:
                    pass

        if tl == 'color' or tl.startswith('vector'):
            nums = []
            di = 0
            dlen = len(default_str)
            while di < dlen:
                while di < dlen and not (default_str[di].isdigit() or default_str[di] == '-' or default_str[di] == '.'):
                    di += 1
                if di >= dlen:
                    break
                num_start = di
                while di < dlen and (default_str[di].isdigit() or default_str[di] == '.' or default_str[di] in '-+eEf'):
                    di += 1
                try:
                    nums.append(float(default_str[num_start:di].rstrip('f')))
                except ValueError:
                    pass
            default_value = nums
        elif tl == 'float' or tl.startswith('range'):
            try:
                default_value = float(default_str.rstrip('f'))
            except ValueError:
                default_value = 0.0
        elif tl == 'int':
            try:
                default_value = int(default_str)
            except ValueError:
                default_value = 0
        elif tl == '2d' or tl == 'cube':
            dq1 = default_str.find('"')
            if dq1 >= 0:
                dq2 = default_str.find('"', dq1 + 1)
                default_value = default_str[dq1 + 1:dq2] if dq2 > dq1 else ""
            else:
                default_value = ""

        pdname = prop_name[1:] if prop_name.startswith('_') else prop_name
        properties.append((prop_name, pdname, prop_type, default_value, [], range_min, range_max))
    return properties


def parse_shader_file(path: str):
    cdef bytes raw
    cdef const char *data
    cdef int length
    cdef int ns_start, ns_end
    cdef int pp, pb, pe
    cdef int sp, sb, se, search_from, pass_pos, pass_brace, pass_end
    cdef int fb, fq1, fq2
    cdef str shader_name, fallback
    cdef list properties, passes
    cdef const char *props_marker = b"Properties"
    cdef int props_marker_len = 10
    cdef const char *subshader_marker = b"SubShader"
    cdef int subshader_marker_len = 9
    cdef const char *pass_marker = b"Pass"
    cdef int pass_marker_len = 4
    cdef const char *fallback_marker = b"Fallback"
    cdef int fallback_marker_len = 8

    try:
        with open(path, 'rb') as f:
            raw = f.read()
    except Exception:
        return None

    data = raw
    length = len(raw)

    ns_start = _find_marker(data, b'Shader "', 8, 0, length)
    if ns_start < 0:
        return None
    ns_start += 8
    ns_end = _find_marker(data, b'"', 1, ns_start, length)
    if ns_end < 0:
        return None
    shader_name = bytes(data[ns_start:ns_end]).decode('utf-8')

    properties = []
    pp = _find_marker(data, props_marker, props_marker_len, 0, length)
    if pp >= 0:
        pb = _find_marker(data, b"{", 1, pp + props_marker_len, length)
        if pb >= 0:
            pe = _find_closing_brace(data, pb, length)
            if pe > pb:
                properties = _parse_properties(data, pb, pe)

    passes = []
    sp = _find_marker(data, subshader_marker, subshader_marker_len, 0, length)
    if sp >= 0:
        sb = _find_marker(data, b"{", 1, sp + subshader_marker_len, length)
        if sb >= 0:
            se = _find_closing_brace(data, sb, length)
            if se > sb:
                search_from = sb + 1
                while search_from < se:
                    pass_pos = _find_marker(data, pass_marker, pass_marker_len, search_from, se)
                    if pass_pos < 0:
                        break
                    pass_brace = _find_marker(data, b"{", 1, pass_pos + pass_marker_len, se)
                    if pass_brace < 0:
                        break
                    pass_end = _find_closing_brace(data, pass_brace, length)
                    if pass_end < 0 or pass_end <= pass_brace:
                        search_from = pass_brace + 1
                        continue
                    _parse_pass_glsl(data, length, pass_brace, pass_end, passes)
                    search_from = pass_end + 1

    fallback = ""
    fb = _find_marker(data, fallback_marker, fallback_marker_len, 0, length)
    if fb >= 0:
        fq1 = _find_marker(data, b'"', 1, fb + fallback_marker_len, length)
        if fq1 >= 0:
            fq2 = _find_marker(data, b'"', 1, fq1 + 1, length)
            if fq2 >= 0:
                fallback = bytes(data[fq1 + 1:fq2]).decode('utf-8')

    return (shader_name, properties, passes, fallback)