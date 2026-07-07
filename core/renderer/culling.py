# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np

try:
    from core._culling import cpu_frustum_cull as _culling_cpu
    from core._culling import extract_frustum_planes_c

    def extract_frustum_planes(view_proj: np.ndarray) -> np.ndarray:
        return extract_frustum_planes_c(
            np.ascontiguousarray(view_proj, dtype=np.float64))

    def cpu_frustum_cull(centers: np.ndarray, radii: np.ndarray,
                         view_proj: np.ndarray) -> np.ndarray:
        c = np.ascontiguousarray(centers, dtype=np.float64)
        r = np.ascontiguousarray(radii, dtype=np.float64)
        vp = np.ascontiguousarray(view_proj, dtype=np.float64)
        return _culling_cpu(c, r, vp)

except ImportError:
    # Pure Python fallback when Cython extension not built
    def extract_frustum_planes(view_proj: np.ndarray) -> np.ndarray:
        vp = view_proj.astype(np.float32)
        planes = np.zeros((6, 4), dtype=np.float32)
        planes[0] = vp[3] + vp[0]
        planes[1] = vp[3] - vp[0]
        planes[2] = vp[3] + vp[1]
        planes[3] = vp[3] - vp[1]
        planes[4] = vp[3] + vp[2]
        planes[5] = vp[3] - vp[2]
        norms = np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        planes /= norms
        return planes

    def cpu_frustum_cull(centers: np.ndarray, radii: np.ndarray,
                         view_proj: np.ndarray) -> np.ndarray:
        if len(centers) == 0:
            return np.zeros(0, dtype=np.intp)
        planes = extract_frustum_planes(view_proj)
        distances = planes[:, :3] @ centers.T + planes[:, 3, None]
        visible = np.all(distances > -radii[None, :], axis=0)
        return np.where(visible)[0]
