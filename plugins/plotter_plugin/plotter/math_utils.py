# MIT License
#
# Copyright (c) 2026 Zarrakun
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations
import math
from bisect import bisect_right
from typing import Callable, Dict, List, Optional, Tuple

_CY = False
_NICE_TICKS_MAX = 64


def derivative(xs: List[float], ys: List[float]) -> Tuple[List[float], List[float]]:
    n = len(xs)
    if n < 2:
        return [], []
    dydx = []
    for i in range(n):
        if i == 0:
            dx = xs[1] - xs[0]
            dy = ys[1] - ys[0]
        elif i == n - 1:
            dx = xs[-1] - xs[-2]
            dy = ys[-1] - ys[-2]
        else:
            dx = xs[i + 1] - xs[i - 1]
            dy = ys[i + 1] - ys[i - 1]
        dydx.append(dy / dx if abs(dx) > 1e-15 else 0.0)
    return list(xs), dydx


def second_derivative(xs: List[float], ys: List[float]) -> Tuple[List[float], List[float]]:
    n = len(xs)
    if n < 3:
        return [], []
    out = []
    for i in range(1, n - 1):
        dx1 = xs[i] - xs[i - 1]
        dx2 = xs[i + 1] - xs[i]
        if abs(dx1) < 1e-15 or abs(dx2) < 1e-15:
            out.append(0.0)
        else:
            out.append(2 * ((ys[i + 1] - ys[i]) / dx2 - (ys[i] - ys[i - 1]) / dx1) / (dx1 + dx2))
    return xs[1:-1], out


def cumulative_integral(xs: List[float], ys: List[float]) -> Tuple[List[float], List[float]]:
    if len(xs) < 2:
        return list(xs), [0.0] * len(xs)
    result = [0.0]
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        result.append(result[-1] + dx * (ys[i - 1] + ys[i]) / 2.0)
    return list(xs), result


def fft_spectrum(xs: List[float], ys: List[float]) -> Tuple[List[float], List[float]]:
    n = len(ys)
    if n < 4:
        return [], []
    dt = (xs[-1] - xs[0]) / (n - 1) if n > 1 else 1.0
    if dt <= 0:
        return [], []
    re = list(ys)
    im = [0.0] * n
    for k in range(n):
        re[k] = sum(ys[j] * math.cos(2 * math.pi * k * j / n) for j in range(n))
        im[k] = -sum(ys[j] * math.sin(2 * math.pi * k * j / n) for j in range(n))
    half = n // 2 + 1
    freqs = [k / (n * dt) for k in range(half)]
    amps = [math.sqrt(re[k] ** 2 + im[k] ** 2) / n * (1 if k == 0 else 2) for k in range(half)]
    return freqs, amps


def fft_spectrum_numpy(xs: List[float], ys: List[float]) -> Tuple[List[float], List[float]]:
    try:
        import numpy as np
        n = len(ys)
        if n < 4:
            return [], []
        dt = (xs[-1] - xs[0]) / (n - 1) if n > 1 else 1.0
        if dt <= 0:
            return [], []
        fft = np.fft.rfft(ys)
        freqs = np.fft.rfftfreq(n, dt).tolist()
        amps = (np.abs(fft) / n * np.array([1] + [2] * (len(fft) - 1))).tolist()
        return freqs, amps
    except ImportError:
        return fft_spectrum(xs, ys)


def histogram(data: List[float], bins: int = 20) -> Tuple[List[float], List[float]]:
    if not data or bins < 1:
        return [], []
    lo, hi = min(data), max(data)
    if lo == hi:
        return [lo], [float(len(data))]
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in data:
        idx = min(int((v - lo) / width), bins - 1)
        counts[idx] += 1
    centers = [lo + (i + 0.5) * width for i in range(bins)]
    return centers, [float(c) for c in counts]


def autocorrelation(ys: List[float], max_lag: Optional[int] = None) -> Tuple[List[int], List[float]]:
    n = len(ys)
    if n < 2:
        return [], []
    lim = min(max_lag or n - 1, n - 1)
    mean = sum(ys) / n
    var = sum((v - mean) ** 2 for v in ys)
    if var < 1e-15:
        return list(range(lim + 1)), [1.0] + [0.0] * lim
    lags = list(range(lim + 1))
    acf = []
    for lag in lags:
        c = sum((ys[i] - mean) * (ys[i + lag] - mean) for i in range(n - lag))
        acf.append(c / var)
    return lags, acf


def normalize(ys: List[float], method: str = "minmax") -> List[float]:
    if not ys:
        return []
    if method == "minmax":
        lo, hi = min(ys), max(ys)
        rng = hi - lo
        return [(v - lo) / rng if rng > 1e-15 else 0.0 for v in ys]
    if method == "zscore":
        n = len(ys)
        mean = sum(ys) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in ys) / n)
        return [(v - mean) / std if std > 1e-15 else 0.0 for v in ys]
    if method == "maxabs":
        ma = max(abs(v) for v in ys)
        return [v / ma if ma > 1e-15 else 0.0 for v in ys]
    return list(ys)


def peak_find(xs: List[float], ys: List[float],
              min_prominence: float = 0.0) -> Tuple[List[float], List[float]]:
    n = len(ys)
    if n < 3:
        return [], []
    y_range = max(ys) - min(ys) if n else 1.0
    px, py = [], []
    for i in range(1, n - 1):
        if ys[i] > ys[i - 1] and ys[i] > ys[i + 1]:
            left_min = min(ys[:i + 1])
            right_min = min(ys[i:])
            prominence = ys[i] - max(left_min, right_min)
            if prominence >= min_prominence * y_range:
                px.append(xs[i])
                py.append(ys[i])
    return px, py


def weighted_mean(ys: List[float], weights: List[float]) -> float:
    sw = sum(weights)
    if sw < 1e-15:
        return float("nan")
    return sum(y * w for y, w in zip(ys, weights)) / sw


def moving_std(ys: List[float], window: int) -> List[float]:
    n = len(ys)
    out = []
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, lo + window)
        seg = ys[lo:hi]
        m = sum(seg) / len(seg)
        out.append(math.sqrt(sum((v - m) ** 2 for v in seg) / len(seg)))
    return out


def _gaussian_fit(x_pts: List[float], y_pts: List[float], x_eval: List[float]) -> Optional[List[float]]:
    if len(x_pts) < 3:
        return None
    C = min(y_pts)
    ys_c = [max(v - C, 1e-15) for v in y_pts]
    total = sum(ys_c)
    if total < 1e-15:
        return None
    mu = sum(x * y for x, y in zip(x_pts, ys_c)) / total
    sigma2 = sum((x - mu) ** 2 * y for x, y in zip(x_pts, ys_c)) / total
    A = max(ys_c)
    return [A * math.exp(-((xi - mu) ** 2) / (2 * sigma2)) + C for xi in x_eval]


def _gaussian_formula(x_pts: List[float], y_pts: List[float]) -> str:
    C = min(y_pts)
    ys_c = [max(v - C, 1e-15) for v in y_pts]
    total = sum(ys_c)
    if total < 1e-15:
        return ""
    mu = sum(x * y for x, y in zip(x_pts, ys_c)) / total
    sigma2 = sum((x - mu) ** 2 * y for x, y in zip(x_pts, ys_c)) / total
    sigma = math.sqrt(max(sigma2, 1e-15))
    A = max(ys_c)
    return f"y = {fmt(A)}·exp(-(x-{fmt(mu)})²/(2·{fmt(sigma)}²)) + {fmt(C)}"


def _logistic_fit(x_pts: List[float], y_pts: List[float], x_eval: List[float]) -> Optional[List[float]]:
    if len(x_pts) < 4:
        return None
    y_min, y_max = min(y_pts), max(y_pts)
    L = y_max - y_min
    C = y_min
    if L < 1e-15:
        return None
    x0 = x_pts[len(x_pts) // 2]
    span = x_pts[-1] - x_pts[0]
    k = 4.0 / span if span > 0 else 1.0
    for _ in range(60):
        grad_k = sum(
            -2 * (y_pts[i] - C - L / (1 + math.exp(-k * (x_pts[i] - x0)))) *
            (L * (x_pts[i] - x0) * math.exp(-k * (x_pts[i] - x0)) /
             (1 + math.exp(-k * (x_pts[i] - x0))) ** 2)
            for i in range(len(x_pts))
        )
        grad_x0 = sum(
            -2 * (y_pts[i] - C - L / (1 + math.exp(-k * (x_pts[i] - x0)))) *
            (-L * k * math.exp(-k * (x_pts[i] - x0)) /
             (1 + math.exp(-k * (x_pts[i] - x0))) ** 2)
            for i in range(len(x_pts))
        )
        step = 1e-4 / (abs(grad_k) + abs(grad_x0) + 1e-15)
        k -= step * grad_k
        x0 -= step * grad_x0
    return [C + L / (1 + math.exp(-k * (xi - x0))) for xi in x_eval]


def _logistic_formula(x_pts: List[float], y_pts: List[float]) -> str:
    y_min, y_max = min(y_pts), max(y_pts)
    L = y_max - y_min
    x0 = x_pts[len(x_pts) // 2]
    span = x_pts[-1] - x_pts[0]
    k = 4.0 / span if span > 0 else 1.0
    return f"y = {fmt(L)}/(1+exp(-{fmt(k)}·(x-{fmt(x0)}))) + {fmt(y_min)}"


def _double_exp_fit(x_pts: List[float], y_pts: List[float], x_eval: List[float]) -> Optional[List[float]]:
    if len(x_pts) < 6:
        return None
    mid = len(x_pts) // 2
    r1 = _exp_fit(x_pts[:mid], y_pts[:mid], x_eval)
    r2 = _exp_fit(x_pts[mid:], y_pts[mid:], x_eval)
    if r1 is None or r2 is None:
        return None
    return [0.5 * (v1 + v2) for v1, v2 in zip(r1, r2)]


def linspace(start: float, stop: float, n: int) -> List[float]:
    if n < 2:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + step * i for i in range(n)]


def _sort_unique(x: List[float], y: List[float]) -> Tuple[List[float], List[float]]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    x2, y2 = [x[i] for i in order], [y[i] for i in order]
    xs: List[float] = []
    ys: List[float] = []
    for xi, yi in zip(x2, y2):
        if not xs or xi != xs[-1]:
            xs.append(xi)
            ys.append(yi)
    return xs, ys


def _gauss_solve(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(b)
    a = [row[:] for row in a]
    b = b[:]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        b[col], b[pivot] = b[pivot], b[col]
        if abs(a[col][col]) < 1e-15:
            continue
        f = a[col][col]
        a[col] = [v / f for v in a[col]]
        b[col] /= f
        for row in range(n):
            if row == col:
                continue
            fac = a[row][col]
            a[row] = [a[row][j] - fac * a[col][j] for j in range(n)]
            b[row] -= fac * b[col]
    return b


def _polyfit(x: List[float], y: List[float], deg: int) -> List[float]:
    d = deg + 1
    vt_v = [[0.0] * d for _ in range(d)]
    vt_y = [0.0] * d
    for xi, yi in zip(x, y):
        row = [xi ** (deg - j) for j in range(d)]
        for r in range(d):
            for c in range(d):
                vt_v[r][c] += row[r] * row[c]
            vt_y[r] += row[r] * yi
    return _gauss_solve(vt_v, vt_y)


def _polyval(coeffs: List[float], x: float) -> float:
    result = 0.0
    for c in coeffs:
        result = result * x + c
    return result


def _pchip_eval(x_pts: List[float], y_pts: List[float], x_eval: List[float]) -> List[float]:
    n = len(x_pts)
    h = [x_pts[i + 1] - x_pts[i] for i in range(n - 1)]
    s = [(y_pts[i + 1] - y_pts[i]) / h[i] for i in range(n - 1)]
    d = [0.0] * n
    d[0] = s[0]
    d[-1] = s[-1]
    for k in range(1, n - 1):
        if s[k - 1] * s[k] <= 0.0:
            d[k] = 0.0
        else:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / s[k - 1] + w2 / s[k])
    result = []
    for xi in x_eval:
        idx = max(0, min(bisect_right(x_pts, xi) - 1, n - 2))
        dx = xi - x_pts[idx]
        hk = h[idx]
        t = dx / hk
        t2 = t * t
        t3 = t2 * t
        result.append(
            (2 * t3 - 3 * t2 + 1) * y_pts[idx]
            + (t3 - 2 * t2 + t) * hk * d[idx]
            + (-2 * t3 + 3 * t2) * y_pts[idx + 1]
            + (t3 - t2) * hk * d[idx + 1]
        )
    return result


def _cubic_spline_eval(x_pts: List[float], y_pts: List[float], x_eval: List[float]) -> List[float]:
    n = len(x_pts)
    h = [float(x_pts[i + 1] - x_pts[i]) for i in range(n - 1)]
    rhs = [0.0] * n
    for i in range(1, n - 1):
        rhs[i] = 3.0 * ((y_pts[i + 1] - y_pts[i]) / h[i] - (y_pts[i] - y_pts[i - 1]) / h[i - 1])
    diag = [2.0] * n
    lo = [0.0] * n
    up = [0.0] * n
    for i in range(1, n - 1):
        lo[i] = h[i - 1]
        up[i] = h[i]
    diag[0] = 1.0; up[0] = 0.0; rhs[0] = 0.0
    diag[-1] = 1.0; lo[-1] = 0.0; rhs[-1] = 0.0
    cu = [0.0] * n
    cr = [0.0] * n
    cu[0] = up[0] / diag[0] if diag[0] else 0.0
    cr[0] = rhs[0] / diag[0] if diag[0] else 0.0
    for i in range(1, n):
        den = diag[i] - lo[i] * cu[i - 1]
        cu[i] = (up[i] / den) if (i < n - 1 and den) else 0.0
        cr[i] = ((rhs[i] - lo[i] * cr[i - 1]) / den) if den else 0.0
    m = [0.0] * n
    m[-1] = cr[-1]
    for i in range(n - 2, -1, -1):
        m[i] = cr[i] - cu[i] * m[i + 1]
    a = list(y_pts[:-1])
    b_c = [(y_pts[i + 1] - y_pts[i]) / h[i] - h[i] * (2 * m[i] + m[i + 1]) / 3.0 for i in range(n - 1)]
    c_c = m[:-1]
    d_c = [(m[i + 1] - m[i]) / (3.0 * h[i]) for i in range(n - 1)]
    result = []
    for xi in x_eval:
        idx = max(0, min(bisect_right(x_pts, xi) - 1, n - 2))
        dx = xi - x_pts[idx]
        result.append(a[idx] + b_c[idx] * dx + c_c[idx] * dx ** 2 + d_c[idx] * dx ** 3)
    return result


def _exp_fit(x_pts: List[float], y_pts: List[float], x_eval: List[float]) -> Optional[List[float]]:
    pairs = [(x, y) for x, y in zip(x_pts, y_pts) if y > 0]
    if len(pairs) < 2:
        return None
    xs2 = [p[0] for p in pairs]
    lys = [math.log(p[1]) for p in pairs]
    n = len(xs2)
    sx = sum(xs2)
    sy = sum(lys)
    sxx = sum(xi * xi for xi in xs2)
    sxy = sum(xi * li for xi, li in zip(xs2, lys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-15:
        return None
    b = (n * sxy - sx * sy) / denom
    a = math.exp((sy - b * sx) / n)
    return [a * math.exp(b * xi) for xi in x_eval]


def _exp_formula(x_pts: List[float], y_pts: List[float]) -> str:
    pairs = [(x, y) for x, y in zip(x_pts, y_pts) if y > 0]
    if len(pairs) < 2:
        return ""
    xs2 = [p[0] for p in pairs]
    lys = [math.log(p[1]) for p in pairs]
    n = len(xs2)
    sx = sum(xs2)
    sy = sum(lys)
    sxx = sum(xi * xi for xi in xs2)
    sxy = sum(xi * li for xi, li in zip(xs2, lys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-15:
        return ""
    b = (n * sxy - sx * sy) / denom
    a = math.exp((sy - b * sx) / n)
    sign = "+" if b >= 0 else "-"
    return f"y = {fmt(a)}·e^({sign}{fmt(abs(b))}·x)"


def _power_fit(x_pts: List[float], y_pts: List[float], x_eval: List[float]) -> Optional[List[float]]:
    pairs = [(x, y) for x, y in zip(x_pts, y_pts) if x > 0 and y > 0]
    if len(pairs) < 2:
        return None
    lxs = [math.log(p[0]) for p in pairs]
    lys = [math.log(p[1]) for p in pairs]
    n = len(lxs)
    sx = sum(lxs)
    sy = sum(lys)
    sxx = sum(xi * xi for xi in lxs)
    sxy = sum(xi * li for xi, li in zip(lxs, lys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-15:
        return None
    b = (n * sxy - sx * sy) / denom
    a = math.exp((sy - b * sx) / n)
    result = []
    for xi in x_eval:
        if xi > 0:
            result.append(a * (xi ** b))
        else:
            result.append(None)
    return result


def _power_formula(x_pts: List[float], y_pts: List[float]) -> str:
    pairs = [(x, y) for x, y in zip(x_pts, y_pts) if x > 0 and y > 0]
    if len(pairs) < 2:
        return ""
    lxs = [math.log(p[0]) for p in pairs]
    lys = [math.log(p[1]) for p in pairs]
    n = len(lxs)
    sx = sum(lxs)
    sy = sum(lys)
    sxx = sum(xi * xi for xi in lxs)
    sxy = sum(xi * li for xi, li in zip(lxs, lys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-15:
        return ""
    b = (n * sxy - sx * sy) / denom
    a = math.exp((sy - b * sx) / n)
    return f"y = {fmt(a)}·x^{fmt(b)}"


def _log_fit(x_pts: List[float], y_pts: List[float], x_eval: List[float]) -> Optional[List[float]]:
    pairs = [(x, y) for x, y in zip(x_pts, y_pts) if x > 0]
    if len(pairs) < 2:
        return None
    lxs = [math.log(p[0]) for p in pairs]
    ys2 = [p[1] for p in pairs]
    n = len(lxs)
    slx = sum(lxs)
    sy = sum(ys2)
    slxlx = sum(lx * lx for lx in lxs)
    slxy = sum(lx * yi for lx, yi in zip(lxs, ys2))
    denom = n * slxlx - slx * slx
    if abs(denom) < 1e-15:
        return None
    b = (n * slxy - slx * sy) / denom
    a = (sy - b * slx) / n
    result = []
    for xi in x_eval:
        if xi > 0:
            result.append(a + b * math.log(xi))
        else:
            result.append(None)
    return result


def _log_formula(x_pts: List[float], y_pts: List[float]) -> str:
    pairs = [(x, y) for x, y in zip(x_pts, y_pts) if x > 0]
    if len(pairs) < 2:
        return ""
    lxs = [math.log(p[0]) for p in pairs]
    ys2 = [p[1] for p in pairs]
    n = len(lxs)
    slx = sum(lxs)
    sy = sum(ys2)
    slxlx = sum(lx * lx for lx in lxs)
    slxy = sum(lx * yi for lx, yi in zip(lxs, ys2))
    denom = n * slxlx - slx * slx
    if abs(denom) < 1e-15:
        return ""
    b = (n * slxy - slx * sy) / denom
    a = (sy - b * slx) / n
    sign = "+" if b >= 0 else "-"
    return f"y = {fmt(a)} {sign} {fmt(abs(b))}·ln(x)"


def _sinusoidal_fit(x_pts: List[float], y_pts: List[float], x_eval: List[float]) -> Optional[List[float]]:
    n = len(x_pts)
    if n < 4:
        return None
    try:
        ys = y_pts
        y_mean = sum(ys) / n
        y_centered = [v - y_mean for v in ys]
        x_range = x_pts[-1] - x_pts[0]
        if x_range <= 0:
            return None
        T_guess = x_range / 2.0
        omega = 2 * math.pi / T_guess
        sc = sum(y_centered[i] * math.sin(omega * x_pts[i]) for i in range(n))
        cc = sum(y_centered[i] * math.cos(omega * x_pts[i]) for i in range(n))
        A = 2 * math.sqrt(sc ** 2 + cc ** 2) / n
        phi = math.atan2(sc, cc)
        return [A * math.sin(omega * xi + phi) + y_mean for xi in x_eval]
    except Exception:
        return None


def _moving_average_fit(x_pts: List[float], y_pts: List[float], x_eval: List[float]) -> List[float]:
    n = len(y_pts)
    win = max(3, n // 5)
    half = win // 2
    smoothed_x = []
    smoothed_y = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed_x.append(x_pts[i])
        smoothed_y.append(sum(y_pts[lo:hi]) / (hi - lo))
    return _pchip_eval(smoothed_x, smoothed_y, x_eval)


FitFn = Callable[[List[float], List[float], List[float]], List[float]]


class FitMode:
    def __init__(self, key: str, label: str, fn: FitFn, min_points: int = 2,
                 formula_fn: Optional[Callable[[List[float], List[float]], str]] = None):
        self.key = key
        self.label = label
        self.fn = fn
        self.min_points = min_points
        self.formula_fn = formula_fn

    def evaluate(self, x_pts: List[float], y_pts: List[float], x_eval: List[float]) -> Optional[List[float]]:
        x_pts, y_pts = _sort_unique(x_pts, y_pts)
        if len(x_pts) < self.min_points:
            return None
        return self.fn(x_pts, y_pts, x_eval)

    def formula(self, x_pts: List[float], y_pts: List[float]) -> str:
        if self.formula_fn is None:
            return ""
        x_pts, y_pts = _sort_unique(x_pts, y_pts)
        if len(x_pts) < self.min_points:
            return ""
        try:
            return self.formula_fn(x_pts, y_pts)
        except Exception:
            return ""


def _fit_linear_origin(x_pts, y_pts, x_eval):
    denom = sum(xi * xi for xi in x_pts)
    k = sum(xi * yi for xi, yi in zip(x_pts, y_pts)) / denom if denom else 0.0
    return [k * xi for xi in x_eval]


def _formula_linear_origin(x_pts, y_pts):
    denom = sum(xi * xi for xi in x_pts)
    k = sum(xi * yi for xi, yi in zip(x_pts, y_pts)) / denom if denom else 0.0
    return f"y = {fmt(k)}·x"


def _fit_linear(x_pts, y_pts, x_eval):
    x_min = x_pts[0]
    x_range = float(x_pts[-1] - x_min) or 1.0
    xn = [(xi - x_min) / x_range for xi in x_pts]
    c = _polyfit(xn, y_pts, 1)
    return [_polyval(c, (xi - x_min) / x_range) for xi in x_eval]


def _formula_linear(x_pts, y_pts):
    x_min = x_pts[0]
    x_range = float(x_pts[-1] - x_min) or 1.0
    xn = [(xi - x_min) / x_range for xi in x_pts]
    c = _polyfit(xn, y_pts, 1)
    a_real = c[0] / x_range
    b_real = c[1] - c[0] * x_min / x_range
    sign = "+" if b_real >= 0 else "-"
    return f"y = {fmt(a_real)}·x {sign} {fmt(abs(b_real))}"


def _make_poly_fit(deg: int) -> FitFn:
    def _fit(x_pts, y_pts, x_eval):
        actual_deg = min(deg, len(x_pts) - 1)
        x_min = x_pts[0]
        x_range = float(x_pts[-1] - x_min) or 1.0
        xn = [(xi - x_min) / x_range for xi in x_pts]
        c = _polyfit(xn, y_pts, actual_deg)
        return [_polyval(c, (xi - x_min) / x_range) for xi in x_eval]
    return _fit


def _make_poly_formula(deg: int):
    def _formula(x_pts, y_pts):
        actual_deg = min(deg, len(x_pts) - 1)
        x_min = x_pts[0]
        x_range = float(x_pts[-1] - x_min) or 1.0
        xn = [(xi - x_min) / x_range for xi in x_pts]
        c = _polyfit(xn, y_pts, actual_deg)
        parts = []
        for i, coef in enumerate(c):
            power = actual_deg - i
            if abs(coef) < 1e-10:
                continue
            if power == 0:
                parts.append(fmt(coef))
            elif power == 1:
                parts.append(f"{fmt(coef)}·x")
            else:
                parts.append(f"{fmt(coef)}·x^{power}")
        return "y = " + (" + ".join(parts) if parts else "0")
    return _formula


def _fit_pchip(x_pts, y_pts, x_eval):
    return _pchip_eval(x_pts, y_pts, x_eval)


def _fit_cubic_spline(x_pts, y_pts, x_eval):
    if len(x_pts) == 2:
        return _pchip_eval(x_pts, y_pts, x_eval)
    return _cubic_spline_eval(x_pts, y_pts, x_eval)


def _fit_exp(x_pts, y_pts, x_eval):
    result = _exp_fit(x_pts, y_pts, x_eval)
    if result is None:
        return [0.0] * len(x_eval)
    return result


def _fit_power(x_pts, y_pts, x_eval):
    result = _power_fit(x_pts, y_pts, x_eval)
    if result is None:
        return [None] * len(x_eval)
    return result


def _fit_log(x_pts, y_pts, x_eval):
    result = _log_fit(x_pts, y_pts, x_eval)
    if result is None:
        return [None] * len(x_eval)
    return result


def trapezoid_integral(xs: List[float], ys: List[float],
                       x_lo: Optional[float] = None, x_hi: Optional[float] = None) -> float:
    if len(xs) < 2:
        return 0.0
    pairs = sorted(zip(xs, ys), key=lambda p: p[0])
    if x_lo is not None:
        pairs = [(x, y) for x, y in pairs if x >= x_lo]
    if x_hi is not None:
        pairs = [(x, y) for x, y in pairs if x <= x_hi]
    if len(pairs) < 2:
        return 0.0
    total = 0.0
    for i in range(len(pairs) - 1):
        x0, y0 = pairs[i]
        x1, y1 = pairs[i + 1]
        total += (x1 - x0) * (y0 + y1) / 2.0
    return total


def rms(ys: List[float]) -> float:
    if not ys:
        return 0.0
    return math.sqrt(sum(v * v for v in ys) / len(ys))


def percentile(data: List[float], p: float) -> float:
    n = len(data)
    if n == 0:
        return float("nan")
    if n == 1:
        return data[0]
    sorted_data = sorted(data)
    pos = (p / 100.0) * (n - 1)
    lo = int(pos)
    hi = lo + 1
    if hi >= n:
        return sorted_data[-1]
    frac = pos - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])


def correlation(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(xs, ys))
    dx2 = sum((xi - mx) ** 2 for xi in xs)
    dy2 = sum((yi - my) ** 2 for yi in ys)
    denom = math.sqrt(dx2 * dy2)
    return num / denom if denom > 1e-15 else None


def r_squared(xs: List[float], ys: List[float], fit_ys: List[float]) -> Optional[float]:
    n = len(ys)
    if n < 2 or len(fit_ys) != n:
        return None
    mean_y = sum(ys) / n
    ss_res = sum((yi - fi) ** 2 for yi, fi in zip(ys, fit_ys))
    ss_tot = sum((yi - mean_y) ** 2 for yi in ys)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else None


def _fit_gaussian(x_pts, y_pts, x_eval):
    r = _gaussian_fit(x_pts, y_pts, x_eval)
    return r if r is not None else [0.0] * len(x_eval)


def _fit_logistic(x_pts, y_pts, x_eval):
    r = _logistic_fit(x_pts, y_pts, x_eval)
    return r if r is not None else [0.0] * len(x_eval)


def _fit_double_exp(x_pts, y_pts, x_eval):
    r = _double_exp_fit(x_pts, y_pts, x_eval)
    return r if r is not None else [0.0] * len(x_eval)


_BUILTIN_MODES: List[FitMode] = [
    FitMode("linear_origin", "Linear (origin)",  _fit_linear_origin,     1,  _formula_linear_origin),
    FitMode("linear",        "Linear",            _fit_linear,             2,  _formula_linear),
    FitMode("poly2",         "Polynomial 2°",     _make_poly_fit(2),       2,  _make_poly_formula(2)),
    FitMode("poly3",         "Polynomial 3°",     _make_poly_fit(3),       2,  _make_poly_formula(3)),
    FitMode("poly4",         "Polynomial 4°",     _make_poly_fit(4),       2,  _make_poly_formula(4)),
    FitMode("poly5",         "Polynomial 5°",     _make_poly_fit(5),       2,  _make_poly_formula(5)),
    FitMode("pchip",         "PCHIP",             _fit_pchip,              2),
    FitMode("spline",        "Cubic Spline",      _fit_cubic_spline,       2),
    FitMode("exp",           "Exponential",       _fit_exp,                2,  _exp_formula),
    FitMode("power",         "Power",             _fit_power,              2,  _power_formula),
    FitMode("log",           "Logarithmic",       _fit_log,                2,  _log_formula),
    FitMode("sine",          "Sinusoidal",        _sinusoidal_fit,         4),
    FitMode("moving_avg",    "Moving Average",    _moving_average_fit,     3),
    FitMode("gaussian",      "Gaussian",          _fit_gaussian,           3,  _gaussian_formula),
    FitMode("logistic",      "Logistic",          _fit_logistic,           4,  _logistic_formula),
    FitMode("double_exp",    "Double Exponential",_fit_double_exp,         6),
]

_REGISTRY: Dict[str, FitMode] = {m.key: m for m in _BUILTIN_MODES}


def register_fit_mode(mode: FitMode):
    _REGISTRY[mode.key] = mode


def get_fit_modes() -> List[FitMode]:
    return list(_REGISTRY.values())


def get_fit_mode(key: str) -> Optional[FitMode]:
    return _REGISTRY.get(key)


def nice_ticks(lo: float, hi: float, n: int = 7) -> List[float]:
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo] if math.isfinite(lo) else []
    span = hi - lo
    raw = span / max(n - 1, 1)
    if raw <= 0 or not math.isfinite(raw):
        return [lo]
    try:
        mag = 10.0 ** math.floor(math.log10(raw))
    except (ValueError, OverflowError):
        return [lo]
    if not math.isfinite(mag) or mag <= 0:
        return [lo]
    step = mag
    for s in (mag, mag * 2, mag * 2.5, mag * 5, mag * 10):
        if span / s <= n + 1:
            step = s
            break
    if not math.isfinite(step) or step <= 0:
        return [lo]
    try:
        start = math.floor(lo / step) * step
    except (OverflowError, ValueError):
        return [lo]
    ticks: List[float] = []
    v = start
    while v <= hi + step * 0.001 and len(ticks) < _NICE_TICKS_MAX:
        if v >= lo - step * 0.001:
            ticks.append(round(v, 10))
        nv = round(v + step, 10)
        if nv <= v:
            break
        v = nv
    return ticks


def nice_log_ticks(lo: float, hi: float) -> List[float]:
    if lo <= 0 or hi <= 0 or lo >= hi:
        return []
    lo_e = math.floor(math.log10(lo))
    hi_e = math.ceil(math.log10(hi))
    ticks = []
    for e in range(int(lo_e), int(hi_e) + 1):
        for m in (1, 2, 5):
            v = m * 10.0 ** e
            if lo <= v <= hi:
                ticks.append(v)
    return ticks if ticks else [lo, hi]


def to_log(v: float) -> float:
    return math.log10(v) if v > 0 else float('-inf')


def decimated(xs: List[float], ys: List[float], max_pts: int) -> Tuple[List[float], List[float]]:
    n = len(xs)
    if n <= max_pts:
        return xs, ys
    step = n / max_pts
    indices = [int(i * step) for i in range(max_pts)]
    indices[-1] = n - 1
    return [xs[i] for i in indices], [ys[i] for i in indices]


def fmt(v: float) -> str:
    if not math.isfinite(v):
        return str(v)
    if v == 0:
        return "0"
    abs_v = abs(v)
    if abs_v >= 1e15:
        return f"{v:.6g}"
    if abs_v >= 1000:
        return f"{v:.0f}"
    if abs_v >= 100:
        return f"{v:.0f}"
    if abs_v >= 10:
        return f"{v:.1f}"
    if abs_v >= 1:
        return f"{v:.4g}"
    for digits in range(1, 10):
        s = f"{v:.{digits}f}".rstrip("0").rstrip(".")
        if float(s) != 0.0:
            return s
    return f"{v:.4g}"