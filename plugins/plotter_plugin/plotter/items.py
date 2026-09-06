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
from typing import Callable, List, Optional, Tuple, Union, TYPE_CHECKING
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QPen
if TYPE_CHECKING:
    from .chart_widget import ChartWidget

try:
    import numpy as _np_items
    _NP_ITEMS = True
except ImportError:
    _np_items = None
    _NP_ITEMS = False


class _InfLine:
    def __init__(self, chart: "ChartWidget", horizontal: bool, value: float, pen: QPen):
        self._chart = chart
        self.horizontal = horizontal
        self.value = value
        self.pen = pen
        self.visible = False
        self.draggable = True
        self.label: str = ""

    def setValue(self, v: float):
        self.value = v
        self._chart.update()

    def setVisible(self, v: bool):
        self.visible = v
        self._chart.update()

    def setDraggable(self, v: bool):
        self.draggable = v

    def setLabel(self, label: str):
        self.label = label
        self._chart.update()


class _LineItem:
    def __init__(self, chart: "ChartWidget", pen: QPen, label: str = ""):
        self._chart = chart
        self.xs: List[float] = []
        self.ys: List[float] = []
        self.pen = pen
        self.label = label
        self.visible = True
        self.raw_visible = True
        self.fill_under = False
        self.fill_alpha = 40
        self.step_mode = False
        self.point_symbol: str = "none"
        self.point_size: int = 6

    def setData(self, xs=None, ys=None):
        self.xs = list(xs) if xs is not None else []
        self.ys = list(ys) if ys is not None else []
        self._chart._schedule_autofit()

    def appendPoint(self, x: float, y: float):
        self.xs.append(x)
        self.ys.append(y)
        self._chart._schedule_autofit()

    def setVisible(self, v: bool):
        self.visible = v
        self._chart.update()

    def setRawVisible(self, v: bool):
        self.raw_visible = v
        self._chart.update()

    def setLabel(self, label: str):
        self.label = label
        self._chart.update()

    def setFillUnder(self, enabled: bool, alpha: int = 40):
        self.fill_under = enabled
        self.fill_alpha = alpha
        self._chart.update()

    def setStepMode(self, enabled: bool):
        self.step_mode = enabled
        self._chart.update()

    def setPointSymbol(self, symbol: str, size: int = 6):
        self.point_symbol = symbol
        self.point_size = size
        self._chart.update()

    def setPen(self, pen: QPen):
        self.pen = pen
        self._chart.update()

    @property
    def x_range(self) -> Optional[Tuple[float, float]]:
        if not self.xs:
            return None
        return (min(self.xs), max(self.xs))

    @property
    def y_range(self) -> Optional[Tuple[float, float]]:
        if not self.ys:
            return None
        return (min(self.ys), max(self.ys))

    def clear(self):
        self.xs = []
        self.ys = []
        self._chart._schedule_autofit()


class _ScatterItem(QObject):
    point_clicked = pyqtSignal(float, float, int)

    def __init__(self, chart: "ChartWidget", size: int, color: QColor, label: str = ""):
        super().__init__()
        self._chart = chart
        self.xs: List[float] = []
        self.ys: List[float] = []
        self.size = size
        self.color = color
        self.label = label
        self.visible = True
        self.raw_visible = True
        self.error_xs: Optional[List[float]] = None
        self.error_ys: Optional[List[float]] = None
        self.annotations: Optional[List[str]] = None
        self.selected_idx: Optional[int] = None
        self.symbol: str = "circle"

    def setData(self, x=None, y=None, **_):
        self.xs = list(x) if x is not None else []
        self.ys = list(y) if y is not None else []
        self.selected_idx = None
        self._chart._schedule_autofit()

    def appendPoint(self, x: float, y: float):
        self.xs.append(x)
        self.ys.append(y)
        self._chart._schedule_autofit()

    def setVisible(self, v: bool):
        self.visible = v
        self._chart.update()

    def setRawVisible(self, v: bool):
        self.raw_visible = v
        self._chart.update()

    def setLabel(self, label: str):
        self.label = label
        self._chart.update()

    def setErrorBars(self, error_x: Optional[List[float]] = None,
                     error_y: Optional[List[float]] = None):
        self.error_xs = list(error_x) if error_x is not None else None
        self.error_ys = list(error_y) if error_y is not None else None
        self._chart.update()

    def setAnnotations(self, labels: Optional[List[str]]):
        self.annotations = list(labels) if labels is not None else None
        self._chart.update()

    def selectPoint(self, idx: Optional[int]):
        self.selected_idx = idx
        self._chart.update()

    def setSymbol(self, symbol: str):
        self.symbol = symbol
        self._chart.update()

    def setColor(self, color: QColor):
        self.color = color
        self._chart.update()

    def setSize(self, size: int):
        self.size = size
        self._chart.update()

    def clear(self):
        self.xs = []
        self.ys = []
        self.selected_idx = None
        self._chart._schedule_autofit()

    @property
    def x_range(self) -> Optional[Tuple[float, float]]:
        if not self.xs:
            return None
        return (min(self.xs), max(self.xs))

    @property
    def y_range(self) -> Optional[Tuple[float, float]]:
        if not self.ys:
            return None
        return (min(self.ys), max(self.ys))


def _screen_decimate(xs, ys, x_lo: float, x_hi: float,
                     pixel_width: int, pixel_height: int) -> tuple:
    try:
        import numpy as _np
    except ImportError:
        return xs, ys
    n = len(xs)
    if n <= pixel_width * 2:
        return xs, ys
    xs_arr = _np.asarray(xs, dtype=_np.float64)
    none_mask = _np.array([v is None for v in ys], dtype=bool)
    ys_arr = _np.array([0.0 if v is None else v for v in ys], dtype=_np.float64)
    x_range = x_hi - x_lo
    y_fin = ys_arr[~none_mask]
    y_range = float(_np.ptp(y_fin)) if len(y_fin) >= 2 else 1.0
    if x_range <= 0 or y_range <= 0:
        return xs, ys
    px = (xs_arr - x_lo) / x_range * pixel_width
    py = ys_arr / y_range * pixel_height
    dpx = _np.diff(px)
    dpy = _np.diff(py)
    dist_sq = dpx * dpx + dpy * dpy
    none_transition = none_mask[:-1] | none_mask[1:]
    keep_mask = _np.ones(n, dtype=bool)
    redundant = (dist_sq < 0.25) & ~none_transition
    keep_mask[1:][redundant] = False
    keep_mask[none_mask] = True
    idx = _np.where(keep_mask)[0]
    xs_out = xs_arr[idx].tolist()
    ys_out: list = [ys[int(i)] for i in idx]
    return xs_out, ys_out


class _FunctionItem:
    def __init__(self, chart: "ChartWidget", fn: Callable[[List[float]], List[float]],
                 pen: QPen, label: str = "", resolution: float = 1.0):
        self._chart = chart
        self.fn = fn
        self.pen = pen
        self.label = label
        self.visible = True
        self.fill_under = False
        self.fill_alpha = 30
        self.resolution = resolution
        self._cached_xs: List[float] = []
        self._cached_ys: List[Optional[float]] = []
        self._cache_key: Optional[Tuple] = None
        self._adaptive: bool = False
        self._expr: str = ""
        self._extra: dict = {}

    @property
    def xs(self) -> List[float]:
        return self._cached_xs

    @property
    def ys(self) -> List[Optional[float]]:
        return self._cached_ys

    def evaluate(self, x_lo: float, x_hi: float, pixel_width: int,
                 pixel_height: int = 600) -> Tuple[List[float], List[Optional[float]]]:
        n_pts = max(4, int(pixel_width * min(self.resolution, 1.0)))
        key = (round(x_lo, 12), round(x_hi, 12), n_pts)
        if key == self._cache_key:
            return self._cached_xs, self._cached_ys
        from .math_utils import linspace as _linspace
        if _NP_ITEMS:
            xs_base = _np_items.linspace(x_lo, x_hi, n_pts, dtype=_np_items.float64).tolist()
        else:
            xs_base = _linspace(x_lo, x_hi, n_pts)
        if self._adaptive and self._expr:
            try:
                from .math_engine import sample_y_adaptive
                xs_ad, ys_ad = sample_y_adaptive(self._expr, xs_base, self._extra)
                self._cached_xs = xs_ad
                self._cached_ys = ys_ad
                self._cache_key = key
                return self._cached_xs, self._cached_ys
            except Exception:
                pass
        try:
            raw = self.fn(xs_base)
        except Exception:
            self._cached_xs = xs_base
            self._cached_ys = [None] * len(xs_base)
            self._cache_key = key
            return self._cached_xs, self._cached_ys
        if _NP_ITEMS:
            raw_arr = _np_items.array(
                [v if v is not None else _np_items.nan for v in raw], dtype=_np_items.float64
            )
            fin = _np_items.isfinite(raw_arr)
            ys: List[Optional[float]] = [
                float(raw_arr[i]) if fin[i] else None for i in range(len(raw_arr))
            ]
        else:
            _isfinite = math.isfinite
            ys = []
            for v in raw:
                if v is None:
                    ys.append(None)
                else:
                    fv = v if type(v) is float else float(v)
                    ys.append(fv if _isfinite(fv) else None)
        self._cached_xs = xs_base
        self._cached_ys = ys
        self._cache_key = key
        return self._cached_xs, self._cached_ys

    def invalidateCache(self):
        self._cache_key = None

    def setFunction(self, fn: Callable[[List[float]], List[float]]):
        self.fn = fn
        self.invalidateCache()
        self._chart.update()

    def setVisible(self, v: bool):
        self.visible = v
        self._chart.update()

    def setLabel(self, label: str):
        self.label = label
        self._chart.update()

    def setFillUnder(self, enabled: bool, alpha: int = 30):
        self.fill_under = enabled
        self.fill_alpha = alpha
        self._chart.update()

    def setPen(self, pen: QPen):
        self.pen = pen
        self._chart.update()

    def setResolution(self, resolution: float):
        self.resolution = max(0.1, resolution)
        self.invalidateCache()
        self._chart.update()


class _RulerItem:
    def __init__(self, chart: "ChartWidget", pen: QPen, handle_radius: int = 6):
        self._chart = chart
        self.pen = pen
        self.handle_radius = handle_radius
        self.visible = False
        self.draggable = True
        self.x0: float = 0.0
        self.y0: float = 0.0
        self.x1: float = 1.0
        self.y1: float = 1.0
        self.changed: Optional[Callable[[], None]] = None

    @property
    def distance(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)

    @property
    def dx(self) -> float:
        return self.x1 - self.x0

    @property
    def dy(self) -> float:
        return self.y1 - self.y0

    @property
    def angle_deg(self) -> float:
        return math.degrees(math.atan2(self.y1 - self.y0, self.x1 - self.x0))

    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def setPoints(self, x0: float, y0: float, x1: float, y1: float):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        if self.changed is not None:
            self.changed()
        self._chart.update()

    def setVisible(self, v: bool):
        self.visible = v
        self._chart.update()

    def setDraggable(self, v: bool):
        self.draggable = v

    def setPen(self, pen: QPen):
        self.pen = pen
        self._chart.update()


class _FitWorker(QThread):
    result_ready = pyqtSignal(object, list, list)

    def __init__(self, fit_item: "_FitItem", x_lo: float, x_hi: float, n_pts: int):
        super().__init__()
        self._fit_item = fit_item
        self._x_lo = x_lo
        self._x_hi = x_hi
        self._n_pts = n_pts

    def run(self):
        from .math_utils import get_fit_mode, linspace
        mode = get_fit_mode(self._fit_item.mode_key)
        if mode is None or not self._fit_item.source.xs:
            self.result_ready.emit(self._fit_item, [], [])
            return
        x_eval = linspace(self._x_lo, self._x_hi, self._n_pts)
        result = mode.evaluate(list(self._fit_item.source.xs), list(self._fit_item.source.ys), x_eval)
        if result is None:
            self.result_ready.emit(self._fit_item, [], [])
        else:
            self.result_ready.emit(self._fit_item, x_eval, result)


class _FitItem:
    def __init__(self, chart: "ChartWidget", source: "_LineItem | _ScatterItem",
                 mode_key: str, pen: QPen, label: str = ""):
        self._chart = chart
        self.source = source
        self.mode_key = mode_key
        self.pen = pen
        self.label = label
        self.visible = True
        self.show_formula = False
        self._xs: List[float] = []
        self._ys: List[float] = []
        self._worker: Optional[_FitWorker] = None
        self._pending_range: Optional[Tuple[float, float, int]] = None

    def _recompute(self, x_range_lo: float, x_range_hi: float,
                   n_pts: int = 400, threaded: bool = False):
        if not threaded:
            from .math_utils import get_fit_mode, linspace
            mode = get_fit_mode(self.mode_key)
            if mode is None or not self.source.xs:
                self._xs = []
                self._ys = []
                return
            x_eval = linspace(x_range_lo, x_range_hi, n_pts)
            result = mode.evaluate(list(self.source.xs), list(self.source.ys), x_eval)
            if result is None:
                self._xs = []
                self._ys = []
            else:
                self._xs = x_eval
                self._ys = result
            return
        self._pending_range = (x_range_lo, x_range_hi, n_pts)
        if self._worker is None or not self._worker.isRunning():
            self._start_worker(x_range_lo, x_range_hi, n_pts)

    def _start_worker(self, x_lo: float, x_hi: float, n_pts: int):
        self._worker = _FitWorker(self, x_lo, x_hi, n_pts)
        self._worker.result_ready.connect(self._on_worker_result)
        self._worker.start()

    def _on_worker_result(self, fit_item: "_FitItem", xs: List[float], ys: List[float]):
        if fit_item is not self:
            return
        self._xs = xs
        self._ys = ys
        self._chart._canvas.update()
        if self._pending_range is not None:
            lo, hi, n = self._pending_range
            if (self._worker is not None and
                    (abs(lo - self._worker._x_lo) > 1e-10 or
                     abs(hi - self._worker._x_hi) > 1e-10)):
                self._start_worker(lo, hi, n)
            else:
                self._pending_range = None

    def getData(self, x_lo: Optional[float] = None, x_hi: Optional[float] = None,
                n_pts: int = 400) -> Tuple[List[float], List[float]]:
        lo = x_lo if x_lo is not None else (min(self.source.xs) if self.source.xs else 0.0)
        hi = x_hi if x_hi is not None else (max(self.source.xs) if self.source.xs else 1.0)
        self._recompute(lo, hi, n_pts, threaded=False)
        return list(self._xs), list(self._ys)

    def evaluate(self, x: Union[float, List[float]]) -> Union[Optional[float], List[Optional[float]]]:
        scalar = isinstance(x, (int, float))
        x_list = [float(x)] if scalar else [float(v) for v in x]
        if not x_list:
            return []
        from .math_utils import get_fit_mode, _sort_unique
        mode = get_fit_mode(self.mode_key)
        if mode is None or not self.source.xs:
            return None if scalar else [None] * len(x_list)
        xs_s, ys_s = _sort_unique(list(self.source.xs), list(self.source.ys))
        if len(xs_s) < mode.min_points:
            return None if scalar else [None] * len(x_list)
        result = mode.fn(xs_s, ys_s, x_list)
        return result[0] if scalar else result

    def getFormula(self) -> str:
        from .math_utils import get_fit_mode
        mode = get_fit_mode(self.mode_key)
        if mode is None or not self.source.xs:
            return ""
        return mode.formula(list(self.source.xs), list(self.source.ys))

    def getRSquared(self) -> Optional[float]:
        from .math_utils import get_fit_mode, r_squared
        mode = get_fit_mode(self.mode_key)
        if mode is None or not self.source.xs or len(self._xs) < 2:
            return None
        return r_squared(list(self._xs), self._ys, self._ys)

    def asDict(self, x_lo: Optional[float] = None, x_hi: Optional[float] = None,
               n_pts: int = 400) -> dict:
        xs, ys = self.getData(x_lo, x_hi, n_pts)
        return {"x": xs, "y": ys}

    def asTuples(self, x_lo: Optional[float] = None, x_hi: Optional[float] = None,
                 n_pts: int = 400) -> List[Tuple[float, float]]:
        xs, ys = self.getData(x_lo, x_hi, n_pts)
        return list(zip(xs, ys))

    def setModeKey(self, key: str):
        self.mode_key = key
        self._chart.update()

    def setVisible(self, v: bool):
        self.visible = v
        self._chart.update()

    def setLabel(self, label: str):
        self.label = label
        self._chart.update()

    def setShowFormula(self, v: bool):
        self.show_formula = v
        self._chart.update()

    @property
    def xs(self) -> List[float]:
        return self._xs

    @property
    def ys(self) -> List[float]:
        return self._ys


class _DerivativeItem:
    def __init__(self, chart: "ChartWidget", source: "_LineItem | _ScatterItem",
                 order: int = 1, pen: QPen = None, label: str = ""):
        from PyQt6.QtGui import QPen as _QPen
        self._chart = chart
        self.source = source
        self.order = order
        self.pen = pen or _QPen()
        self.label = label
        self.visible = True
        self._xs: List[float] = []
        self._ys: List[float] = []

    def recompute(self):
        from .math_utils import derivative, second_derivative, _sort_unique
        xs, ys = _sort_unique(list(self.source.xs), list(self.source.ys))
        if len(xs) < 2:
            self._xs, self._ys = [], []
            return
        if self.order == 2:
            self._xs, self._ys = second_derivative(xs, ys)
        else:
            self._xs, self._ys = derivative(xs, ys)
        self._chart.update()

    def setVisible(self, v: bool):
        self.visible = v
        self._chart.update()

    def setLabel(self, label: str):
        self.label = label
        self._chart.update()

    def setPen(self, pen: QPen):
        self.pen = pen
        self._chart.update()

    @property
    def xs(self) -> List[float]:
        return self._xs

    @property
    def ys(self) -> List[float]:
        return self._ys

    @property
    def x_range(self) -> Optional[Tuple[float, float]]:
        return (min(self._xs), max(self._xs)) if self._xs else None

    @property
    def y_range(self) -> Optional[Tuple[float, float]]:
        return (min(self._ys), max(self._ys)) if self._ys else None


class _IntegralItem:
    def __init__(self, chart: "ChartWidget", source: "_LineItem | _ScatterItem",
                 pen: QPen = None, label: str = ""):
        from PyQt6.QtGui import QPen as _QPen
        self._chart = chart
        self.source = source
        self.pen = pen or _QPen()
        self.label = label
        self.visible = True
        self._xs: List[float] = []
        self._ys: List[float] = []

    def recompute(self):
        from .math_utils import cumulative_integral, _sort_unique
        xs, ys = _sort_unique(list(self.source.xs), list(self.source.ys))
        if len(xs) < 2:
            self._xs, self._ys = [], []
            return
        self._xs, self._ys = cumulative_integral(xs, ys)
        self._chart.update()

    def setVisible(self, v: bool):
        self.visible = v
        self._chart.update()

    def setLabel(self, label: str):
        self.label = label
        self._chart.update()

    def setPen(self, pen: QPen):
        self.pen = pen
        self._chart.update()

    @property
    def xs(self) -> List[float]:
        return self._xs

    @property
    def ys(self) -> List[float]:
        return self._ys

    @property
    def x_range(self) -> Optional[Tuple[float, float]]:
        return (min(self._xs), max(self._xs)) if self._xs else None

    @property
    def y_range(self) -> Optional[Tuple[float, float]]:
        return (min(self._ys), max(self._ys)) if self._ys else None


class _HistogramItem:
    def __init__(self, chart: "ChartWidget", color: QColor = None,
                 bins: int = 20, label: str = ""):
        from PyQt6.QtGui import QColor as _QColor
        self._chart = chart
        self.color = color or _QColor(100, 149, 237)
        self.bins = bins
        self.label = label
        self.visible = True
        self.normalize = False
        self._centers: List[float] = []
        self._counts: List[float] = []
        self._raw: List[float] = []

    def setData(self, data: List[float]):
        from .math_utils import histogram
        self._raw = list(data)
        self._centers, self._counts = histogram(self._raw, self.bins)
        if self.normalize and self._counts:
            total = sum(self._counts)
            self._counts = [c / total if total > 0 else 0.0 for c in self._counts]
        self._chart._schedule_autofit()

    def setBins(self, bins: int):
        self.bins = max(1, bins)
        if self._raw:
            self.setData(self._raw)

    def setNormalize(self, enabled: bool):
        self.normalize = enabled
        if self._raw:
            self.setData(self._raw)

    def setColor(self, color: QColor):
        self.color = color
        self._chart.update()

    def setVisible(self, v: bool):
        self.visible = v
        self._chart.update()

    def setLabel(self, label: str):
        self.label = label
        self._chart.update()

    def clear(self):
        self._raw = []
        self._centers = []
        self._counts = []
        self._chart._schedule_autofit()

    @property
    def centers(self) -> List[float]:
        return self._centers

    @property
    def counts(self) -> List[float]:
        return self._counts

    @property
    def x_range(self) -> Optional[Tuple[float, float]]:
        return (min(self._centers), max(self._centers)) if self._centers else None

    @property
    def y_range(self) -> Optional[Tuple[float, float]]:
        return (0.0, max(self._counts)) if self._counts else None


class _SpectrumItem:
    def __init__(self, chart: "ChartWidget", source: "_LineItem | _ScatterItem",
                 pen: QPen = None, label: str = ""):
        from PyQt6.QtGui import QPen as _QPen
        self._chart = chart
        self.source = source
        self.pen = pen or _QPen()
        self.label = label
        self.visible = True
        self.log_scale = False
        self._freqs: List[float] = []
        self._amps: List[float] = []

    def recompute(self):
        from .math_utils import fft_spectrum_numpy, _sort_unique
        xs, ys = _sort_unique(list(self.source.xs), list(self.source.ys))
        if len(xs) < 4:
            self._freqs, self._amps = [], []
            return
        self._freqs, self._amps = fft_spectrum_numpy(xs, ys)
        if self.log_scale:
            self._amps = [math.log10(max(a, 1e-30)) for a in self._amps]
        self._chart.update()

    def setLogScale(self, enabled: bool):
        self.log_scale = enabled
        self.recompute()

    def setVisible(self, v: bool):
        self.visible = v
        self._chart.update()

    def setLabel(self, label: str):
        self.label = label
        self._chart.update()

    def setPen(self, pen: QPen):
        self.pen = pen
        self._chart.update()

    def dominantFrequency(self) -> Optional[float]:
        if not self._amps:
            return None
        idx = max(range(len(self._amps)), key=lambda i: self._amps[i])
        return self._freqs[idx] if self._freqs else None

    @property
    def freqs(self) -> List[float]:
        return self._freqs

    @property
    def amps(self) -> List[float]:
        return self._amps

    @property
    def x_range(self) -> Optional[Tuple[float, float]]:
        return (min(self._freqs), max(self._freqs)) if self._freqs else None

    @property
    def y_range(self) -> Optional[Tuple[float, float]]:
        return (0.0, max(self._amps)) if self._amps else None


class _ErrorBandItem:
    def __init__(self, chart: "ChartWidget", color: QColor = None,
                 alpha: int = 50, label: str = ""):
        from PyQt6.QtGui import QColor as _QColor
        self._chart = chart
        self.color = color or _QColor(100, 180, 255)
        self.alpha = alpha
        self.label = label
        self.visible = True
        self.xs: List[float] = []
        self.ys_lo: List[float] = []
        self.ys_hi: List[float] = []

    def setData(self, xs: List[float], ys_lo: List[float], ys_hi: List[float]):
        self.xs = list(xs)
        self.ys_lo = list(ys_lo)
        self.ys_hi = list(ys_hi)
        self._chart._schedule_autofit()

    def setFromMeanStd(self, xs: List[float], ys: List[float], n_sigma: float = 1.0):
        from .math_utils import moving_std
        window = max(3, len(ys) // 10)
        stds = moving_std(ys, window)
        self.xs = list(xs)
        self.ys_lo = [y - n_sigma * s for y, s in zip(ys, stds)]
        self.ys_hi = [y + n_sigma * s for y, s in zip(ys, stds)]
        self._chart._schedule_autofit()

    def setColor(self, color: QColor, alpha: int = 50):
        self.color = color
        self.alpha = alpha
        self._chart.update()

    def setVisible(self, v: bool):
        self.visible = v
        self._chart.update()

    def setLabel(self, label: str):
        self.label = label
        self._chart.update()

    def clear(self):
        self.xs = []
        self.ys_lo = []
        self.ys_hi = []
        self._chart._schedule_autofit()

    @property
    def x_range(self) -> Optional[Tuple[float, float]]:
        return (min(self.xs), max(self.xs)) if self.xs else None

    @property
    def y_range(self) -> Optional[Tuple[float, float]]:
        if not self.ys_lo and not self.ys_hi:
            return None
        return (min(self.ys_lo + self.ys_hi), max(self.ys_lo + self.ys_hi))