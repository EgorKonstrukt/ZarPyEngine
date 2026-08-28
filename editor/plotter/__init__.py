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
from .chart_widget import ChartWidget
from .items import (
    _LineItem, _ScatterItem, _FitItem, _InfLine, _FunctionItem, _RulerItem,
    _DerivativeItem, _IntegralItem, _HistogramItem, _SpectrumItem, _ErrorBandItem,
)
from .math_utils import (
    FitMode, register_fit_mode, get_fit_modes, get_fit_mode,
    derivative, second_derivative, cumulative_integral,
    fft_spectrum, fft_spectrum_numpy, histogram, autocorrelation,
    normalize, peak_find, weighted_mean, moving_std,
)
from .sidebar import SidebarLabel, SidebarButton
from .palette import set_palette, reset_colors
from .i18n import tr, set_tr, update_strings

__all__ = [
    "ChartWidget",
    "_LineItem",
    "_ScatterItem",
    "_FitItem",
    "_InfLine",
    "_FunctionItem",
    "_RulerItem",
    "_DerivativeItem",
    "_IntegralItem",
    "_HistogramItem",
    "_SpectrumItem",
    "_ErrorBandItem",
    "FitMode",
    "register_fit_mode",
    "get_fit_modes",
    "get_fit_mode",
    "derivative",
    "second_derivative",
    "cumulative_integral",
    "fft_spectrum",
    "fft_spectrum_numpy",
    "histogram",
    "autocorrelation",
    "normalize",
    "peak_find",
    "weighted_mean",
    "moving_std",
    "SidebarLabel",
    "SidebarButton",
    "set_palette",
    "reset_colors",
    "tr",
    "set_tr",
    "update_strings",
]
__version__ = "6.1.0"