# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math

from core.foundation.curve import Curve, TangentMode


def _sample(func, n: int = 33) -> Curve:
    c = Curve()
    for i in range(n + 1):
        t = i / n
        c.add_key(round(t, 4), float(func(t)))
    return c


def _linear(pairs) -> Curve:
    c = Curve()
    for t, v in pairs:
        k = c.add_key(round(t, 4), float(v))
        k.tangent_mode = TangentMode.LINEAR
    return c


def _constant(v: float) -> Curve:
    c = Curve()
    c.add_key(0.0, float(v))
    return c


_PRESETS: dict[str, callable] = {}


def _reg(name: str):
    def deco(f):
        _PRESETS[name] = f
        return f
    return deco


# --------------------------------------------------------------------------
# Linear / constant / steps
# --------------------------------------------------------------------------
@_reg("Linear")
def _p_linear() -> Curve:
    return _sample(lambda t: t)


@_reg("Flat 0")
def _p_flat0() -> Curve:
    return _constant(0.0)


@_reg("Flat 1")
def _p_flat1() -> Curve:
    return _constant(1.0)


@_reg("Step 2")
def _p_step2() -> Curve:
    return _linear([(0.0, 0.0), (0.49, 0.0), (0.51, 1.0), (1.0, 1.0)])


@_reg("Step 4")
def _p_step4() -> Curve:
    return _linear([
        (0.0, 0.0), (0.235, 0.0), (0.265, 0.333),
        (0.485, 0.333), (0.515, 0.667), (0.735, 0.667),
        (0.765, 1.0), (1.0, 1.0),
    ])


# --------------------------------------------------------------------------
# Ease In
# --------------------------------------------------------------------------
@_reg("Ease In Sine")
def _p_ein_sine() -> Curve:
    return _sample(lambda t: 1.0 - math.cos((t * math.pi) / 2.0))


@_reg("Ease In Quad")
def _p_ein_quad() -> Curve:
    return _sample(lambda t: t * t)


@_reg("Ease In Cubic")
def _p_ein_cubic() -> Curve:
    return _sample(lambda t: t ** 3)


@_reg("Ease In Quart")
def _p_ein_quart() -> Curve:
    return _sample(lambda t: t ** 4)


@_reg("Ease In Quint")
def _p_ein_quint() -> Curve:
    return _sample(lambda t: t ** 5)


@_reg("Ease In Expo")
def _p_ein_expo() -> Curve:
    return _sample(lambda t: 0.0 if t == 0.0 else (2.0 ** (20.0 * t - 20.0)))


@_reg("Ease In Circ")
def _p_ein_circ() -> Curve:
    return _sample(lambda t: 1.0 - math.sqrt(1.0 - t * t))


# --------------------------------------------------------------------------
# Ease Out
# --------------------------------------------------------------------------
@_reg("Ease Out Sine")
def _p_eout_sine() -> Curve:
    return _sample(lambda t: math.sin((t * math.pi) / 2.0))


@_reg("Ease Out Quad")
def _p_eout_quad() -> Curve:
    return _sample(lambda t: 1.0 - (1.0 - t) ** 2)


@_reg("Ease Out Cubic")
def _p_eout_cubic() -> Curve:
    return _sample(lambda t: 1.0 - (1.0 - t) ** 3)


@_reg("Ease Out Quart")
def _p_eout_quart() -> Curve:
    return _sample(lambda t: 1.0 - (1.0 - t) ** 4)


@_reg("Ease Out Quint")
def _p_eout_quint() -> Curve:
    return _sample(lambda t: 1.0 - (1.0 - t) ** 5)


@_reg("Ease Out Expo")
def _p_eout_expo() -> Curve:
    return _sample(lambda t: 1.0 if t == 1.0 else 1.0 - 2.0 ** (-20.0 * t))


@_reg("Ease Out Circ")
def _p_eout_circ() -> Curve:
    return _sample(lambda t: math.sqrt(1.0 - (1.0 - t) ** 2))


# --------------------------------------------------------------------------
# Ease In-Out
# --------------------------------------------------------------------------
@_reg("Ease In-Out Sine")
def _p_eio_sine() -> Curve:
    return _sample(lambda t: -(math.cos(math.pi * t) - 1.0) / 2.0)


@_reg("Smoothstep")
def _p_smooth() -> Curve:
    return _sample(lambda t: t * t * (3.0 - 2.0 * t))


@_reg("Smootherstep")
def _p_smoother() -> Curve:
    return _sample(lambda t: t ** 3 * (t * (t * 6.0 - 15.0) + 10.0))


@_reg("Ease In-Out Quad")
def _p_eio_quad() -> Curve:
    return _sample(lambda t: 2.0 * t * t if t < 0.5 else 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0)


@_reg("Ease In-Out Cubic")
def _p_eio_cubic() -> Curve:
    return _sample(lambda t: 4.0 * t ** 3 if t < 0.5 else 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0)


@_reg("Ease In-Out Quart")
def _p_eio_quart() -> Curve:
    return _sample(lambda t: 8.0 * t ** 4 if t < 0.5 else 1.0 - (-2.0 * t + 2.0) ** 4 / 2.0)


@_reg("Ease In-Out Quint")
def _p_eio_quint() -> Curve:
    return _sample(lambda t: 16.0 * t ** 5 if t < 0.5 else 1.0 - (-2.0 * t + 2.0) ** 5 / 2.0)


@_reg("Ease In-Out Expo")
def _p_eio_expo() -> Curve:
    return _sample(lambda t: (2.0 ** (20.0 * t - 10.0)) / 2.0 if t < 0.5
                   else 1.0 - 2.0 ** (-20.0 * t + 10.0) / 2.0)


@_reg("Ease In-Out Circ")
def _p_eio_circ() -> Curve:
    return _sample(lambda t: (1.0 - math.sqrt(1.0 - (2.0 * t) ** 2)) / 2.0 if t < 0.5
                   else (math.sqrt(1.0 - (-2.0 * t + 2.0) ** 2) + 1.0) / 2.0)


# --------------------------------------------------------------------------
# Back / Elastic / Bounce
# --------------------------------------------------------------------------
_C1 = 1.70158
_C2 = _C1 * 1.525


@_reg("Back In")
def _p_back_in() -> Curve:
    return _sample(lambda t: ((_C1 + 1.0) * t ** 3 - _C1 * t ** 2))


@_reg("Back Out")
def _p_back_out() -> Curve:
    return _sample(lambda t: 1.0 + (_C1 + 1.0) * (t - 1.0) ** 3 + _C1 * (t - 1.0) ** 2)


@_reg("Back In-Out")
def _p_back_io() -> Curve:
    return _sample(lambda t: ((2.0 * t) ** 2 * ((_C2 + 1.0) * 2.0 * t - _C2)) / 2.0 if t < 0.5
                   else ((2.0 * t - 2.0) ** 2 * ((_C2 + 1.0) * (2.0 * t - 2.0) + _C2) + 2.0) / 2.0)


def _bounce_out(t: float) -> float:
    n1 = 7.5625
    d1 = 2.75
    if t < 1.0 / d1:
        return n1 * t * t
    elif t < 2.0 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    elif t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    else:
        t -= 2.625 / d1
        return n1 * t * t + 0.984375


@_reg("Bounce Out")
def _p_bounce_out() -> Curve:
    return _sample(_bounce_out)


@_reg("Bounce In")
def _p_bounce_in() -> Curve:
    return _sample(lambda t: 1.0 - _bounce_out(1.0 - t))


@_reg("Bounce In-Out")
def _p_bounce_io() -> Curve:
    return _sample(lambda t: (1.0 - _bounce_out(1.0 - 2.0 * t)) / 2.0 if t < 0.5
                   else (1.0 + _bounce_out(2.0 * t - 1.0)) / 2.0)


_C4 = (2.0 * math.pi) / 3.0
_C5 = (2.0 * math.pi) / 4.5


def _ease_out_elastic(t: float) -> float:
    if t == 0.0:
        return 0.0
    if t == 1.0:
        return 1.0
    return 2.0 ** (-10.0 * t) * math.sin((t * 10.0 - 0.75) * _C4) + 1.0


def _ease_in_elastic(t: float) -> float:
    if t == 0.0:
        return 0.0
    if t == 1.0:
        return 1.0
    return -(2.0 ** (10.0 * t - 10.0)) * math.sin((t * 10.0 - 0.75) * _C4)


@_reg("Elastic Out")
def _p_elast_out() -> Curve:
    return _sample(_ease_out_elastic)


@_reg("Elastic In")
def _p_elast_in() -> Curve:
    return _sample(_ease_in_elastic)


@_reg("Elastic In-Out")
def _p_elast_io() -> Curve:
    return _sample(lambda t: 0.0 if t == 0.0 else 1.0 if t == 1.0
                   else (-(2.0 ** (20.0 * t - 10.0) * math.sin((20.0 * t - 11.125) * _C5)) / 2.0) if t < 0.5
                   else (2.0 ** (-20.0 * t + 10.0) * math.sin((20.0 * t - 11.125) * _C5)) / 2.0 + 0.5)


# --------------------------------------------------------------------------
# Waves / oscillation
# --------------------------------------------------------------------------
@_reg("Sine Wave")
def _p_sine1() -> Curve:
    return _sample(lambda t: 0.5 - 0.5 * math.cos(2.0 * math.pi * t))


@_reg("Sine Wave 2")
def _p_sine2() -> Curve:
    return _sample(lambda t: 0.5 - 0.5 * math.cos(4.0 * math.pi * t))


@_reg("Triangle")
def _p_triangle() -> Curve:
    return _linear([(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)])


@_reg("Sawtooth")
def _p_saw() -> Curve:
    return _linear([(0.0, 0.0), (0.92, 0.92), (1.0, 0.0)])


@_reg("Sawtooth Down")
def _p_saw_down() -> Curve:
    return _linear([(0.0, 1.0), (0.92, 0.08), (1.0, 1.0)])


@_reg("Square Wave")
def _p_square() -> Curve:
    return _linear([(0.0, 0.0), (0.46, 0.0), (0.54, 1.0), (1.0, 1.0)])


@_reg("Pulse")
def _p_pulse() -> Curve:
    return _linear([(0.0, 0.0), (0.12, 0.0), (0.18, 1.0), (0.44, 1.0), (0.5, 0.0), (1.0, 0.0)])


@_reg("Bell")
def _p_bell() -> Curve:
    return _sample(lambda t: math.exp(-(((t - 0.5) / 0.16) ** 2)))


@_reg("Plateau")
def _p_plateau() -> Curve:
    return _sample(lambda t: 0.0 if t < 0.2 else 1.0 if t > 0.8 else (t - 0.2) / 0.6)


@_reg("Inverted")
def _p_inverted() -> Curve:
    return _sample(lambda t: 1.0 - t)


PRESET_CATEGORIES: list[tuple[str, list[str]]] = [
    ("Linear & Steps", ["Linear", "Flat 0", "Flat 1", "Step 2", "Step 4"]),
    ("Ease In", ["Ease In Sine", "Ease In Quad", "Ease In Cubic", "Ease In Quart",
                 "Ease In Quint", "Ease In Expo", "Ease In Circ"]),
    ("Ease Out", ["Ease Out Sine", "Ease Out Quad", "Ease Out Cubic", "Ease Out Quart",
                  "Ease Out Quint", "Ease Out Expo", "Ease Out Circ"]),
    ("Ease In-Out", ["Ease In-Out Sine", "Smoothstep", "Smootherstep", "Ease In-Out Quad",
                     "Ease In-Out Cubic", "Ease In-Out Quart", "Ease In-Out Quint",
                     "Ease In-Out Expo", "Ease In-Out Circ"]),
    ("Back / Elastic / Bounce", ["Back In", "Back Out", "Back In-Out", "Elastic In",
                                 "Elastic Out", "Elastic In-Out", "Bounce In", "Bounce Out",
                                 "Bounce In-Out"]),
    ("Waves & Shapes", ["Sine Wave", "Sine Wave 2", "Triangle", "Sawtooth", "Sawtooth Down",
                        "Square Wave", "Pulse", "Bell", "Plateau", "Inverted"]),
]


def build_preset(name: str) -> Curve:
    maker = _PRESETS.get(name)
    if maker is None:
        return Curve()
    return maker()


def preset_names() -> list[str]:
    return list(_PRESETS.keys())
