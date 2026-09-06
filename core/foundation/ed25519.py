# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import hashlib
import os

_b = 256
_q = (1 << 255) - 19
_l = (1 << 252) + 27742317777372353535851937790883648493
_d = (-121665 * pow(121666, _q - 2, _q)) % _q
_I = pow(2, (_q - 1) // 4, _q)
_Bx = 15112221349535400772501151409588531511454012693041857206046113283949847762202
_By = 46316835694926478169428394003475163141307993866256225615783033603165251855960
_B = (_Bx % _q, _By % _q)
_ID = (0, 1)


def _H(m: bytes) -> bytes:
    return hashlib.sha512(m).digest()


def _bit(h: bytes, i: int) -> int:
    return (h[i // 8] >> (i % 8)) & 1


def _inv(x: int) -> int:
    return pow(x, _q - 2, _q)


def _xrecover(y: int) -> int:
    xx = (y * y - 1) * _inv(_d * y * y + 1)
    x = pow(xx % _q, (_q + 3) // 8, _q)
    if (x * x - xx) % _q != 0:
        x = (x * _I) % _q
    if x % 2 != 0:
        x = _q - x
    return x


def _edwards(P, Q):
    x1, y1 = P
    x2, y2 = Q
    x3 = (x1 * y2 + x2 * y1) * _inv(1 + _d * x1 * x2 * y1 * y2)
    y3 = (y1 * y2 + x1 * x2) * _inv(1 - _d * x1 * x2 * y1 * y2)
    return (x3 % _q, y3 % _q)


def _scalarmult(P, e: int):
    if e == 0:
        return _ID
    Q = _scalarmult(P, e // 2)
    Q = _edwards(Q, Q)
    if e & 1:
        Q = _edwards(Q, P)
    return Q


def _encodeint(y: int) -> bytes:
    return bytes((y >> (8 * i)) & 0xFF for i in range(_b // 8))


def _encodepoint(P) -> bytes:
    x, y = P
    bits = [(y >> i) & 1 for i in range(_b - 1)] + [x & 1]
    return bytes(sum(bits[i * 8 + j] << j for j in range(8)) for i in range(_b // 8))


def _decodeint(s: bytes) -> int:
    return sum(s[i] << (8 * i) for i in range(_b // 8))


def _decodepoint(s: bytes):
    y = sum(2 ** i * _bit(s, i) for i in range(_b - 1))
    x = _xrecover(y)
    if x & 1 != _bit(s, _b - 1):
        x = _q - x
    P = (x, y)
    x1, y1 = P
    if (-x1 * x1 + y1 * y1 - 1 - _d * x1 * x1 * y1 * y1) % _q != 0:
        return None
    return P


def _Hint(m: bytes) -> int:
    h = _H(m)
    return sum(2 ** i * _bit(h, i) for i in range(2 * _b))


def publickey_hex(priv_hex: str) -> str:
    sk = bytes.fromhex(priv_hex)
    h = _H(sk)
    a = 2 ** (_b - 2) + sum(2 ** i * _bit(h, i) for i in range(3, _b - 2))
    return _encodepoint(_scalarmult(_B, a)).hex()


def generate_hex() -> tuple:
    sk = os.urandom(32)
    return (sk.hex(), publickey_hex(sk.hex()))


def sign_hex(msg: bytes, priv_hex: str) -> str:
    sk = bytes.fromhex(priv_hex)
    h = _H(sk)
    a = 2 ** (_b - 2) + sum(2 ** i * _bit(h, i) for i in range(3, _b - 2))
    r = _Hint(h[_b // 8:_b // 4] + msg)
    R = _scalarmult(_B, r)
    pk = _encodepoint(_scalarmult(_B, a))
    S = (r + _Hint(_encodepoint(R) + pk + msg) * a) % _l
    return (_encodepoint(R) + _encodeint(S)).hex()


def verify_hex(sig_hex: str, msg: bytes, pub_hex: str) -> bool:
    try:
        sig = bytes.fromhex(sig_hex)
        pk = bytes.fromhex(pub_hex)
    except Exception:
        return False
    if len(sig) != _b // 4 or len(pk) != _b // 8:
        return False
    R = _decodepoint(sig[:_b // 8])
    A = _decodepoint(pk)
    if R is None or A is None:
        return False
    S = _decodeint(sig[_b // 8:_b // 4])
    h = _Hint(_encodepoint(R) + pk + msg)
    return _scalarmult(_B, S) == _edwards(R, _scalarmult(A, h))