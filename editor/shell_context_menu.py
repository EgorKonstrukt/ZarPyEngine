# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import os
import ctypes
import ctypes.wintypes as wt
from ctypes import (POINTER, byref, c_void_p, c_uint, c_int, c_ulong, c_wchar,
                     c_ubyte, c_ushort, c_int64, Structure, WINFUNCTYPE, cast)

shell32 = ctypes.windll.shell32
ole32 = ctypes.windll.ole32
user32 = ctypes.windll.user32


class GUID(Structure):
    _fields_ = [("Data1", c_ulong), ("Data2", c_ushort), ("Data3", c_ushort),
                ("Data4", c_ubyte * 8)]


class WNDCLASSEX(Structure):
    _fields_ = [
        ("cbSize", c_uint),
        ("style", c_uint),
        ("lpfnWndProc", c_void_p),
        ("cbClsExtra", c_int),
        ("cbWndExtra", c_int),
        ("hInstance", c_void_p),
        ("hIcon", c_void_p),
        ("hCursor", c_void_p),
        ("hbrBackground", c_void_p),
        ("lpszMenuName", c_void_p),
        ("lpszClassName", c_void_p),
        ("hIconSm", c_void_p),
    ]


ole32.IIDFromString.argtypes = [wt.LPCWSTR, POINTER(GUID)]
ole32.IIDFromString.restype = c_int
ole32.CoInitializeEx.argtypes = [c_void_p, c_ulong]
ole32.CoInitializeEx.restype = c_int
ole32.CoUninitialize.argtypes = []
ole32.CoUninitialize.restype = None

shell32.SHParseDisplayName.argtypes = [wt.LPCWSTR, c_void_p, POINTER(c_void_p), wt.ULONG, POINTER(wt.ULONG)]
shell32.SHParseDisplayName.restype = c_int
shell32.SHGetDesktopFolder.argtypes = [POINTER(c_void_p)]
shell32.SHGetDesktopFolder.restype = c_int
shell32.SHBindToParent.argtypes = [c_void_p, POINTER(GUID), POINTER(c_void_p), POINTER(c_void_p)]
shell32.SHBindToParent.restype = c_int
shell32.ILFree.argtypes = [c_void_p]
shell32.ILFree.restype = None

user32.CreatePopupMenu.restype = wt.HMENU
user32.DestroyMenu.argtypes = [wt.HMENU]
user32.DestroyMenu.restype = c_int
user32.AppendMenuW.argtypes = [wt.HMENU, c_uint, c_uint, wt.LPCWSTR]
user32.AppendMenuW.restype = c_int
user32.GetMenuItemCount.argtypes = [wt.HMENU]
user32.GetMenuItemCount.restype = c_int
user32.GetMenuStringW.argtypes = [wt.HMENU, c_int, wt.LPCWSTR, c_int, c_uint]
user32.GetMenuStringW.restype = c_int
user32.TrackPopupMenuEx.argtypes = [wt.HMENU, c_uint, c_int, c_int, wt.HWND, c_void_p]
user32.TrackPopupMenuEx.restype = c_uint
user32.RegisterClassExW.argtypes = [POINTER(WNDCLASSEX)]
user32.RegisterClassExW.restype = c_ushort
user32.CreateWindowExW.argtypes = [c_uint, c_void_p, wt.LPCWSTR, c_uint,
                                   c_int, c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p]
user32.CreateWindowExW.restype = wt.HWND
kernel32 = ctypes.windll.kernel32
kernel32.GetModuleHandleW.argtypes = [wt.LPCWSTR]
kernel32.GetModuleHandleW.restype = wt.HINSTANCE
user32.DefWindowProcW.argtypes = [c_void_p, c_uint, c_void_p, c_void_p]
user32.DefWindowProcW.restype = c_int

GWLP_WNDPROC = -4
user32.GetWindowLongPtrW.argtypes = [c_void_p, c_int]
user32.GetWindowLongPtrW.restype = c_void_p
user32.SetWindowLongPtrW.argtypes = [c_void_p, c_int, c_void_p]
user32.SetWindowLongPtrW.restype = c_void_p
user32.CallWindowProcW.argtypes = [c_void_p, c_void_p, c_uint, c_void_p, c_void_p]
user32.CallWindowProcW.restype = c_int64

COINIT_APARTMENTTHREADED = 0x2
CMF_NORMAL = 0x0
CMF_EXPLORE = 0x4
CMIC_MASK_UNICODE = 0x4000
SW_NORMAL = 1
TPM_RETURNCMD = 0x100
TPM_RIGHTBUTTON = 0x2
HWND_MESSAGE = c_void_p(-3)
WM_INITMENUPOPUP = 0x117
WM_MEASUREITEM = 0x2C
WM_DRAWITEM = 0x2B
WM_MENUCHAR = 0x120
MF_STRING = 0x0
MF_SEPARATOR = 0x800
MF_ENABLED = 0x0


def _iid_from_string(text: str) -> GUID:
    g = GUID()
    ole32.IIDFromString(ctypes.create_unicode_buffer(text), byref(g))
    return g


IID_IShellFolder = _iid_from_string("{000214E6-0000-0000-C000-000000000046}")
IID_IContextMenu = _iid_from_string("{000214E4-0000-0000-C000-000000000046}")
IID_IContextMenu2 = _iid_from_string("{000214F4-0000-0000-C000-000000000046}")
IID_IContextMenu3 = _iid_from_string("{000214F1-0000-0000-C000-000000000046}")


class CMINVOKECOMMANDINFOEX(Structure):
    _fields_ = [
        ("cbSize", wt.DWORD),
        ("fMask", wt.DWORD),
        ("hwnd", wt.HWND),
        ("lpVerb", wt.LPCWSTR),
        ("lpParameters", wt.LPCWSTR),
        ("lpDirectory", wt.LPCWSTR),
        ("nShow", c_int),
        ("dwHotKey", wt.DWORD),
        ("hIcon", wt.HANDLE),
        ("lpTitle", wt.LPCWSTR),
        ("lpVerbW", wt.LPCWSTR),
        ("lpParametersW", wt.LPCWSTR),
        ("lpDirectoryW", wt.LPCWSTR),
        ("lpTitleW", wt.LPCWSTR),
        ("ptInvoke", wt.POINT),
    ]


def _vt_call(pv, index, restype, argtypes, *args):
    obj = cast(pv, POINTER(c_void_p))
    vtbl = cast(obj[0], POINTER(c_void_p))
    fptr = vtbl[index]
    proto = WINFUNCTYPE(restype, *([c_void_p] + list(argtypes)))
    func = cast(fptr, proto)
    return func(pv, *args)


def _query_interface(pv, iid) -> c_void_p | None:
    out = c_void_p()
    hr = _vt_call(pv, 0, c_int, [POINTER(type(iid)), POINTER(c_void_p)],
                  byref(iid), byref(out))
    if hr < 0 or not out:
        return None
    return out


_SUBCLASS_REFS: dict[int, tuple] = {}


def _make_wnd_proc(old_wndproc, icm2, icm3):
    proto = WINFUNCTYPE(c_int64, c_void_p, c_uint, c_void_p, c_void_p)

    def _wnd_proc(hwnd, msg, wparam, lparam):
        if msg in (WM_INITMENUPOPUP, WM_MEASUREITEM, WM_DRAWITEM, WM_MENUCHAR):
            if icm3:
                res = c_void_p()
                _vt_call(icm3, 7, c_int,
                         [c_uint, c_void_p, c_void_p, POINTER(c_void_p)],
                         msg, wparam, lparam, byref(res))
                return int(res.value) if res.value else 0
            if icm2:
                _vt_call(icm2, 6, c_int, [c_uint, c_void_p, c_void_p],
                         msg, wparam, lparam)
                return 0
        return user32.CallWindowProcW(old_wndproc, hwnd, msg, wparam, lparam)

    return proto(_wnd_proc)


def _subclass(owner_hwnd, icm2, icm3):
    old = user32.GetWindowLongPtrW(owner_hwnd, GWLP_WNDPROC)
    if not old:
        return None
    new = _make_wnd_proc(old, icm2, icm3)
    user32.SetWindowLongPtrW(owner_hwnd, GWLP_WNDPROC, cast(new, c_void_p))
    _SUBCLASS_REFS[int(owner_hwnd)] = (old, new)
    return old


def _unsubclass(owner_hwnd):
    entry = _SUBCLASS_REFS.get(int(owner_hwnd))
    if entry:
        old, _new = entry
        user32.SetWindowLongPtrW(owner_hwnd, GWLP_WNDPROC, old)
        del _SUBCLASS_REFS[int(owner_hwnd)]


def show_shell_context_menu(paths, hwnd_owner_int, x, y, extra_actions=None):
    if not paths:
        return False
    dirs = [os.path.dirname(os.path.abspath(p)) for p in paths]
    try:
        parent_dir = os.path.commonpath(dirs)
    except ValueError:
        return False
    if not parent_dir or not os.path.isdir(parent_dir):
        return False

    _ensure_com()
    return _show_impl(paths, parent_dir, int(hwnd_owner_int), x, y, extra_actions)


_com_initialized = False


def _ensure_com():
    global _com_initialized
    if _com_initialized:
        return
    ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
    _com_initialized = True


def _show_impl(paths, parent_dir, owner_hwnd, x, y, extra_actions):
    parent_folder = None
    child_pidls = []
    try:
        for p in paths:
            abs_pidl = c_void_p()
            hr = shell32.SHParseDisplayName(
                ctypes.create_unicode_buffer(os.path.abspath(p)),
                None, byref(abs_pidl), 0, byref(c_ulong(0)))
            if hr < 0 or not abs_pidl:
                continue
            par = c_void_p()
            child = c_void_p()
            bhr = shell32.SHBindToParent(abs_pidl, byref(IID_IShellFolder),
                                        byref(par), byref(child))
            shell32.ILFree(abs_pidl)
            if bhr < 0 or not par or not child:
                if par:
                    _release(par)
                continue
            if parent_folder is None:
                parent_folder = par
            elif par.value != parent_folder.value:
                _release(par)
            child_pidls.append(child)
        if not parent_folder or not child_pidls:
            return False

        arr = (c_void_p * len(child_pidls))(*child_pidls)
        icm = c_void_p()
        hr = _vt_call(parent_folder, 10, c_int,
                      [wt.HWND, c_uint, POINTER(c_void_p), POINTER(GUID),
                       c_void_p, POINTER(c_void_p)],
                      owner_hwnd, len(child_pidls), arr, byref(IID_IContextMenu),
                      None, byref(icm))
        if hr < 0 or not icm:
            return False

        icm2 = _query_interface(icm, IID_IContextMenu2)
        icm3 = _query_interface(icm, IID_IContextMenu3)
        ctx_ptr = icm3 or icm2 or icm

        _subclass(owner_hwnd, icm2, icm3)

        hmenu = user32.CreatePopupMenu()
        flags = CMF_EXPLORE
        added = _vt_call(ctx_ptr, 3, c_int,
                         [wt.HWND, c_uint, c_uint, c_uint, c_uint],
                         hmenu, 0, 1, 0x6FFF, flags)
        if added < 0:
            user32.DestroyMenu(hmenu)
            return False

        extra_base = 0x7000
        if extra_actions:
            user32.AppendMenuW(hmenu, MF_SEPARATOR, 0, None)
            for i, (label, _cb) in enumerate(extra_actions):
                user32.AppendMenuW(hmenu, MF_STRING | MF_ENABLED,
                                    extra_base + i, ctypes.create_unicode_buffer(label))

        cmd = user32.TrackPopupMenuEx(hmenu, TPM_RETURNCMD | TPM_RIGHTBUTTON,
                                       x, y, owner_hwnd, None)

        if cmd and extra_actions and cmd >= extra_base:
            idx = cmd - extra_base
            if 0 <= idx < len(extra_actions):
                label, cb = extra_actions[idx]
                cb()
        elif cmd:
            info = CMINVOKECOMMANDINFOEX()
            info.cbSize = ctypes.sizeof(CMINVOKECOMMANDINFOEX)
            info.fMask = CMIC_MASK_UNICODE
            info.hwnd = owner_hwnd
            offset = cmd - 1
            info.lpVerb = c_void_p(offset)
            info.lpVerbW = c_void_p(offset)
            info.nShow = SW_NORMAL
            _vt_call(ctx_ptr, 4, c_int, [POINTER(CMINVOKECOMMANDINFOEX)], byref(info))

        user32.DestroyMenu(hmenu)
        return True
    finally:
        if int(owner_hwnd) in _SUBCLASS_REFS:
            _unsubclass(owner_hwnd)
        for c in child_pidls:
            shell32.ILFree(c)
        _release(parent_folder)
        _release(icm)
        _release(icm2)
        _release(icm3)


def _release(pv):
    if pv:
        _vt_call(pv, 2, c_int, [], )
