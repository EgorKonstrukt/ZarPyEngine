# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Callable, Any
from core.ecs.ecs import Entity


_RPC_REGISTRY: dict[str, Callable] = {}


def server_rpc(func: Callable) -> Callable:
    func._rpc_type = "server"
    _RPC_REGISTRY[func.__name__] = func
    return func


def client_rpc(func: Callable) -> Callable:
    func._rpc_type = "client"
    _RPC_REGISTRY[func.__name__] = func
    return func


def target_rpc(func: Callable) -> Callable:
    func._rpc_type = "target"
    _RPC_REGISTRY[func.__name__] = func
    return func


def invoke_rpc(entity: Entity, method: str, args: dict, sender: int = -1):
    if entity is None or not method:
        return
    for comp in entity.get_all_components():
        fn = getattr(comp, method, None)
        if callable(fn):
            try:
                if isinstance(args, dict):
                    fn(args)
                else:
                    fn()
                return
            except Exception:
                try:
                    fn()
                    return
                except Exception:
                    pass
        alt = getattr(comp, f"rpc_{method}", None)
        if callable(alt):
            try:
                if isinstance(args, dict):
                    alt(**args)
                else:
                    alt(args)
                return
            except Exception:
                pass


def send_server_rpc(net_id: int, method: str, args: dict | None = None):
    try:
        from core.network.transport import get_transport
        from core.network.protocol import MessageType
        get_transport().broadcast(MessageType.NET_RPC, {"net_id": int(net_id), "method": str(method), "args": dict(args) if args else {}})
    except Exception:
        pass


def send_client_rpc(net_id: int, method: str, args: dict | None = None):
    try:
        from core.network.transport import get_transport
        from core.network.protocol import MessageType
        get_transport().broadcast(MessageType.NET_RPC, {"net_id": int(net_id), "method": str(method), "args": dict(args) if args else {}})
    except Exception:
        pass


def send_target_rpc(peer_id: int, net_id: int, method: str, args: dict | None = None):
    try:
        from core.network.transport import get_transport
        from core.network.protocol import MessageType
        get_transport().send_to(int(peer_id), MessageType.NET_RPC, {"net_id": int(net_id), "method": str(method), "args": dict(args) if args else {}, "target": int(peer_id)})
    except Exception:
        pass
