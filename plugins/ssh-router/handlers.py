import os
import json as _json
from datetime import datetime

from . import context as ctx
from .config import _resolve_device, _list_devices, _read_state, _write_state, STATE_FILE, _load_devices, get_remote_home
from .connection import _test_ssh, _clear_current_environment, set_ssh_env
from .liveness import _invalidate_liveness_cache
from .patching import _ensure_patch, _verify_patches

def handle_connect_to(args: dict, **kwargs) -> str:
    _ensure_patch()

    try:
        target = _resolve_device(args)
    except ValueError as e:
        return _json.dumps({
            "status": "error",
            "error": str(e),
            "available_devices": _list_devices(),
        })

    ok, msg = _test_ssh(
        target["host"], target["user"],
        target["key"], target["port"],
    )
    if not ok:
        return _json.dumps({"status": "error", "error": msg})

    _clear_current_environment()
    _invalidate_liveness_cache()

    _write_state({
        "mode": "ssh",
        "host": target["host"],
        "user": target["user"],
        "port": target["port"],
        "key": os.path.expanduser(target["key"]) if target["key"] else "",
        "label": target["label"],
        "connected_at": datetime.now().isoformat(),
        "local_cwd": os.getcwd(),
    })

    set_ssh_env(target["host"], target["user"], target["port"], target["key"])
    ctx._ssh_active.set(True)

    return _json.dumps({
        "status": "ok",
        "message": msg,
        "mode": "ssh",
        "target": f"{target['user']}@{target['host']}",
    })


def handle_disconnect(args: dict, **kwargs) -> str:
    state = _read_state()

    if state.get("mode") != "ssh":
        return _json.dumps({
            "status": "ok",
            "message": "Already in local mode",
            "mode": "local",
        })

    target = f"{state.get('user', '')}@{state.get('host', 'unknown')}"

    _clear_current_environment()
    _invalidate_liveness_cache()

    ctx._ssh_active.set(False)

    from .connection import clear_ssh_env
    clear_ssh_env()

    saved_local_cwd = state.get("local_cwd", "")
    if saved_local_cwd:
        try:
            os.chdir(saved_local_cwd)
        except OSError:
            pass

    try:
        STATE_FILE.unlink(missing_ok=True)
    except OSError:
        pass

    return _json.dumps({
        "status": "ok",
        "message": f"Disconnected from {target}. Back to local.",
        "mode": "local",
    })

def handle_status(args: dict, **kwargs) -> str:
    _ensure_patch()
    state = _read_state()
    mode = state.get("mode", "local")

    active_mode = "ssh" if (mode == "ssh" and ctx._ssh_active.get()) else "local"
    info = {"status": "ok", "mode": active_mode, "persisted_mode": mode}

    if active_mode == "ssh":
        info["target"] = f"{state.get('user', '')}@{state.get('host', '')}:{state.get('port', 22)}"
        info["label"] = state.get("label", "")
        info["connected_at"] = state.get("connected_at", "")
    else:
        info["target"] = "local"

    devices = _load_devices()
    if devices:
        info["available_devices"] = list(devices.keys())

    patch_warnings = _verify_patches()
    if patch_warnings:
        info["patch_warnings"] = patch_warnings
        info["status"] = "degraded"

    return _json.dumps(info)
