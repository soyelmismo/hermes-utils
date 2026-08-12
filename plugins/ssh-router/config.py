import json
import logging
from pathlib import Path

logger = logging.getLogger("ssh-router")

STATE_DIR = Path.home() / ".hermes"
STATE_FILE = STATE_DIR / "ssh_target.json"
CONFIG_FILE = STATE_DIR / "config.yaml"

def _load_devices() -> dict:
    """Read device definitions from plugins.ssh-router.devices in config.yaml."""
    try:
        import yaml
        if CONFIG_FILE.exists():
            raw = yaml.safe_load(CONFIG_FILE.read_text())
            devices = (raw or {}).get("plugins", {}).get("ssh-router", {}).get("devices", {})
            if devices:
                logger.debug("Loaded %d devices from config.yaml", len(devices))
            return devices
    except Exception as e:
        logger.warning("Failed to load devices from config.yaml: %s", e)
    return {}

def _resolve_device(args: dict) -> dict:
    """Resolve connection parameters from either a named device or explicit args."""
    device_name = args.get("device", "")
    if device_name:
        devices = _load_devices()
        device = devices.get(device_name)
        if not device:
            raise ValueError(
                f"Unknown device '{device_name}'. Available: "
                + ", ".join(sorted(devices.keys()) or ["(none defined)"])
            )
        return {
            "host": device["host"],
            "user": device.get("user", "root"),
            "port": device.get("port", 22),
            "key": device.get("key", ""),
            "label": device_name,
        }

    host = args.get("host", "").strip()
    if not host:
        raise ValueError("Either 'device' or 'host' is required")
    return {
        "host": host,
        "user": args.get("user", "root").strip(),
        "port": args.get("port", 22),
        "key": args.get("key", "").strip(),
        "label": f"{args.get('user', 'root')}@{host}",
    }

def _list_devices() -> str:
    """Return a formatted list of defined devices for error messages/info."""
    devices = _load_devices()
    if not devices:
        return "(no devices configured in plugins.ssh-router.devices)"
    lines = []
    for name, cfg in sorted(devices.items()):
        user = cfg.get("user", "root")
        host = cfg["host"]
        port = cfg.get("port", 22)
        port_str = f":{port}" if port != 22 else ""
        lines.append(f"  • {name} → {user}@{host}{port_str}")
    return "\n".join(lines)

def _read_state() -> dict:
    """Read persisted SSH routing state."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception as e:
            logger.warning("Failed to read state file: %s", e)
    return {"mode": "local"}

def _write_state(state: dict) -> None:
    """Persist SSH routing state."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))

def get_remote_home(user: str) -> str:
    """Helper to determine remote home directory path"""
    return "/root" if user == "root" else f"/home/{user}"
