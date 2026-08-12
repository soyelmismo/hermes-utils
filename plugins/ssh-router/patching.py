import os
import sys
import logging

from . import context as ctx
from .config import _load_devices, _resolve_device, _read_state, get_remote_home
from .connection import _test_ssh, set_ssh_env, clear_ssh_env

logger = logging.getLogger("ssh-router")

def _patched_get_env_config():
    result = ctx._original_get_env_config()

    with ctx._cron_ssh_lock:
        cron_active = ctx._cron_ssh_counter > 0
    ssh_ctx = ctx._ssh_active.get()

    log_extra = ""
    if cron_active or ssh_ctx:
        cron_tid = ctx._cron_task_id.get()
        if cron_tid:
            env_t = ctx._get_task_ssh_env(cron_tid)
            result["env_type"] = "ssh"
            result["ssh_host"] = env_t.get("TERMINAL_SSH_HOST", "")
            result["ssh_user"] = env_t.get("TERMINAL_SSH_USER", "")
            result["ssh_port"] = int(env_t.get("TERMINAL_SSH_PORT", "22"))
            result["ssh_key"] = env_t.get("TERMINAL_SSH_KEY", "")
            log_extra = "CRON_SSH"
        elif ssh_ctx:
            result["env_type"] = "ssh"
            result["ssh_host"] = os.environ.get("TERMINAL_SSH_HOST", "")
            result["ssh_user"] = os.environ.get("TERMINAL_SSH_USER", "")
            result["ssh_port"] = int(os.environ.get("TERMINAL_SSH_PORT", "22"))
            result["ssh_key"] = os.environ.get("TERMINAL_SSH_KEY", "")
            log_extra = "MAIN_SSH"
        else:
            result["env_type"] = "local"
            result["ssh_host"] = ""
            result["ssh_user"] = ""
            result["ssh_port"] = 22
            result["ssh_key"] = ""
            log_extra = "CRON_RUNNING_BUT_LOCAL"
    elif result.get("env_type") == "ssh":
        result["env_type"] = "local"
        result["ssh_host"] = ""
        result["ssh_user"] = ""
        result["ssh_port"] = 22
        result["ssh_key"] = ""
        cwd = result.get("cwd", "")
        ssh_from_env = (
            os.environ.get("TERMINAL_ENV") == "ssh"
            or bool(os.environ.get("TERMINAL_SSH_HOST"))
        )
        if cwd and ssh_from_env and (
            cwd.startswith("/root") or cwd.startswith("/home/")
        ):
            result["cwd"] = os.getcwd()
        log_extra = "STALE_SSH→LOCAL"

    logger.debug(
        "_patched_get_env_config: env_type=%s counter=%d ssh_ctx=%s %s cwd=%s",
        result.get("env_type"), ctx._cron_ssh_counter, ssh_ctx, log_extra,
        result.get("cwd"),
    )
    return result

def _ensure_patch():
    if ctx._patch_applied:
        return

    ttt = None
    try:
        import tools.terminal_tool as ttt
    except ImportError:
        ttt = sys.modules.get("tools.terminal_tool")

    if ttt is None:
        logger.debug("_ensure_patch: tools.terminal_tool not available yet, will retry")
        return

    if ctx._original_get_env_config is None:
        ctx._original_get_env_config = ttt._get_env_config

    ttt._get_env_config = _patched_get_env_config

    if not hasattr(ttt._resolve_container_task_id, "__patched__"):
        _orig_resolve = ttt._resolve_container_task_id
        def _patched_resolve(task_id):
            cron_tid = ctx._cron_task_id.get()
            if cron_tid:
                return cron_tid
            return _orig_resolve(task_id)
        _patched_resolve.__patched__ = True
        ttt._resolve_container_task_id = _patched_resolve

    ctx._patch_applied = True
    logger.info("Monkey-patch applied to _get_env_config + _resolve_container_task_id")

def _patch_cron_scheduler():
    if ctx._cron_scheduler_patched:
        logger.debug("ssh-router: cron scheduler already patched — skipping")
        return

    try:
        import cron.scheduler as cs
    except ImportError:
        logger.warning("ssh-router: cron.scheduler not importable — auto-SSH for crons disabled")
        return
    ctx._cron_scheduler_patched = True

    original_run_job = cs.run_job

    def _patched_run_job(job):
        job_name = str(job.get("name") or job.get("prompt") or job.get("id", ""))
        device = None
        _cron_task_token = None
        _ssh_configured = False

        if ":" in job_name:
            prefix = job_name.split(":", 1)[0].strip().lower()
            devices = _load_devices()
            if prefix in devices:
                device = prefix

        if device:
            try:
                target = _resolve_device({"device": device})
                _ensure_patch()
                ok, msg = _test_ssh(
                    target["host"], target["user"],
                    target["key"], target["port"],
                )
                if ok:
                    set_ssh_env(target["host"], target["user"], target["port"], target["key"])
                    
                    tid = f"cron_{job.get('id', 'unknown')}"
                    remote_home = get_remote_home(target["user"])
                    ctx._set_task_ssh_env(tid, {
                        "TERMINAL_SSH_HOST": target["host"],
                        "TERMINAL_SSH_USER": target["user"],
                        "TERMINAL_SSH_PORT": str(target["port"]),
                        "TERMINAL_SSH_KEY": os.path.expanduser(target["key"]) if target["key"] else "",
                        "TERMINAL_CWD": remote_home,
                    })

                    with ctx._cron_ssh_lock:
                        ctx._cron_ssh_counter += 1
                        _counter_after = ctx._cron_ssh_counter
                    _ssh_configured = True

                    _cron_task_token = ctx._cron_task_id.set(tid)
                    logger.debug("Cron '%s': set cron task_id=%s", job_name, ctx._cron_task_id.get())
                    logger.info("Cron '%s': auto-connected to '%s' (counter=%d, module_id=%s)", job_name, device, _counter_after, id(__name__))
                else:
                    logger.error("Cron '%s': SSH FAILED for '%s' — skipping agent run: %s", job_name, device, msg)
                    doc = (f"# Cron Job: {job_name}\n\n**Status:** SKIPPED — SSH connection failed\n**Device:** {device}\n**Error:** {msg}\n\n")
                    return False, doc, "", msg
            except Exception as e:
                logger.error("Cron '%s': SSH FAILED for '%s' — skipping agent run: %s", job_name, device, e)
                doc = (f"# Cron Job: {job_name}\n\n**Status:** SKIPPED — SSH connection failed\n**Device:** {device}\n**Error:** {e}\n\n")
                return False, doc, "", str(e)

        try:
            return original_run_job(job)
        finally:
            if device:
                if _cron_task_token is not None:
                    try:
                        ctx._cron_task_id.reset(_cron_task_token)
                    except ValueError:
                        pass

                remaining = ctx._cron_ssh_counter
                if _ssh_configured:
                    with ctx._cron_ssh_lock:
                        ctx._cron_ssh_counter -= 1
                        remaining = ctx._cron_ssh_counter
                main_connected = _read_state().get("mode") == "ssh"
                ctx._clear_task_ssh_env(f"cron_{job.get('id', 'unknown')}")
                if remaining == 0 and not main_connected:
                    clear_ssh_env()
                logger.info("Cron '%s': disconnected from '%s'", job_name, device)

    cs.run_job = _patched_run_job
    logger.info("Cron scheduler patched for auto-SSH routing via job name prefix")


def _verify_patches() -> list[str]:
    warnings = []
    try:
        import tools.terminal_tool as ttt
        fn = ttt._get_env_config
        if not hasattr(fn, "__wrapped__"):
            if fn.__code__.co_name != "_patched_get_env_config":
                warnings.append("terminal _get_env_config patch MISSING (name mismatch)")
        else:
            warnings.append("terminal _get_env_config patch MISSING (wrapped)")
    except Exception as e:
        warnings.append(f"terminal _get_env_config patch check failed: {e}")

    try:
        import cron.scheduler as cs2
        if getattr(cs2.run_job, "__name__", "") != "_patched_run_job":
            warnings.append("cron scheduler run_job patch MISSING")
    except Exception as e:
        warnings.append(f"cron scheduler patch check failed: {e}")

    return warnings
