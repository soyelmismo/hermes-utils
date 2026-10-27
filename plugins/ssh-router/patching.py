import os
import sys
import inspect
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

def _resolve_core_run_job(fn):
    """Return the real ``cron.scheduler.run_job``, unwrapping stale ssh-router wrappers.

    A plugin reload re-runs ``register()`` while ``cs.run_job`` still points at the
    previous wrapper. Capturing that as the "original" nests one wrapper inside the
    next, and the stale inner signature then breaks every cron fire with TypeError.

    Following ``__ssh_router_original__`` alone is not enough: wrappers installed
    before that tag existed carry no attribute, and their closure is the only path
    back to the core function. Walk the closure graph instead and stop at the
    function that actually lives in ``cron.scheduler``.
    """
    stack = [fn]
    seen = set()
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        if _is_core_run_job(cur):
            return cur
        for attr in ("__ssh_router_original__", "__wrapped__"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                stack.append(nxt)
        for cell in getattr(cur, "__closure__", None) or ():
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if callable(val):
                stack.append(val)
    return fn


def _is_core_run_job(fn) -> bool:
    """True when ``fn`` is the scheduler's own ``run_job``, not an ssh-router wrapper."""
    code = getattr(fn, "__code__", None)
    if code is None:
        return False
    if not str(code.co_filename).replace("\\", "/").endswith("cron/scheduler.py"):
        return False
    if getattr(fn, "__name__", "") != "run_job":
        return False
    # Final guard: the real scheduler accepts the execution_id kwarg. Refusing to
    # return a narrow function is what keeps the TypeError from ever coming back.
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "execution_id" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


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
    # A plugin reload (control-socket `reload-plugins`) re-runs register() while
    # cs.run_job still points at the PREVIOUS wrapper. Capturing that as the
    # "original" nests a stale wrapper, and its narrow signature then breaks every
    # cron fire with TypeError. Unwrap to the real core function instead.
    original_run_job = _resolve_core_run_job(original_run_job)

    def _patched_run_job(job, **kwargs):
        # Forward every kwarg the core passes (execution_id, extra_prompt,
        # cancel_event, defer_agent_teardown). A narrow (job) signature breaks
        # every cron fire with TypeError once the scheduler grows a new kwarg.
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
            return original_run_job(job, **kwargs)
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

    # Tag the wrapper with the real core function so a later reload can unwrap.
    setattr(_patched_run_job, "__ssh_router_original__", original_run_job)
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
