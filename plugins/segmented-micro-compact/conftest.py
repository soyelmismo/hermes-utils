"""Pytest bootstrap: make the Hermes source tree importable so the engine's
subclass of ``agent.context_compressor`` resolves (hermes_yaml + agent.*)."""
import sys
from pathlib import Path
_HERMES_SRC = Path("/usr/local/lib/hermes-agent")
if str(_HERMES_SRC) not in sys.path:
    sys.path.append(str(_HERMES_SRC))
