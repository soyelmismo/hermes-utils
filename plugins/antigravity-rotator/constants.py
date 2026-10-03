"""Shared constants for antigravity-rotator."""

from pathlib import Path

# OAuth token file names in priority order
TOKEN_FILENAMES = ("jetski-standalone-oauth-token", "antigravity-oauth-token")

# Google accounts state filename
GOOGLE_ACCOUNTS_FILE = "google_accounts.json"

# DirectSDK isolated workspace search patterns under /tmp
ISOLATED_TMP_ROOT = Path("/tmp")
ISOLATED_HOME_GLOBS = ("agy_*", "hermes_agy_*")

# OpenProxy defaults
DEFAULT_API_URL = "http://localhost:8787/admin/api"
DEFAULT_SWITCH_SCRIPT_PATH = Path("/root/switch_account.sh")
SWITCH_SCRIPT_ENV_VAR = "ANTIGRAVITY_ROTATOR_SWITCH_SCRIPT"
