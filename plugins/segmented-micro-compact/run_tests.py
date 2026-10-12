#!/usr/bin/env python3
"""Run the plugin's behavioural tests and verify_tool_thread checks in a clean runner."""
import os
import subprocess
import sys

HERMES_SRC = "/usr/local/lib/hermes-agent"
HERMES_PYTHON = "/root/.hermes/tools/python-3.14.7+20260901-linux-arm64/bin/python3"
HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{HERMES_SRC}:{HERE}:{env.get('PYTHONPATH', '')}"

    py_bin = HERMES_PYTHON if os.path.isfile(HERMES_PYTHON) else sys.executable

    print("=== Running test_segmented_micro_compact.py ===")
    r1 = subprocess.run(
        [py_bin, "-c", "import hermes_bootstrap, sys; import test_segmented_micro_compact; sys.exit(test_segmented_micro_compact._run_all())"],
        cwd=HERE,
        env=env,
    ).returncode

    print("\n=== Running verify_tool_thread.py ===")
    r2 = subprocess.run(
        [py_bin, "-c", "import hermes_bootstrap, runpy; runpy.run_path('verify_tool_thread.py', run_name='__main__')"],
        cwd=HERE,
        env=env,
    ).returncode

    if r1 != 0 or r2 != 0:
        print(f"\nTests failed: segmented_micro_compact={r1}, tool_thread={r2}")
        return 1
    print("\nAll plugin test suites passed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
