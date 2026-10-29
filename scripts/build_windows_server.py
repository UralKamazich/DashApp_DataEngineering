"""Build the self-contained Python sidecar used by packaged Windows Electron."""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "python-build"
OUTPUT = ROOT / "python-dist"


def main() -> int:
    if sys.platform != "win32":
        raise SystemExit("Windows server runtime must be built on a Windows runner.")

    for directory in (WORK, OUTPUT):
        if directory.exists():
            shutil.rmtree(directory)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        "dataanalize-server",
        "--distpath",
        str(OUTPUT),
        "--workpath",
        str(WORK),
        "--specpath",
        str(WORK),
        "--paths",
        str(ROOT),
        "--add-data",
        f"{ROOT / 'assets'}{os.pathsep}assets",
        "--collect-all",
        "dash",
        "--collect-all",
        "dash_mantine_components",
        "--collect-all",
        "dash_iconify",
        "--collect-all",
        "plotly",
        "--collect-all",
        "catboost",
        "--collect-all",
        "sklearn",
        "--collect-all",
        "torch",
        str(ROOT / "run_server.py"),
    ]
    subprocess.run(command, cwd=ROOT, check=True)

    executable = OUTPUT / "dataanalize-server" / "dataanalize-server.exe"
    if not executable.is_file():
        raise SystemExit(f"PyInstaller did not create {executable}")
    print(f"Windows Python server ready: {executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
