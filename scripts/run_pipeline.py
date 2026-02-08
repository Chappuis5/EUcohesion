#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    build_script = project_root / "scripts" / "build_dataset.py"

    subprocess.run(
        [sys.executable, str(build_script)],
        cwd=project_root,
        check=True,
    )


if __name__ == "__main__":
    main()
