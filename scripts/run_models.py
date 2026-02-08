#!/usr/bin/env python3

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import run_models_pipeline


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    outputs = run_models_pipeline()
    for name, path in sorted(outputs.items()):
        print(f"[run_models] {name}: {path}")


if __name__ == "__main__":
    main()
