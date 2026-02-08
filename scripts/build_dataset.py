#!/usr/bin/env python3

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import build_dataset_pipeline


def main() -> None:
    build_dataset_pipeline(write_panel_csv=True)


if __name__ == "__main__":
    main()
