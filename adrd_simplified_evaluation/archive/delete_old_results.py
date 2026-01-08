#!/usr/bin/env python3

import shutil
from pathlib import Path

import yaml

ROOT = Path("/projectnb/vkolagrp/bellitti/adrd-foundation-model/adrd_simplified_evaluation/results")  # <-- set this
DRY_RUN = False               # <-- set to False to actually delete

TARGET_VALUES = {"NACC-3B", "NACC-3B-SCE"}


def should_delete_dir(dir_path: Path) -> bool:
    for yml in dir_path.glob("*.yml"):
        try:
            with yml.open() as f:
                data = yaml.safe_load(f)
        except Exception:
            continue

        if isinstance(data, dict) and data.get("run_readable_name") in TARGET_VALUES:
            return True

    return False


def main():
    # Walk bottom-up so deleting does not interfere with traversal
    for d in sorted(ROOT.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if not d.is_dir():
            continue

        if should_delete_dir(d):
            if DRY_RUN:
                print(f"[DRY RUN] Would delete: {d}")
            else:
                print(f"Deleting: {d}")
                shutil.rmtree(d)


if __name__ == "__main__":
    main()
