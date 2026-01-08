from pathlib import Path
import yaml
import re

root = Path("/projectnb/vkolagrp/bellitti/adrd-foundation-model/adrd_simplified_evaluation/configs")  # top-level directory
dry_run = False              # set to False to actually rename

def sanitize(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\-_. ]", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name

for yml_path in root.rglob("*.yml"):
    with open(yml_path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "run_readable_name" not in data:
        continue

    new_name = sanitize(str(data["run_readable_name"])) + ".yml"
    new_path = yml_path.with_name(new_name)

    if new_path.exists() and new_path != yml_path:
        print(f"SKIP (exists): {yml_path} -> {new_path}")
        continue

    if yml_path == new_path:
        continue

    if dry_run:
        print(f"DRY-RUN: {yml_path} -> {new_path}")
    else:
        print(f"RENAME: {yml_path} -> {new_path}")
        yml_path.rename(new_path)