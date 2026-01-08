#!/usr/bin/env bash

ROOT_DIR="/projectnb/vkolagrp/bellitti/adrd-foundation-model/adrd_simplified_evaluation/configs"

find "$ROOT_DIR" -type f \( -name "*.yml" -o -name "*.yaml" \) | while read -r file; do
    if grep -Eq '^[[:space:]]*run_readable_name:[[:space:]]*"(NACC-3B|NACC-3B-SCE)"([[:space:]]*#.*)?$' "$file"; then
        qsub ./run_benchmarks.sh \"$file\"
    fi
done