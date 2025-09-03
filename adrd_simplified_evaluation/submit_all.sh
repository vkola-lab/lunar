#!/bin/bash -l 

# Run benchmarks on multiple models. This script will read all config files (*.yml)
# in the specified directories, and pass them one by one to run_benchmarks.sh
#
# Usage: ./submit_all.sh dir1 dir2 ...

# Check if at least one directory is passed
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 dir1 [dir2 ...]"
    exit 1
fi

# Loop over all specified directories
for DIR in "$@"; do
    # Find all .yml files in the directory (non-recursive)
    for FILE in "$DIR"/*.yml; do
        # Check if the glob matched any files
        if [ -e "$FILE" ]; then
            echo "Submitting: $FILE"
            qsub run_benchmarks.sh "$FILE"
        fi
    done
done
