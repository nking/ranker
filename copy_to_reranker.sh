#!/usr/bin/env bash

#=========================================================================
#
#  after running write_reranker_parquet.py
#     run this script to copy files to the reranker project
#
#=========================================================================

# Exit immediately if a command exits with a non-zero status, treat unset variables as errors
set -euo pipefail

SRC_PATTERN="bin/reranker_*.parquet"
BASE_DEST="../reranker"
DEST_DIR="$BASE_DEST/src/test/resources/data"

echo "=== Reranker Parquet Copy Script ==="

# Check if the base ../reranker/ directory exists
if [ ! -d "$BASE_DEST" ]; then
    echo "Error: Destination root directory '$BASE_DEST/' does not exist. Aborting." >&2
    exit 1
fi

# Check if source files matching the pattern actually exist
shopt -s nullglob
src_files=( $SRC_PATTERN )
shopt -u nullglob

if [ ${#src_files[@]} -eq 0 ]; then
    echo "Error: No source files matching '$SRC_PATTERN' found. Aborting." >&2
    exit 1
fi

# Ensure the full destination directory path exists (creates it if missing)
mkdir -p "$DEST_DIR"

# Copy files safely with verbose output
echo "Found ${#src_files[@]} file(s). Copying to $DEST_DIR..."
cp -v "${src_files[@]}" "$DEST_DIR/"

SRC_PATTERN="bin/tiny3/reranker_*.parquet"
DEST_DIR="$BASE_DEST/src/test/resources/data/tiny3"
shopt -s nullglob
src_files=( $SRC_PATTERN )
shopt -u nullglob

if [ ${#src_files[@]} -eq 0 ]; then
    echo "Error: No source files matching '$SRC_PATTERN' found. Aborting." >&2
    exit 1
fi

mkdir -p "$DEST_DIR"

echo "Found ${#src_files[@]} file(s). Copying to $DEST_DIR..."
cp -v "${src_files[@]}" "$DEST_DIR/"

echo "All files copied successfully!"
