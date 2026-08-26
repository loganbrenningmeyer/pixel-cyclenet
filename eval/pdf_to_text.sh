#!/usr/bin/env bash

# Recursively find all .pdf files under the given directory
# and create a corresponding .txt file containing the base64
# encoding of each PDF, preserving the same filename/location.
#
# Usage:
#   ./pdf_to_base64_txt.sh /path/to/root_directory
#
# Example:
#   ./pdf_to_base64_txt.sh ./documents
#
# Result:
#   report.pdf  -> report.txt
#   subdir/file.pdf -> subdir/file.txt

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/root_directory"
    exit 1
fi

ROOT_DIR="$1"

if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: Directory does not exist: $ROOT_DIR"
    exit 1
fi

find "$ROOT_DIR" -type f \( -iname "*.pdf" \) | while IFS= read -r pdf_file; do
    txt_file="${pdf_file%.*}.txt"

    echo "Converting:"
    echo "  PDF: $pdf_file"
    echo "  TXT: $txt_file"

    # GNU/Linux version:
    base64 "$pdf_file" > "$txt_file"

    # If you're on macOS and line wrapping causes issues,
    # you can use:
    # base64 -i "$pdf_file" > "$txt_file"
done

echo "Done."