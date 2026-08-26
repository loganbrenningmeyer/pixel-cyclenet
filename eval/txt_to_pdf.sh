#!/usr/bin/env bash

# Recursively find all .txt files under the given directory
# and create a corresponding .pdf file by base64-decoding
# each TXT file, preserving the same filename/location.
#
# macOS version using:
#   base64 -D -i input.txt -o output.pdf
#
# Usage:
#   ./base64_txt_to_pdf.sh /path/to/root_directory
#
# Example:
#   ./base64_txt_to_pdf.sh ./documents
#
# Result:
#   report.txt  -> report.pdf
#   subdir/file.txt -> subdir/file.pdf

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

find "$ROOT_DIR" -type f \( -iname "*.txt" \) | while IFS= read -r txt_file; do
    pdf_file="${txt_file%.*}.pdf"

    echo "Converting:"
    echo "  TXT: $txt_file"
    echo "  PDF: $pdf_file"

    # macOS base64 decode
    base64 -d "$txt_file" > "$pdf_file"
done

echo "Done."