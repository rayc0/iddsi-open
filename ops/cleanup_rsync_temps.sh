#!/usr/bin/env bash
# Safely remove rsync temporary files for FLUX.1-schnell.
#
# The script scans FLUX_DIR for files named
# .flux1-schnell.safetensors.XXXXXX. It deletes them only when --apply is
# supplied and the completed flux1-schnell.safetensors in the same directory
# is at least 23,000,000,000 bytes.

set -euo pipefail

readonly MIN_FINAL_BYTES=23000000000
readonly DEFAULT_FLUX_DIR="${HOME}/iddsi/models_local/FLUX.1-schnell"
FLUX_DIR="${FLUX_DIR:-$DEFAULT_FLUX_DIR}"
APPLY=0
SKIPPED=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [--apply | --dry-run]

Scan FLUX_DIR for .flux1-schnell.safetensors.XXXXXX temporary files.
FLUX_DIR defaults to: $DEFAULT_FLUX_DIR

  --dry-run  print files that would be deleted (default)
  --apply    delete files whose completed sibling is at least
             $MIN_FINAL_BYTES bytes
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --apply)
            APPLY=1
            ;;
        --dry-run)
            APPLY=0
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

if [ ! -d "$FLUX_DIR" ]; then
    echo "ERROR: FLUX_DIR does not exist: $FLUX_DIR" >&2
    exit 1
fi

if [ "$APPLY" -eq 1 ]; then
    MODE="APPLY (will delete verified temps)"
else
    MODE="DRY-RUN (no deletions; pass --apply to delete)"
fi
echo "cleanup_rsync_temps: mode=$MODE target=$FLUX_DIR min_final_bytes=$MIN_FINAL_BYTES"

# Return a file size in bytes on GNU and BSD/macOS systems.
file_size() {
    stat -c %s "$1" 2>/dev/null || stat -f %z "$1" 2>/dev/null || echo 0
}

# Print the verified final-file description and succeed only at the size floor.
final_file_ok() {
    local directory=$1
    local final_file="$directory/flux1-schnell.safetensors"
    local size

    [ -f "$final_file" ] || return 1
    size=$(file_size "$final_file")
    [ "$size" -ge "$MIN_FINAL_BYTES" ] || return 1
    printf '%s (%s bytes)\n' "$final_file" "$size"
}

FOUND=0
while IFS= read -r -d '' temp_file; do
    FOUND=1
    directory=$(dirname "$temp_file")
    temp_size=$(file_size "$temp_file")

    if verified_final=$(final_file_ok "$directory"); then
        if [ "$APPLY" -eq 1 ]; then
            rm -f -- "$temp_file"
            echo "DELETED  $temp_file ($temp_size bytes) — final verified: $verified_final"
        else
            echo "WOULD-DELETE  $temp_file ($temp_size bytes) — final verified: $verified_final"
        fi
    else
        SKIPPED=$((SKIPPED + 1))
        echo "SKIP  $temp_file ($temp_size bytes) — WARNING: flux1-schnell.safetensors is missing or smaller than $MIN_FINAL_BYTES bytes in $directory; leaving temp in place" >&2
    fi
done < <(find "$FLUX_DIR" -type f -name '.flux1-schnell.safetensors.?*' -print0 2>/dev/null)

if [ "$FOUND" -eq 0 ]; then
    echo "No .flux1-schnell.safetensors.* temp files found under $FLUX_DIR — nothing to do."
fi

if [ "$SKIPPED" -gt 0 ]; then
    echo "WARNING: $SKIPPED temp file(s) skipped by safety check." >&2
    exit 2
fi

exit 0
