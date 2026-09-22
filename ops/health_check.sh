#!/usr/bin/env bash
# W35: one-screen health status for the DGX Spark (IDDSI-Open). Runs ON Spark as tun.
# Read-only. No parameters. Sections are delimited by "=== SECTION: <name> ==="
# and machine lines start with a TAB-separated keyword so ops/parse_health.py can
# parse this output. Lines starting with "#" are human commentary only.
#
# set -u (not -e): a failing section must degrade to "[unavailable: reason]",
# never kill the whole screen.
set -u

IDDSI="${HOME}/iddsi"
HUB="${IDDSI}/hf/hub"

section() { printf '=== SECTION: %s ===\n' "$1"; }

# ---------------------------------------------------------------- 1. containers
section containers
if command -v docker >/dev/null 2>&1; then
  if docker ps -a --format '{{.Names}}\t{{.Status}}' 2>/dev/null \
      | awk -F'\t' '{printf "CONTAINER\t%s\t%s\n", $1, $2}'; then
    if [ -z "$(docker ps -a --format '{{.Names}}' 2>/dev/null)" ]; then
      printf '#   (no containers listed)\n'
    fi
  else
    printf '[unavailable: docker ps failed (daemon down or permission?)\n'
  fi
else
  printf '[unavailable: docker not installed]\n'
fi

# ---------------------------------------------------------------- 2. models
section models
# name|expected_bytes (expected sizes are approximations from the WP spec:
# resnet18 ~45 MB, siglip2-base ~900 MB, Qwen3-VL-2B ~5 GB, FLUX.1-schnell ~23 GB)
EXPECTED_MODELS="
models--timm--resnet18.a1_in1k|45000000
models--google--siglip2-base-patch16-224|900000000
models--Qwen--Qwen3-VL-2B-Instruct|4000000000
models--black-forest-labs--FLUX.1-schnell|23000000000
"
if [ -d "$HUB" ]; then
  printf '#   hub root: %s\n' "$HUB"
  while IFS='|' read -r mname mexpected; do
    [ -z "$mname" ] && continue
    mdir="${HUB}/${mname}"
    if [ ! -d "$mdir" ]; then
      printf 'MODEL\t%s\t0\tMISSING\n' "$mname"
      continue
    fi
    mbytes="$(du -sb "$mdir" 2>/dev/null | awk '{print $1}')"
    [ -z "${mbytes:-}" ] && mbytes=0
    if [ "$mbytes" -ge "$mexpected" ]; then
      mstatus=PRESENT
    else
      mstatus=PARTIAL
    fi
    mhuman="$(du -sh "$mdir" 2>/dev/null | awk '{print $1}')"
    printf 'MODEL\t%s\t%s\t%s\n' "$mname" "$mbytes" "$mstatus"
    printf '#   %s -> %s (expected >= %s bytes)\n' "$mname" "$mhuman" "$mexpected"
  done <<EOF
$EXPECTED_MODELS
EOF

  # FLUX.1-schnell specifics: flag the big safetensors file if present and >= 23 GB.
  flux_dir="$HOME/iddsi/models_local/FLUX.1-schnell"
  flux_file="$(find "$flux_dir" -type f -name '*.safetensors' -size +23000000000c 2>/dev/null | head -1)"
  if [ -n "$flux_file" ]; then
    flux_bytes="$(stat -c %s "$flux_file" 2>/dev/null || echo 0)"
    printf 'FLUX_SAFETENSORS\tFOUND\t%s\n' "$flux_bytes"
    printf '#   FLUX.1-schnell safetensors OK: %s (%s bytes)\n' "$flux_file" "$flux_bytes"
  else
    printf 'FLUX_SAFETENSORS\tMISSING\t0\n'
    printf '#   WARNING: no FLUX.1-schnell *.safetensors >= 23,000,000,000 bytes found\n'
  fi
  # leftover download temp files (e.g. .flux1-schnell.safetensors.part / .incomplete)
  flux_temps="$(find "$HUB" -type f -name '.flux1-schnell.safetensors.*' 2>/dev/null)"
  flux_temp_count="$(printf '%s' "$flux_temps" | grep -c . || true)"
  printf 'FLUX_TEMP_FILES\t%s\n' "${flux_temp_count:-0}"
  if [ "${flux_temp_count:-0}" -gt 0 ]; then
    printf '%s\n' "$flux_temps" | sed 's/^/#   leftover: /'
    printf '#   WARNING: leftover FLUX download temp files - see ops/cleanup_rsync_temps.sh\n'
  fi
else
  printf '[unavailable: hub dir %s not found]\n' "$HUB"
fi

# ---------------------------------------------------------------- 3. gpu
section gpu
printf '#   GB10 (GB200-class) unified-memory Spark: GPU MIG memory query may be N/A;\n'
printf '#   host RAM below is the real memory budget.\n'
if command -v nvidia-smi >/dev/null 2>&1; then
  gpumem="$(nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' | sed 's/\[//g; s/\]//g')"
  if [ -n "${gpumem:-}" ]; then
    printf 'GPU_MEM\t%s\n' "$gpumem"
  else
    printf 'GPU_MEM\tunavailable\n'
  fi
  printf '#   top GPU consumers:\n'
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '{printf "GPU_APP\t%s\t%s\t%s\n", $1, $2, $3}'
else
  printf 'GPU_MEM\tunavailable\n'
  printf '[unavailable: nvidia-smi not found]\n'
fi
if command -v free >/dev/null 2>&1; then
  free -h 2>/dev/null | awk '/^Mem:/{printf "HOST_RAM\t%s\t%s\t%s\t%s\t%s\t%s\n",$2,$3,$4,$5,$6,$7}'
else
  printf 'HOST_RAM\tunavailable\n'
fi

# ---------------------------------------------------------------- 4. heartbeat
section heartbeat
newest_log=""
newest_mtime=0
shopt -s nullglob
for f in "${IDDSI}/night_pilot.log" "${IDDSI}"/logs/*.log "${IDDSI}"/*.log; do
  [ -f "$f" ] || continue
  mt="$(stat -c %Y "$f" 2>/dev/null || echo 0)"
  if [ "$mt" -gt "$newest_mtime" ]; then
    newest_mtime="$mt"
    newest_log="$f"
  fi
done
shopt -u nullglob
if [ -n "$newest_log" ]; then
  now="$(date +%s)"
  age_min=$(( (now - newest_mtime) / 60 ))
  printf 'HEARTBEAT\t%s\t%s\n' "$newest_log" "$age_min"
  printf '#   newest log %s modified %s min ago\n' "$newest_log" "$age_min"
else
  printf 'HEARTBEAT\tnone\n'
  printf '#   no heartbeat logs found under %s/*.log or %s/logs/\n' "$IDDSI" "$IDDSI"
fi

# ---------------------------------------------------------------- 5. night summary
section night_summary
latest_night="$(ls -t "${IDDSI}"/reports/NIGHT_*.md 2>/dev/null | head -1 || true)"
if [ -n "$latest_night" ]; then
  printf 'NIGHT_SUMMARY\t%s\n' "$(basename "$latest_night")"
  printf '#   first lines of %s:\n' "$latest_night"
  head -10 "$latest_night" 2>/dev/null | sed 's/^/#   /'
else
  printf 'NIGHT_SUMMARY\tnone\n'
  printf '#   no %s/reports/NIGHT_*.md found\n' "$IDDSI"
fi

printf '=== END ===\n'
