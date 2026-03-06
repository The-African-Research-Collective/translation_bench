#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${1:-$ROOT_DIR/configs}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

if [[ ! -d "$CONFIG_DIR" ]]; then
  echo "Config directory not found: $CONFIG_DIR" >&2
  exit 1
fi

shopt -s nullglob
configs=("$CONFIG_DIR"/*.yaml "$CONFIG_DIR"/*.yml)
shopt -u nullglob

if [[ ${#configs[@]} -eq 0 ]]; then
  echo "No YAML config files found in: $CONFIG_DIR" >&2
  exit 1
fi

IFS=$'\n' configs=($(printf '%s\n' "${configs[@]}" | sort))
unset IFS

if command -v translation-pipeline >/dev/null 2>&1; then
  RUN_CMD=(translation-pipeline)
elif [[ -x "$ROOT_DIR/.venv/bin/translation-pipeline" ]]; then
  RUN_CMD=("$ROOT_DIR/.venv/bin/translation-pipeline")
else
  RUN_CMD=(python -m translation_bench.pipeline)
fi

echo "Runner: ${RUN_CMD[*]}"
echo "Config directory: $CONFIG_DIR"
echo "Found ${#configs[@]} config file(s)"
echo

ok=0
failed=0
failed_files=()

for cfg in "${configs[@]}"; do
  echo "============================================================"
  echo "Running: $cfg"
  echo "============================================================"

  if "${RUN_CMD[@]}" "$cfg"; then
    echo "Status: OK ($cfg)"
    ok=$((ok + 1))
  else
    echo "Status: FAILED ($cfg)" >&2
    failed=$((failed + 1))
    failed_files+=("$cfg")

    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      break
    fi
  fi

  echo
done

echo "============================== SUMMARY =============================="
echo "Succeeded: $ok"
echo "Failed:    $failed"

if [[ $failed -gt 0 ]]; then
  echo "Failed configs:"
  for cfg in "${failed_files[@]}"; do
    echo "  - $cfg"
  done
  exit 1
fi

echo "All configs completed successfully."
