#!/usr/bin/env bash
set -u

OUT_DIR="${1:-data_dac/hypothesis_smoke/v261_rebuilt}"
MODEL="${OPENROUTER_AGENT_MODEL:-}"

if [[ -z "$MODEL" ]]; then
  echo "OPENROUTER_AGENT_MODEL is not set." >&2
  exit 1
fi
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is not set." >&2
  exit 1
fi

cases=(missing_mediator candidate alignment partial_absence weak_empty)
mkdir -p "$OUT_DIR/runs"

printf "%-18s %-8s\n" "CASE" "EXIT"
printf "%-18s %-8s\n" "------------------" "--------"

for case_name in "${cases[@]}"; do
  context="$OUT_DIR/${case_name}.context.json"
  prefix="$OUT_DIR/runs/${case_name}"
  if [[ ! -f "$context" ]]; then
    echo "Missing context: $context" >&2
    continue
  fi

  python -m scripts.run_hypothesis_maker \
    --context "$context" \
    --model "$MODEL" \
    --base-url "https://openrouter.ai/api/v1" \
    --api-key-env OPENROUTER_API_KEY \
    --output-prefix "$prefix" \
    --save-prompt
  rc=$?
  printf "%-18s %-8s\n" "$case_name" "$rc"
done
