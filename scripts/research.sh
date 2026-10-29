#!/usr/bin/env bash
# Headless trading-research run: drives the `trading-research` skill through one agent runtime.
# Usage: scripts/research.sh TICKER [DATE] [--runtime claude|codex|omp] [--depth N]
set -euo pipefail

usage() {
  echo "usage: $0 TICKER [DATE] [--runtime claude|codex|omp] [--depth N]" >&2
  exit 2
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME=claude
DEPTH=1
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime) [[ $# -ge 2 ]] || usage; RUNTIME="$2"; shift 2 ;;
    --depth) [[ $# -ge 2 ]] || usage; DEPTH="$2"; shift 2 ;;
    -h|--help) usage ;;
    -*) echo "unknown option: $1" >&2; usage ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
[[ ${#POSITIONAL[@]} -ge 1 && ${#POSITIONAL[@]} -le 2 ]] || usage
TICKER="${POSITIONAL[0]}"
DATE="${POSITIONAL[1]:-$(date +%F)}"
[[ "$DEPTH" =~ ^[0-9]+$ && "$DEPTH" -ge 1 ]] || { echo "--depth must be a positive integer" >&2; exit 2; }
[[ "$DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || { echo "DATE must be YYYY-MM-DD" >&2; exit 2; }

command -v uv >/dev/null || { echo "uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }
command -v "$RUNTIME" >/dev/null || { echo "runtime CLI not found: $RUNTIME" >&2; exit 1; }

PROMPT="Use the trading-research skill. Ticker: ${TICKER}. Date: ${DATE}. Depth: ${DEPTH}. \
Run the full pipeline to completion and finish with the rating, price target and report path."

# Run dirs, reports, results and the memory log live under ~/.tradingagents by default.
DATA_HOME="$HOME/.tradingagents"
mkdir -p "$DATA_HOME"
cd "$ROOT"

case "$RUNTIME" in
  claude)
    # dontAsk: anything not allow-listed is denied, so the run cannot stall on a prompt.
    exec claude -p "$PROMPT" \
      --permission-mode dontAsk \
      --allowedTools "Bash" "Read" "Write" "Agent" "Task" "Skill" \
      --add-dir "$DATA_HOME"
    ;;
  codex)
    # Role subagents are nested `codex exec` processes: they need network (model API + data
    # vendors) and write access to the data home and to ~/.codex (auth, sessions).
    exec codex exec -C "$ROOT" -s workspace-write \
      -c sandbox_workspace_write.network_access=true \
      --add-dir "$DATA_HOME" --add-dir "$HOME/.codex" \
      -m gpt-6.1-sol "$PROMPT"
    ;;
  omp)
    exec omp -p --no-session --cwd "$ROOT" "$PROMPT"
    ;;
  *)
    echo "unknown runtime: $RUNTIME (expected claude|codex|omp)" >&2
    usage
    ;;
esac
