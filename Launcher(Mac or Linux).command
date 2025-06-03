#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"         
VENV=".venv"                 
PY="$VENV/bin/python"

if [ ! -d "$VENV" ]; then
  echo "Creating isolated Python environment …"
  python3 -m venv "$VENV"
fi

echo "Ensuring required packages are installed …"
"$PY" -m pip install --upgrade pip >/dev/null
"$PY" -m pip install -r requirements.txt

echo "🚀  Launching QAutoEval …"
exec "$PY" "QAutoEval app.py" "$@"
