#!/usr/bin/env bash
# One-liner launcher for the ETNA live demo.
#
# Usage:
#   ./run_demo.sh                 # starts on http://localhost:8501
#   PORT=8502 ./run_demo.sh       # override the port
#
set -euo pipefail

cd "$(dirname "$0")"

PORT="${PORT:-8501}"

# Pick python executable.
PY="${PYTHON:-python3}"

# Sanity check: ensure every runtime dependency is importable.
$PY - <<'PY'
import importlib, sys
REQUIRED = (
    "numpy", "scipy", "cv2", "torch", "torchvision", "kornia",
    "streamlit", "plotly",
)
missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
if missing:
    sys.stderr.write(
        "\nMissing packages: " + ", ".join(missing) + "\n"
        "Install with:  pip install -r requirements-demo.txt\n\n"
    )
    sys.exit(1)
PY

exec $PY -m streamlit run app.py \
    --server.port "$PORT" \
    --server.headless true \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false
