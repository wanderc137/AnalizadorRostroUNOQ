#!/usr/bin/env bash
# Arranca el servidor de AnalizadorRostroUNOQ desde la raíz del proyecto.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi

if [ -z "${PYTHON_BIN}" ]; then
  echo "ERROR: no hay python3 ni .venv. Ejecuta scripts/setup_board.sh." >&2
  exit 1
fi

echo ">> Iniciando AnalizadorRostroUNOQ con ${PYTHON_BIN}"
exec "${PYTHON_BIN}" -m app.main
