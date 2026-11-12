#!/usr/bin/env bash
# Prepara el entorno Python de la app DENTRO de la UNO Q (lado Linux).
# Idempotente: puede ejecutarse varias veces sin romper nada.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WITH_MODELS=0
WITH_ONNX=0

for arg in "$@"; do
  case "${arg}" in
    --with-model) WITH_ONNX=1 ;;
    --with-models) WITH_MODELS=1 ;;
    -h|--help)
      echo "Uso: $0 [--with-model] [--with-models]"
      echo "  --with-model   descarga el modelo propio UTKFace (ONNX)"
      echo "  --with-models  descarga además los modelos Caffe de fallback"
      exit 0
      ;;
    *)
      echo "Argumento desconocido: ${arg}" >&2
      exit 2
      ;;
  esac
done

cd "${PROJECT_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: no se encontró '${PYTHON_BIN}'." >&2
  exit 1
fi

echo "Python: $("${PYTHON_BIN}" --version 2>&1)"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo ">> Creando venv en ${VENV_DIR}"
  if ! "${PYTHON_BIN}" -m venv "${VENV_DIR}"; then
    echo "ERROR: no se pudo crear el venv." >&2
    echo "       En Debian instala python3-venv (requiere autorización):" >&2
    echo "         sudo apt-get install -y python3-venv python3-pip" >&2
    exit 1
  fi
else
  echo "== venv ya existe"
fi

echo ">> Actualizando pip"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel >/dev/null

echo ">> Instalando requirements.txt"
"${VENV_DIR}/bin/python" -m pip install -r "${PROJECT_ROOT}/requirements.txt"

if [ "${WITH_ONNX}" -eq 1 ]; then
  echo ">> Descargando modelo propio UTKFace (ONNX)"
  bash "${PROJECT_ROOT}/scripts/download_model.sh"
fi

if [ "${WITH_MODELS}" -eq 1 ]; then
  echo ">> Descargando modelos fallback"
  bash "${PROJECT_ROOT}/scripts/download_fallback_models.sh"
fi

if [ ! -f "${PROJECT_ROOT}/.env" ] && [ -f "${PROJECT_ROOT}/.env.example" ]; then
  cp "${PROJECT_ROOT}/.env.example" "${PROJECT_ROOT}/.env"
  echo "== .env creado a partir de .env.example"
fi

echo
echo "Entorno listo. Arranca la app con:"
echo "  ${PROJECT_ROOT}/scripts/run.sh"
