#!/usr/bin/env bash
# Descarga el modelo propio UTKFace (ONNX) desde el release de GitHub.
#
# Uso:
#   ./scripts/download_model.sh              # descarga a models/utkface_mobilenetv2.onnx
#   ./scripts/download_model.sh --force      # fuerza la descarga aunque exista
#   MODEL_URL=... ./scripts/download_model.sh
#
# El .onnx no se versiona en git (ver .gitignore): se distribuye como asset del
# release v1.0.0 y se valida con SHA256 antes de instalarlo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST="${PROJECT_ROOT}/models/utkface_mobilenetv2.onnx"
MODEL_URL="${MODEL_URL:-https://github.com/wanderc137/AnalizadorRostroUNOQ/releases/download/v1.0.0/utkface_mobilenetv2.onnx}"
MODEL_SHA256="${MODEL_SHA256:-390ade0e1f237616aca4636976f3223f1b8396e4bb539c27a4b55848a30c1661}"
MIN_BYTES=9000000
FORCE=0

for arg in "$@"; do
  case "${arg}" in
    --force) FORCE=1 ;;
    -h|--help)
      echo "Uso: $0 [--force]"
      echo "  --force  vuelve a descargar el modelo aunque ya exista"
      exit 0
      ;;
    *)
      echo "Argumento desconocido: ${arg}" >&2
      exit 2
      ;;
  esac
done

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: se requiere 'curl' para descargar el modelo." >&2
  exit 1
fi

sha256_of() {
  local file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${file}" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${file}" | awk '{print $1}'
  elif command -v python3 >/dev/null 2>&1; then
    python3 - "${file}" <<'PY'
import hashlib
import sys

h = hashlib.sha256()
with open(sys.argv[1], "rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
        h.update(chunk)
print(h.hexdigest())
PY
  else
    echo "ERROR: no hay sha256sum, shasum ni python3 para verificar el modelo." >&2
    return 1
  fi
}

verify_file() {
  local file="$1"
  if [ ! -s "${file}" ]; then
    return 1
  fi
  local size
  size="$(wc -c <"${file}" | tr -d ' ')"
  if [ "${size}" -lt "${MIN_BYTES}" ]; then
    echo "ERROR: ${file} mide ${size} bytes; se esperaban al menos ${MIN_BYTES}." >&2
    return 1
  fi
  local actual
  actual="$(sha256_of "${file}")"
  if [ "${actual}" != "${MODEL_SHA256}" ]; then
    echo "ERROR: SHA256 de ${file} no coincide." >&2
    echo "  esperado: ${MODEL_SHA256}" >&2
    echo "  obtenido: ${actual}" >&2
    return 1
  fi
  return 0
}

mkdir -p "$(dirname "${DEST}")"

if [ "${FORCE}" -eq 0 ] && verify_file "${DEST}" 2>/dev/null; then
  echo "== Modelo ya presente y verificado: ${DEST}"
  echo "   SHA256: ${MODEL_SHA256}"
  exit 0
fi

TMP="${DEST}.part"
rm -f "${TMP}"
echo ">> Descargando modelo UTKFace"
echo "   URL:  ${MODEL_URL}"
echo "   Dest: ${DEST}"
if ! curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 --max-time 900 \
  -o "${TMP}" "${MODEL_URL}"; then
  rm -f "${TMP}"
  echo "ERROR: no se pudo descargar el modelo." >&2
  echo "       Verifica el release: https://github.com/wanderc137/AnalizadorRostroUNOQ/releases" >&2
  exit 1
fi

if ! verify_file "${TMP}"; then
  rm -f "${TMP}"
  exit 1
fi

mv "${TMP}" "${DEST}"
chmod 644 "${DEST}"
echo "OK: modelo instalado en ${DEST}"
echo "    Tamaño: $(wc -c <"${DEST}" | tr -d ' ') bytes"
echo "    SHA256: ${MODEL_SHA256}"
