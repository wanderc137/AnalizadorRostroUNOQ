#!/usr/bin/env bash
# Descarga los 4 archivos del fallback Caffe (Levi & Hassner) a models/fallback/.
#
# Nota: el repositorio upstream nombra los prototxt como
#   age_deploy.prototxt / gender_deploy.prototxt
# pero la app espera los nombres usados por la mayoría de ejemplos:
#   deploy_age.prototxt / deploy_gender.prototxt
# Se descargan y se guardan con el nombre local esperado por la configuración.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_DIR="${1:-${PROJECT_ROOT}/models/fallback}"
BASE_URL="https://github.com/smahesh29/Gender-and-Age-Detection/raw/master"

MODEL_LOCALS=("age_net.caffemodel" "gender_net.caffemodel")
# local_name|remote_name_1|remote_name_2 (alternativas por si cambia el repo)
ASSETS=(
  "age_net.caffemodel|age_net.caffemodel"
  "gender_net.caffemodel|gender_net.caffemodel"
  "deploy_age.prototxt|age_deploy.prototxt|deploy_age.prototxt"
  "deploy_gender.prototxt|gender_deploy.prototxt|deploy_gender.prototxt"
)
MIN_MODEL_BYTES=1000000

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: se requiere 'curl' para descargar los modelos Caffe." >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"
echo "Destino: ${DEST_DIR}"
echo "Origen:  ${BASE_URL}"

download_asset() {
  local local_name="$1"
  shift
  local target="${DEST_DIR}/${local_name}"
  local tmp="${target}.part"

  if [ -s "${target}" ]; then
    echo "== ${local_name} ya existe, se omite"
    return 0
  fi

  local remote_name
  for remote_name in "$@"; do
    echo ">> Descargando ${remote_name} -> ${local_name}"
    rm -f "${tmp}"
    if curl -fsSL --retry 2 --retry-delay 2 --connect-timeout 20 --max-time 900 \
      -o "${tmp}" "${BASE_URL}/${remote_name}"; then
      mv "${tmp}" "${target}"
      return 0
    fi
    rm -f "${tmp}"
    echo "   ../${remote_name} no disponible, probando alternativa..."
  done

  echo "ERROR: no se pudo descargar ${local_name} desde ${BASE_URL}" >&2
  echo "       La URL del repositorio puede haber cambiado. Descarga manual:" >&2
  echo "       https://github.com/smahesh29/Gender-and-Age-Detection" >&2
  exit 1
}

for asset in "${ASSETS[@]}"; do
  IFS='|' read -r -a parts <<<"${asset}"
  download_asset "${parts[@]}"
done

for file in "${MODEL_LOCALS[@]}"; do
  size="$(wc -c <"${DEST_DIR}/${file}" | tr -d ' ')"
  if [ "${size}" -lt "${MIN_MODEL_BYTES}" ]; then
    echo "ERROR: ${file} mide ${size} bytes, se esperaban > ${MIN_MODEL_BYTES}." >&2
    echo "       Descarga incompleta o página de error de GitHub." >&2
    rm -f "${DEST_DIR}/${file}"
    exit 1
  fi
  echo "   ${file}: ${size} bytes"
done

for file in "deploy_age.prototxt" "deploy_gender.prototxt"; do
  if [ ! -s "${DEST_DIR}/${file}" ]; then
    echo "ERROR: ${file} quedó vacío." >&2
    exit 1
  fi
done

echo
echo "Modelos Caffe listos en ${DEST_DIR}"
echo "Activa el backend con MODEL_BACKEND=caffe (o deja 'auto')."
