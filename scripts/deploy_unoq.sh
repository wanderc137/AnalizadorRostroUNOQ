#!/usr/bin/env bash
# Copia el proyecto al lado Linux de la UNO Q por ADB o SSH.
#
# Uso:
#   ./scripts/deploy_unoq.sh                 # ADB si está disponible; si no, SSH
#   ./scripts/deploy_unoq.sh --adb --setup   # copia y prepara venv en la placa
#   ./scripts/deploy_unoq.sh --ssh --model   # SSH + modelo ONNX UTKFace
#   ./scripts/deploy_unoq.sh --ssh --models  # SSH + descarga modelos Caffe
#
# SSH (opcional): exporta UNOQ_HOST=192.168.x.x [UNOQ_USER=arduino]
#                 [UNOQ_SSH_KEY=~/.ssh/unoq_key]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE_APP_DIR="${REMOTE_APP_DIR:-/home/arduino/ArduinoApps/AnalizadorRostroUNOQ}"

MODE="auto"
DO_SETUP=0
WITH_MODELS=0
WITH_ONNX=0

ITEMS=(
  "app"
  "python"
  "static"
  "scripts"
  "training"
  "models"
  "requirements.txt"
  "app.yaml"
  ".env.example"
)

usage() {
  cat <<EOF
Uso: $0 [--adb|--ssh] [--setup] [--model] [--models]

  --adb       fuerza transferencia por ADB
  --ssh       fuerza transferencia por SSH/rsync
  --setup     ejecuta scripts/setup_board.sh en la placa tras copiar
  --model     descarga el ONNX UTKFace en el host antes de copiar
  --models    al preparar, descarga también los modelos Caffe fallback

Variables: REMOTE_APP_DIR, UNOQ_HOST, UNOQ_USER, UNOQ_SSH_KEY
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --adb) MODE="adb" ;;
    --ssh) MODE="ssh" ;;
    --setup) DO_SETUP=1 ;;
    --model) WITH_ONNX=1 ;;
    --models) WITH_MODELS=1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Argumento desconocido: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

remote_setup_cmd() {
  local cmd="cd '${REMOTE_APP_DIR}' && bash scripts/setup_board.sh"
  if [ "${WITH_MODELS}" -eq 1 ]; then
    cmd="${cmd} --with-models"
  fi
  printf '%s' "${cmd}"
}

push_adb() {
  if ! command -v adb >/dev/null 2>&1; then
    echo "ADB: comando 'adb' no encontrado." >&2
    return 1
  fi
  local devices
  devices="$(adb devices | awk 'NR>1 && $2=="device" {print $1}')"
  if [ -z "${devices}" ]; then
    echo "ADB: no hay dispositivos enumerados." >&2
    return 1
  fi

  echo ">> ADB detectado: $(printf '%s ' "${devices}")"
  adb shell "mkdir -p '${REMOTE_APP_DIR}'"
  for item in "${ITEMS[@]}"; do
    if [ ! -e "${PROJECT_ROOT}/${item}" ]; then
      continue
    fi
    echo "   push ${item}"
    adb push "${PROJECT_ROOT}/${item}" "${REMOTE_APP_DIR}/" >/dev/null
  done
  echo "OK: proyecto copiado a ${REMOTE_APP_DIR}"

  if [ "${DO_SETUP}" -eq 1 ]; then
    echo ">> Preparando entorno en la placa (setup_board.sh)"
    adb shell "$(remote_setup_cmd)"
  fi
  return 0
}

push_ssh() {
  if [ -z "${UNOQ_HOST:-}" ]; then
    echo "SSH: define UNOQ_HOST (ej: export UNOQ_HOST=192.168.1.50)." >&2
    return 1
  fi
  command -v ssh >/dev/null 2>&1 || {
    echo "SSH: comando 'ssh' no encontrado." >&2
    return 1
  }

  local user="${UNOQ_USER:-arduino}"
  local target="${user}@${UNOQ_HOST}"
  local ssh_opts=(-o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new)
  if [ -n "${UNOQ_SSH_KEY:-}" ]; then
    ssh_opts+=(-i "${UNOQ_SSH_KEY}")
  fi

  echo ">> SSH a ${target}"
  ssh "${ssh_opts[@]}" "${target}" "mkdir -p '${REMOTE_APP_DIR}'"

  if command -v rsync >/dev/null 2>&1; then
    local rsh="ssh ${ssh_opts[*]}"
    rsync -az \
      --exclude '.venv' --exclude 'venv' --exclude '__pycache__' \
      --exclude '.git' --exclude 'snapshots' --exclude 'dataset' \
      --exclude '.env' --exclude '*.pyc' \
      -e "${rsh}" "${PROJECT_ROOT}/" "${target}:${REMOTE_APP_DIR}/"
  else
    for item in "${ITEMS[@]}"; do
      if [ ! -e "${PROJECT_ROOT}/${item}" ]; then
        continue
      fi
      echo "   scp ${item}"
      scp -r "${ssh_opts[@]}" "${PROJECT_ROOT}/${item}" "${target}:${REMOTE_APP_DIR}/"
    done
  fi
  echo "OK: proyecto copiado a ${target}:${REMOTE_APP_DIR}"

  if [ "${DO_SETUP}" -eq 1 ]; then
    echo ">> Preparando entorno en la placa (setup_board.sh)"
    ssh "${ssh_opts[@]}" "${target}" "$(remote_setup_cmd)"
  fi
  return 0
}

if [ "${WITH_ONNX}" -eq 1 ]; then
  echo ">> Descargando modelo UTKFace para incluirlo en la copia"
  bash "${PROJECT_ROOT}/scripts/download_model.sh"
fi

case "${MODE}" in
  adb)
    push_adb || {
      echo "ERROR: la transferencia por ADB falló." >&2
      exit 1
    }
    ;;
  ssh)
    push_ssh || {
      echo "ERROR: la transferencia por SSH falló." >&2
      exit 1
    }
    ;;
  auto)
    if push_adb; then
      :
    else
      echo ">> ADB no disponible; intentando SSH..."
      if [ -n "${UNOQ_HOST:-}" ]; then
        push_ssh || {
          echo "ERROR: la transferencia por SSH falló." >&2
          exit 1
        }
      else
        echo "ERROR: no se detectó ninguna UNO Q accesible." >&2
        echo "  - Conecta la placa por USB (debe aparecer en 'adb devices')." >&2
        echo "  - O define UNOQ_HOST y UNOQ_USER para usar SSH." >&2
        exit 1
      fi
    fi
    ;;
esac

echo
echo "Siguiente paso:"
echo "  adb shell 'cd ${REMOTE_APP_DIR} && ./scripts/run.sh'"
echo "  (o: ssh ${UNOQ_USER:-arduino}@<host> 'cd ${REMOTE_APP_DIR} && ./scripts/run.sh')"
