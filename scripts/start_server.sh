#!/usr/bin/env bash
# Inicia el servidor AnalizadorRostroUNOQ (servicio systemd unoq-app).
set -euo pipefail

SERVICE="unoq-app"
LOCAL_HEALTH="http://127.0.0.1:8000/healthz"
LOCAL_STATUS="http://127.0.0.1:8000/api/status"
PUBLIC_URL="${PUBLIC_URL:-http://192.0.2.30:8000}"

if systemctl is-active --quiet "$SERVICE"; then
    echo "El servidor ya está activo."
else
    sudo -n systemctl start "$SERVICE"
    echo "Servicio $SERVICE iniciado."
fi

for _ in $(seq 1 30); do
    if curl -fsS --max-time 2 "$LOCAL_HEALTH" >/dev/null 2>&1; then
        echo "Servidor disponible en $PUBLIC_URL"

        if STATE=$(curl -fsS --max-time 2 "$LOCAL_STATUS" 2>/dev/null); then
            printf '%s' "$STATE" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    raise SystemExit(0)
cam = "OK" if d.get("camera_ok") else "offline"
mod = "OK" if d.get("model_loaded") else "no cargado"
pipe = "activo" if d.get("pipeline_running") else "detenido"
print(f"Cámara: {cam} | Modelo: {mod} | Pipeline: {pipe}")
' || true
        fi
        exit 0
    fi
    sleep 0.5
done

echo "El servicio arrancó, pero $LOCAL_HEALTH no responde." >&2
exit 1
