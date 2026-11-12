#!/usr/bin/env bash
# Detiene el servidor AnalizadorRostroUNOQ (servicio systemd unoq-app).
set -euo pipefail

SERVICE="unoq-app"

if ! systemctl is-active --quiet "$SERVICE"; then
    echo "El servidor ya está detenido."
    exit 0
fi

sudo -n systemctl stop "$SERVICE"

for _ in $(seq 1 20); do
    if ! systemctl is-active --quiet "$SERVICE"; then
        echo "Servidor detenido."
        exit 0
    fi
    sleep 0.25
done

echo "No se pudo detener $SERVICE." >&2
exit 1
