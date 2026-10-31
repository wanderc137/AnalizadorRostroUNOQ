# Bitácora — AnalizadorRostroUNOQ

Registro de problemas, decisiones, enfoques descartados y lecciones aprendidas durante el desarrollo. El objetivo es que un futuro proyecto (o otra persona) no repita los mismos caminos.

> **Nota de honestidad:** la placa UNO Q **no estuvo accesible** durante el desarrollo (`adb devices` vacío, SSH con timeout). Todo lo que sigue se basa en análisis estático, ejecución en host y revisión de código; **no** en pruebas de hardware. Las afirmaciones de rendimiento son objetivos de diseño, no mediciones.

---

## 1. Contexto y objetivo

- **Placa:** Arduino UNO Q, MPU Qualcomm QRB2210 (ARM64, Debian), variante de 8 GB reportada por el usuario.
- **Cámara:** Logitech Brio USB vía hub.
- **Objetivo:** detectar rostros y estimar género + rango de edad, con UI web en español y procesamiento local.
- **Restricción clave:** la app debía correr **en el MPU Linux**, no en el MCU STM32U585/Zephyr.

---

## 2. Cronología resumida

| Fase | Qué se hizo | Resultado |
|---|---|---|
| 1. Análisis | Lectura del requerimiento, especificación de la UNO Q, elección de stack. | `PLAN.md` con arquitectura, API y criterios de aceptación. |
| 2. Implementación | FastAPI + OpenCV + Haar + dos backends; UI vanilla; scripts; training. | Código completo en `app/`, `static/`, `scripts/`, `training/`. |
| 3. Modelos | Descarga del fallback Caffe; diseño del entrenamiento UTKFace. | Caffe presente (~88 MB); ONNX pendiente de entrenar. |
| 4. Auditoría ciclo 1 | Revisión estática. | 9 incidencias (4 medias, 5 bajas). |
| 5. Correcciones | Se resolvieron las 9 incidencias. | `AUDIT_REPORT.md` ciclo 2: aprobado sin incidencias. |
| 6. Despliegue | Intento de acceso a la placa por ADB y SSH. | **No fue posible**; despliegue no verificado en hardware. |
| 7. Documentación | README, datasheet, developer, esta bitácora. | Documentación completa. |

---

## 3. Problemas encontrados y cómo se resolvieron

### P-01 — La placa no estuvo accesible

**Síntoma**

```text
$ adb devices
List of devices attached
(vacío)
```

SSH a la IP supuesta: timeout.

**Impacto**

- No se pudo ejecutar `deploy_unoq.sh` contra hardware.
- No se pudo medir FPS, temperatura ni comportamiento real de `/dev/video0`.
- No se pudo validar App Lab (`app.yaml`).

**Decisión**

- No inventar resultados. Documentar el estado como "no verificado en hardware" en README y datasheet.
- Dejar los scripts listos y verificados estáticamente (`bash -n`).
- Añadir mensajes claros en `deploy_unoq.sh` cuando no hay dispositivo o `UNOQ_HOST`.

**Lección**

Un proyecto puede estar completo a nivel de código y aun así no estar validado en placa. La documentación debe distinguir **"verificado en host"** de **"verificado en hardware"**.

---

### P-02 — `adb devices` vacío o `unauthorized`

**Causas probables identificadas**

1. Cable USB solo de carga.
2. Puerto/hub sin datos.
3. Diálogo de depuración no aceptado en la placa.
4. Servidor ADB en estado inconsistente.
5. Falta de regla udev en Linux (dispositivo visible en `lsusb` pero no en `adb`).

**Mitigación documentada**

```bash
adb kill-server
adb start-server
adb devices
lsusb
```

- Probar otro cable/puerto.
- Aceptar el diálogo en la placa.
- Verificar el VID:PID real antes de crear reglas udev; **no copiar reglas de terceros a ciegas**.

**Lección**

`adb` es un requisito duro para el despliegue por USB. Si no enumera, no hay nada que hacer: no se debe "adivinar" un puerto ni forzar `adb push`.

---

### P-03 — SSH con timeout

**Causas probables**

- IP incorrecta o cambiada por DHCP.
- Equipos en redes/VLAN distintas.
- Servicio SSH apagado en la placa.
- Firewall.
- Usuario/clave incorrectos.

**Mitigación documentada**

```bash
ping -c 3 "$UNOQ_HOST"
ssh -v "${UNOQ_USER:-arduino}@${UNOQ_HOST}"
```

- Obtener la IP por `adb shell 'hostname -I'` si ADB funciona.
- Usar `UNOQ_SSH_KEY` si se requiere clave.
- `deploy_unoq.sh` exige `UNOQ_HOST` para el modo SSH y falla con mensaje claro.

---

### P-04 — Modelo ONNX ausente al inicio (resuelto en v1.0.0)

**Situación inicial:** el backend `utkface_onnx` requiere `models/utkface_mobilenetv2.onnx`, que no existía; es un artefacto de entrenamiento excluido por `.gitignore`.

**Solución aplicada**

- Se entrenó MobileNetV2 con UTKFace en GitHub Actions (10 épocas, CPU) y se exportó el ONNX NCHW.
- Métricas de test: género 84.56 %, edad exacta 48.17 %, edad ±1 bucket 83.8 %.
- El modelo se publicó como asset del release `v1.0.0` y se descarga con `scripts/download_model.sh` (verifica SHA256).
- `MODEL_BACKEND=auto` prefiere ONNX y cae a Caffe si no está.
- Si ninguno existe, la app **no crashea**: modo degradado con detección Haar y etiqueta `Rostro detectado | modelo no cargado`.
- `/api/status` expone `model_error` para saber exactamente qué falta.

**Lección**

Un backend opcional debe fallar de forma **observable pero no fatal**. El modo degradado es una funcionalidad, no un error.

---

### P-05 — Buckets de edad incompatibles entre backends

**Situación:** UTKFace usa 8 buckets propios (`0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60+`), pero el modelo Caffe de Levi & Hassner usa otros (`0-2, 4-6, 8-12, 15-20, 25-32, 38-43, 48-53, 60-100`).

**Error que se habría cometido:** reutilizar `AGE_BUCKETS` para ambos backends. Las etiquetas de Caffe habrían sido incorrectas (p. ej., decir "30-39" cuando el modelo dijo "25-32").

**Solución**

- `AGE_BUCKETS` solo se aplica a `OnnxBackend`.
- `CAFFE_AGE_BUCKETS` es una constante separada en `models_loader.py`.
- Documentado en README y en `models/README.md`.

**Lección**

Los "buckets" son parte del **contrato del modelo**, no de la configuración global. Mezclarlos produce errores silenciosos y creíbles.

---

### P-06 — Coste de inferencia multiplicado por cliente

**Riesgo:** si cada conexión a `/video` ejecutara Haar + red + JPEG, N navegadores multiplicarían la CPU.

**Solución**

- Un **único** `VideoPipeline` produce el JPEG.
- Los clientes solo leen el último JPEG publicado (`latest_jpeg()`), sin inferencia por cliente.
- Contador `stream_clients`; sin clientes, modo idle (sin Haar, sin red, sin JPEG; lectura cada 250 ms).

**Lección**

En streaming, separa **producción** de **distribución**. El fan-out debe ser barato.

---

### P-07 — Hilos que mueren silenciosamente o se "resucitan"

**Riesgos detectados en auditoría**

1. Una excepción no capturada mataba el worker de vídeo.
2. Un `stop`/`start` rápido podía reutilizar el `Event` de parada y revivir al worker anterior.
3. Un hilo de cámara antiguo podía liberar el `VideoCapture` de un `start()` posterior.

**Soluciones**

| Problema | Solución |
|---|---|
| Worker muere | `try/except Exception` alrededor de cada iteración; registra `last_error`, espera 0.5 s y continúa. |
| Event reutilizado | `start()` crea **nuevos** `threading.Event` por ejecución. |
| Liberación cruzada | Solo `stop()` libera `VideoCapture`; el hilo no libera al salir. |
| Cámara colgada | `is_ok` exige frame < 3 s; 20 fallos → release + reconexión. |

**Lección**

En hilos de larga duración: captura total de excepciones, eventos por ejecución y propiedad única de los recursos.

---

### P-08 — Snapshot sin autenticación ni límites

**Riesgo:** cualquiera en la red podía guardar JPG de rostros en la placa; además, ráfagas de POST podían llenar el disco.

**Solución**

- `API_TOKEN` opcional; si está definido, los POST exigen `X-Api-Token` (`hmac.compare_digest`).
- `SNAPSHOT_MIN_INTERVAL` (2 s) con HTTP 429 + `Retry-After`, aplicado de forma **atómica** antes de tocar disco.
- `SNAP_FOLDER` vacío por defecto (deshabilitado).
- `/api/config` solo expone `auth_enabled`, nunca el token.

**Lección**

Los endpoints de escritura necesitan tres cosas: autenticación, rate-limit y un default seguro (deshabilitado).

---

### P-09 — Path traversal en `SNAP_FOLDER`

**Riesgo:** `SNAP_FOLDER=../../..` o una ruta absoluta externa permitía escribir fuera del proyecto.

**Solución**

- `_resolve_snap_folder()` resuelve y comprueba `relative_to(PROJECT_ROOT)`.
- `ensure_snap_folder()` **revalida en cada snapshot** (por si la ruta o un symlink cambian).
- Respuesta 400 con el motivo; warning en logs.

**Lección**

Validar rutas una vez al arrancar no basta si la ruta se puede manipular después. Revalida en el punto de uso.

---

### P-10 — Regex de UTKFace demasiado estricto

**Situación:** los nombres reales de UTKFace incluyen timestamps de 13–17 dígitos y sufijos como `.chip.jpg`. Un regex que exigía un formato exacto descartaba imágenes válidas.

**Solución**

- Limpieza iterativa del stem (`.chip`, `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
- Regex tolerante `^(\d{1,3})_(\d)_\d+_(.+)$` sobre el stem.
- Self-test con 9 casos, incluido `26_1_3_20170116174519429.jpg.chip.jpg`.
- `python train_utkface.py --self-test` no requiere TensorFlow ni dataset.

**Lección**

Los datasets reales son sucios. Incluye un **self-test** del parser: es barato y evita horas de depuración con TensorFlow de por medio.

---

### P-11 — Reintento inútil en ONNX

**Situación:** `_forward()` reintentaba la misma llamada tras un `cv2.error`. Un fallo de OpenCV DNN no se recupera repitiendo la operación sin cambiar entradas ni estado; solo duplicaba el log y el coste.

**Solución:** un único `forward`; si falla, se registra una vez y se devuelve `None` (predicción degradada).

**Lección**

No reintentes errores deterministas. Reintentar tiene sentido con timeouts/red, no con un grafo inválido.

---

### P-12 — Quoting en mensajes ADB

**Situación:** `echo ">> ADB detectado: $(printf '%s ' ${devices})"` sin comillas podía romper con múltiples dispositivos o espacios.

**Solución:** `"${devices}"` citado; `printf '%s '` para listar varios seriales.

**Lección**

En bash, cita siempre variables, incluso cuando "sabes" que no tienen espacios.

---

## 4. Enfoques descartados y por qué

### D-01 — Implementarlo como sketch `.ino` en el MCU

**Por qué se descartó**

- El STM32U585 no tiene RAM/CPU para MobileNetV2 ni para OpenCV DNN.
- No hay cámara USB en el MCU.
- No hay red HTTP trivial ni filesystem para modelos.
- El requerimiento pedía una página web servida desde la placa.

**Consecuencia:** no se usa `arduino-cli`, ni `arduino:zephyr:unoq`, ni `/dev/ttyACM0`.

---

### D-02 — Inferencia en el MCU con TensorFlow Lite Micro

**Por qué se descartó**

- El modelo (MobileNetV2 + dos cabezas) no cabe con holgura y requeriría cuantización agresiva.
- El pipeline de cámara USB no existe en el MCU.
- La UI web y el streaming MJPEG son responsabilidad del MPU de todos modos.
- Añadiría complejidad de firmware y un protocolo MPU↔MCU sin beneficio claro.

**Cuándo tendría sentido:** un producto final de bajo consumo que solo necesite un LED/buzzer y no streaming.

---

### D-03 — Un hilo/proceso de inferencia por cliente MJPEG

**Por qué se descartó**

- Multiplica CPU y memoria con cada navegador.
- Genera resultados inconsistentes entre clientes.
- Complica el ciclo de vida de la cámara.

**Alternativa elegida:** un productor único y fan-out de bytes.

---

### D-04 — Abrir/cerrar la cámara en cada petición

**Por qué se descartó**

- `VideoCapture` tarda en abrir y pierde frames.
- Varias peticiones concurrentes pelearían por `/dev/video0`.
- El stream MJPEG necesita un flujo continuo.

**Alternativa:** hilo de captura persistente con reconexión.

---

### D-05 — Usar `AGE_BUCKETS` también para Caffe

**Por qué se descartó**

- Los buckets de Caffe son parte del modelo entrenado.
- Reetiquetar con buckets de UTKFace produce resultados incorrectos pero plausibles (peor que un error visible).

**Alternativa:** constantes separadas y documentación explícita.

---

### D-06 — Guardar snapshots por defecto

**Por qué se descartó**

- Son datos biométricos.
- Llenan el disco sin que el usuario lo pida.
- Aumentan la superficie legal/privacidad.

**Alternativa:** `SNAP_FOLDER` vacío; el usuario habilita conscientemente.

---

### D-07 — `opencv-python` completo en la placa

**Por qué se descartó**

- Arrastra dependencias gráficas (GTK/Qt) que no existen en un servidor sin X.
- Más peso y posibles fallos de importación.

**Alternativa:** `opencv-python-headless`.

---

### D-08 — Múltiples workers de Uvicorn

**Por qué se descartó**

- Cada worker abriría la cámara y cargaría el modelo.
- El estado (pipeline, contadores, rate-limit) dejaría de ser consistente.

**Alternativa:** `workers=1` explícito en `python/main.py` y en la documentación.

---

### D-09 — Descargar modelos automáticamente al arrancar

**Por qué se descartó**

- Requiere red y credenciales en el arranque.
- Un fallo de red retrasaría o rompería el inicio.
- El usuario debe decidir qué modelo y cuándo.

**Alternativa:** scripts explícitos (`download_fallback_models.sh`, entrenamiento manual) y modo degradado.

---

### D-10 — Hardcodear IP/puerto de la placa

**Por qué se descartó**

- La IP cambia por DHCP.
- El puerto puede estar ocupado.

**Alternativa:** `HOST`/`PORT` en `.env`, `adb shell 'hostname -I'` y `UNOQ_HOST` para SSH.

---

### D-11 — Autenticar también `/video` con `X-Api-Token`

**Por qué se descartó (por ahora)**

- `<img src="/video">` no puede enviar headers personalizados.
- Implementar login/cookie ampliaría el alcance sin requisito.

**Alternativa documentada:** proxy inverso con auth delante del puerto 8000.

---

## 5. Decisiones técnicas y justificación

| Decisión | Alternativas | Por qué |
|---|---|---|
| **FastAPI + Uvicorn** | Flask, aiohttp | Tipado, documentación automática, `StreamingResponse`, ecosistema. |
| **OpenCV DNN** | ONNX Runtime, TensorFlow | Ya se necesita OpenCV para cámara/Haar; evita otra dependencia pesada en ARM64. |
| **Haar cascade** | YuNet, MTCNN, RetinaFace | Incluido en OpenCV, sin descarga, suficiente para frontal. Se documenta YuNet como mejora. |
| **Un worker de vídeo** | Hilo por cliente | Coste constante, estado consistente. |
| **MJPEG** | WebRTC, HLS, RTSP | Trivial de servir, compatible con `<img>`, sin build. |
| **JPEG quality 80** | 95 | Equilibrio calidad/ancho de banda. |
| **Buckets de 8 clases** | Regresión de edad | Clasificación más estable y fácil de entrenar; la regresión habría requerido más datos/ajuste. |
| **ONNX opset 13** | 17 | Compatibilidad con OpenCV DNN y tf2onnx. |
| **Dos cabezas** | Dos modelos separados | Una sola pasada por la CNN; menos cómputo. |
| **Venv en la placa** | `pip --user`, sistema | Aislamiento, reproducibilidad, no ensucia el Python del sistema. |
| **ADB + SSH** | Solo ADB | ADB puede no estar disponible; SSH cubre redes. |
| **`.env` no se copia** | Copiar todo | Evita filtrar tokens entre máquinas. |
| **API_TOKEN opcional** | Auth obligatoria | No rompe el uso en LAN de confianza; se activa cuando hace falta. |
| **Rate-limit 2 s** | Sin límite | Evita abuso y llenado de disco sin molestar en uso normal. |
| **Snapshots dentro del proyecto** | Cualquier ruta | Contención y facilidad de borrado. |
| **Modo idle sin clientes** | Codificar siempre | Ahorra CPU y batería/térmica. |
| **Placeholder cada 1 s** | Cada frame | Evita codificar en bucle cuando no hay cámara. |
| **Logs en español** | Inglés | Coherencia con la UI y el usuario objetivo. |

---

## 6. Verificación realizada

| Prueba | Comando | Resultado |
|---|---|---|
| Sintaxis Python | `python3 -m py_compile app/*.py python/*.py training/*.py` | OK (según `AUDIT_REPORT.md`). |
| Sintaxis bash | `bash -n scripts/*.sh` | OK. |
| Sintaxis JS | `node --check static/app.js` | OK. |
| Self-test parser UTKFace | `python3 training/train_utkface.py --self-test` | 9/9 OK. |
| Entrenamiento UTKFace | GitHub Actions run `41117352948` | **SUCCESS**: género 84.56 %, edad exacta 48.17 %, ±1 bucket 83.8 %. |
| ONNX NCHW | Validación del workflow + OpenCV DNN | Entrada `[1,3,224,224]`, salidas `gender (1,1)` y `age (1,8)`. |
| Release v1.0.0 | `scripts/download_model.sh --force` | Descarga y SHA256 `390ade...` verificados. |
| Auditoría estática | Revisión de código | 9/9 incidencias resueltas; 0 nuevas. |
| `adb devices` | En el host de documentación | **Vacío** (sin dispositivos). |
| Ejecución en placa | — | **No realizada**. |

### Lo que falta verificar en hardware

- [ ] `deploy_unoq.sh --adb --setup` de extremo a extremo.
- [ ] `deploy_unoq.sh --ssh --setup` de extremo a extremo.
- [ ] Apertura de `/dev/video0` con la Brio y permisos del usuario `arduino`.
- [ ] FPS real con Caffe y con ONNX.
- [ ] Temperatura en sesión de 30+ minutos.
- [ ] `app.yaml` bajo Arduino App Lab.
- [ ] Reconexión real al desconectar/conectar la cámara.
- [ ] Comportamiento con 2–3 clientes MJPEG simultáneos.

---

## 7. Lecciones aprendidas

1. **Separa "verificado" de "esperado".** Un README que promete FPS medidos sin hardware es peor que uno que dice "objetivo de diseño".
2. **El modo degradado es una feature.** Sin cámara o sin modelo, la app debe seguir siendo útil y explicar qué falta.
3. **El contrato del modelo manda.** Buckets, orden de clases, normalización y `swapRB` deben coincidir exactamente; un desajuste no crashea, produce etiquetas falsas.
4. **Un solo productor, muchos consumidores.** Es la arquitectura correcta para streaming con análisis caro.
5. **Los eventos de parada se crean por ejecución.** Reutilizarlos provoca hilos zombis difíciles de reproducir.
6. **Cita siempre en bash.** Un espacio en un serial ADB puede romper un script de despliegue.
7. **Valida rutas en el punto de uso.** La validación en el arranque no cubre cambios posteriores.
8. **Rate-limit atómico.** Actualiza el timestamp antes de trabajar, no después.
9. **Self-test del parser de dataset.** Ahorra depuración costosa y documenta el formato real.
10. **No reintentes errores deterministas.** Un grafo ONNX inválido no se arregla repitiendo `forward`.
11. **La placa puede no estar.** Diseña scripts que reporten con claridad y no bloqueen la entrega del código.
12. **La privacidad se diseña, no se documenta al final.** Defaults seguros (sin snapshots, sin nube) desde el primer commit.

---

## 8. Pendientes y próximos pasos

### Bloqueantes para "listo en producción"

1. **Acceso a la placa** por ADB o SSH.
2. **Desplegar y ejecutar** en hardware; corregir lo que falle.
3. **Medir FPS** con Caffe y con ONNX; ajustar `INFER_EVERY_N` y resolución.
4. **Verificar permisos** de `/dev/video0` para el usuario que ejecuta la app.

### Mejoras deseables

5. Validar el ONNX UTKFace en condiciones reales de cámara e iluminación.
6. Migrar la detección a YuNet para perfiles y luz difícil.
7. Añadir tracking con IDs para estabilizar etiquetas.
8. Proteger `/video` con proxy inverso si se despliega fuera de una LAN de confianza.
9. Servicio systemd con `Restart=on-failure`.
10. Tests unitarios de `parse_filename`, `_split_outputs` y rate-limit.
11. Añadir un `LICENSE` al repositorio.

---

## 9. Anexo — Comandos útiles durante el desarrollo

```bash
# Verificación estática completa
python3 -m py_compile app/*.py python/*.py training/*.py
bash -n scripts/*.sh
node --check static/app.js
python3 training/train_utkface.py --self-test

# Descargar Caffe
bash scripts/download_fallback_models.sh

# Probar la app en el host
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m app.main

# Estado en local
curl -s http://127.0.0.1:8000/api/status | python3 -m json.tool

# Despliegue (cuando la placa esté disponible)
adb devices
./scripts/deploy_unoq.sh --adb --setup
adb shell 'cd /home/arduino/ArduinoApps/AnalizadorRostroUNOQ && ./scripts/run.sh'

# Diagnóstico en la placa
adb shell 'ls -l /dev/video*'
adb shell 'hostname -I'
adb shell 'pgrep -af app.main'
```

---

## 10. Anexo — Glosario

| Término | Significado |
|---|---|
| **MPU** | Microprocesador de aplicación (QRB2210) que corre Linux. |
| **MCU** | Microcontrolador (STM32U585) que corre Zephyr. |
| **V4L2** | Video4Linux2, API de captura de vídeo en Linux. |
| **UVC** | USB Video Class, estándar de cámaras USB. |
| **MJPEG** | Stream de imágenes JPEG concatenadas con boundaries. |
| **Haar cascade** | Detector clásico de objetos de Viola-Jones. |
| **ONNX** | Formato abierto de modelos de redes neuronales. |
| **Caffe** | Framework clásico de deep learning; aquí solo para inferencia. |
| **Bucket** | Intervalo de edad discreto (clase). |
| **EMA** | Media móvil exponencial (suavizado de FPS). |
| **Modo degradado** | Funcionamiento sin cámara o sin modelo, con avisos. |
