# AnalizadorRostroUNOQ — Manual de usuario (datasheet)

**Versión:** 1.0.0
**Plataforma:** Arduino UNO Q — MPU Linux (Qualcomm QRB2210, ARM64, Debian)
**Tipo de aplicación:** servicio web local de análisis de rostros (detección + estimación de género y rango de edad)
**Idioma de la interfaz:** español

---

## 1. Descripción general

AnalizadorRostroUNOQ es una aplicación que se ejecuta **dentro de la Arduino UNO Q**, en su procesador Linux. Toma vídeo de una **cámara USB (Logitech Brio)** o de una **cámara IP/MJPEG por WiFi** (por ejemplo, IP Webcam), detecta rostros, estima **género** (Masculino/Femenino) y **rango de edad** (8 intervalos), y publica el resultado como una página web accesible desde cualquier navegador de la misma red.

No requiere instalar nada en el navegador, no usa Internet para inferir y no guarda imágenes por defecto.

### 1.1 Uso previsto

- Demostraciones educativas de visión por computador y sistemas embebidos Linux.
- Experimentos de interacción persona-máquina en ferias, aulas o laboratorios.
- Base para proyectos que necesiten un stream de vídeo analizado en local.

### 1.2 Uso NO previsto

- Control de acceso, verificación de identidad o vigilancia.
- Diagnóstico médico, estimación legal de edad o decisiones automatizadas sobre personas.
- Entornos donde se requiera precisión certificada o equidad demostrada.
- Procesamiento de menores sin base legal y consentimiento.

> La salida es una **estimación estadística**, no un hecho. El sistema no identifica personas.

---

## 2. Especificaciones técnicas

### 2.1 Hardware

| Parámetro | Valor |
|---|---|
| Placa | Arduino UNO Q |
| MPU | Qualcomm QRB2210, ARM64 (AArch64) |
| Sistema operativo del MPU | Debian Linux |
| RAM | 8 GB (variante reportada por el usuario) |
| MCU | STM32U585, Cortex-M33 con TrustZone |
| RTOS del MCU | Zephyr |
| Uso del MCU en esta app | **Ninguno** (sin firmware, sin GPIO, sin ADC) |
| Cámara | **Recomendada:** IP Webcam / cámara IP por WiFi (MJPEG/RTSP). **USB:** Logitech Brio (UVC), requiere host mode con hub USB-C PD o VIN + JCTL. |
| Dispositivo de vídeo | `/dev/video0` por defecto (`CAMERA_INDEX=0`) o URL en `CAMERA_SOURCE` |
| Red | Wi-Fi o Ethernet en la placa |
| Alimentación | Fuente oficial de la UNO Q |

### 2.2 Software

| Componente | Versión / requisito |
|---|---|
| Python | 3.10+ (validado con 3.13 en ARM64) |
| FastAPI | `>=0.115.0,<0.116` |
| Uvicorn | `>=0.30.0,<0.31` |
| OpenCV | `opencv-python-headless>=4.12.0,<5` |
| NumPy | `>=2.2.0,<3` |
| Pillow | `>=11.0.0,<13` |
| python-dotenv | `>=1.0.0,<2.0` |
| python-multipart | `>=0.0.9,<0.1` |
| Navegador | Chrome, Firefox, Edge o Safari actualizados |

Las versiones exactas de runtime están fijadas en [`../requirements.txt`](../requirements.txt); el procedimiento de instalación con entorno virtual está en la sección 14.

### 2.3 Servicio

| Parámetro | Valor por defecto |
|---|---|
| Bind | `0.0.0.0` |
| Puerto | `8000/tcp` |
| Stream | MJPEG, `multipart/x-mixed-replace; boundary=frame` |
| Resolución | 640×480 |
| FPS objetivo | 15 |
| Inferencia | 1 de cada 5 frames (`INFER_EVERY_N=5`) |
| Umbral de confianza | 0.6 |
| Calidad JPEG | 80 |
| Procesos | 1 worker (estado en memoria) |

### 2.4 Modelos

| Backend | Archivos | Entrada | Edad |
|---|---|---|---|
| `utkface_onnx` | `models/utkface_mobilenetv2.onnx` (v1.0.0) | NCHW `[N,3,224,224]`, `(x/127.5)-1`, RGB | 8 buckets configurables (`AGE_BUCKETS`) |
| `caffe` | `models/fallback/deploy_age.prototxt`, `age_net.caffemodel`, `deploy_gender.prototxt`, `gender_net.caffemodel` | 227×227, media BGR, sin swapRB | Buckets fijos: `0-2, 4-6, 8-12, 15-20, 25-32, 38-43, 48-53, 60-100` |
| `none` | — | — | Solo detección, sin etiquetas |

---

## 3. Contenido del producto

| Elemento | ¿Incluido? |
|---|---|
| Código de la aplicación (`app/`, `python/`, `static/`) | Sí |
| Scripts de despliegue y setup (`scripts/`) | Sí |
| Script de entrenamiento UTKFace (`training/`) | Sí |
| Modelos Caffe fallback (`models/fallback/`, ~88 MB) | Presentes en el equipo de desarrollo, **excluidos de Git** |
| Modelo ONNX UTKFace | **Entrenado v1.0.0**, disponible como asset del release v1.0.0; en la placa de referencia ya está instalado; no se versiona en Git |
| Cámara, hub, placa y fuente | No (hardware aparte) |

---

## 4. Instalación rápida

```bash
# 1) En el host, con la placa conectada por USB
adb devices

# 2) Descargar el modelo y copiar/preparar el proyecto en la placa
./scripts/deploy_unoq.sh --adb --setup --model

# 3) Arrancar
adb shell 'cd /home/arduino/ArduinoApps/AnalizadorRostroUNOQ && ./scripts/run.sh'

# 4) Abrir en el navegador
# http://<IP_DE_LA_PLACA>:8000/
```

Si usas SSH:

```bash
export UNOQ_HOST=192.0.2.20
export UNOQ_USER=arduino
./scripts/deploy_unoq.sh --ssh --setup --model
```

Instalación detallada: [`../README.md`](../README.md).

---

## 5. Operación

### 5.1 Encendido y arranque

1. Elegí la fuente de cámara:
   - **IP Webcam (recomendada):** iniciá el servidor en el teléfono y configurá `CAMERA_SOURCE=http://IP:8080/video`.
   - **Brio USB:** conectá la fuente PD al hub, el hub al USB-C de la UNO Q y la Brio a un USB-A del hub. Sin hub PD, usá VIN 7–24 V + JCTL.
2. Alimenta la UNO Q.
3. Espera a que el sistema Linux arranque.
4. Lanza la app (`./scripts/run.sh` o desde App Lab).
5. Abre `http://<IP>:8000/`.

### 5.2 Elementos de la pantalla

```text
┌─────────────────────────────────────────────────────────────┐
│ Analizador de Rostros        [Cámara] [Modelo] [FPS]        │
├──────────────────────────────────┬──────────────────────────┤
│                                  │ Estado del sistema       │
│      ANUNCIO CONTEXTUAL          │  Cámara:     Activa      │
│  (según género y banda de edad)  │  Backend:    UTKFace     │
│                                  │  Modelo:     Cargado     │
│   [Producto]  [Precio]           │  Resolución: 640 × 480   │
│   [Ver oferta]                   │  Inferencia: 1 de 5      │
│                                  │  FPS:        12.4        │
│                                  ├──────────────────────────┤
│                                  │ Vista en vivo (pequeña)  │
│                                  │  [stream MJPEG]          │
│                                  │  Rostros: 1 · FPS: 12.4  │
│                                  │  [Iniciar] [Detener]     │
│                                  │  [Snapshot] [Token API]  │
├──────────────────────────────────┴──────────────────────────┤
│ Aviso de privacidad · v1.0 · STM32U585 no interviene        │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Controles

| Control | Acción | Requisitos |
|---|---|---|
| **Iniciar cámara** | Reinicia cámara y pipeline; espera hasta 8 s a que haya frames. | Token si `API_TOKEN` está definido. |
| **Detener** | Detiene el worker y libera el dispositivo de vídeo. | Token si aplica. |
| **Snapshot** | Guarda un JPG en `SNAP_FOLDER`. | `SNAP_FOLDER` configurado + token si aplica. |
| **Token API** | Guarda el token en la sesión del navegador. | Solo visible si `auth_enabled`. |

Los botones se deshabilitan automáticamente según el estado (no puedes iniciar si ya está activa ni detener si está apagada).

Elementos nuevos del layout:

- **Anuncio contextual**: ocupa el gran formato de la izquierda. Cambia según el género y la banda de edad del rostro más grande; si no hay rostro (o la confianza es baja), muestra el anuncio de viajes "El destino te está esperando". Los productos, precios y marcas son ficticios.
- **Vista en vivo**: miniatura a la derecha, debajo del estado. Allí están el stream, el contador de rostros, los FPS y los controles.

### 5.4 Secuencia típica de uso

1. Abrir la página → el stream se conecta solo.
2. Verificar que el badge **Cámara** está en verde y **Modelo** indica el backend.
3. Colocarse a 40–80 cm, de frente, con luz difusa frontal.
4. Leer las etiquetas sobre cada rostro.
5. Observar cómo el anuncio grande cambia según la persona detectada.
6. Pulsar **Detener** al terminar (libera la cámara y reduce CPU).
7. Si se guardaron snapshots, borrarlos.

### 5.5 Apagado

- Desde la terminal: `Ctrl+C` en el proceso de `run.sh`.
- Si se lanzó con `nohup`: `pkill -f "app.main"` (verifica antes con `pgrep -af app.main`).
- Al detener, el `lifespan` de FastAPI detiene el pipeline y libera la cámara.

---

## 6. Interpretación de resultados

### 6.1 Etiquetas

| Texto en pantalla | Lectura |
|---|---|
| `Masculino \| 20-29 (87%)` | Género estimado masculino, edad estimada 20–29, confianza media 87 %. |
| `Femenino \| 50-59 (72%)` | Género estimado femenino, edad 50–59, confianza media 72 %. |
| `~Género? \| 30-39 (55%)` | La cabeza de género no alcanzó `CONF_THRESHOLD`; se muestra la clase estimada con `~` y `?`. |
| `Masculino \| ~20-29? (63%)` | La cabeza de edad no alcanzó el umbral; se muestra la clase estimada con `~` y `?`. |
| `Rostro detectado \| modelo no cargado` | No hay modelo; solo se detectó la cara. |

**Cómo se calcula el porcentaje:** es el promedio de la confianza de género y la confianza de edad (`(conf_genero + conf_edad) / 2`), redondeado. No es una probabilidad calibrada ni una medida de calidad de la detección.

### 6.2 Colores

| Color | Significado |
|---|---|
| Caja teal/verde | Rostro con predicción disponible. |
| Caja ámbar | Rostro sin predicción (modelo ausente, ROI inválida o error de inferencia). |
| Badge verde | Funcionando. |
| Badge ámbar | Advertencia (sin cámara, FPS bajo). |
| Badge rojo | Fallo (sin modelo, sin conexión). |

### 6.3 HUD

- `Rostros: N` — número de caras del último frame procesado.
- `FPS: x.x` — frames por segundo suavizados del pipeline.
- `CAMARA OFFLINE` — no llegan frames; se muestra un placeholder.

### 6.4 Anuncio contextual

| Rostro detectado | Anuncio |
|---|---|
| Masculino < 20 | Teléfono |
| Masculino 20–40 | Computador |
| Masculino > 40 | Corbatas |
| Femenino < 20 | Teléfono |
| Femenino 20–40 | Pañuelo de cuello |
| Femenino > 40 | Auto |
| Sin rostro o confianza baja | "El destino te está esperando" (viajes) |

- La banda se calcula con el punto medio del bucket de edad.
- El anuncio es una **demostración**: los productos, precios y marcas no son ofertas reales.
- No se guarda historial de anuncios ni se crea un perfil publicitario.
- Si la etiqueta del vídeo lleva `~`, el anuncio puede seguir mostrándose: es una estimación orientativa, no una certeza.

---

## 7. Estados y mensajes del sistema

`/api/status` devuelve `message`; la UI lo muestra en el panel.

| `message` | Significado | Acción |
|---|---|---|
| `OK` | Cámara y modelo operativos. | Ninguna. |
| `Modelo no cargado: modo solo detección` | No hay backend válido. | `bash scripts/download_model.sh` (ONNX) o `download_fallback_models.sh`; revisar `model_error`. |
| `Cámara no disponible` | El dispositivo no entrega frames. | Revisar conexión, `CAMERA_INDEX`, permisos. |
| `Cámara detenida` | Se pulsó Detener o el hilo no está capturando. | Pulsar Iniciar cámara. |
| `Pipeline detenido` | El worker fue detenido. | Iniciar cámara. |
| `Pipeline detenido inesperadamente` | El hilo murió pese a `_started=true`. | Reiniciar la app; revisar logs. |
| `Detector no disponible: ...` | Haar cascade no cargó. | Revisar `HAAR_PATH` / instalación de OpenCV. |
| `Error en pipeline: ...` | Excepción capturada por el worker; sigue vivo. | Revisar `pipeline_error` y logs. |

### 7.1 Badges

| Badge | Estado |
|---|---|
| Cámara activa / Cámara offline / Sin conexión | `camera_ok` y conectividad con el servidor. |
| UTKFace ONNX / Caffe · Levi & Hassner / Sin modelo | Backend activo. |
| FPS x.x | Verde si > 1, ámbar si ≤ 1. |

### 7.2 Códigos HTTP

| Código | Significado | Qué hacer |
|---|---|---|
| `200` | Correcto. | — |
| `400` | `SNAP_FOLDER` inválido (fuera del proyecto). | Usar una subcarpeta relativa. |
| `401` | Token ausente o incorrecto. | Enviar `X-Api-Token` correcto. |
| `429` | Snapshot demasiado frecuente. | Esperar `SNAPSHOT_MIN_INTERVAL` (2 s). |
| `500` | Error al guardar el snapshot. | Revisar permisos/espacio. |
| `503` | Cámara no abre, snapshot deshabilitado o sin frame. | Revisar configuración/cámara. |

---

## 8. Configuración

Toda la configuración se hace por variables de entorno en `.env` (raíz del proyecto). Tabla completa en [`../README.md#configuración-env`](../README.md#configuración-env).

Valores que un usuario final suele tocar:

| Variable | Para qué |
|---|---|
| `CAMERA_SOURCE` | URL de cámara IP/DroidCam (prioridad sobre `CAMERA_INDEX`). |
| `CAMERA_INDEX` | Elegir otra cámara USB (`/dev/video1`, etc.). |
| `MODEL_BACKEND` | Forzar `utkface_onnx`, `caffe` o `auto`. |
| `INFER_EVERY_N` | Suavidad vs. CPU. |
| `CONF_THRESHOLD` | Rigor de las etiquetas. |
| `SNAP_FOLDER` | Habilitar snapshots (vacío = deshabilitado). |
| `API_TOKEN` | Proteger los POST. |
| `PORT` | Cambiar el puerto web. |

Después de editar `.env`, **reinicia la app**.

---

## 9. Mantenimiento

| Tarea | Frecuencia | Procedimiento |
|---|---|---|
| Limpiar el objetivo de la Brio | Según uso | Paño de microfibra; no disolventes. |
| Revisar espacio en disco | Mensual | `df -h`; borrar snapshots antiguos. |
| Actualizar dependencias | Solo con pruebas | Recrear `.venv` y `pip install -r requirements.txt`. |
| Rotar `API_TOKEN` | Si se comparte el acceso | Nuevo valor en `.env` + reinicio + actualizar clientes. |
| Borrar snapshots | Tras cada sesión | `rm -f snapshots/*.jpg` (verifica la carpeta). |
| Revisar logs | Ante fallos | `adb shell 'tail -n 100 /tmp/analizador.log'` o consola de `run.sh`. |

> No hay base de datos ni estado persistente salvo los snapshots opcionales.

---

## 10. Privacidad

- Procesamiento **100 % local** en la UNO Q.
- **No** se envían imágenes, vídeo ni resultados a Internet.
- **No** se guardan imágenes salvo que `SNAP_FOLDER` esté configurado y se pulse Snapshot.
- El stream MJPEG está accesible a cualquiera que alcance el puerto 8000: usa una red de confianza y `API_TOKEN`.
- Los snapshots son **datos biométricos**: applícala base legal correspondiente (RGPD art. 9), informa y elimina los archivos al terminar.
- La app no identifica personas ni crea plantillas biométricas persistentes.
- El anuncio contextual se calcula en memoria con el frame actual; no se guarda historial ni se envía a terceros.
- Las imágenes de producto están empaquetadas localmente en `static/ads/`; no se descargan en tiempo de ejecución.
- Los anuncios son ficticios: no hay perfilado publicitario persistente ni compra real.

---

## 11. Rendimiento

| Escenario | Expectativa de diseño |
|---|---|
| 0 clientes MJPEG | CPU mínima: solo lectura de cámara cada 250 ms, sin inferencia ni JPEG. |
| 1 cliente, 1 rostro, 640×480 | ~15 FPS de UI medidos con IP Webcam 1080p reducida a 640×360; no es un benchmark formal. |
| Varios clientes | El coste no se multiplica: todos leen el mismo JPEG publicado. |
| Varios rostros | El coste de inferencia crece por rostro. |

> **Medición parcial.** El escenario IP Webcam con un cliente se observó en la placa de referencia (~15 FPS de UI, inferencia ONNX cada 5 frames). No se midieron Brio USB, Caffe, varios clientes, latencia de inferencia ni temperatura sostenida. Ajusta `INFER_EVERY_N`, resolución y `JPEG_QUALITY` según lo que reporte `/api/status`.

---

## 12. Solución de problemas (resumen)

| Problema | Solución rápida |
|---|---|
| No carga la página | `curl http://IP:8000/healthz`; revisar que el proceso siga vivo y el puerto. |
| Cámara offline | Revisar `CAMERA_SOURCE` (misma red/IP/puerto) o `ls -l /dev/video*` y `CAMERA_INDEX` para USB. |
| Sin modelo | `bash scripts/download_model.sh` (ONNX) o `download_fallback_models.sh`; revisar `model_error`. |
| FPS bajo | Subir `INFER_EVERY_N`; bajar resolución; cerrar clientes. |
| Puerto ocupado | `ss -ltnp \| grep 8000`; cambiar `PORT`. |
| 401 | Enviar `X-Api-Token` o vaciar `API_TOKEN`. |
| 429 | Esperar 2 s entre snapshots. |
| 503 snapshot | Configurar `SNAP_FOLDER`. |
| `adb` vacío | Otro cable/puerto; `adb kill-server`; aceptar depuración. |
| SSH timeout | Verificar IP/red/servicio SSH. |

Detalle ampliado en [`../README.md#solución-de-problemas`](../README.md#solución-de-problemas).

---

## 13. Compatibilidad con otras placas

| Placa / plataforma | ¿Funciona? | Notas |
|---|---|---|
| **Arduino UNO Q (MPU Linux)** | ✅ Objetivo principal | ARM64, Debian, cámara USB. |
| Raspberry Pi 4/5 (ARM64) | ✅ Muy probable | Mismos requisitos; ajustar `CAMERA_INDEX` y permisos. |
| PC Linux x86_64 | ✅ Para desarrollo | `opencv-python-headless` funciona igual. |
| macOS / Windows (host) | ✅ Para desarrollo | La ruta del dispositivo la gestiona OpenCV; `CAMERA_INDEX` puede variar. |
| Arduino UNO R3 / AVR | ❌ No | Sin Linux, sin Python, sin RAM para el modelo. |
| Arduino GIGA R1 | ❌ No tal cual | Tiene Linux? No; el sketch no aplica. |
| ESP32 / ESP8266 | ❌ No | No ejecuta este stack Python/OpenCV. |
| Arduino UNO Q (MCU Zephyr) | ❌ No | El MCU no ejecuta Python ni OpenCV. |
| Android/Termux | ⚠️ No probado | Podría funcionar con cámara accesible; sin soporte. |

---

## 14. Librerías: instalación

### 14.1 Runtime (placa)

Se instalan automáticamente con `scripts/setup_board.sh` dentro de `.venv`. Manualmente:

```bash
cd /home/arduino/ArduinoApps/AnalizadorRostroUNOQ
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 14.2 Entrenamiento (PC/Colab)

```bash
cd training
pip install -r requirements-train.txt
```

### 14.3 Notas

- No instales `opencv-python` completo en la placa: requiere librerías gráficas.
- Usa las versiones de `requirements.txt` (OpenCV 4.12+ y NumPy 2.2+). Las combinaciones antiguas OpenCV 4.9/4.10 con NumPy 2 no son compatibles.
- Si `python3 -m venv` falla, falta `python3-venv` (requiere autorización para instalar).

---

## 15. Preguntas frecuentes

**¿Necesita Internet para funcionar?**
No. Solo para descargar modelos o dependencias durante la instalación.

**¿Guarda vídeo?**
No. El stream se genera en memoria. Solo se guardan JPG si activas `SNAP_FOLDER` y pulsas Snapshot.

**¿Puedo verlo desde el móvil?**
Sí, si el móvil está en la misma red y abres `http://IP:8000/`.

**¿Cuántas personas puede analizar a la vez?**
No hay límite lógico; el coste crece con cada rostro. En la práctica, 2–4 rostros suele ser cómodo.

**¿Por qué mi edad no coincide?**
Son buckets gruesos y el modelo tiene error. No es una medición real.

**¿Por qué no detecta de perfil?**
El cascade es `frontalface_default`: está entrenado para caras frontales.

**¿Puedo cambiar los buckets de edad?**
Solo en el backend ONNX (`AGE_BUCKETS`, exactamente 8). En Caffe están fijados por el modelo.

**¿Se puede usar con otra cámara?**
Sí. Por USB, cualquier cámara UVC que exponga `/dev/videoN`, siempre que la UNO Q esté en host mode (hub USB-C PD o VIN + JCTL). Por red, cualquier fuente MJPEG/RTSP accesible desde la placa, configurada en `CAMERA_SOURCE`.

**¿Por qué el MCU no hace nada?**
Porque el proyecto es una app Linux. El STM32U585 queda libre para otro firmware.

**¿Es seguro exponerlo en la red?**
Es aceptable en una LAN de confianza con `API_TOKEN`. No lo publiques en Internet: no tiene TLS.

---

## 16. Advertencias de seguridad

1. No conectes señales de 5 V a los GPIO del MCU (3.3 V).
2. No alimentes la Brio desde un puerto USB débil; usá un hub USB-C con PD o alimentá por VIN con JCTL puenteado.
3. No ejecutes la app como root sin necesidad.
4. No uses `--workers >1` en Uvicorn: rompe el estado compartido.
5. No compartas el `.env` ni el token.
6. No dejes snapshots biométricos en la placa.
7. Verifica la temperatura en sesiones prolongadas.
8. La app no tiene autenticación para `GET /video`: cualquiera en la red puede ver el stream.

---

## 17. Soporte y diagnóstico

Recopila esta información antes de reportar un problema:

```bash
# Estado de la app
curl -s http://127.0.0.1:8000/api/status | python3 -m json.tool
curl -s http://127.0.0.1:8000/api/config | python3 -m json.tool

# Dispositivo de vídeo
ls -l /dev/video*

# Proceso
pgrep -af "app.main"

# Log (si se lanzó con nohup)
tail -n 100 /tmp/analizador.log
```

| Documento | Uso |
|---|---|
| [`../README.md`](../README.md) | Instalación, API, configuración, problemas. |
| [`developer.md`](developer.md) | Detalles internos y extensión. |
| [`BITACORA.md`](BITACORA.md) | Decisiones y estado real del proyecto. |
| [`../models/README.md`](../models/README.md) | Contrato de modelos. |
| [`../training/README.md`](../training/README.md) | Entrenamiento. |
