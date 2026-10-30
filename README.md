# AnalizadorRostroUNOQ

**Detección de rostros y estimación de género + rango de edad en la Arduino UNO Q, con interfaz web en español y procesamiento 100 % local.**

Aplicación **Python / FastAPI** que corre en el **MPU Linux (Qualcomm QRB2210, ARM64, Debian)** de la Arduino UNO Q. Captura vídeo de una **Logitech Brio USB**, detecta rostros con el clasificador **Haar** de OpenCV y clasifica **género** y **rango de edad** con uno de dos backends: un modelo **ONNX propio entrenado con UTKFace** o el **fallback Caffe de Levi & Hassner**. El vídeo anotado se sirve como **stream MJPEG** a cualquier navegador de la red local.

> **No es un sketch `.ino`.** Este proyecto no se compila con `arduino-cli` ni ejecuta inferencia en el MCU STM32U585. Todo el runtime vive en el lado Linux de la placa. Ver [¿Por qué no se usa arduino-cli?](#por-qué-no-se-usa-arduino-cli).

---

## Tabla de contenidos

1. [Estado del proyecto](#estado-del-proyecto)
2. [Qué es y qué no es](#qué-es-y-qué-no-es)
3. [Arquitectura](#arquitectura)
4. [Principio de funcionamiento](#principio-de-funcionamiento)
5. [Hardware requerido](#hardware-requerido)
6. [Software requerido](#software-requerido)
7. [Puesta en marcha paso a paso](#puesta-en-marcha-paso-a-paso)
8. [Recursos, "pinout" y conexión](#recursos-pinout-y-conexión)
9. [Configuración (.env)](#configuración-env)
10. [API HTTP](#api-http)
11. [Operación de la interfaz web](#operación-de-la-interfaz-web)
12. [Calibración y rendimiento](#calibración-y-rendimiento)
13. [Entrenamiento del modelo UTKFace](#entrenamiento-del-modelo-utkface)
14. [Ejemplos de uso realistas](#ejemplos-de-uso-realistas)
15. [Solución de problemas](#solución-de-problemas)
16. [Privacidad y datos biométricos](#privacidad-y-datos-biométricos)
17. [Limitaciones conocidas](#limitaciones-conocidas)
18. [Atribuciones y licencias](#atribuciones-y-licencias)
19. [Advertencias específicas de la UNO Q](#advertencias-específicas-de-la-uno-q)
20. [Estructura del repositorio](#estructura-del-repositorio)
21. [Documentación adicional](#documentación-adicional)

---

## Estado del proyecto

| Aspecto | Estado |
|---|---|
| Código de la aplicación | Completo; auditoría estática **APROBADA sin incidencias** (`AUDIT_REPORT.md`, ciclo 2). |
| Verificación en host | `py_compile` de todos los módulos, `bash -n` de los scripts, `node --check` del frontend y `train_utkface.py --self-test` (9/9) ejecutados correctamente según `AUDIT_REPORT.md`. |
| Modelos Caffe fallback | **Presentes localmente** en `models/fallback/` (~88 MB: `age_net.caffemodel` 45.7 MB, `gender_net.caffemodel` 45.6 MB, 2 prototxt). **Excluidos de Git** por `.gitignore`. |
| Modelo ONNX UTKFace | **No existe todavía.** Debe entrenarse fuera de la placa (Colab/PC) y colocarse en `models/utkface_mobilenetv2.onnx`. Mientras tanto, `MODEL_BACKEND=auto` usa Caffe. |
| Despliegue en hardware | **NO verificado en la placa.** Durante el desarrollo la UNO Q no estuvo accesible: `adb devices` no enumeró dispositivos y SSH agotó el tiempo de espera. Los scripts están listos, pero no se ha confirmado su ejecución real sobre hardware. |
| Rendimiento real | **No medido en QRB2210.** Las cifras de FPS de esta documentación son objetivos de diseño y recomendaciones de ajuste, no mediciones. |

> **Honestidad técnica:** cualquier tabla, captura o JSON de ejemplo con cifras concretas está marcado como *ilustrativo* y no debe interpretarse como una medición hecha en la placa.

---

## Qué es y qué no es

| ✅ Es | ❌ No es |
|---|---|
| Una app web autocontenida servida desde el MPU Linux de la UNO Q. | Un sketch Arduino / `.ino`. |
| Detección de rostros frontales con Haar cascade. | Un sistema de reconocimiento facial (no identifica personas). |
| Estimación de género binario y rango de edad en 8 buckets. | Un sensor de edad exacta ni una medición médica. |
| Inferencia local en CPU con OpenCV DNN (ONNX o Caffe). | Inferencia en el MCU STM32U585 / Zephyr / TensorFlow Lite Micro. |
| Stream MJPEG para varios navegadores con un solo worker. | Un servidor WebRTC, RTSP o HLS. |
| Snapshots **desactivados por defecto**. | Un sistema de videovigilancia con grabación continua. |

**Resultado esperado:** una página en `http://<IP_UNO_Q>:8000/` con el vídeo en vivo, cajas sobre los rostros y etiquetas del tipo `Femenino | 20-29 (84%)`, un contador de rostros y FPS.

---

## Arquitectura

```text
┌──────────────────────────────────────────────────────────────┐
│ NAVEGADOR (PC / móvil en la misma red)                        │
│  index.html + app.js + style.css                             │
│    ├── <img src="/video">      stream MJPEG                   │
│    ├── GET  /api/status        polling cada 1.2 s             │
│    └── POST /api/snapshot|start|stop   (X-Api-Token opcional) │
└───────────────────────────┬──────────────────────────────────┘
                            │ HTTP :8000
┌───────────────────────────▼──────────────────────────────────┐
│ ARDUINO UNO Q — MPU Linux (Qualcomm QRB2210, ARM64, Debian)  │
│                                                              │
│  Uvicorn + FastAPI  (app/main.py)                            │
│    ├── VideoPipeline: 1 hilo worker único                    │
│    │     frame → (cada INFER_EVERY_N) Haar + clasificación   │
│    │           → overlay → JPEG → publica bajo lock          │
│    ├── Camera: hilo de captura V4L2 + reconexión             │
│    ├── FaceDetector: Haar frontalface + dibujo               │
│    └── ModelHandle: ONNX UTKFace (224×224)                   │
│                     o Caffe Levi & Hassner (227×227)         │
│                     o modo degradado (solo detección)        │
│                                                              │
│  OpenCV VideoCapture(index=CAMERA_INDEX) → /dev/video0       │
└───────────────────────────┬──────────────────────────────────┘
                            │ USB (V4L2 / UVC)
                    ┌───────▼────────┐
                    │ Hub USB        │  (recomendado, alimentado)
                    └───────┬────────┘
                    ┌───────▼────────┐
                    │ Logitech Brio  │  cámara USB UVC
                    └────────────────┘

   El MCU STM32U585 (Zephyr) NO participa: no hay GPIO, ADC ni
   comunicación MCU↔MPU en esta versión.
```

### Módulos

| Archivo | Responsabilidad |
|---|---|
| `app/main.py` | App FastAPI, endpoints, worker de vídeo, generador MJPEG, control de cámara, snapshots. |
| `app/config.py` | Carga de `.env`, validación, rutas seguras, configuración pública. |
| `app/camera.py` | `cv2.VideoCapture` en hilo dedicado, reconexión, lectura thread-safe. |
| `app/detector.py` | Haar cascade, ROI con margen, overlay de cajas/etiquetas/HUD. |
| `app/models_loader.py` | Backends ONNX y Caffe, preprocesado y contrato de predicción. |
| `python/main.py` | Entrypoint compatible con Arduino App Lab. |
| `static/` | UI sin build step (HTML + CSS + JS vanilla). |
| `training/` | Entrenamiento UTKFace fuera de la placa. |
| `scripts/` | Descarga de modelos, setup en placa, despliegue y arranque. |

---

## Principio de funcionamiento

1. **Arranque.** `load_settings()` lee `.env` y el entorno. `load_model()` intenta cargar el backend pedido (`auto` → ONNX → Caffe → ninguno). `FaceDetector` carga `haarcascade_frontalface_default.xml` (incluido en OpenCV).
2. **Captura.** `Camera` abre `/dev/video0` con backend `CAP_V4L2` (Linux) y, si falla, `CAP_ANY`. Fija resolución, FPS y `BUFFERSIZE=1` para minimizar latencia. Un hilo guarda siempre el último frame y se reconecta si el dispositivo desaparece (20 lecturas fallidas → reapertura cada 2 s).
3. **Pipeline único.** `VideoPipeline` es el **único** hilo que lee cámara, detecta, clasifica, dibuja y codifica. Publica el último JPEG protegido por lock. Los clientes de `/video` solo copian ese JPEG: **N navegadores no multiplican la CPU**.
4. **Detección.** Cada `INFER_EVERY_N` frames (o si aún no hay caras), el frame se convierte a gris, se ecualiza (`equalizeHist`) y se pasa a `detectMultiScale(scaleFactor=1.1, minNeighbors=5, minSize=(60,60))`. Cada caja se recorta con un **margen del 6 %** para incluir mentón/frente.
5. **Clasificación.** Cada ROI se envía al backend activo. Género se devuelve como `Masculino`/`Femenino`; edad como uno de los 8 buckets configurados. Si la confianza individual de una cabeza no supera `CONF_THRESHOLD`, la etiqueta muestra `Género?` o `Edad?`.
6. **Overlay.** Se dibujan cajas (verde/teal si hay predicción, ámbar si no), etiqueta `Género | Edad (conf%)` y HUD `Rostros: N | FPS: x.x`. El porcentaje mostrado es el **promedio** de las confianzas de género y edad.
7. **Publicación.** El frame anotado se codifica a JPEG (`JPEG_QUALITY`, por defecto 80) y se publica. `/video` emite chunks `multipart/x-mixed-replace; boundary=frame`.
8. **Reposo.** Si no hay clientes MJPEG, el worker entra en modo idle: lee la cámara cada 250 ms, **no ejecuta Haar ni la red y no codifica JPEG**. Esto mantiene el frame disponible para snapshots con CPU casi nula. Al conectarse un cliente, `add_client()` despierta al worker.

### Backends

| Backend | Entrada | Preprocesado | Salidas | Buckets de edad |
|---|---|---|---|---|
| `utkface_onnx` | 224×224 | `blobFromImage(face, 1/127.5, (224,224), (127.5,127.5,127.5), swapRB=True)` | `gender` sigmoide (0=M, 1=F), `age` softmax 8 | `AGE_BUCKETS` (configurable, 8 valores) |
| `caffe` | 227×227 | `blobFromImage(face, 1.0, (227,227), mean BGR, swapRB=False)` | `gender_net` 2 clases (0=M, 1=F), `age_net` 8 clases | **Fijos en código**: `0-2, 4-6, 8-12, 15-20, 25-32, 38-43, 48-53, 60-100` |
| `none` (degradado) | — | — | Etiqueta `Rostro detectado \| modelo no cargado` | — |

> ⚠️ Los buckets del modelo Caffe **no son** los de UTKFace. `AGE_BUCKETS` solo se aplica al backend ONNX. Si mezclas ambos verás rangos distintos en pantalla.

---

## Hardware requerido

| Elemento | Detalle |
|---|---|
| **Arduino UNO Q** | MPU Qualcomm QRB2210, ARM64/AArch64, Debian Linux. Variante reportada por el usuario: **8 GB RAM**. |
| **Cámara** | Logitech Brio (UVC). Cualquier cámara USB compatible con V4L2 que exponga `/dev/video0` puede servir. |
| **Hub USB** | Recomendado, preferiblemente **alimentado**. La Brio consume ancho de banda y corriente; un hub pasivo puede provocar desconexiones. |
| **Alimentación de la placa** | Fuente oficial de la UNO Q. |
| **Red** | Wi-Fi o Ethernet en la UNO Q para abrir la UI desde otro equipo. |
| **Host de desarrollo** | PC Linux/macOS/Windows para editar y desplegar (ADB o SSH). No es imprescindible si trabajas directamente en la placa. |
| **Opcional: GPU** | Solo para entrenar el modelo UTKFace (Colab o PC). No se usa en la placa. |

### Especificaciones de la UNO Q relevantes

| Bloque | Detalle | ¿Se usa aquí? |
|---|---|---|
| MPU | Qualcomm QRB2210, ARM64, Debian | ✅ Todo el runtime |
| MCU | STM32U585 (Cortex-M33, TrustZone) | ❌ No |
| RTOS del MCU | Zephyr | ❌ No |
| GPIO del MCU | 3.3 V | ❌ No se conecta nada |
| ADC del MCU | 12 bits (0–4095) | ❌ No se usa |
| FQBN del MCU | `arduino:zephyr:unoq` | ❌ No aplica |
| Serie del MCU | `/dev/ttyACM0` | ❌ No requerido |
| Cámara | USB UVC → `/dev/video0` | ✅ |
| Servidor | TCP `0.0.0.0:8000` | ✅ |

---

## Software requerido

### En la placa (runtime)

| Componente | Versión / notas |
|---|---|
| Debian Linux del MPU | El que trae la UNO Q. |
| Python | 3.9 o superior (según las suposiciones del plan). |
| `python3-venv` | Necesario para crear `.venv`. Si falta, se instala con autorización explícita: `sudo apt-get install -y python3-venv python3-pip`. |
| Dependencias | `requirements.txt`: FastAPI, Uvicorn, OpenCV headless, NumPy, Pillow, python-dotenv, python-multipart. |
| `curl` | Para `scripts/download_fallback_models.sh`. |
| `adb` / `ssh` | Solo en el host que despliega. |

> `opencv-python-headless` evita dependencias gráficas (no hay servidor X en la placa). No cambies a `opencv-python` "normal" en la UNO Q.

### En el host de desarrollo

| Herramienta | Uso |
|---|---|
| `adb` | Despliegue por USB. |
| `ssh` + `rsync` (opcional) | Despliegue por red. |
| `python3` | Verificaciones de sintaxis en host. |
| Navegador moderno | UI (Chrome, Firefox, Safari, Edge). |

---

## Puesta en marcha paso a paso

### 0. Obtener el proyecto en el host

```bash
cd /ruta/a/tu/workspace
ls AnalizadorRostroUNOQ
```

### 1. Obtener los modelos

#### Opción A — Fallback Caffe (rápida, ya disponible localmente)

Los 4 archivos ya están en `models/fallback/` (~88 MB) pero **no se versionan**. Si clonas el repo en otro equipo:

```bash
cd AnalizadorRostroUNOQ
bash scripts/download_fallback_models.sh
```

Descarga desde `https://github.com/smahesh29/Gender-and-Age-Detection/raw/master` y valida que los `.caffemodel` superen 1 MB. Resultado esperado:

```text
age_net.caffemodel: 45661480 bytes
gender_net.caffemodel: 45649168 bytes
Modelos Caffe listos en /.../models/fallback
```

Con `MODEL_BACKEND=auto` (por defecto), la app usará Caffe si no existe el ONNX.

#### Opción B — Modelo propio UTKFace (ONNX, requiere entrenamiento)

El archivo `models/utkface_mobilenetv2.onnx` **no existe en el repositorio**. Se genera fuera de la placa:

```bash
cd training
pip install -r requirements-train.txt
python train_utkface.py --download --epochs 50 --batch-size 32
# ...
cp output/utkface_mobilenetv2.onnx ../models/utkface_mobilenetv2.onnx
```

Guía completa: [`training/README.md`](training/README.md).

### 2. Conectar la cámara

```text
Logitech Brio → cable USB → hub USB (alimentado) → puerto USB de la UNO Q
```

Verifica en la placa que aparezca el dispositivo:

```bash
ls -l /dev/video*
# esperado: /dev/video0 (y quizá /dev/video1 para metadata)
```

### 3. Desplegar el proyecto en la placa

#### Por ADB (USB)

```bash
adb devices
# Debe listar el dispositivo, p. ej.:
# List of devices attached
# <serial>    device

./scripts/deploy_unoq.sh --adb --setup
```

El script copia `app`, `python`, `static`, `scripts`, `training`, `models`, `requirements.txt`, `app.yaml` y `.env.example` a `/home/arduino/ArduinoApps/AnalizadorRostroUNOQ`, y luego ejecuta `scripts/setup_board.sh` en la placa.

> Si `adb devices` sale vacío o muestra `unauthorized`, ver [Solución de problemas](#solución-de-problemas). **Sin dispositivo enumerado no hay nada que desplegar.**

#### Por SSH (red)

```bash
export UNOQ_HOST=192.168.1.50
export UNOQ_USER=arduino            # valor por defecto del script
# export UNOQ_SSH_KEY=~/.ssh/unoq_key   # si usas clave
./scripts/deploy_unoq.sh --ssh --setup
```

#### Modo automático

```bash
./scripts/deploy_unoq.sh --setup
```

Intenta ADB; si falla, usa SSH (solo si `UNOQ_HOST` está definido).

> El despliegue **no copia `.env`** (está excluido en rsync; ADB tampoco lo incluye). `setup_board.sh` crea `.env` a partir de `.env.example` si no existe. Edita el `.env` de la placa para añadir `API_TOKEN`, cambiar `MODEL_BACKEND`, etc.

### 4. Preparar el entorno en la placa

Si no usaste `--setup`:

```bash
adb shell 'cd /home/arduino/ArduinoApps/AnalizadorRostroUNOQ && bash scripts/setup_board.sh'
# o, si quieres que además descargue Caffe en la placa:
adb shell 'cd /home/arduino/ArduinoApps/AnalizadorRostroUNOQ && bash scripts/setup_board.sh --with-models'
```

`setup_board.sh` es idempotente: crea `.venv`, actualiza `pip`, instala `requirements.txt`, copia `.env.example` → `.env` si falta y, con `--with-models`, descarga Caffe.

### 5. Ejecutar

```bash
adb shell 'cd /home/arduino/ArduinoApps/AnalizadorRostroUNOQ && ./scripts/run.sh'
```

Salida esperada (ilustrativa):

```text
>> Iniciando AnalizadorRostroUNOQ con /home/arduino/ArduinoApps/AnalizadorRostroUNOQ/.venv/bin/python
INFO:     Started server process [1234]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
2026-01-01 12:00:00,000 INFO analizador: Analizador listo en http://0.0.0.0:8000 (backend=caffe, camara=/dev/video0, auth=off)
```

> La app queda en primer plano. Para dejarla en segundo plano desde ADB, usa `nohup` (ver [Ejemplos de uso](#ejemplos-de-uso-realistas)). App Lab la gestiona por sí mismo si usas `app.yaml`.
>
> Si tras el `adb push` el bit de ejecución se perdió, arranca con `bash scripts/run.sh` en lugar de `./scripts/run.sh`.

### 6. Abrir la interfaz

Averigua la IP de la placa:

```bash
adb shell 'hostname -I'
# o en la placa: ip -4 addr show
```

Abre en el navegador:

```text
http://192.168.1.50:8000/
```

Sustituye la IP por la real. La página muestra el stream, los badges de cámara/modelo, el contador de rostros y el panel de estado.

### 7. Alternativa: ejecutar en un PC para probar la UI

También puedes validar la app en tu equipo de desarrollo (con una webcam local):

```bash
cd AnalizadorRostroUNOQ
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m app.main
# abrir http://127.0.0.1:8000/
```

Si tu webcam no es el índice 0:

```bash
CAMERA_INDEX=1 python -m app.main
```

---

## Recursos, "pinout" y conexión

Esta aplicación **no usa los pines GPIO del MCU**. El "pinout" relevante es el de **recursos del sistema Linux**.

### Tabla de recursos

| Recurso | Valor / dispositivo | Tipo | Notas |
|---|---|---|---|
| MPU Linux | Qualcomm QRB2210 ARM64 | Procesador | Ejecuta FastAPI, OpenCV y la inferencia. |
| MCU STM32U585 | Zephyr RTOS | MCU | **No se usa.** Sin firmware propio. |
| Cámara USB | Logitech Brio (UVC) | USB | Conectada por hub USB. |
| Dispositivo de vídeo | `/dev/video0` | V4L2 | Índice configurable con `CAMERA_INDEX=0`. |
| Backend de captura | `cv2.CAP_V4L2` → `CAP_ANY` | Software | Fallback automático si V4L2 no abre. |
| Puerto HTTP | `8000/tcp` | Red | `PORT=8000`. |
| Bind | `0.0.0.0` | Red | Expuesto a la LAN; protégete con `API_TOKEN`. |
| Stream MJPEG | `GET /video` | HTTP | `multipart/x-mixed-replace; boundary=frame`. |
| Snapshots | `SNAP_FOLDER` (vacío) | FS | Desactivados por defecto; deben quedar dentro del proyecto. |
| Carpeta en placa | `/home/arduino/ArduinoApps/AnalizadorRostroUNOQ` | FS | `REMOTE_APP_DIR` configurable. |
| Entorno Python | `.venv/` | FS | Creado por `setup_board.sh`. |
| Buckets de edad | 8 clases | Config | `AGE_BUCKETS` (solo ONNX). |

### Pines del MCU (no usados)

| Pin / bloque | Conexión | Estado |
|---|---|---|
| Headers digitales D0–D13 | — | Sin uso. |
| Headers analógicos A0–A5 | — | Sin uso (el ADC de 12 bits no interviene). |
| `LED_BUILTIN` (D13) | — | Sin uso; solo se menciona como posible evolución futura. |
| 3V3 / 5V / GND de headers | — | No se alimenta nada desde ellos. |
| `/dev/ttyACM0` | — | No se abre ni se requiere. |

> Si en el futuro quieres un LED de estado controlado por el MCU, necesitarás firmware Zephyr y un canal de comunicación MPU↔MCU (USB/serie). Eso **no** forma parte de este proyecto.

### Diagrama de conexión

```text
        ┌──────────────────────────────┐
        │        Logitech Brio         │
        │  (USB-C o USB-A según cable) │
        └──────────────┬───────────────┘
                       │  USB (datos + alimentación)
                       ▼
        ┌──────────────────────────────┐
        │  Hub USB alimentado          │  ← recomendado
        │  (evita caídas de la Brio)   │
        └──────────────┬───────────────┘
                       │  USB
                       ▼
        ┌──────────────────────────────┐
        │  Arduino UNO Q               │
        │  puerto USB host del MPU     │
        │  Linux ve /dev/video0        │
        └──────────────────────────────┘

  Alimentación: fuente oficial de la UNO Q.
  No hay conexiones a GPIO, ADC, I2C, SPI ni UART del MCU.
```

**Pasos físicos:**

1. Conecta la Brio al hub USB.
2. Conecta el hub al puerto USB de la UNO Q (lado MPU Linux).
3. Alimenta la UNO Q con su fuente.
4. Comprueba `ls -l /dev/video*` en la placa.
5. Opcional: `v4l2-ctl --list-devices` si está instalado, para confirmar el nombre del dispositivo.

> Si conectas la Brio directamente a la UNO Q y funciona, perfecto; el hub solo aporta estabilidad. Evita hubs pasivos baratos y puertos USB 2.0 saturados si ves cortes.

---

## Configuración (.env)

Copia `.env.example` a `.env` y ajusta. Los valores por defecto también están en `app/config.py`.

```bash
cp .env.example .env
```

| Variable | Defecto | Rango / valores | Descripción |
|---|---|---|---|
| `CAMERA_INDEX` | `0` | entero ≥ 0 | Índice de `/dev/videoN`. |
| `WIDTH` | `640` | ≥ 64 | Ancho solicitado a la cámara. |
| `HEIGHT` | `480` | ≥ 64 | Alto solicitado. |
| `FPS_TARGET` | `15` | ≥ 1 | Ritmo objetivo del worker y de `CAP_PROP_FPS`. |
| `HOST` | `0.0.0.0` | IP | Bind de Uvicorn. |
| `PORT` | `8000` | 1–65535 | Puerto HTTP. |
| `MODEL_BACKEND` | `auto` | `auto`, `utkface_onnx`, `caffe` | Selección de backend. |
| `UTKFACE_ONNX_PATH` | `models/utkface_mobilenetv2.onnx` | ruta | Modelo propio. |
| `CAFFE_AGE_PROTO` | `models/fallback/deploy_age.prototxt` | ruta | Prototxt de edad. |
| `CAFFE_AGE_MODEL` | `models/fallback/age_net.caffemodel` | ruta | Pesos de edad. |
| `CAFFE_GENDER_PROTO` | `models/fallback/deploy_gender.prototxt` | ruta | Prototxt de género. |
| `CAFFE_GENDER_MODEL` | `models/fallback/gender_net.caffemodel` | ruta | Pesos de género. |
| `INFER_EVERY_N` | `5` | ≥ 1 | Cada cuántos frames se ejecuta Haar + red. |
| `CONF_THRESHOLD` | `0.6` | 0.0–1.0 | Umbral para mostrar género/edad o `?`. |
| `HAAR_PATH` | vacío | ruta | Vacío = cascade incluido en OpenCV. |
| `SNAP_FOLDER` | vacío | subcarpeta | Vacío = snapshots deshabilitados. |
| `SNAPSHOT_MIN_INTERVAL` | `2` | ≥ 0 s | Rate-limit entre snapshots (HTTP 429). |
| `AGE_BUCKETS` | `0-2,3-9,10-19,20-29,30-39,40-49,50-59,60+` | 8 valores | Orden exacto del ONNX. |
| `JPEG_QUALITY` | `80` | 30–95 | Calidad del stream. |
| `API_TOKEN` | vacío | texto | Si se define, los POST exigen `X-Api-Token`. |

### Reglas de validación

- `MODEL_BACKEND` inválido → se usa `auto` y se registra un warning.
- `AGE_BUCKETS` con un número de valores distinto de 8 → se restauran los 8 por defecto.
- `CONF_THRESHOLD` se recorta a `[0,1]`; `JPEG_QUALITY` a `[30,95]`.
- `SNAP_FOLDER` se resuelve contra la raíz del proyecto. Si escapa del árbol (`../`, ruta absoluta externa), se **rechaza** y el snapshot responde HTTP 400.
- `API_TOKEN` nunca se expone en `/api/config`; solo se indica `auth_enabled: true/false`.

### Generar un token

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Guárdalo solo en el `.env` de la placa (nunca en Git).

---

## API HTTP

| Método | Ruta | Autenticación | Respuesta |
|---|---|---|---|
| `GET` | `/` | No | `text/html` con `static/index.html`. |
| `GET` | `/static/*` | No | CSS/JS estáticos. |
| `GET` | `/video` | No | `multipart/x-mixed-replace; boundary=frame`. |
| `GET` | `/api/status` | No | JSON de estado. |
| `GET` | `/api/config` | No | JSON de configuración pública. |
| `GET` | `/healthz` | No | `{"ok": true}`. |
| `POST` | `/api/camera/start` | `X-Api-Token` si `API_TOKEN` | JSON; 503 si la cámara no abre en 1.5 s. |
| `POST` | `/api/camera/stop` | `X-Api-Token` si `API_TOKEN` | JSON. |
| `POST` | `/api/snapshot` | `X-Api-Token` si `API_TOKEN` | JSON `{ok, path}`; 429 rate-limit; 400 carpeta inválida; 503 deshabilitado/sin frame; 500 error de escritura. |

### `GET /api/status` — ejemplo ilustrativo

```json
{
  "camera_ok": true,
  "model_backend": "caffe",
  "model_backend_requested": "auto",
  "model_loaded": true,
  "degraded": false,
  "fps": 12.4,
  "faces_count": 2,
  "resolution": [640, 480],
  "message": "OK",
  "model_error": "",
  "camera_error": "",
  "pipeline_error": "",
  "pipeline_running": true,
  "stream_clients": 1
}
```

> Cifras ilustrativas del contrato, **no medidas en hardware**.

Campos clave:

| Campo | Significado |
|---|---|
| `camera_ok` | Hay frames frescos (< 3 s) del dispositivo. |
| `model_backend` | Backend realmente cargado (`utkface_onnx`, `caffe`, `none`). |
| `model_backend_requested` | Lo pedido en `.env`. |
| `model_loaded` | `false` = modo degradado (solo detección). |
| `degraded` | `true` si falta cámara o modelo. |
| `fps` | FPS suavizado del pipeline (0 en reposo). |
| `faces_count` | Rostros del último frame procesado. |
| `resolution` | `[ancho, alto]` reales del frame. |
| `message` | Estado legible en español. |
| `model_error` / `camera_error` / `pipeline_error` | Motivos de degradación. |
| `pipeline_running` | El hilo worker está vivo. |
| `stream_clients` | Navegadores conectados a `/video`. |

### `GET /api/config` — ejemplo

```json
{
  "width": 640,
  "height": 480,
  "fps_target": 15,
  "model_backend": "auto",
  "infer_every_n": 5,
  "conf_threshold": 0.6,
  "snap_enabled": false,
  "snapshot_min_interval": 2.0,
  "auth_enabled": false,
  "age_buckets": ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+"],
  "camera_control": true,
  "snapshot_endpoint": "/api/snapshot",
  "jpeg_quality": 80
}
```

### Códigos de error

| Código | Cuándo | `detail` típico |
|---|---|---|
| `400` | `SNAP_FOLDER` inválido o fuera del proyecto. | `SNAP_FOLDER inválido: ...` |
| `401` | Falta o no coincide `X-Api-Token`. | `Token de API ausente o inválido...` |
| `429` | Snapshot antes de `SNAPSHOT_MIN_INTERVAL`. | `Espera N s antes de guardar otro snapshot.` |
| `503` | Cámara no abre; snapshot deshabilitado; sin frame. | `No se pudo abrir la cámara en 1.5 s...` |
| `500` | Fallo al escribir el JPG. | `No se pudo guardar el snapshot: ...` |

---

## Operación de la interfaz web

| Elemento | Función |
|---|---|
| **Iniciar cámara** | `POST /api/camera/start`. Reconstruye cámara + pipeline y espera a que haya frames. |
| **Detener** | `POST /api/camera/stop`. Detiene el worker y libera `/dev/video0`. |
| **Snapshot** | Solo visible si `SNAP_FOLDER` está configurado. Guarda un JPG y muestra la ruta. |
| **Token API** | Campo visible si `API_TOKEN` está definido. Se guarda en `sessionStorage` (no en cookies). |
| **Badges** | Cámara, modelo y FPS con colores ok/warn/bad. |
| **Panel de estado** | Cámara, backend, modelo, resolución, frecuencia de inferencia y FPS. |
| **Aviso de privacidad** | Pie de página: procesamiento local y snapshots desactivados por defecto. |

### Interpretación de etiquetas

| Etiqueta | Significado |
|---|---|
| `Masculino \| 20-29 (87%)` | Género y bucket de edad; 87 % es el promedio de ambas confianzas. |
| `Femenino \| 50-59 (72%)` | Igual, con confianza media. |
| `Género? \| 30-39 (55%)` | La confianza de género no supera `CONF_THRESHOLD`. |
| `Masculino \| Edad? (58%)` | La confianza de edad no supera el umbral. |
| `Rostro detectado \| modelo no cargado` | No hay backend cargado; solo detección. |
| Caja **teal** | Rostro con predicción. |
| Caja **ámbar** | Rostro sin predicción (modelo ausente o ROI no válida). |
| HUD `Rostros: 3 \| FPS: 11.8` | Contador y FPS del último frame. |
| HUD `CAMARA OFFLINE` | Frame placeholder sin cámara. |

> El porcentaje **no es una probabilidad calibrada**; es una media aritmética de dos salidas de red. Úsalo como indicador cualitativo, nunca como certeza.

---

## Calibración y rendimiento

### Parámetros que puedes tocar

| Parámetro | Efecto al subirlo | Efecto al bajarlo | Recomendación |
|---|---|---|---|
| `INFER_EVERY_N` | Menos CPU, etiquetas más "pegajosas". | Más CPU, reacción más rápida. | 5 en uso normal; 8–10 si el FPS cae. |
| `CONF_THRESHOLD` | Más etiquetas `?`, menos falsos positivos. | Más etiquetas, más errores. | 0.6 por defecto; 0.7–0.8 en demos exigentes. |
| `WIDTH`/`HEIGHT` | Más detalle, más CPU y ancho de banda. | Más FPS, rostros más pequeños. | 640×480; 480×360 si hay poco CPU. |
| `JPEG_QUALITY` | Mejor imagen, más bytes. | Más fluidez en Wi-Fi malo. | 80; 65–70 en red congestionada. |
| `FPS_TARGET` | Ritmo objetivo más alto. | Menos carga. | 15; 10 si la placa va justa. |
| `CAMERA_INDEX` | — | — | Cambia si la Brio no es `/dev/video0`. |

### Notas de rendimiento

- La inferencia corre en **CPU** (OpenCV DNN no usa GPU en el QRB2210). MobileNetV2 a 224×224 tarda decenas de ms por rostro según carga.
- Haar es comparativamente barato; el coste dominante es la red y la codificación JPEG.
- El diseño de **un solo worker** evita que varios navegadores multipliquen el coste.
- Sin clientes MJPEG no se codifica JPEG ni se infiere: el consumo en reposo es mínimo.
- No se ha medido el rendimiento real en la placa; ajusta con `/api/status` (`fps`, `faces_count`) y observa la temperatura.

### Cómo saber si está bien calibrado

1. Abre `/api/status`.
2. Con 0–1 rostros, el FPS debería acercarse a `FPS_TARGET`. Si está muy por debajo, sube `INFER_EVERY_N` o baja resolución.
3. Si las etiquetas cambian demasiado rápido o "parpadean", baja `INFER_EVERY_N` (más inferencia) o sube `CONF_THRESHOLD`.
4. Si aparecen muchas etiquetas `?`, baja `CONF_THRESHOLD` ligeramente o mejora la iluminación.
5. Si hay falsos positivos de detección (cajas en objetos), sube `minNeighbors`... **no es configurable por `.env`**; requiere editar `app/detector.py` (ver [`docs/developer.md`](docs/developer.md)).

---

## Entrenamiento del modelo UTKFace

Resumen; la guía completa está en [`training/README.md`](training/README.md).

```bash
cd training
pip install -r requirements-train.txt

# Autotest del parser de nombres (no requiere TensorFlow)
python train_utkface.py --self-test

# Entrenamiento (Colab o PC con GPU recomendado)
python train_utkface.py --download --epochs 50 --batch-size 32 --output-dir output
```

- **Dataset:** UTKFace (`jangedoo/utkface-new` en Kaggle), nombres `[edad]_[genero]_[raza]_[timestamp].jpg`.
- **Etiquetas:** edad 0–116 → 8 buckets; género `0` = masculino, `1` = femenino.
- **Modelo:** MobileNetV2 `include_top=False` + dos cabezas (sigmoide para género, softmax 8 para edad).
- **Split:** 80/10/10 estratificado.
- **Salida:** `output/utkface_mobilenetv2.onnx` (opset 13), más `evaluation.json`, matriz de confusión e historial.
- **Cópialo a** `models/utkface_mobilenetv2.onnx` y usa `MODEL_BACKEND=utkface_onnx` (o `auto`).

**Contrato crítico:** la entrada es RGB 224×224 normalizada `x/127.5 - 1`; la salida de edad debe tener exactamente 8 clases en el orden de `AGE_BUCKETS`. Si cambias los buckets en el entrenamiento, cámbialos también en `.env`.

---

## Ejemplos de uso realistas

### 1. Comprobar que la app responde

```bash
curl -s http://192.168.1.50:8000/healthz
# {"ok":true}
```

### 2. Ver el estado en JSON legible

```bash
curl -s http://192.168.1.50:8000/api/status | python3 -m json.tool
```

Salida esperada (ilustrativa):

```json
{
  "camera_ok": true,
  "model_backend": "caffe",
  "model_backend_requested": "auto",
  "model_loaded": true,
  "degraded": false,
  "fps": 12.4,
  "faces_count": 1,
  "resolution": [640, 480],
  "message": "OK",
  "pipeline_running": true,
  "stream_clients": 1
}
```

### 3. Abrir el stream desde la terminal

```bash
# Guarda 10 segundos de MJPEG para inspeccionarlo con un reproductor
timeout 10 curl -sN http://192.168.1.50:8000/video -o /tmp/stream.mjpeg
ls -lh /tmp/stream.mjpeg
```

### 4. Guardar un snapshot con token

```bash
export API_TOKEN="el-token-que-pusiste-en-el-.env"
curl -s -X POST \
  -H "X-Api-Token: ${API_TOKEN}" \
  http://192.168.1.50:8000/api/snapshot | python3 -m json.tool
```

```json
{
  "ok": true,
  "path": "/home/arduino/ArduinoApps/AnalizadorRostroUNOQ/snapshots/snapshot_20260101_120000_123.jpg"
}
```

> Requiere `SNAP_FOLDER=snapshots` en el `.env` de la placa. Sin esa variable, responde 503.

### 5. Detener y reiniciar la cámara por API

```bash
curl -s -X POST -H "X-Api-Token: ${API_TOKEN}" \
  http://192.168.1.50:8000/api/camera/stop | python3 -m json.tool

curl -s -X POST -H "X-Api-Token: ${API_TOKEN}" \
  http://192.168.1.50:8000/api/camera/start | python3 -m json.tool
```

### 6. Cliente Python mínimo (sin dependencias externas)

```python
import json
import urllib.request

BASE = "http://192.168.1.50:8000"

with urllib.request.urlopen(f"{BASE}/api/status", timeout=5) as response:
    status = json.load(response)

print(f"Cámara: {'OK' if status['camera_ok'] else 'sin señal'}")
print(f"Backend: {status['model_backend']}")
print(f"Rostros: {status['faces_count']}  FPS: {status['fps']}")
```

### 7. Arrancar en segundo plano desde ADB

```bash
adb shell 'cd /home/arduino/ArduinoApps/AnalizadorRostroUNOQ && \
  nohup ./scripts/run.sh > /tmp/analizador.log 2>&1 & echo $!'

# Ver el log
adb shell 'tail -n 40 /tmp/analizador.log'
```

> `run.sh` usa `exec`, por lo que el PID que devuelve `nohup` es el del proceso Python.

### 8. Verificación estática antes de desplegar

```bash
python3 -m py_compile app/main.py app/camera.py app/config.py \
  app/detector.py app/models_loader.py python/main.py training/train_utkface.py

bash -n scripts/deploy_unoq.sh scripts/setup_board.sh scripts/run.sh \
  scripts/download_fallback_models.sh

node --check static/app.js
python3 training/train_utkface.py --self-test
```

Todo debe salir sin errores. `--self-test` imprime 9/9 casos correctos.

---

## Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| Página no carga | App no arrancada, IP/puerto incorrectos, firewall. | Verifica el log de Uvicorn; prueba `curl http://IP:8000/healthz`. |
| `/api/status` → `camera_ok: false` | Cámara ausente, ocupada o índice incorrecto. | `ls -l /dev/video*`; prueba `CAMERA_INDEX=1`; cierra otros programas que usen la cámara. |
| Stream con "CAMARA NO DISPONIBLE" | Sin frames del dispositivo. | Reconecta la Brio/hub; revisa permisos; reinicia la app. |
| `model_loaded: false` | Faltan modelos o rutas mal. | Consulta `model_error`; ejecuta `download_fallback_models.sh` o coloca el ONNX. |
| Etiqueta `Género?` / `Edad?` constante | Confianza baja o umbral alto. | Mejora la luz/frontalidad; baja `CONF_THRESHOLD` a 0.5. |
| FPS muy bajo | CPU saturada, resolución alta, varios rostros. | Sube `INFER_EVERY_N`, baja `WIDTH/HEIGHT`, baja `JPEG_QUALITY`. |
| El puerto 8000 está ocupado | Otro servicio. | `ss -ltnp \| grep 8000`; cambia `PORT` en `.env`. |
| `adb devices` vacío | Cable/permiso/modo USB. | Ver abajo. |
| `adb devices` → `unauthorized` | Clave RSA no aceptada. | Ver abajo. |
| SSH timeout | IP incorrecta, red distinta, servicio SSH apagado. | Ver abajo. |
| `No module named app` | Ejecutado fuera de la raíz. | `cd` a la raíz o usa `scripts/run.sh`. |
| `Permission denied: /dev/video0` | Usuario sin grupo `video`. | Ver abajo. |
| `pip install` falla por NumPy 2 | OpenCV < 4.11 no soporta NumPy 2. | Respeta `numpy<2` de `requirements.txt`. |

### Cámara no detectada

```bash
# En la placa
ls -l /dev/video*
cat /proc/asound/cards          # la Brio también expone audio
lsusb                           # debe aparecer Logitech
```

- Si no aparece nada: prueba otro cable/puerto/hub; descarta un hub pasivo.
- Si aparece `/dev/video1` o superior: ajusta `CAMERA_INDEX`.
- Si aparece y desaparece: problema de alimentación o cable; usa hub alimentado.

### Permisos de `/dev/video0`

```bash
id
ls -l /dev/video0
# crw-rw---- 1 root video 81, 0 ... /dev/video0
```

Si tu usuario no está en `video`:

```bash
sudo usermod -aG video "$USER"
# cierra sesión y vuelve a entrar (o reinicia) para aplicar el grupo
```

> Requiere autorización del usuario; es un cambio en el sistema de la placa.

### Modelo ausente

```bash
curl -s http://127.0.0.1:8000/api/status | python3 -m json.tool
# revisa "model_error"
ls -l models/fallback/ models/utkface_mobilenetv2.onnx 2>/dev/null
```

Causas típicas:

- `Modelo ONNX no encontrado: ...` → entrena y copia el `.onnx`, o usa `MODEL_BACKEND=caffe`.
- `Faltan modelos Caffe: ...` → `bash scripts/download_fallback_models.sh`.
- `Error cargando ONNX: ...` → ONNX corrupto o incompatible con OpenCV; reexporta con opset 13.

### FPS bajo

1. Comprueba `/api/status` (`fps`, `resolution`, `stream_clients`).
2. Sube `INFER_EVERY_N` a 8–10.
3. Baja a 480×360 o 320×240.
4. Baja `JPEG_QUALITY` a 65–70.
5. Cierra clientes MJPEG innecesarios (el modo idle ahorra CPU sin clientes).
6. Verifica que no haya otro proceso consumiendo CPU: `top` / `htop`.

### Puerto ocupado

```bash
ss -ltnp | grep ':8000'
# o
lsof -i :8000
```

Cambia `PORT=8080` en `.env` y reinicia. Recuerda abrir `http://IP:8080/`.

### ADB `unauthorized` o sin dispositivos

```bash
adb kill-server
adb start-server
adb devices
```

- Acepta el diálogo "Allow USB debugging" en la placa si aparece.
- Prueba otro cable USB de datos (muchos cables solo cargan).
- Verifica que la placa esté encendida y en el modo correcto.
- En Linux, si el dispositivo aparece en `lsusb` pero no en `adb`, puede faltar una regla udev. Comprueba el VID:PID real con `lsusb` y revisa si existe una regla del fabricante:

```bash
lsusb
ls /etc/udev/rules.d/ | grep -i -E 'arduino|unoq'
```

> No copies reglas udev de terceros sin verificar el VID:PID. Si necesitas crear una, hazlo con autorización y recarga con `sudo udevadm control --reload-rules && sudo udevadm trigger`.

### SSH timeout

```bash
ping -c 3 "$UNOQ_HOST"
ssh -v "${UNOQ_USER:-arduino}@${UNOQ_HOST}"
```

- Confirma la IP con `adb shell 'hostname -I'` si ADB funciona.
- Asegura que ambos equipos están en la misma red/VLAN.
- Verifica que el servicio SSH esté activo en la placa.
- Si usas clave: `UNOQ_SSH_KEY=~/.ssh/tu_clave`.
- Si el firewall bloquea, abre el puerto 22 en la red local.

### El snapshot devuelve 503

- `SNAP_FOLDER` está vacío → configura `SNAP_FOLDER=snapshots`.
- No hay frame todavía → espera a que la cámara entregue vídeo.
- La carpeta no se pudo crear → revisa permisos y que la ruta esté dentro del proyecto.

### El snapshot devuelve 400

La ruta resuelta escapa de la raíz del proyecto. Usa una subcarpeta relativa, por ejemplo `SNAP_FOLDER=snapshots`.

### El snapshot devuelve 429

Espera `SNAPSHOT_MIN_INTERVAL` segundos (2 por defecto) o ajústalo.

### El snapshot devuelve 401

Define el header `X-Api-Token` con el valor exacto de `API_TOKEN`, o deja `API_TOKEN` vacío en redes de confianza.

---

## Privacidad y datos biométricos

> **Aviso:** la estimación de género y edad a partir de rostros es tratamiento de **datos biométricos** y de **categorías especiales** según el RGPD (art. 9). No es una decisión neutral ni una verdad objetiva.

Buenas prácticas que aplica el proyecto:

- **Procesamiento local:** no se envían imágenes a ningún servicio externo. No hay telemetría ni nube.
- **Sin almacenamiento por defecto:** `SNAP_FOLDER` está vacío; no se guardan JPG salvo que lo actives.
- **Snapshots confinados:** la carpeta debe quedar dentro del árbol del proyecto; se rechazan rutas externas.
- **Token opcional:** `API_TOKEN` protege inicio/parada de cámara y snapshots.
- **Rate-limit:** evita ráfagas de capturas.
- **Logs sin imágenes:** los errores se registran como texto.
- **UI informada:** el pie de página muestra el aviso de privacidad.

Responsabilidades del operador:

1. **Informar y obtener consentimiento** de las personas ante la cámara, especialmente si se muestran resultados en una pantalla pública.
2. **Minimizar:** no actives snapshots si no los necesitas; bórralos cuando termines.
3. **Limitar exposición:** no publiques el puerto 8000 en Internet. En una LAN de confianza, aun así considera `API_TOKEN`.
4. **No tomar decisiones automatizadas** (acceso, contratación, diagnóstico, precio) basadas en estas etiquetas.
5. **Respetar la normativa local** sobre videovigilancia y biometría.
6. **No usar con menores** sin base legal y salvaguardas adecuadas.

> El modelo devuelve una **categoría estimada**, no una identidad. No hay reconocimiento facial, no hay embeddings de identidad y no se compara contra una base de datos.

---

## Limitaciones conocidas

| Limitación | Detalle |
|---|---|
| Solo rostros frontales | Haar `frontalface_default` falla con perfiles, cabezas giradas, oclusiones y rostros muy pequeños (< 60 px). |
| Género binario | El modelo solo distingue Masculino/Femenino; no representa identidades no binarias. |
| Edad en buckets | No es edad exacta; los límites son gruesos y el error aumenta en extremos. |
| Sesgo del dataset | UTKFace tiene sesgos demográficos, de iluminación y de etiquetado; el modelo los hereda. |
| Caffe es de 2015 | `age_net`/`gender_net` son modelos clásicos con precisión limitada en condiciones modernas. |
| Sin seguimiento | No hay tracking entre frames; las cajas se recalculan y pueden "saltar". |
| Sin anti-spoofing | Una foto o una pantalla pueden detectarse como rostro. |
| Sin identificación | No reconoce personas; no apto para control de acceso. |
| CPU solamente | Sin GPU en el QRB2210 para OpenCV DNN; el FPS depende de la carga. |
| Iluminación | Luz frontal difusa funciona mejor; contraluz y sombras duras degradan todo. |
| Distancia | La Brio tiene buen campo de visión, pero rostros lejanos caen por debajo de `minSize`. |
| Un solo proceso | `workers=1` es obligatorio: hay estado compartido en memoria (cámara, pipeline, modelo). |
| App Lab | `app.yaml` es un descriptor opcional; no se ha validado en hardware. |

---

## Atribuciones y licencias

### UTKFace

- Zhang, Z., Song, Y., & Qi, H. (2017). *Age Progression / Regression in the Wild*. ICCV Workshops.
- Dataset distribuido en Kaggle (`jangedoo/utkface-new`). **Uso académico**; revisa la licencia del dataset antes de redistribuir imágenes o modelos derivados.

### Modelos Caffe (Levi & Hassner)

- Levi, G., & Hassner, T. (2015). *Age and Gender Classification using Convolutional Neural Networks*. CVPR Workshops.
- Pesos obtenidos de `smahesh29/Gender-and-Age-Detection`. Revisa la licencia del repositorio original antes de redistribuirlos.
- Los `.caffemodel` **no se versionan** en este repositorio.

### OpenCV

- Clasificador `haarcascade_frontalface_default.xml`, incluido en `opencv-python`. OpenCV es Apache 2.0.
- Haar cascade original: Viola, P., & Jones, M. (2001). *Rapid Object Detection using a Boosted Cascade of Simple Features*. CVPR.

### Código del proyecto

- No se incluye un archivo `LICENSE` en el repositorio. Antes de redistribuir el código, define y añade la licencia que corresponda.
- FastAPI (MIT), Uvicorn (BSD), NumPy (BSD), Pillow (HPND), python-dotenv (BSD).

---

## Advertencias específicas de la UNO Q

1. **La placa opera a 3.3 V en los GPIO del MCU.** Aunque esta app no usa pines, si añades hardware al MCU no conectes señales de 5 V sin level shifter.
2. **El ADC del MCU es de 12 bits (0–4095)**, no de 10 bits como el Uno clásico. Irrelevante aquí porque no se usa, pero crítico si amplías el proyecto.
3. **El MCU ejecuta Zephyr RTOS.** Las librerías AVR clásicas (`avr/io.h`, `Arduino.h` de AVR, `SoftwareSerial`, etc.) no aplican. Este proyecto no compila firmware.
4. **El despliegue no es `arduino-cli upload`.** Es copia de archivos por **ADB o SSH** y ejecución de Python en el MPU. Si `adb` no ve la placa, no hay despliegue posible.
5. **No alimentes la Brio desde un puerto débil.** Usa hub alimentado; las desconexiones intermitentes suelen ser eléctricas.
6. **No publiques el puerto 8000 en Internet.** Es una app de laboratorio sin TLS. Usa una red de confianza y `API_TOKEN`.
7. **Vigila la temperatura** en ejecuciones largas: la inferencia sostenida en CPU calienta el MPU.
8. **No ejecutes `pip install` como root** dentro del proyecto; usa el `.venv`.
9. **La app es un proceso único.** No uses `uvicorn --workers >1`: duplicaría cámara, modelo y estado.
10. **El snapshot es dato biométrico.** Actívalo solo con base legal y bórralo al terminar.

### ¿Por qué no se usa arduino-cli?

Porque **no hay ningún sketch `.ino`**. El FQBN `arduino:zephyr:unoq` y el puerto `/dev/ttyACM0` corresponden al MCU STM32U585, que aquí no ejecuta nada. Los comandos clásicos **no aplican**:

```bash
# ❌ NO usar en este proyecto (no hay sketch .ino que compilar)
# arduino-cli compile --fqbn arduino:zephyr:unoq .
# arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:zephyr:unoq .
```

El "build" real es la instalación de dependencias Python; el "upload" real es `scripts/deploy_unoq.sh`. Si en el futuro añades un sketch para el MCU, esos comandos volverían a tener sentido **para ese sketch**, no para esta app.

---

## Estructura del repositorio

```text
AnalizadorRostroUNOQ/
├── README.md                     ← este documento
├── PLAN.md                       ← plan de implementación (interno)
├── AUDIT_REPORT.md               ← auditoría (interno)
├── .env.example                  ← plantilla de configuración
├── requirements.txt              ← dependencias runtime (placa)
├── app.yaml                      ← descriptor opcional Arduino App Lab
├── .gitignore
├── app/
│   ├── __init__.py
│   ├── main.py                   ← FastAPI + VideoPipeline
│   ├── config.py                 ← settings + validación + rutas seguras
│   ├── camera.py                 ← captura V4L2 con hilo y reconexión
│   ├── detector.py               ← Haar + overlay
│   └── models_loader.py          ← ONNX / Caffe / degradado
├── python/
│   └── main.py                   ← entrypoint App Lab
├── static/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── training/
│   ├── train_utkface.py
│   ├── README.md
│   └── requirements-train.txt
├── models/
│   ├── README.md
│   ├── .gitkeep
│   └── fallback/                 ← Caffe (88 MB, gitignored)
├── scripts/
│   ├── download_fallback_models.sh
│   ├── setup_board.sh
│   ├── deploy_unoq.sh
│   └── run.sh
└── docs/
    ├── datasheet.md              ← manual de usuario
    ├── developer.md              ← guía técnica
    └── BITACORA.md               ← decisiones y lecciones
```

---

## Documentación adicional

| Documento | Contenido |
|---|---|
| [`docs/datasheet.md`](docs/datasheet.md) | Manual de usuario: operación, etiquetas, estados, errores, privacidad. |
| [`docs/developer.md`](docs/developer.md) | Arquitectura interna, hilos/locks, contrato de API, backends, extensión. |
| [`docs/BITACORA.md`](docs/BITACORA.md) | Problemas, decisiones técnicas descartadas y lecciones aprendidas. |
| [`training/README.md`](training/README.md) | Entrenamiento UTKFace paso a paso (Colab/Kaggle). |
| [`models/README.md`](models/README.md) | Contrato de ambos backends y verificación rápida. |
| [`PLAN.md`](PLAN.md) | Especificación funcional y criterios de aceptación. |
| [`AUDIT_REPORT.md`](AUDIT_REPORT.md) | Resultado de la auditoría estática. |

---

**Resumen en una línea:** una webcam USB, una UNO Q y un FastAPI con OpenCV que detecta caras y estima género/rango de edad en local, sin nube, sin MCU y sin guardar imágenes por defecto.
