# Referencias — AnalizadorRostroUNOQ

Recopilación de documentación oficial y comunitaria usada para diseñar,
implementar y verificar este proyecto. Los enlaces se agrupan por tema y se
indica **para qué sirve cada uno** dentro del repositorio.

> **Nota:** las versiones citadas corresponden al momento de la escritura
> (2026). Las APIs de OpenCV, FastAPI y ONNX evolucionan; si actualizas
> `requirements.txt`, contrasta primero la documentación de la versión que
> instales. Los enlaces externos no forman parte del repositorio y pueden
> cambiar.

---

## 1. Detección de rostros

| Recurso | Enlace | Utilidad en este proyecto |
|---|---|---|
| OpenCV — Cascade Classifier (tutorial) | <https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html> | Explica `CascadeClassifier` y `detectMultiScale`, usados en `app/detector.py`. |
| OpenCV — `haarcascade_frontalface_default.xml` | <https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml> | Cascada frontal incluida en `opencv-python`; es el detector por defecto (`FaceDetector`). |
| OpenCV — `CascadeClassifier` (API) | <https://docs.opencv.org/4.x/db/d28/classcv_1_1CascadeClassifier.html> | Referencia de `detectMultiScale`, `scaleFactor`, `minNeighbors` y `minSize`. |
| Viola & Jones (2001) | <https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf> | Paper original del detector Haar que usa el proyecto; útil para entender sus límites. |
| OpenCV Zoo — YuNet | <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet> | Detector alternativo (CNN ligera) recomendado para perfiles y luz difícil. **No está integrado** en esta versión; la guía de extensión está en `docs/developer.md`. |
| OpenCV — `FaceDetectorYN` (API) | <https://docs.opencv.org/4.x/df/d20/classcv_1_1FaceDetectorYN.html> | API que usaría una futura integración de YuNet (`setInputSize`, `detect`). |
| OpenCV — Face Detection with YuNet (tutorial) | <https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html> | Ejemplo oficial de YuNet + landmarks; base para el plan de migración. |

**Decisión actual:** se usa Haar por ser cero-dependencias (viene con OpenCV),
barato en CPU y suficiente para rostros frontales. YuNet queda documentado como
evolución, no como característica presente.

---

## 2. Modelos de edad y género

| Recurso | Enlace | Utilidad en este proyecto |
|---|---|---|
| UTKFace — página oficial | <https://susanqq.github.io/UTKFace/> | Descripción del dataset, etiquetas y condiciones de uso. |
| UTKFace — paper (Zhang, Song & Qi, 2017) | <https://arxiv.org/abs/1702.02722> | *Age Progression / Regression in the Wild*; base académica del entrenamiento. |
| UTKFace — Kaggle (`jangedoo/utkface-new`) | <https://www.kaggle.com/datasets/jangedoo/utkface-new> | Descarga alternativa usada por `training/train_utkface.py --download`. |
| UTKFace — Hugging Face (`nlphuji/utk_faces`) | <https://huggingface.co/datasets/nlphuji/utk_faces> | Origen que descarga el workflow de GitHub Actions. |
| MobileNetV2 — paper | <https://arxiv.org/abs/1801.04381> | Arquitectura base del modelo ONNX (backbone congelado + dos cabezas). |
| TensorFlow / Keras — MobileNetV2 | <https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2> | `include_top=False`, pesos ImageNet y preprocesado usados en el entrenamiento. |
| Levi & Hassner (2015) — paper | <https://openaccess.thecvf.com/content_cvpr_workshops_2015/W08/papers/Levi_Age_and_Gender_2015_CVPR_paper.pdf> | Modelo Caffe de fallback (`age_net` / `gender_net`). |
| `smahesh29/Gender-and-Age-Detection` | <https://github.com/smahesh29/Gender-and-Age-Detection> | Repositorio del que `scripts/download_fallback_models.sh` obtiene los `.caffemodel` y prototxt. Revisa su licencia antes de redistribuir. |
| Caffe — modelo y formato | <https://caffe.berkeleyvision.org/> | Referencia histórica del formato `.caffemodel` / `.prototxt` que consume `CaffeBackend`. |

**Advertencia de licencia:** UTKFace se distribuye para uso académico. El modelo
derivado se publica con fines de investigación y demostración; **no** para
vigilancia, perfilado biométrico ni decisiones automatizadas. Las imágenes del
dataset no se incluyen en el repositorio.

---

## 3. ONNX y ejecución de redes en CPU

| Recurso | Enlace | Utilidad en este proyecto |
|---|---|---|
| ONNX — sitio oficial | <https://onnx.ai/> | Formato abierto de grafos usado por `models/utkface_mobilenetv2.onnx`. |
| ONNX — especificación | <https://github.com/onnx/onnx> | Operadores, versionado y reglas del formato. |
| ONNX — operadores | <https://onnx.ai/onnx/operators/> | Consulta de `Sigmoid`, `Softmax`, `Transpose`, `Reshape`, etc. |
| ONNX — changelog de opsets | <https://onnx.ai/onnx/repo-docs/Changelog.html> | El modelo se exporta con **opset 13**; aquí se verifican cambios entre opsets. |
| OpenCV — módulo `dnn` | <https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html> | `cv2.dnn.readNetFromONNX` y ejecución en CPU sin ONNX Runtime. |
| OpenCV — `blobFromImage` | <https://docs.opencv.org/4.x/d6/d0f/group__dnn.html> | Preprocesado `(x/127.5)-1`, `swapRB=True`, tamaño 224×224. |
| ONNX Runtime | <https://onnxruntime.ai/> | Alternativa de inferencia; **no** se usa en la placa para evitar otra dependencia pesada. |
| tf2onnx | <https://github.com/onnx/tf2onnx> | Exportación Keras → ONNX en `training/train_utkface.py`. |

**Contrato crítico:** entrada NCHW `[N,3,224,224]`, RGB normalizado
`x/127.5 - 1`, salidas `gender` (sigmoide, 1 valor) y `age` (softmax, 8 clases).
Cualquier cambio de orden de buckets, `swapRB` o normalización invalida las
etiquetas aunque el grafo siga ejecutándose.

---

## 4. Servidor web y streaming

| Recurso | Enlace | Utilidad en este proyecto |
|---|---|---|
| FastAPI — documentación | <https://fastapi.tiangolo.com/> | Framework HTTP de `app/main.py`. |
| FastAPI — `StreamingResponse` | <https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse> | Base del endpoint `/video` (generador asíncrono). |
| FastAPI — `StaticFiles` | <https://fastapi.tiangolo.com/tutorial/static-files/> | Montaje de `static/` en `/static`. |
| Starlette — `StreamingResponse` | <https://www.starlette.io/responses/#streamingresponse> | Implementación subyacente y manejo de `media_type`. |
| Uvicorn — documentación | <https://www.uvicorn.org/> | Servidor ASGI; se ejecuta con **un solo worker**. |
| Uvicorn — settings | <https://www.uvicorn.org/settings/> | `host`, `port`, `log_level` y por qué no usar `--workers > 1` aquí. |
| RFC 2046 — Multipart | <https://www.rfc-editor.org/rfc/rfc2046> | Definición de `multipart/x-mixed-replace`, el formato del stream MJPEG. |
| MDN — MIME types | <https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/MIME_types> | Contexto de `multipart/x-mixed-replace; boundary=frame`. |
| MDN — `<img>` | <https://developer.mozilla.org/en-US/docs/Web/HTML/Element/img> | El stream se consume con `<img src="/video">`; no se necesitan WebSockets. |

**Por qué MJPEG:** es trivial de servir, funciona en cualquier navegador con
`<img>`, no requiere build ni codecs en el cliente y encaja con el patrón
“un productor, muchos consumidores”. La contrapartida es que no hay audio,
control de congestión ni baja latencia comparable a WebRTC.

---

## 5. Hardware: Arduino UNO Q

| Recurso | Enlace | Utilidad en este proyecto |
|---|---|---|
| Arduino UNO Q — documentación | <https://docs.arduino.cc/hardware/uno-q> | Especificaciones de la placa, MPU, MCU y alimentación. |
| Arduino UNO Q — host mode | <https://docs.arduino.cc/tutorials/uno-q/host-mode> | Requisito para usar una cámara USB (hub USB-C PD o VIN + JCTL). |
| Arduino App Lab | <https://docs.arduino.cc/software/app-lab/> | Descriptor opcional `app.yaml` y entrypoint `python/main.py`. |
| Qualcomm QRB2210 | <https://www.qualcomm.com/products/internet-of-things/industrial/industrial-controls/qrb2210> | Características del SoC ARM64 que ejecuta Linux, OpenCV y la inferencia. |
| Debian — documentación | <https://www.debian.org/doc/> | Sistema base del MPU; gestión de paquetes, grupos y systemd. |
| OpenCV — `VideoCapture` | <https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html> | `CAP_V4L2`, `CAP_FFMPEG`, `CAP_PROP_FPS`, `CAP_PROP_BUFFERSIZE`. |
| V4L2 — kernel docs | <https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/v4l2.html> | Interfaz `/dev/videoN` usada por cámaras UVC. |

**Aclaración importante:** el MCU STM32U585 (Zephyr) **no participa**. No hay
sketch `.ino`, ni GPIO, ni ADC, ni comunicación MPU↔MCU en esta versión.

---

## 6. Dependencias Python

| Recurso | Enlace | Utilidad en este proyecto |
|---|---|---|
| Python — `venv` | <https://docs.python.org/3/library/venv.html> | Entorno aislado de `requirements.txt`. |
| pip — `requirements.txt` | <https://pip.pypa.io/en/stable/user_guide/#requirements-files> | Formato de dependencias y rangos de versión. |
| NumPy | <https://numpy.org/doc/stable/> | Arrays BGR, preprocesado y salidas del modelo. |
| Pillow | <https://pillow.readthedocs.io/> | Guardado de snapshots JPEG con calidad 92. |
| python-dotenv | <https://github.com/theskumar/python-dotenv> | Carga de `.env`; hay un parser mínimo de respaldo en `app/config.py`. |
| python-multipart | <https://github.com/Kludex/python-multipart> | Dependencia de FastAPI para formularios/multipart. |

Las versiones exactas y sus rangos están en [`../requirements.txt`](../requirements.txt).
No se instala `opencv-python` completo en la placa: se usa
`opencv-python-headless` para evitar dependencias gráficas.

---

## 7. Entrenamiento y CI

| Recurso | Enlace | Utilidad en este proyecto |
|---|---|---|
| GitHub Actions | <https://docs.github.com/actions> | Workflow `training/train-utkface.workflow.yml` (copiar a `.github/workflows/`). |
| `actions/setup-python` | <https://github.com/actions/setup-python> | Fija Python 3.11 en el runner de entrenamiento. |
| GitHub Actions — artifacts | <https://docs.github.com/actions/using-workflows/storing-workflow-data-as-artifacts> | Publicación del ONNX entrenado. |
| GitHub CLI — `gh release` | <https://cli.github.com/manual/gh_release> | Descarga del asset `utkface_mobilenetv2.onnx` desde el release `v1.0.0`. |
| scikit-learn — métricas | <https://scikit-learn.org/stable/modules/model_evaluation.html> | `classification_report` y matriz de confusión en `evaluation.json`. |
| Matplotlib | <https://matplotlib.org/> | Gráficos de entrenamiento y matriz de confusión. |

---

## 8. Seguridad, privacidad y normativa

| Recurso | Enlace | Utilidad en este proyecto |
|---|---|---|
| RGPD — artículo 9 | <https://gdpr-info.eu/art-9-gdpr/> | Datos biométricos como categoría especial; base del aviso de privacidad. |
| AEPD — biometría | <https://www.aepd.es/> | Guías de la autoridad española sobre tratamiento de datos biométricos. |
| RFC 5737 — direcciones de documentación | <https://www.rfc-editor.org/rfc/rfc5737> | `192.0.2.0/24`, `198.51.100.0/24` y `203.0.113.0/24` se usan en los ejemplos para no exponer IPs reales. |
| OWASP — Path Traversal | <https://owasp.org/www-community/attacks/Path_Traversal> | Justifica la validación de `SNAP_FOLDER` contra la raíz del proyecto. |
| OWASP — ASVS | <https://owasp.org/www-project-application-security-verification-standard/> | Marco para revisar autenticación, validación y exposición de la app. |
| Tor Project — onion services | <https://community.torproject.org/onion-services/> | No se usa aquí; referenciado solo si se quisiera exponer la UI fuera de la LAN (no recomendado sin TLS/auth). |

**Modelo de amenaza resumido:** app de laboratorio en LAN de confianza, sin TLS,
sin autenticación en `GET /video`, con token opcional para POST. No debe
publicarse directamente en Internet.

---

## 9. Documentación interna del repositorio

| Documento | Contenido |
|---|---|
| [`../README.md`](../README.md) | Visión general, instalación, API, rendimiento y límites. |
| [`datasheet.md`](datasheet.md) | Manual de usuario: operación, etiquetas, estados y mantenimiento. |
| [`developer.md`](developer.md) | Arquitectura interna, hilos, contrato de API y guía de extensión. |
| [`BITACORA.md`](BITACORA.md) | Decisiones, problemas reales y lecciones aprendidas. |
| [`../models/README.md`](../models/README.md) | Contrato de los backends ONNX y Caffe. |
| [`../training/README.md`](../training/README.md) | Entrenamiento UTKFace paso a paso. |

---

## 10. Cómo citar este proyecto

Si usas este trabajo en un contexto académico o educativo, puedes citar el
repositorio y, por separado, los trabajos originales de UTKFace, MobileNetV2,
Viola–Jones y Levi–Hassner. No atribuyas los modelos de terceros a este
repositorio: se distribuyen desde sus fuentes originales y conservan sus
licencias.
