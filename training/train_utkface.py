#!/usr/bin/env python3
"""Entrenamiento del modelo propio UTKFace (género + rango de edad).

Pensado para Google Colab o PC con GPU. NO se ejecuta en la Arduino UNO Q.

Pipeline:
  1. (Opcional) descarga del dataset desde Kaggle.
  2. Parseo de nombres `[edad]_[genero]_[raza]_[fecha].jpg`.
  3. Filtrado de imágenes corruptas.
  4. Bucketización de edad a 8 clases.
  5. Split 80 / 10 / 10 (train / val / test).
  6. MobileNetV2 + dos cabezas (gender sigmoid, age softmax 8).
  7. Exportación ONNX (opset 13) con salidas nombradas "gender" y "age".

Uso típico en Colab:
    !pip install -r requirements-train.txt
    !python train_utkface.py --epochs 50 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Debe coincidir EXACTAMENTE con AGE_BUCKETS de app/config.py.
AGE_BUCKETS: List[str] = [
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60+",
]
GENDER_LABELS = ["Masculino", "Femenino"]
IMG_SIZE: Tuple[int, int] = (224, 224)
# UTKFace real: [edad]_[genero]_[raza]_[timestamp].jpg, con timestamps de
# 13-17 dígitos y variantes que añaden sufijos como ".chip.jpg". El regex se
# aplica sobre el stem ya normalizado (ver _clean_stem).
FILENAME_RE = re.compile(r"^(\d{1,3})_(\d)_\d+_(.+)$")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
# Sufijos que pueden quedar dentro del stem al usar Path.stem.
_STEM_SUFFIXES = (".chip", ".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --------------------------------------------------------------------- datos


def parse_buckets(spec: str) -> List[str]:
    buckets = [item.strip() for item in spec.split(",") if item.strip()]
    if len(buckets) != 8:
        raise ValueError(
            f"Se esperaban 8 buckets de edad, se recibieron {len(buckets)}: {buckets}"
        )
    return buckets


def _bucket_bounds(bucket: str) -> Tuple[int, int]:
    if bucket.endswith("+"):
        return int(bucket[:-1]), 200
    low, _, high = bucket.partition("-")
    return int(low), int(high)


def bucket_for_age(age: int, buckets: Sequence[str]) -> int:
    for index, bucket in enumerate(buckets):
        low, high = _bucket_bounds(bucket)
        if low <= age <= high:
            return index
    return len(buckets) - 1


def _clean_stem(path: Path) -> str:
    """Normaliza el stem quitando extensiones y sufijos tipo ``.chip``.

    ``Path("26_1_3_20170116174519429.jpg.chip.jpg").stem`` es
    ``26_1_3_20170116174519429.jpg.chip``; hay que eliminar ``.chip`` y el
    ``.jpg`` interno antes de aplicar el regex.
    """
    stem = Path(path).stem
    changed = True
    while changed:
        changed = False
        lowered = stem.lower()
        for suffix in _STEM_SUFFIXES:
            if lowered.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
                break
    return stem


def parse_filename(path: Path) -> Tuple[int, int] | None:
    """Devuelve ``(edad, genero)`` o ``None`` si el nombre no es UTKFace."""
    match = FILENAME_RE.match(_clean_stem(path))
    if not match:
        return None
    age = int(match.group(1))
    gender = int(match.group(2))
    if age < 0 or age > 120 or gender not in (0, 1):
        return None
    return age, gender


SELF_TEST_CASES: Sequence[Tuple[str, Tuple[int, int] | None]] = (
    ("26_1_3_20170116174519429.jpg", (26, 1)),
    ("26_1_3_20170116174519429.jpg.chip.jpg", (26, 1)),
    ("1_0_0_20161219140623097.jpg", (1, 0)),
    ("116_1_2_20170116174525125.jpg", (116, 1)),
    ("116_1_2_20170116174525125.jpg.chip.jpg", (116, 1)),
    ("0_0_4_20170103235929388.jpg", (0, 0)),
    ("not_a_utkface.jpg", None),
    ("26_2_3_20170116174519429.jpg", None),  # género inválido
    ("200_1_3_20170116174519429.jpg", None),  # edad fuera de rango
)


def run_self_test(verbose: bool = True) -> bool:
    """Valida el parser con nombres reales de UTKFace y sufijos ``.chip.jpg``."""
    failures: List[str] = []
    for name, expected in SELF_TEST_CASES:
        got = parse_filename(Path(name))
        if got != expected:
            failures.append(f"{name}: esperado {expected}, obtenido {got}")
        elif verbose:
            print(f"  OK  {name} -> {got}")
    if failures:
        print("Self-test del parser FALLÓ:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return False
    if verbose:
        print(f"Self-test del parser OK ({len(SELF_TEST_CASES)} casos).")
    return True


def collect_samples(dataset_dir: Path, validate: bool = True) -> Tuple[List[str], List[int], List[int]]:
    import cv2  # import local: el host sin dependencias puede compilar el archivo

    files = [
        path
        for path in sorted(dataset_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not files:
        raise SystemExit(
            f"No se encontraron imágenes en {dataset_dir}. "
            "Descarga UTKFace o ajusta --dataset-dir."
        )

    paths: List[str] = []
    genders: List[int] = []
    ages: List[int] = []
    skipped_names = 0
    skipped_corrupt = 0

    for count, path in enumerate(files, start=1):
        parsed = parse_filename(path)
        if parsed is None:
            skipped_names += 1
            continue
        if validate:
            image = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
            if image is None:
                skipped_corrupt += 1
                continue
        age, gender = parsed
        paths.append(str(path))
        ages.append(age)
        genders.append(gender)
        if count % 2000 == 0:
            print(f"  ... revisadas {count} imágenes ({len(paths)} válidas)")

    if not paths:
        raise SystemExit("Ninguna imagen pasó el parseo/validación.")

    print(
        f"Dataset: {len(paths)} imágenes válidas "
        f"(descartadas por nombre: {skipped_names}, corruptas: {skipped_corrupt})"
    )
    return paths, genders, ages


def maybe_download(dataset_dir: Path, slug: str) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if any(dataset_dir.rglob("*.jpg")):
        print(f"dataset_dir={dataset_dir} ya contiene imágenes; se omite la descarga.")
        return
    print(f"Descargando dataset Kaggle '{slug}' en {dataset_dir} ...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(dataset_dir), "--unzip"],
            check=True,
        )
    except FileNotFoundError:
        raise SystemExit(
            "El comando 'kaggle' no está instalado. Instala requirements-train.txt "
            "y configura ~/.kaggle/kaggle.json."
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Falló la descarga de Kaggle (código {exc.returncode}).")


def split_dataset(
    paths: Sequence[str],
    genders: Sequence[int],
    ages: Sequence[int],
    buckets: Sequence[str],
    seed: int,
) -> Dict[str, Tuple[List[str], List[int], List[int]]]:
    from sklearn.model_selection import train_test_split

    labels = [bucket_for_age(age, buckets) for age in ages]
    counts = {label: labels.count(label) for label in set(labels)}
    stratify = labels if min(counts.values()) >= 3 else None
    if stratify is None:
        print("Aviso: clases de edad escasas; el split no será estratificado.")

    x_train, x_rest, g_train, g_rest, a_train, a_rest = train_test_split(
        list(paths),
        list(genders),
        list(ages),
        test_size=0.2,
        random_state=seed,
        stratify=stratify,
    )
    rest_labels = [bucket_for_age(age, buckets) for age in a_rest]
    rest_counts = {label: rest_labels.count(label) for label in set(rest_labels)}
    stratify_rest = rest_labels if min(rest_counts.values()) >= 2 else None
    x_val, x_test, g_val, g_test, a_val, a_test = train_test_split(
        x_rest,
        g_rest,
        a_rest,
        test_size=0.5,
        random_state=seed,
        stratify=stratify_rest,
    )
    print(
        f"Split -> train={len(x_train)} val={len(x_val)} test={len(x_test)}"
    )
    return {
        "train": (x_train, g_train, a_train),
        "val": (x_val, g_val, a_val),
        "test": (x_test, g_test, a_test),
    }


# ------------------------------------------------------------------ tensorflow


def build_datasets(split, buckets, batch_size: int, seed: int):
    import tensorflow as tf

    autotune = tf.data.AUTOTUNE

    def make_loader(num_buckets: int):
        def load(path, gender, age):
            raw = tf.io.read_file(path)
            image = tf.io.decode_jpeg(raw, channels=3)
            image = tf.image.resize(image, IMG_SIZE, antialias=True)
            image = tf.cast(image, tf.float32)
            # Misma normalización que la inferencia: x / 127.5 - 1
            image = image / 127.5 - 1.0
            labels = {
                "gender": tf.cast(gender, tf.float32),
                "age": tf.one_hot(tf.cast(age, tf.int32), num_buckets),
            }
            return image, labels

        return load

    augmenter = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(20.0 / 360.0, fill_mode="nearest"),
            tf.keras.layers.RandomZoom(0.15, fill_mode="nearest"),
            tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomBrightness(0.2, value_range=(-1.0, 1.0)),
        ],
        name="augmentation",
    )

    def make_dataset(data, training: bool):
        paths, genders, ages = data
        ds = tf.data.Dataset.from_tensor_slices(
            (
                tf.constant(paths),
                tf.constant(genders, dtype=tf.float32),
                tf.constant(ages, dtype=tf.int32),
            )
        )
        if training:
            ds = ds.shuffle(min(len(paths), 4096), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(make_loader(len(buckets)), num_parallel_calls=autotune)
        if training:
            ds = ds.map(
                lambda image, labels: (augmenter(image, training=True), labels),
                num_parallel_calls=autotune,
            )
        return ds.batch(batch_size).prefetch(autotune)

    return (
        make_dataset(split["train"], True),
        make_dataset(split["val"], False),
        make_dataset(split["test"], False),
    )


def build_model(num_buckets: int, learning_rate: float):
    import tensorflow as tf
    from tensorflow.keras import layers

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="image")
    features = base(inputs, training=False)
    pooled = layers.GlobalAveragePooling2D(name="gap")(features)

    gender = layers.Dense(128, activation="relu", name="gender_fc")(pooled)
    gender = layers.Dropout(0.5, name="gender_dropout")(gender)
    gender = layers.Dense(1, activation="sigmoid", name="gender")(gender)

    age = layers.Dense(128, activation="relu", name="age_fc")(pooled)
    age = layers.Dropout(0.5, name="age_dropout")(age)
    age = layers.Dense(num_buckets, activation="softmax", name="age")(age)

    model = tf.keras.Model(inputs, {"gender": gender, "age": age}, name="utkface_mobilenetv2")
    compile_model(model, learning_rate)
    return model, base


def compile_model(model, learning_rate: float) -> None:
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "gender": "binary_crossentropy",
            "age": "categorical_crossentropy",
        },
        loss_weights={"gender": 1.0, "age": 2.0},
        metrics={
            "gender": [tf.keras.metrics.BinaryAccuracy(name="acc")],
            "age": [tf.keras.metrics.CategoricalAccuracy(name="acc")],
        },
    )


def unfreeze_top(base, layers_to_unfreeze: int) -> None:
    base.trainable = True
    if layers_to_unfreeze <= 0:
        return
    for layer in base.layers[:-layers_to_unfreeze]:
        layer.trainable = False
    print(f"Fine-tuning: {layers_to_unfreeze} capas superiores de MobileNetV2 liberadas.")


# --------------------------------------------------------------- export/métricas


def ensure_onnx_output_names(path: Path) -> List[str]:
    """Garantiza que el ONNX exponga salidas llamadas 'gender' y 'age'."""
    try:
        import onnx
        from onnx import helper
    except ImportError:
        print("Aviso: onnx no está instalado; no se renombran las salidas.")
        return []

    model = onnx.load(str(path))
    graph = model.graph
    existing = [output.name for output in graph.output]
    if len(graph.output) == 2 and set(existing) >= {"gender", "age"}:
        print(f"ONNX exportado con salidas: {existing}")
        return existing

    gender_out = None
    age_out = None
    for output in graph.output:
        dims = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        last = dims[-1] if dims else None
        if last == 1 and gender_out is None:
            gender_out = output
        elif last == 8 and age_out is None:
            age_out = output

    if gender_out is None or age_out is None:
        print(
            "Aviso: no se pudieron identificar las salidas por forma; "
            f"se conservan {existing}. La inferencia las mapea por nombre/tamaño."
        )
        return existing

    for source, target in ((gender_out, "gender"), (age_out, "age")):
        source_name = source.name
        graph.node.append(helper.make_node("Identity", [source_name], [target]))
        source.name = target

    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    final = [output.name for output in graph.output]
    print(f"ONNX exportado con salidas renombradas: {final}")
    return final


def _build_nchw_export_model(model):
    """Envuelve el modelo NHWC con una entrada NCHW para OpenCV DNN.

    La app usa ``cv2.dnn.blobFromImage``, que produce un blob NCHW
    ``(1, 3, 224, 224)``. El modelo Keras entrena con NHWC, así que se
    inserta un ``Transpose`` interno al exportar. De este modo el ONNX
    acepta exactamente el tensor que genera la app sin tocar su código.
    """
    import tensorflow as tf

    nchw_input = tf.keras.Input(
        shape=(3, IMG_SIZE[0], IMG_SIZE[1]), name="input"
    )
    nhwc = tf.transpose(nchw_input, perm=(0, 2, 3, 1))
    outputs = model(nhwc, training=False)
    return tf.keras.Model(
        inputs=nchw_input,
        outputs=outputs,
        name="utkface_nchw_export",
    )


def _onnx_shape(value_info) -> List[int]:
    return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]


def validate_onnx_model(path: Path) -> bool:
    """Valida la entrada NCHW, las salidas y una inferencia real.

    Falla con ``RuntimeError`` si el ONNX no es compatible con la app.
    Si ``onnxruntime`` no está instalado, valida la estructura con ``onnx``
    y omite la inferencia (en GitHub Actions sí se instala).
    """
    import numpy as np
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    if len(model.graph.input) != 1:
        raise RuntimeError(
            f"El ONNX debe tener una entrada, tiene {len(model.graph.input)}."
        )
    input_dims = _onnx_shape(model.graph.input[0])
    if len(input_dims) != 4 or input_dims[1:] != [3, IMG_SIZE[0], IMG_SIZE[1]]:
        raise RuntimeError(
            f"Entrada ONNX inesperada {input_dims}; se esperaba [N,3,"
            f"{IMG_SIZE[0]},{IMG_SIZE[1]}] (NCHW)."
        )

    output_dims = {out.name: _onnx_shape(out) for out in model.graph.output}
    if set(output_dims) != {"gender", "age"}:
        raise RuntimeError(
            f"Salidas ONNX inesperadas {sorted(output_dims)}; "
            "se esperaban exactamente 'gender' y 'age'."
        )
    if output_dims["gender"][-1] != 1:
        raise RuntimeError(
            f"Salida 'gender' con forma {output_dims['gender']}; se esperaba 1."
        )
    if output_dims["age"][-1] != len(AGE_BUCKETS):
        raise RuntimeError(
            f"Salida 'age' con forma {output_dims['age']}; se esperaban "
            f"{len(AGE_BUCKETS)} clases."
        )

    try:
        import onnxruntime as ort
    except ImportError:
        print(
            "Aviso: onnxruntime no está instalado; se omite la inferencia "
            "de validación del ONNX."
        )
        return True

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.random.default_rng(0).uniform(
        -1.0, 1.0, (1, 3, IMG_SIZE[0], IMG_SIZE[1])
    ).astype(np.float32)
    values = session.run(None, {input_name: dummy})
    result = {
        output.name: value for output, value in zip(session.get_outputs(), values)
    }
    if result["gender"].shape[-1] != 1:
        raise RuntimeError(
            f"Inferencia gender con forma {result['gender'].shape}; se esperaba 1."
        )
    if result["age"].shape[-1] != len(AGE_BUCKETS):
        raise RuntimeError(
            f"Inferencia age con forma {result['age'].shape}; se esperaban "
            f"{len(AGE_BUCKETS)} clases."
        )
    if not np.isfinite(result["gender"]).all() or not np.isfinite(result["age"]).all():
        raise RuntimeError("El ONNX produjo valores no finitos en la validación.")
    print(
        "ONNX validado: entrada "
        f"{input_dims}, gender {tuple(result['gender'].shape)}, "
        f"age {tuple(result['age'].shape)}"
    )
    return True


def export_onnx(model, output_path: Path) -> Path:
    import tensorflow as tf
    import tf2onnx

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Entrada NCHW: es el layout que produce cv2.dnn.blobFromImage en la app.
    signature = (
        tf.TensorSpec(
            (None, 3, IMG_SIZE[0], IMG_SIZE[1]), tf.float32, name="input"
        ),
    )
    export_model = _build_nchw_export_model(model)
    try:
        tf2onnx.convert.from_keras(
            export_model,
            input_signature=signature,
            opset=13,
            output_path=str(output_path),
        )
    except Exception as exc:
        print(f"Exportación directa falló ({exc}); reintentando con el modelo congelado...")
        frozen = tf.keras.models.clone_model(model)
        frozen.set_weights(model.get_weights())
        export_model = _build_nchw_export_model(frozen)
        tf2onnx.convert.from_keras(
            export_model,
            input_signature=signature,
            opset=13,
            output_path=str(output_path),
        )
    ensure_onnx_output_names(output_path)
    validate_onnx_model(output_path)
    print(f"ONNX listo: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return output_path


def export_tflite(model, output_path: Path) -> None:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    output_path.write_bytes(converter.convert())
    print(f"TFLite opcional: {output_path}")


def evaluate(model, test_ds, truth, buckets: Sequence[str], output_dir: Path) -> None:
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix

    metrics = model.evaluate(test_ds, return_dict=True, verbose=1)
    print("\nMétricas de test:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    predictions = model.predict(test_ds, verbose=0)
    gender_prob = np.asarray(predictions["gender"]).reshape(-1)
    age_prob = np.asarray(predictions["age"])
    gender_pred = (gender_prob >= 0.5).astype(int)
    age_pred = np.argmax(age_prob, axis=1)

    test_paths, test_genders, test_ages = truth
    age_true = np.array([bucket_for_age(age, buckets) for age in test_ages])

    print("\nReporte de género:")
    print(
        classification_report(
            test_genders,
            gender_pred,
            target_names=GENDER_LABELS,
            zero_division=0,
        )
    )
    print("Reporte de edad:")
    print(
        classification_report(
            age_true,
            age_pred,
            labels=list(range(len(buckets))),
            target_names=list(buckets),
            zero_division=0,
        )
    )
    matrix = confusion_matrix(age_true, age_pred, labels=list(range(len(buckets))))
    print("Matriz de confusión (edad):")
    print(matrix)

    summary = {
        "metrics": {key: float(value) for key, value in metrics.items()},
        "age_confusion_matrix": matrix.tolist(),
        "age_buckets": list(buckets),
        "gender_labels": GENDER_LABELS,
        "test_size": len(test_paths),
    }
    (output_dir / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(buckets)))
        ax.set_yticks(range(len(buckets)))
        ax.set_xticklabels(buckets, rotation=45, ha="right")
        ax.set_yticklabels(buckets)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title("Matriz de confusión — edad")
        fig.colorbar(image)
        fig.tight_layout()
        fig.savefig(output_dir / "confusion_matrix.png", dpi=140)
        plt.close(fig)
        print(f"Gráfica: {output_dir / 'confusion_matrix.png'}")
    except Exception as exc:  # pragma: no cover - matplotlib opcional
        print(f"Aviso: no se generó la gráfica ({exc}).")


def save_history(history, output_dir: Path) -> None:
    payload = {key: [float(v) for v in values] for key, values in history.history.items()}
    (output_dir / "training_history.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        if "loss" in payload:
            axes[0].plot(payload["loss"], label="train")
        if "val_loss" in payload:
            axes[0].plot(payload["val_loss"], label="val")
        axes[0].set_title("Pérdida")
        axes[0].legend()

        for key in ("gender_acc", "age_acc"):
            if key in payload:
                axes[1].plot(payload[key], label=key)
            if f"val_{key}" in payload:
                axes[1].plot(payload[f"val_{key}"], label=f"val_{key}")
        axes[1].set_title("Exactitud por cabeza")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(output_dir / "training_history.png", dpi=140)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"Aviso: no se generó la gráfica de historial ({exc}).")


# ------------------------------------------------------------------------ CLI


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena género + edad con UTKFace.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/utkface"),
        help="Carpeta con las imágenes UTKFace.",
    )
    parser.add_argument("--download", action="store_true", help="Descarga el dataset de Kaggle.")
    parser.add_argument(
        "--kaggle-dataset", default="jangedoo/utkface-new", help="Slug del dataset en Kaggle."
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--fine-tune-layers", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--buckets", default=",".join(AGE_BUCKETS), help="8 buckets separados por coma."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"), help="Carpeta de artefactos."
    )
    parser.add_argument(
        "--onnx-name", default="utkface_mobilenetv2.onnx", help="Nombre del ONNX final."
    )
    parser.add_argument("--tflite", action="store_true", help="Exportar también TFLite.")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="No verificar imágenes corruptas (más rápido).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Valida el parser de nombres UTKFace y sale (no requiere TensorFlow).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1

    random.seed(args.seed)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    try:
        import numpy as np
        import tensorflow as tf
    except ImportError as exc:
        print(
            "TensorFlow no está disponible. Instala training/requirements-train.txt "
            f"(detalle: {exc})",
            file=sys.stderr,
        )
        return 1

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    buckets = parse_buckets(args.buckets)
    if args.download:
        maybe_download(args.dataset_dir, args.kaggle_dataset)

    paths, genders, ages = collect_samples(args.dataset_dir, validate=not args.no_validate)
    split = split_dataset(paths, genders, ages, buckets, args.seed)
    train_ds, val_ds, test_ds = build_datasets(split, buckets, args.batch_size, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, base = build_model(len(buckets), args.learning_rate)
    model.summary()

    checkpoint = args.output_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    save_history(history, args.output_dir)

    if args.fine_tune_epochs > 0 and args.fine_tune_layers > 0:
        unfreeze_top(base, args.fine_tune_layers)
        compile_model(model, args.learning_rate / 10.0)
        fine_history = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=len(history.history.get("loss", [])),
            epochs=len(history.history.get("loss", [])) + args.fine_tune_epochs,
            callbacks=callbacks,
            verbose=1,
        )
        fine_dir = args.output_dir / "fine_tune"
        fine_dir.mkdir(parents=True, exist_ok=True)
        save_history(fine_history, fine_dir)

    model.save(args.output_dir / "utkface_mobilenetv2.keras")
    evaluate(model, test_ds, split["test"], buckets, args.output_dir)

    onnx_path = args.output_dir / args.onnx_name
    export_onnx(model, onnx_path)
    if args.tflite:
        export_tflite(model, args.output_dir / "utkface_mobilenetv2.tflite")

    print("\nListo. Copia el ONNX a la UNO Q en models/utkface_mobilenetv2.onnx")
    print(f"  {onnx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
