from __future__ import annotations

import argparse
import pathlib

import tensorflow as tf

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
GATE_CLASSES = (
    "not_animal",
    "dog",
    "cat",
    "insect",
    "bird",
    "fish",
    "arachnid",
    "other_animal",
)
SPECIALISTS = ("insect", "bird", "fish", "arachnid")


def make_dataset(directory: pathlib.Path, class_names: list[str], training: bool) -> tf.data.Dataset:
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=training,
        seed=42,
    ).map(
        lambda images, labels: (
            tf.keras.applications.mobilenet_v2.preprocess_input(images),
            labels,
        ),
        num_parallel_calls=AUTOTUNE,
    ).prefetch(AUTOTUNE)


def build_model(class_count: int) -> tf.keras.Model:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
        ]
    )
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    features = base(augmentation(inputs), training=False)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dropout(0.2)(features)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(features)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def export_tflite(model: tf.keras.Model, output: pathlib.Path, labels: list[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    output.write_bytes(converter.convert())
    output.with_name(f"{output.stem}_labels.txt").write_text("\n".join(labels) + "\n")


def train_one(
    name: str,
    root: pathlib.Path,
    output: pathlib.Path,
    class_names: list[str],
    epochs: int,
) -> None:
    train = make_dataset(root / "train", class_names, training=True)
    validation = make_dataset(root / "val", class_names, training=False)
    test = make_dataset(root / "test", class_names, training=False)
    model = build_model(len(class_names))
    model.fit(train, validation_data=validation, epochs=epochs)
    loss, accuracy = model.evaluate(test, verbose=0)
    print(f"{name}: test_accuracy={accuracy:.4f} test_loss={loss:.4f}")
    export_tflite(model, output / f"{name}.tflite", class_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train gate and animal specialist TFLite models.")
    parser.add_argument("--data", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    train_one("animal_gate", args.data / "gate", args.output, list(GATE_CLASSES), args.epochs)
    for specialist in SPECIALISTS:
        train_one(
            f"{specialist}_classifier",
            args.data / specialist,
            args.output,
            sorted(path.name for path in (args.data / specialist / "train").iterdir() if path.is_dir()),
            args.epochs,
        )


if __name__ == "__main__":
    main()
