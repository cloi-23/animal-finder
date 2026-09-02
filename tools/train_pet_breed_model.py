from __future__ import annotations

import argparse
import pathlib
import random
from typing import Sequence

import tensorflow as tf


# ============================================================
# Oxford-IIIT Pet breeds
# Class IDs in the Oxford dataset are 1..37.
# We convert them to 0..36 for TensorFlow.
# ============================================================

BREEDS = [
    "Abyssinian",
    "American Bulldog",
    "American Pit Bull Terrier",
    "Basset Hound",
    "Beagle",
    "Bengal",
    "Birman",
    "Bombay",
    "Boxer",
    "British Shorthair",
    "Chihuahua",
    "Egyptian Mau",
    "English Cocker Spaniel",
    "English Setter",
    "German Shorthaired",
    "Great Pyrenees",
    "Havanese",
    "Japanese Chin",
    "Keeshond",
    "Leonberger",
    "Maine Coon",
    "Miniature Pinscher",
    "Newfoundland",
    "Persian",
    "Pomeranian",
    "Pug",
    "Ragdoll",
    "Russian Blue",
    "Saint Bernard",
    "Samoyed",
    "Scottish Terrier",
    "Shiba Inu",
    "Siamese",
    "Sphynx",
    "Staffordshire Bull Terrier",
    "Wheaten Terrier",
    "Yorkshire Terrier",
]


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
SEED = 42
UNKNOWN_LABEL = "Unknown"


# ============================================================
# Read Oxford annotation files
# ============================================================

def read_split(
    annotation_file: pathlib.Path,
) -> tuple[list[str], list[int]]:

    paths = []
    labels = []

    for line in annotation_file.read_text().splitlines():

        if not line:
            continue

        if line.startswith("#"):
            continue

        fields = line.split()

        if len(fields) < 2:
            continue

        image_name = fields[0]
        class_id = int(fields[1]) - 1

        if class_id < 0 or class_id >= len(BREEDS):
            raise ValueError(
                f"Invalid Oxford class ID: {class_id + 1}"
            )

        paths.append(image_name + ".jpg")
        labels.append(class_id)

    return paths, labels


def read_unknown_images(
    image_dir: pathlib.Path,
) -> list[pathlib.Path]:

    extensions = {".jpg", ".jpeg", ".png"}

    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def split_items(
    paths: Sequence[pathlib.Path | str],
    labels: Sequence[int],
    validation_fraction: float = 0.10,
) -> tuple[list[pathlib.Path | str], list[int], list[pathlib.Path | str], list[int]]:

    items = list(zip(paths, labels))
    random.Random(SEED).shuffle(items)

    split_index = max(1, int(len(items) * (1 - validation_fraction)))
    train_items = items[:split_index]
    validation_items = items[split_index:]

    return (
        [item[0] for item in train_items],
        [item[1] for item in train_items],
        [item[0] for item in validation_items],
        [item[1] for item in validation_items],
    )


# ============================================================
# Load image
#
# IMPORTANT:
# We DO NOT normalize here.
#
# Images remain:
#
#     0 .. 255
#
# The model performs normalization itself.
# ============================================================

def load_image(
    path: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:

    image = tf.io.read_file(path)

    image = tf.io.decode_jpeg(
        image,
        channels=3,
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    return image, label


def augment_image(
    image: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=20.0)
    image = tf.image.random_contrast(image, lower=0.80, upper=1.20)
    image = tf.image.random_saturation(image, lower=0.80, upper=1.20)
    image = tf.clip_by_value(image, 0.0, 255.0)

    return image, label


# ============================================================
# Create tf.data dataset
# ============================================================

def make_dataset(
    image_dir: pathlib.Path,
    paths: Sequence[pathlib.Path | str],
    labels: list[int],
    training: bool,
) -> tf.data.Dataset:

    full_paths = [
        str(image_dir / path)
        for path in paths
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            full_paths,
            labels,
        )
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE,
    )

    if training:
        dataset = dataset.map(
            augment_image,
            num_parallel_calls=AUTOTUNE,
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        AUTOTUNE
    )

    return dataset


# ============================================================
# Build transfer-learning classifier
#
# Training uses Oxford-IIIT Pet plus an optional explicit Unknown class.
# ============================================================

def build_model(
    class_count: int,
) -> tuple[tf.keras.Model, tf.keras.Model]:

    inputs = tf.keras.Input(
        shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3,
        ),
        dtype=tf.float32,
        name="image",
    )

    # --------------------------------------------------------
    # Better backbone: pretrained MobileNetV2 on ImageNet.
    #
    # Input from camera/image stays in the standard 0..255 range.
    # The MobileNetV2 preprocessing matches the ImageNet weights.
    # --------------------------------------------------------

    x = tf.keras.layers.Lambda(
        lambda image: tf.keras.applications.mobilenet_v2.preprocess_input(image),
        name="preprocess",
    )(inputs)

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3,
        ),
        include_top=False,
        weights="imagenet",
        pooling=None,
    )

    backbone.trainable = False

    x = backbone(x)

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="feature_pool",
    )(x)

    x = tf.keras.layers.Dropout(
        0.50,
        name="dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        class_count,
        activation=None,
        name="logits",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="oxford_iiit_pet_breed_model",
    )

    return model, backbone


def set_fine_tuning(
    backbone: tf.keras.Model,
    trainable_layers: int,
) -> None:

    backbone.trainable = True

    for layer in backbone.layers[:-trainable_layers]:
        layer.trainable = False

    for layer in backbone.layers[-trainable_layers:]:
        layer.trainable = not isinstance(
            layer,
            tf.keras.layers.BatchNormalization,
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images",
        type=pathlib.Path,
        required=True,
    )

    parser.add_argument(
        "--annotations",
        type=pathlib.Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
    )

    parser.add_argument(
        "--unknown-images",
        type=pathlib.Path,
        help=(
            "Directory of non-Oxford animal and non-animal images. "
            "These become an explicit Unknown class."
        ),
    )

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--warmup-epochs", type=int, default=5)

    args = parser.parse_args()

    # --------------------------------------------------------
    # Set random seeds
    # --------------------------------------------------------

    random.seed(SEED)

    tf.random.set_seed(
        SEED
    )

    # --------------------------------------------------------
    # Read Oxford train/test files
    # --------------------------------------------------------

    train_paths, train_labels = read_split(
        args.annotations / "trainval.txt"
    )

    test_paths, test_labels = read_split(
        args.annotations / "test.txt"
    )

    train_paths, train_labels, validation_paths, validation_labels = split_items(
        train_paths,
        train_labels,
    )

    unknown_train_paths: list[pathlib.Path] = []
    unknown_validation_paths: list[pathlib.Path] = []
    unknown_test_paths: list[pathlib.Path] = []

    if args.unknown_images:
        unknown_paths = read_unknown_images(args.unknown_images)
        if len(unknown_paths) < 5:
            raise ValueError("--unknown-images must contain at least 5 images")

        random.Random(SEED).shuffle(unknown_paths)
        unknown_split = min(
            len(unknown_paths) - 2,
            max(1, int(len(unknown_paths) * 0.80)),
        )
        validation_split = min(
            len(unknown_paths) - 1,
            max(unknown_split + 1, int(len(unknown_paths) * 0.90)),
        )
        unknown_train_paths = unknown_paths[:unknown_split]
        unknown_validation_paths = unknown_paths[unknown_split:validation_split]
        unknown_test_paths = unknown_paths[validation_split:]

        unknown_class = len(BREEDS)
        train_paths += unknown_train_paths
        train_labels += [unknown_class] * len(unknown_train_paths)
        validation_paths += unknown_validation_paths
        validation_labels += [unknown_class] * len(unknown_validation_paths)

    class_names = [*BREEDS]
    if args.unknown_images:
        class_names.append(UNKNOWN_LABEL)

    print()
    print("==============================")
    print("OXFORD-IIIT PET DATASET")
    print("==============================")

    print(
        "Training images:",
        len(train_paths),
    )

    print(
        "Validation images:",
        len(validation_paths),
    )

    print(
        "Test images:",
        len(test_paths),
    )

    print(
        "Classes:",
        len(class_names),
    )

    # --------------------------------------------------------
    # Print class mapping
    # --------------------------------------------------------

    print()
    print("CLASS MAPPING")
    print("==============================")

    for index, breed in enumerate(class_names):

        print(
            f"{index:2d} -> {breed}"
        )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train = make_dataset(
        args.images,
        train_paths,
        train_labels,
        training=True,
    )

    validation = make_dataset(
        args.images,
        validation_paths,
        validation_labels,
        training=False,
    )

    test = make_dataset(
        args.images,
        test_paths,
        test_labels,
        training=False,
    )

    if unknown_test_paths:
        test = make_dataset(
            args.images,
            test_paths + unknown_test_paths,
            test_labels + [len(BREEDS)] * len(unknown_test_paths),
            training=False,
        )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model, backbone = build_model(len(class_names))

    model.summary()

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def compile_model(learning_rate: float) -> None:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=["accuracy"],
        )

    def callbacks() -> list[tf.keras.callbacks.Callback]:
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=6,
                mode="max",
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-7,
            ),
        ]

    warmup_epochs = min(args.warmup_epochs, args.epochs)
    fine_tune_epochs = max(0, args.epochs - warmup_epochs)

    print()
    print("==============================")
    print("HEAD WARMUP")
    print("==============================")

    compile_model(1e-3)
    model.fit(
        train,
        validation_data=validation,
        epochs=warmup_epochs,
        callbacks=callbacks(),
    )

    if fine_tune_epochs > 0:
        print()
        print("==============================")
        print("FINE-TUNING LAST 40 LAYERS")
        print("==============================")

        set_fine_tuning(backbone, trainable_layers=40)
        compile_model(1e-5)
        model.fit(
            train,
            validation_data=validation,
            initial_epoch=warmup_epochs,
            epochs=args.epochs,
            callbacks=callbacks(),
        )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print()
    print("==============================")
    print("TESTING")
    print("==============================")

    test_loss, test_accuracy = model.evaluate(
        test,
        verbose=1,
    )

    print()
    print(
        f"Test accuracy: {test_accuracy * 100:.2f}%"
    )

    if args.unknown_images:
        unknown_loss, unknown_accuracy = model.evaluate(
            make_dataset(
                args.images,
                unknown_test_paths,
                [len(BREEDS)] * len(unknown_test_paths),
                training=False,
            ),
            verbose=0,
        )
        print(
            f"Unknown rejection accuracy: {unknown_accuracy * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Make output directory
    # --------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save labels
    # --------------------------------------------------------

    labels_path = (
        args.output.parent
        / "breed_labels.txt"
    )

    labels_path.write_text(
        "\n".join(class_names) + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # EXPORT
    #
    # The MobileNetV2 backbone is already an inference model.
    # We export the trained model directly to keep the graph simple
    # and avoid unsupported random augmentation layers.
    # --------------------------------------------------------

    export_model = model

    # --------------------------------------------------------
    # Convert to TFLite
    #
    # Float32.
    # No quantization.
    # --------------------------------------------------------

    print()
    print("==============================")
    print("TFLITE CONVERSION")
    print("==============================")

    converter = tf.lite.TFLiteConverter.from_keras_model(
        export_model
    )

    # Keep the model Float32.
    converter.optimizations = []

    # Disable the newer MLIR-based converter.
    converter.experimental_new_converter = False

    tflite_model = converter.convert()

    args.output.write_bytes(
        tflite_model
    )

    # --------------------------------------------------------
    # Verify TFLite model
    # --------------------------------------------------------

    print()
    print("==============================")
    print("TFLITE VERIFICATION")
    print("==============================")

    interpreter = tf.lite.Interpreter(
        model_path=str(args.output)
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter.get_input_details()
    )

    output_details = (
        interpreter.get_output_details()
    )

    print()
    print("INPUT DETAILS")

    for detail in input_details:
        print(detail)

    print()
    print("OUTPUT DETAILS")

    for detail in output_details:
        print(detail)

    print()
    print("Model saved:")
    print(args.output)

    print()
    print("Labels saved:")
    print(labels_path)

    print()
    print("==============================")
    print("DONE")
    print("==============================")


if __name__ == "__main__":
    main()