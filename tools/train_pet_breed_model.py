from __future__ import annotations

import argparse
import pathlib
import random

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


# ============================================================
# Create tf.data dataset
# ============================================================

def make_dataset(
    image_dir: pathlib.Path,
    paths: list[str],
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

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        AUTOTUNE
    )

    return dataset


# ============================================================
# Build CNN
#
# NO MobileNet
# NO ImageNet
# NO external pretrained model
#
# Training uses Oxford-IIIT Pet only.
# ============================================================

def build_model() -> tf.keras.Model:

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
    # PREPROCESSING
    #
    # Input from camera/image:
    #
    #     0 .. 255
    #
    # becomes:
    #
    #     -1 .. 1
    #
    # This layer is INSIDE the model and therefore also
    # exists inside the exported TFLite model.
    # --------------------------------------------------------

    x = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="preprocess",
    )(inputs)

    # --------------------------------------------------------
    # Data augmentation
    # --------------------------------------------------------

    x = tf.keras.layers.RandomFlip(
        "horizontal"
    )(x)

    x = tf.keras.layers.RandomRotation(
        0.08
    )(x)

    x = tf.keras.layers.RandomZoom(
        0.10
    )(x)

    # --------------------------------------------------------
    # CNN BLOCK 1
    # --------------------------------------------------------

    x = tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    # --------------------------------------------------------
    # CNN BLOCK 2
    # --------------------------------------------------------

    x = tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    # --------------------------------------------------------
    # CNN BLOCK 3
    # --------------------------------------------------------

    x = tf.keras.layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    # --------------------------------------------------------
    # CNN BLOCK 4
    # --------------------------------------------------------

    x = tf.keras.layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(
        0.50
    )(x)

    # --------------------------------------------------------
    # 37 Oxford-IIIT Pet classes
    # --------------------------------------------------------

    outputs = tf.keras.layers.Dense(
        len(BREEDS),
        activation=None,
        name="logits",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="oxford_iiit_pet_breed_model",
    )

    return model


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
        "--epochs",
        type=int,
        default=100,
    )

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

    # --------------------------------------------------------
    # Shuffle training data
    # --------------------------------------------------------

    train_items = list(
        zip(
            train_paths,
            train_labels,
        )
    )

    random.Random(SEED).shuffle(
        train_items
    )

    train_paths = [
        item[0]
        for item in train_items
    ]

    train_labels = [
        item[1]
        for item in train_items
    ]

    # --------------------------------------------------------
    # 90% training
    # 10% validation
    # --------------------------------------------------------

    split_index = int(
        len(train_paths) * 0.90
    )

    validation_paths = train_paths[
        split_index:
    ]

    validation_labels = train_labels[
        split_index:
    ]

    train_paths = train_paths[
        :split_index
    ]

    train_labels = train_labels[
        :split_index
    ]

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
        len(BREEDS),
    )

    # --------------------------------------------------------
    # Print class mapping
    # --------------------------------------------------------

    print()
    print("CLASS MAPPING")
    print("==============================")

    for index, breed in enumerate(BREEDS):

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

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model()

    model.summary()

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0003
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
        metrics=[
            "accuracy"
        ],
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        mode="max",
        restore_best_weights=True,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print("==============================")
    print("TRAINING")
    print("==============================")

    model.fit(
        train,
        validation_data=validation,
        epochs=args.epochs,
        callbacks=[
            early_stopping,
            reduce_lr,
        ],
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
        "\n".join(BREEDS) + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # EXPORT
    #
    # The training model already contains:
    #
    #     0..255 -> -1..1
    #
    # and augmentation layers.
    #
    # For TFLite we create an inference model without
    # random augmentation.
    # --------------------------------------------------------

    export_inputs = tf.keras.Input(
        shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3,
        ),
        dtype=tf.float32,
        name="image",
    )

    # Same preprocessing as training.
    export_x = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="preprocess",
    )(export_inputs)

    # Reuse the trained convolutional layers.
    #
    # The trained model layers are:
    #
    # preprocess
    # augmentation
    # conv...
    #
    # We deliberately skip augmentation during inference.

    x = model.get_layer(
        "conv2d"
    )(export_x)

    x = model.get_layer(
        "batch_normalization"
    )(x)

    x = model.get_layer(
        "max_pooling2d"
    )(x)

    x = model.get_layer(
        "conv2d_1"
    )(x)

    x = model.get_layer(
        "batch_normalization_1"
    )(x)

    x = model.get_layer(
        "max_pooling2d_1"
    )(x)

    x = model.get_layer(
        "conv2d_2"
    )(x)

    x = model.get_layer(
        "batch_normalization_2"
    )(x)

    x = model.get_layer(
        "max_pooling2d_2"
    )(x)

    x = model.get_layer(
        "conv2d_3"
    )(x)

    x = model.get_layer(
        "batch_normalization_3"
    )(x)

    x = model.get_layer(
        "max_pooling2d_3"
    )(x)

    x = model.get_layer(
        "global_average_pooling2d"
    )(x)

    # During inference Dropout is disabled.
    x = model.get_layer(
        "dropout"
    )(x, training=False)

    outputs = model.get_layer(
        "logits"
    )(x)

    export_model = tf.keras.Model(
        export_inputs,
        outputs,
        name="oxford_iiit_pet_inference",
    )

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