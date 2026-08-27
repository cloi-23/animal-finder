from __future__ import annotations

import argparse
import pathlib
import tarfile

import tensorflow as tf

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


def read_split(annotation_file: pathlib.Path) -> tuple[list[str], list[int]]:
    paths: list[str] = []
    labels: list[int] = []
    for line in annotation_file.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        paths.append(fields[0] + ".jpg")
        labels.append(int(fields[1]) - 1)
    return paths, labels


def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


def make_dataset(
    image_dir: pathlib.Path,
    paths: list[str],
    labels: list[int],
    training: bool,
) -> tf.data.Dataset:
    full_paths = [str(image_dir / path) for path in paths]
    dataset = tf.data.Dataset.from_tensor_slices((full_paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=42)
    return dataset.map(load_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=pathlib.Path, required=True)
    parser.add_argument("--annotations", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    train_paths, train_labels = read_split(args.annotations / "trainval.txt")
    test_paths, test_labels = read_split(args.annotations / "test.txt")
    train = make_dataset(args.images, train_paths, train_labels, training=True)
    test = make_dataset(args.images, test_paths, test_labels, training=False)

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
    outputs = tf.keras.layers.Dense(len(BREEDS), activation="softmax")(features)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train, validation_data=test, epochs=args.epochs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    concrete_function = tf.function(model).get_concrete_function(
        tf.TensorSpec([1, *IMAGE_SIZE, 3], tf.float32)
    )
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete_function], model
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    model_bytes = converter.convert()
    args.output.write_bytes(model_bytes)
    args.output.with_name("breed_labels.txt").write_text("\n".join(BREEDS) + "\n")


if __name__ == "__main__":
    main()
