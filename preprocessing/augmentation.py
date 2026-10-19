from pathlib import Path
from PIL import Image, ImageEnhance
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NORMALIZED_DIR = PROJECT_ROOT / "dataset" / "processed" / "normalized"
AUGMENTED_DIR = PROJECT_ROOT / "dataset" / "processed" / "augmented"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

AUGMENTATIONS_PER_IMAGE = 2
MOSAIC_PROBABILITY = 0.0
MIXUP_PROBABILITY = 0.0


def augment_image(image):

    if random.random() < 0.5:
        image = image.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )

    angle = random.uniform(-15, 15)

    image = image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False
    )

    brightness_factor = random.uniform(0.85, 1.15)

    image = ImageEnhance.Brightness(
        image
    ).enhance(brightness_factor)

    contrast_factor = random.uniform(0.85, 1.15)

    image = ImageEnhance.Contrast(
        image
    ).enhance(contrast_factor)

    color_factor = random.uniform(0.90, 1.10)

    image = ImageEnhance.Color(
        image
    ).enhance(color_factor)

    return image


def create_mosaic(images):

    if len(images) != 4:
        raise ValueError(
            "Mosaic requires exactly 4 images."
        )

    width, height = images[0].size

    mosaic = Image.new(
        "RGB",
        (width * 2, height * 2)
    )

    positions = [
        (0, 0),
        (width, 0),
        (0, height),
        (width, height)
    ]

    for image, position in zip(
        images,
        positions
    ):

        image = image.resize(
            (width, height),
            Image.Resampling.BILINEAR
        )

        mosaic.paste(
            image,
            position
        )

    return mosaic


def create_mixup(image1, image2):

    alpha = random.uniform(
        0.3,
        0.7
    )

    return Image.blend(
        image1,
        image2,
        alpha
    )


def get_image_paths(source_dir):

    return [
        path
        for path in source_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower()
        in IMAGE_EXTENSIONS
    ]


def save_image(image, output_path):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    image.save(
        output_path,
        quality=95
    )


def augment_training_images(
    source_dir,
    destination_dir
):

    image_paths = get_image_paths(
        source_dir
    )

    total = 0

    # Original images
    for image_path in image_paths:

        relative_path = (
            image_path.relative_to(
                source_dir
            )
        )

        output_path = (
            destination_dir / relative_path
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            save_image(
                image,
                output_path
            )

            total += 1

        except Exception as error:

            print(
                f"ERROR processing "
                f"{image_path}: {error}"
            )

    # Geometric + photometric augmentation
    for image_path in image_paths:

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            relative_path = (
                image_path.relative_to(
                    source_dir
                )
            )

            output_parent = (
                destination_dir
                / relative_path.parent
            )

            for i in range(
                AUGMENTATIONS_PER_IMAGE
            ):

                augmented = augment_image(
                    image.copy()
                )

                output_path = (
                    output_parent
                    / f"{image_path.stem}"
                    f"_aug_{i + 1}"
                    f"{image_path.suffix}"
                )

                save_image(
                    augmented,
                    output_path
                )

                total += 1

        except Exception as error:

            print(
                f"ERROR augmenting "
                f"{image_path}: {error}"
            )

    # Mosaic augmentation
    mosaic_count = max(
        1,
        len(image_paths) // 4
    )

    for i in range(mosaic_count):

        if random.random() > MOSAIC_PROBABILITY:
            continue

        selected_paths = random.choices(
            image_paths,
            k=4
        )

        try:

            images = [
                Image.open(
                    path
                ).convert("RGB")
                for path in selected_paths
            ]

            mosaic = create_mosaic(
                images
            )

            output_path = (
                destination_dir
                / "mosaic"
                / f"mosaic_{i + 1}.jpg"
            )

            save_image(
                mosaic,
                output_path
            )

            total += 1

        except Exception as error:

            print(
                f"ERROR creating Mosaic: "
                f"{error}"
            )

    # MixUp augmentation
    mixup_count = max(
        1,
        len(image_paths) // 4
    )

    for i in range(mixup_count):

        if random.random() > MIXUP_PROBABILITY:
            continue

        path1, path2 = random.choices(
            image_paths,
            k=2
        )

        try:

            image1 = Image.open(
                path1
            ).convert("RGB")

            image2 = Image.open(
                path2
            ).convert("RGB")

            image2 = image2.resize(
                image1.size,
                Image.Resampling.BILINEAR
            )

            mixup = create_mixup(
                image1,
                image2
            )

            output_path = (
                destination_dir
                / "mixup"
                / f"mixup_{i + 1}.jpg"
            )

            save_image(
                mixup,
                output_path
            )

            total += 1

        except Exception as error:

            print(
                f"ERROR creating MixUp: "
                f"{error}"
            )

    return total


def copy_dataset(
    source_dir,
    destination_dir
):

    total = 0

    for image_path in source_dir.rglob("*"):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_path = (
            image_path.relative_to(
                source_dir
            )
        )

        output_path = (
            destination_dir / relative_path
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            save_image(
                image,
                output_path
            )

            total += 1

        except Exception as error:

            print(
                f"ERROR copying "
                f"{image_path}: {error}"
            )

    return total


if __name__ == "__main__":

    print("=" * 60)
    print("PALM RIPENESS PREDICTION")
    print("ADAPTIVE DATA AUGMENTATION")
    print("=" * 60)

    train_count = augment_training_images(
        NORMALIZED_DIR / "train",
        AUGMENTED_DIR / "train"
    )

    print(
        f"Training images generated: "
        f"{train_count}"
    )

    valid_count = copy_dataset(
        NORMALIZED_DIR / "valid",
        AUGMENTED_DIR / "valid"
    )

    print(
        f"Validation images: {valid_count}"
    )

    test_count = copy_dataset(
        NORMALIZED_DIR / "test",
        AUGMENTED_DIR / "test"
    )

    print(
        f"Test images: {test_count}"
    )

    print("=" * 60)
    print("ADAPTIVE AUGMENTATION COMPLETED")
    print("=" * 60)