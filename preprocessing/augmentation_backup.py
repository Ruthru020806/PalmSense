from pathlib import Path
from PIL import Image, ImageEnhance
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NORMALIZED_DIR = PROJECT_ROOT / "dataset" / "processed" / "normalized"
AUGMENTED_DIR = PROJECT_ROOT / "dataset" / "processed" / "augmented"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}

# Number of augmented versions created for each training image
AUGMENTATIONS_PER_IMAGE = 2


def augment_image(image):
    """
    Apply random transformations to an image.
    """

    # Random horizontal flip
    if random.random() < 0.5:
        image = image.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )

    # Random small rotation
    angle = random.uniform(-15, 15)

    image = image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False
    )

    # Random brightness adjustment
    brightness_factor = random.uniform(
        0.85,
        1.15
    )

    image = ImageEnhance.Brightness(
        image
    ).enhance(brightness_factor)

    # Random contrast adjustment
    contrast_factor = random.uniform(
        0.85,
        1.15
    )

    image = ImageEnhance.Contrast(
        image
    ).enhance(contrast_factor)

    # Random color adjustment
    color_factor = random.uniform(
        0.90,
        1.10
    )

    image = ImageEnhance.Color(
        image
    ).enhance(color_factor)

    return image


def augment_training_images(
    source_dir,
    destination_dir
):

    total = 0

    for image_path in source_dir.rglob("*"):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_path = image_path.relative_to(
            source_dir
        )

        original_output = (
            destination_dir / relative_path
        )

        original_output.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            # Keep the original normalized image
            image.save(
                original_output,
                quality=95
            )

            total += 1

            # Create augmented versions
            for i in range(
                AUGMENTATIONS_PER_IMAGE
            ):

                augmented = augment_image(
                    image.copy()
                )

                output_name = (
                    f"{image_path.stem}"
                    f"_aug_{i + 1}"
                    f"{image_path.suffix}"
                )

                output_path = (
                    original_output.parent
                    / output_name
                )

                augmented.save(
                    output_path,
                    quality=95
                )

                total += 1

        except Exception as error:

            print(
                f"ERROR processing "
                f"{image_path}: {error}"
            )

    return total


if __name__ == "__main__":

    print("=" * 60)
    print("PALM RIPENESS PREDICTION")
    print("DATA AUGMENTATION")
    print("=" * 60)

    train_source = (
        NORMALIZED_DIR / "train"
    )

    train_destination = (
        AUGMENTED_DIR / "train"
    )

    print()
    print("Processing: train")

    train_count = augment_training_images(
        train_source,
        train_destination
    )

    print(
        f"Training images after augmentation: "
        f"{train_count}"
    )

    print()
    print("=" * 60)
    print("VALIDATION AND TEST SETS")
    print("No augmentation applied")
    print("=" * 60)

    print()
    print("Copying validation images...")

    valid_source = (
        NORMALIZED_DIR / "valid"
    )

    valid_destination = (
        AUGMENTED_DIR / "valid"
    )

    valid_count = 0

    for image_path in valid_source.rglob("*"):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_path = image_path.relative_to(
            valid_source
        )

        output_path = (
            valid_destination / relative_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image.save(
            output_path,
            quality=95
        )

        valid_count += 1

    print(
        f"Validation images: {valid_count}"
    )

    print()
    print("Copying test images...")

    test_source = (
        NORMALIZED_DIR / "test"
    )

    test_destination = (
        AUGMENTED_DIR / "test"
    )

    test_count = 0

    for image_path in test_source.rglob("*"):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_path = image_path.relative_to(
            test_source
        )

        output_path = (
            test_destination / relative_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image.save(
            output_path,
            quality=95
        )

        test_count += 1

    print(
        f"Test images: {test_count}"
    )

    print()
    print("=" * 60)
    print("DATA AUGMENTATION COMPLETED")
    print("=" * 60)

    print(
        f"Original training images: 598"
    )

    print(
        f"Training images after augmentation: "
        f"{train_count}"
    )

    print(
        f"Validation images: {valid_count}"
    )

    print(
        f"Test images: {test_count}"
    )

    print("=" * 60)