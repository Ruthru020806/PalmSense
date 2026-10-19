import os
import cv2
import random
import numpy as np

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "ffb_clahe",
    "train"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "ffb_augmented",
    "train"
)

AUGMENTATIONS_PER_IMAGE = 2

random.seed(42)
np.random.seed(42)


def horizontal_flip(image):
    return cv2.flip(image, 1)


def rotate_image(image, angle):
    height, width = image.shape[:2]

    center = (width // 2, height // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REFLECT
    )


def adjust_brightness_contrast(image):
    alpha = random.uniform(0.85, 1.15)
    beta = random.randint(-20, 20)

    return cv2.convertScaleAbs(
        image,
        alpha=alpha,
        beta=beta
    )


def adjust_color(image):
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    saturation_scale = random.uniform(
        0.90,
        1.10
    )

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1].astype(np.float32)
        * saturation_scale,
        0,
        255
    ).astype(np.uint8)

    return cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )


def augment(image):

    output = image.copy()

    # Geometric augmentation
    if random.random() < 0.5:
        output = horizontal_flip(output)

    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        output = rotate_image(
            output,
            angle
        )

    # Photometric augmentation
    if random.random() < 0.7:
        output = adjust_brightness_contrast(
            output
        )

    if random.random() < 0.5:
        output = adjust_color(output)

    return output


def main():

    print("=" * 60)
    print("FFB ADAPTIVE AUGMENTATION")
    print("=" * 60)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    total_original = 0
    total_augmented = 0

    for filename in files:

        input_path = os.path.join(
            INPUT_DIR,
            filename
        )

        image = cv2.imread(input_path)

        if image is None:
            continue

        # Keep original training image
        original_name = (
            os.path.splitext(filename)[0]
            + "_original.jpg"
        )

        cv2.imwrite(
            os.path.join(
                OUTPUT_DIR,
                original_name
            ),
            image
        )

        total_original += 1

        # Generate augmented images
        base_name = os.path.splitext(
            filename
        )[0]

        for i in range(
            1,
            AUGMENTATIONS_PER_IMAGE + 1
        ):

            augmented = augment(image)

            output_name = (
                f"{base_name}_aug{i}.jpg"
            )

            cv2.imwrite(
                os.path.join(
                    OUTPUT_DIR,
                    output_name
                ),
                augmented
            )

            total_augmented += 1

    print()
    print("=" * 60)
    print("FFB AUGMENTATION COMPLETED")
    print("=" * 60)
    print(f"Original images:   {total_original}")
    print(f"Augmented images:  {total_augmented}")
    print(f"Total output:      {total_original + total_augmented}")
    print()
    print("Output:")
    print(OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()