import os
import cv2

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "annotations",
    "roi"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "ffb_clahe"
)

IMAGE_SIZE = (224, 224)

CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)


def process_image(input_path, output_path, clahe):

    image = cv2.imread(input_path)

    if image is None:
        return False

    # Resize
    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # Convert BGR -> LAB
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    # Apply CLAHE to L channel
    l_channel, a_channel, b_channel = cv2.split(lab)

    l_channel = clahe.apply(l_channel)

    lab = cv2.merge(
        [l_channel, a_channel, b_channel]
    )

    # Convert LAB -> BGR
    processed = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    cv2.imwrite(
        output_path,
        processed
    )

    return True


def main():

    print("=" * 60)
    print("FFB ROI RESIZE + CLAHE PREPROCESSING")
    print("=" * 60)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_GRID_SIZE
    )

    total_processed = 0
    total_skipped = 0

    for split in ["train", "valid", "test"]:

        input_split = os.path.join(
            INPUT_DIR,
            split
        )

        output_split = os.path.join(
            OUTPUT_DIR,
            split
        )

        os.makedirs(
            output_split,
            exist_ok=True
        )

        if not os.path.exists(input_split):
            print(f"\nMissing input folder: {input_split}")
            continue

        files = [
            f for f in os.listdir(input_split)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]

        processed = 0
        skipped = 0

        for filename in files:

            input_path = os.path.join(
                input_split,
                filename
            )

            output_name = os.path.splitext(
                filename
            )[0] + ".jpg"

            output_path = os.path.join(
                output_split,
                output_name
            )

            success = process_image(
                input_path,
                output_path,
                clahe
            )

            if success:
                processed += 1
            else:
                skipped += 1

        print(
            f"{split}: "
            f"Processed {processed}, "
            f"Skipped {skipped}"
        )

        total_processed += processed
        total_skipped += skipped

    print()
    print("=" * 60)
    print("FFB PREPROCESSING COMPLETED")
    print("=" * 60)
    print(f"Total processed: {total_processed}")
    print(f"Total skipped:   {total_skipped}")
    print()
    print("Output:")
    print(OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()