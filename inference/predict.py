import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.multitask_model import MultiTaskFFBModel


# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "training"
    / "best_multitask_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "inference"
    / "outputs"
)

IMAGE_SIZE = 224

CLASS_NAMES = [
    "Empty Bunch",
    "Overripe",
    "Ripe",
    "Underripe"
]

NUM_CLASSES = len(CLASS_NAMES)

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


# ==========================================================
# DEVICE
# ==========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    print("Loading model...")

    model = MultiTaskFFBModel(
        num_classes=NUM_CLASSES
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    return model


# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

def preprocess_image(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image:\n"
            f"{image_path}"
        )

    original_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    resized_image = cv2.resize(
        original_image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    normalized = (
        resized_image.astype(
            np.float32
        ) / 255.0
    )

    normalized = (
        normalized - MEAN
    ) / STD

    normalized = np.transpose(
        normalized,
        (2, 0, 1)
    )

    tensor = torch.from_numpy(
        normalized
    ).unsqueeze(0)

    return (
        original_image,
        tensor.to(DEVICE)
    )


# ==========================================================
# FIND BOUNDING BOX FROM MASK
# ==========================================================

def get_bounding_box(binary_mask):

    ys, xs = np.where(
        binary_mask > 0
    )

    if len(xs) == 0:
        return None

    x_min = int(xs.min())
    y_min = int(ys.min())
    x_max = int(xs.max())
    y_max = int(ys.max())

    return (
        x_min,
        y_min,
        x_max,
        y_max
    )


# ==========================================================
# CREATE VISUALIZATION
# ==========================================================

def create_visualization(
    original_image,
    binary_mask,
    predicted_class,
    confidence,
    estimated_count
):

    height, width = (
        original_image.shape[:2]
    )

    # Resize predicted mask to original
    mask_original = cv2.resize(
        binary_mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    # ------------------------------------------------------
    # Bounding box
    # ------------------------------------------------------

    bbox = get_bounding_box(
        mask_original
    )

    # ------------------------------------------------------
    # Create overlay
    # ------------------------------------------------------

    visualization = (
        original_image.copy()
    )

    mask_area = (
        mask_original > 0
    )

    # Highlight predicted mask
    overlay = np.zeros_like(
        visualization
    )

    overlay[:, :, 1] = 255

    visualization[
        mask_area
    ] = cv2.addWeighted(
        visualization[
            mask_area
        ],
        0.65,
        overlay[
            mask_area
        ],
        0.35,
        0
    )

    # ------------------------------------------------------
    # Draw bounding box
    # ------------------------------------------------------

    if bbox is not None:

        x_min, y_min, x_max, y_max = (
            bbox
        )

        cv2.rectangle(
            visualization,
            (x_min, y_min),
            (x_max, y_max),
            (255, 0, 0),
            4
        )

        label = (
            f"{predicted_class} "
            f"{confidence * 100:.1f}%"
        )

        cv2.putText(
            visualization,
            label,
            (x_min, max(35, y_min - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

    # ------------------------------------------------------
    # Information panel
    # ------------------------------------------------------

    panel_height = 130

    panel = np.ones(
        (
            panel_height,
            width,
            3
        ),
        dtype=np.uint8
    ) * 255

    cv2.putText(
        panel,
        f"Ripeness: {predicted_class}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        panel,
        f"Confidence: {confidence * 100:.2f}%",
        (20, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        panel,
        f"Estimated FFB count: {estimated_count:.2f}",
        (20, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    final_image = np.vstack(
        [
            visualization,
            panel
        ]
    )

    return final_image


# ==========================================================
# PREDICTION
# ==========================================================

@torch.no_grad()
def predict(image_path):

    model = load_model()

    original_image, image_tensor = (
        preprocess_image(
            image_path
        )
    )

    outputs = model(
        image_tensor
    )

    # ------------------------------------------------------
    # RIPENESS
    # ------------------------------------------------------

    ripeness_logits = outputs[
        "ripeness_logits"
    ]

    probabilities = F.softmax(
        ripeness_logits,
        dim=1
    )

    predicted_class_id = (
        torch.argmax(
            probabilities,
            dim=1
        ).item()
    )

    predicted_class = (
        CLASS_NAMES[
            predicted_class_id
        ]
    )

    confidence = (
        probabilities[
            0,
            predicted_class_id
        ].item()
    )

    # ------------------------------------------------------
    # COUNT
    # ------------------------------------------------------

    estimated_count = (
        outputs[
            "density_count"
        ].item()
    )

    estimated_count = max(
        0.0,
        estimated_count
    )

    # ------------------------------------------------------
    # MASK
    # ------------------------------------------------------

    mask_probability = torch.sigmoid(
        outputs[
            "mask_logits"
        ]
    )

    mask = (
        mask_probability[0, 0]
        .cpu()
        .numpy()
    )

    binary_mask = (
        mask > 0.5
    ).astype(
        np.uint8
    ) * 255

    # ------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------

    visualization = create_visualization(
        original_image,
        binary_mask,
        predicted_class,
        confidence,
        estimated_count
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / "prediction_result.jpg"
    )

    visualization_bgr = cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        str(output_path),
        visualization_bgr
    )

    return {
        "image_path": str(image_path),
        "ripeness": predicted_class,
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[i]:
                probabilities[
                    0, i
                ].item()
            for i in range(NUM_CLASSES)
        },
        "estimated_count": estimated_count,
        "mask": binary_mask,
        "output_path": str(output_path)
    }


# ==========================================================
# COMMAND LINE
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("FFB SINGLE IMAGE PREDICTION")
    print("=" * 70)

    if len(sys.argv) < 2:

        print()
        print("Usage:")

        print(
            'python "inference\\predict.py" '
            '"path\\to\\image.jpg"'
        )

        sys.exit(1)

    image_path = Path(
        sys.argv[1]
    )

    print()
    print(
        "Input image:",
        image_path
    )

    print(
        "Device:",
        DEVICE
    )

    result = predict(
        image_path
    )

    print()
    print("=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)

    print()
    print(
        "Predicted ripeness:",
        result["ripeness"]
    )

    print(
        "Confidence:",
        f"{result['confidence'] * 100:.2f}%"
    )

    print()
    print("Class probabilities:")

    for class_name, probability in (
        result["probabilities"].items()
    ):

        print(
            f"  {class_name:<15}"
            f"{probability * 100:.2f}%"
        )

    print()
    print(
        "Estimated FFB count:",
        f"{result['estimated_count']:.2f}"
    )

    print()
    print(
        "Visualization saved to:"
    )

    print(
        result["output_path"]
    )

    print()
    print("=" * 70)
    print("PREDICTION COMPLETED")
    print("=" * 70)