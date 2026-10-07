from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_SIZE = (224, 224)
MAX_IMAGE_PIXELS = 20_000_000

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


class ImageValidationError(ValueError):
    """Raised when uploaded bytes do not contain an acceptable image."""


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ImageValidationError("The uploaded file is empty.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ImageValidationError(
                    f"Image dimensions exceed the {MAX_IMAGE_PIXELS:,}-pixel limit."
                )
            if getattr(image, "n_frames", 1) > 1:
                raise ImageValidationError("Animated or multi-frame images are not supported.")
            image.load()
            resized = image.convert("RGB").resize(
                IMAGE_SIZE,
                Image.Resampling.BILINEAR,
            )
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageValidationError("The uploaded file is not a valid supported image.") from exc

    # The trained model includes its own Rescaling(1/255) layer.
    image_array = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


def format_class_name(class_name: str) -> tuple[str, str]:
    try:
        plant, disease = class_name.split("___", maxsplit=1)
    except ValueError as exc:
        raise ValueError(f"Invalid model class label: {class_name!r}") from exc

    return _humanize(plant), _humanize(disease)


def _humanize(label: str) -> str:
    return " ".join(label.replace("_", " ").split()).title()
