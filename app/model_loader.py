from pathlib import Path
from typing import Protocol

import numpy as np

from app.utils import CLASS_NAMES, IMAGE_SIZE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "plant_disease_model.keras"


class PlantDiseaseModel(Protocol):
    input_shape: tuple[int | None, ...]
    output_shape: tuple[int | None, ...]

    def predict(self, images: np.ndarray, verbose: int = 0) -> np.ndarray: ...


def load_plant_model() -> PlantDiseaseModel:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file was not found at {MODEL_PATH}. "
            "Place plant_disease_model.keras in the project's models directory."
        )

    try:
        from keras.models import load_model
    except ImportError as exc:
        raise RuntimeError(
            "Keras/TensorFlow is not available in the active Python environment. "
            "Install the project requirements in the selected environment."
        ) from exc

    model = load_model(MODEL_PATH, compile=False)
    expected_input_shape = (None, *IMAGE_SIZE, 3)
    expected_output_shape = (None, len(CLASS_NAMES))

    if tuple(model.input_shape) != expected_input_shape:
        raise ValueError(
            f"Expected model input shape {expected_input_shape}, "
            f"got {model.input_shape}."
        )
    if tuple(model.output_shape) != expected_output_shape:
        raise ValueError(
            f"Expected model output shape {expected_output_shape}, "
            f"got {model.output_shape}."
        )

    return model
