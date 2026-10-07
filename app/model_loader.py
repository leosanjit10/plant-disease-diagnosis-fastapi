import os
import urllib.request
from keras.models import load_model

MODEL_PATH = "models/plant_disease_model.keras"

MODEL_URL = (
    "https://huggingface.co/leosanjit10/plant-disease-cnn/resolve/main/plant_disease_model.keras"
)



def load_plant_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found locally. Downloading from Hugging Face...")
        os.makedirs("models", exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")

    print("Loading model...")
    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")
    return model