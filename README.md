# Plant Disease Detection API

CNN-based plant disease diagnosis service using TensorFlow, FastAPI, Docker,
and the PlantVillage dataset. It classifies leaf images into 38 classes.

## Project layout

```text
Plant_Disease_Diagnosis/
├── app/
│   ├── main.py
│   ├── model_loader.py
│   └── utils.py
├── models/
│   └── plant_disease_model.keras
├── data/
├── notebooks/
├── requirements.txt
└── README.md
```

The model is resolved relative to `app/model_loader.py`, so the server can be
started from the project root or another working directory.

## Run locally (Windows PowerShell)

Use Python 3.12 with the TensorFlow and Keras versions pinned in
`requirements.txt`. The saved model was serialized by Keras 3.15.1, so loading
it requires a compatible Keras version.

```powershell
cd C:\Users\VICTUS\Desktop\Plant_Disease_Diagnosis
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

If PowerShell blocks virtual-environment activation, run the environment's
Python directly instead:

```powershell
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The model loads once per Uvicorn worker during application startup. For a
single local process, do not add `--workers`.

## Run with Docker

Build and start the API from the project root:

```powershell
docker build -t plant-disease-api .
docker run --rm -d --name plant-disease-api -p 10000:10000 plant-disease-api
```

Verify the pinned runtime and model load:

```powershell
docker run --rm plant-disease-api python -c "import tensorflow as tf, keras; print('TensorFlow:', tf.__version__); print('Keras:', keras.__version__); assert tf.__version__ == '2.21.0' and keras.__version__ == '3.15.1'"
Invoke-RestMethod http://127.0.0.1:10000/health
docker stop plant-disease-api
```

## Check the API

In another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@C:\path\to\leaf.jpg"
```

`POST /predict` accepts one image (up to 10 MiB) and returns a cleaned plant
and disease name with confidence as a percentage:

```json
{
  "plant": "Apple",
  "disease": "Apple Scab",
  "confidence": 95.6
}
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.
