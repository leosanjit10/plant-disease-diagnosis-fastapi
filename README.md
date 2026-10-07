# 🌿 Plant Disease Detection API

A deep learning-powered API for plant disease classification using TensorFlow, FastAPI, and Docker.

The model is trained on the PlantVillage dataset and can classify plant leaf images into 38 disease categories, returning the predicted plant, disease name, and confidence score.

---

## 🚀 Live Deployment

### API Base URL

https://plant-disease-diagnosis-fastapi.onrender.com

### Interactive API Documentation

https://plant-disease-diagnosis-fastapi.onrender.com/docs

---

## ✨ Features

- Plant disease classification from leaf images
- 38 plant disease classes
- TensorFlow/Keras CNN model
- FastAPI REST API
- Dockerized deployment
- Hosted on Render
- Interactive Swagger documentation
- Health check endpoint

---

## 🛠 Tech Stack

### Machine Learning

- TensorFlow 2.21
- Keras 3.15.1
- NumPy
- PlantVillage Dataset

### Backend

- FastAPI
- Uvicorn

### Deployment

- Docker
- Render

---

## 📂 Project Structure

```text
Plant_Disease_Diagnosis/
├── app/
│   ├── main.py
│   ├── model_loader.py
│   └── utils.py
│
├── models/
│   └── plant_disease_model.keras
│
├── data/
├── notebooks/
├── requirements.txt
└── README.md
```

The model path is resolved relative to `app/model_loader.py`, allowing the server to run correctly regardless of the current working directory.

---

## 🤖 Model Information

| Property | Value |
|-----------|---------|
| Model Type | Convolutional Neural Network (CNN) |
| Framework | TensorFlow / Keras |
| Dataset | PlantVillage |
| Classes | 38 |
| Input Size | 224 × 224 × 3 |
| Output | Plant Disease Prediction |

---

## 📡 API Endpoints

### Home

```http
GET /
```

Example Response:

```json
{
  "message": "Plant Disease Detection API is running"
}
```

---

### Health Check

```http
GET /health
```

Example Response:

```json
{
  "status": "healthy"
}
```

---

### Disease Prediction

```http
POST /predict
```

Upload a plant leaf image and receive a prediction.

#### Example Response

```json
{
  "plant": "Apple",
  "disease": "Apple Scab",
  "confidence": 95.6
}
```

---

## 💻 Run Locally

### Clone Repository

```bash
git clone https://github.com/leosanjit10/Plant_Disease_Diagnosis.git
cd Plant_Disease_Diagnosis
```

### Create Virtual Environment

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
```

### Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Start API

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## 🐳 Docker

### Build Image

```bash
docker build -t plant-disease-api .
```

### Run Container

```bash
docker run --rm -d --name plant-disease-api -p 10000:10000 plant-disease-api
```

### Health Check

```powershell
Invoke-RestMethod http://127.0.0.1:10000/health
```

### Stop Container

```bash
docker stop plant-disease-api
```

---

## 🧪 Testing the API

### Health Endpoint

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

### Prediction Endpoint

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" ^
-F "file=@C:\path\to\leaf.jpg"
```

Example Response:

```json
{
  "plant": "Apple",
  "disease": "Apple Scab",
  "confidence": 95.6
}
```

---

## 📈 Future Improvements

- Disease treatment recommendations
- Batch image predictions
- User authentication
- Mobile application integration
- Model monitoring and analytics
- Improved model accuracy with transfer learning

---

## 👨‍💻 Author

**Sanjit Sitaula**

GitHub: https://github.com/leosanjit10

---

## 📄 License

This project is licensed under the MIT License.
