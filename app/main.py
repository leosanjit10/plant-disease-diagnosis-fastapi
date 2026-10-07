from contextlib import asynccontextmanager
from typing import AsyncIterator

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.model_loader import load_plant_model
from app.utils import CLASS_NAMES, ImageValidationError, format_class_name, preprocess_image

MAX_IMAGE_BYTES = 10 * 1024 * 1024


class PredictionResponse(BaseModel):
    plant: str
    disease: str
    confidence: float


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    application.state.model = load_plant_model()
    yield
    application.state.model = None


app = FastAPI(
    title="Plant Disease Detection API",
    description="Classifies a plant leaf image into one of 38 PlantVillage classes.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["status"])
async def home() -> dict[str, str]:
    return {"message": "Plant Disease Detection API"}


@app.get("/health", tags=["status"])
async def health(request: Request) -> dict[str, str | bool]:
    if getattr(request.app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="The prediction model is not loaded.")
    return {"status": "ok", "model_loaded": True}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["predictions"],
)
async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    try:
        image_bytes = await file.read(MAX_IMAGE_BYTES + 1)
    finally:
        await file.close()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB upload limit.",
        )

    try:
        image_batch = preprocess_image(image_bytes)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    scores = await run_in_threadpool(
        request.app.state.model.predict,
        image_batch,
        verbose=0,
    )
    scores = np.asarray(scores)
    if scores.shape != (1, len(CLASS_NAMES)) or not np.isfinite(scores).all():
        raise HTTPException(
            status_code=500,
            detail="The model returned invalid prediction scores.",
        )

    class_index = int(np.argmax(scores[0]))
    confidence = float(scores[0, class_index])
    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(
            status_code=500,
            detail="The model returned a confidence outside the expected range.",
        )

    plant, disease = format_class_name(CLASS_NAMES[class_index])
    return PredictionResponse(
        plant=plant,
        disease=disease,
        confidence=round(confidence * 100, 1),
    )
