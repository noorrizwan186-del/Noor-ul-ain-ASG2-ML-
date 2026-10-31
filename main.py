from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

app = FastAPI(title="Pharmaceutical Machine Learning App")

# Allow the local HTML page to call the API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

regression_model = joblib.load(MODEL_DIR / "regression_model.pkl")
classification_model = joblib.load(MODEL_DIR / "classification_model.pkl")

class RegressionInput(BaseModel):
    Name: str = Field(..., examples=["Zestril"])
    Company: str = Field(..., examples=["ICI Pakistan Limited"])
    Price_before_numeric: float = Field(..., ge=0, examples=[202])
    Discount_percent: float = Field(0, ge=0, le=100, examples=[10])
    Pack_Size: str = Field(..., examples=["1x14's"])

class ClassificationInput(BaseModel):
    Name: str = Field(..., examples=["Zestril"])
    Company: str = Field(..., examples=["ICI Pakistan Limited"])
    Price_before_numeric: float = Field(..., ge=0, examples=[202])
    Discount_percent: float = Field(10, ge=0, le=100, examples=[10])
    Pack_Size: str = Field(..., examples=["1x14's"])
    Price_After: float = Field(..., ge=0, examples=[182])

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")

@app.get("/health")
def health():
    return {"status": "running", "message": "Pharmaceutical ML API is ready"}

@app.post("/predict/regression")
def predict_regression(data: RegressionInput):
    row = pd.DataFrame([data.model_dump()])
    prediction = float(regression_model.predict(row)[0])
    return {"predicted_price_pkr": round(prediction, 2)}

@app.post("/predict/classification")
def predict_classification(data: ClassificationInput):
    row = pd.DataFrame([data.model_dump()])
    prediction = str(classification_model.predict(row)[0])
    return {"predicted_availability": prediction}
