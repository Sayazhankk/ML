import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Iris ML API")

artifact = joblib.load("model.joblib")
model = artifact["model"]
target_names = artifact["target_names"]


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, examples=[5.1])
    sepal_width: float = Field(..., gt=0, examples=[3.5])
    petal_length: float = Field(..., gt=0, examples=[1.4])
    petal_width: float = Field(..., gt=0, examples=[0.2])


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(features: IrisFeatures):
    x = [[features.sepal_length, features.sepal_width,
          features.petal_length, features.petal_width]]
    idx = int(model.predict(x)[0])
    proba = float(model.predict_proba(x)[0][idx])
    return {"prediction": idx, "class_name": target_names[idx], "probability": round(proba, 4)}
