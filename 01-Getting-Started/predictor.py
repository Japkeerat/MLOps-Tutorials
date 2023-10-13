import pickle

import pandas as pd
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model_path = "model.pkl"


class InputData(BaseModel):
    Location: str
    MinTemp: Optional[float] = None
    MaxTemp: Optional[float] = None
    Rainfall: Optional[float] = None
    Evaporation: Optional[float] = None
    Sunshine: Optional[float] = None
    WindGustDir: Optional[str] = None
    WindGustSpeed: Optional[float] = None
    WindDir9am: Optional[str] = None
    WindDir3pm: Optional[str] = None
    WindSpeed9am: Optional[float] = None
    WindSpeed3pm: Optional[float] = None
    Humidity9am: Optional[float] = None
    Humidity3pm: Optional[float] = None
    Pressure9am: Optional[float] = None
    Pressure3pm: Optional[float] = None
    Cloud9am: Optional[float] = None
    Cloud3pm: Optional[float] = None
    Temp9am: Optional[float] = None
    Temp3pm: Optional[float] = None
    RainToday: Optional[str] = None


# Load the model
with open(model_path, 'rb') as file:
    pipeline = pickle.load(file)


@app.post("/predict")
async def predict(input_data: InputData):
    input_df = pd.DataFrame.from_records([dict(input_data)])
    predictions = pipeline.predict(input_df)
    return {"prediction": int(predictions[0])}
