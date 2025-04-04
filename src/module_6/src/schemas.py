from pydantic import BaseModel

class PredictionRequest(BaseModel):
    user_id: str

class Prediction(BaseModel):
    predicted_price: float