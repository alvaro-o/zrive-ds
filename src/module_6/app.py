import uvicorn
from fastapi import FastAPI, HTTPException, Response
import logging
from time import time

from src.basket_model.feature_store import FeatureStore
from src.basket_model.basket_model import BasketModel
from src.metrics import Metrics
from src.schemas import PredictionRequest, Prediction
from src.exceptions import UserNotFoundException, PredictionException


app = FastAPI()

# Status
@app.get("/status")
async def get_status():
    return {"status": "200"}


# Predict
feature_store = FeatureStore()
model = BasketModel()
metrics = Metrics()


@app.post("/predict")
async def make_prediction(request: PredictionRequest) -> Prediction:
    metrics.increase_request()
    start_time = time()
    user_id = request.user_id
    try:
        features = feature_store.get_features(user_id)
        predicted_price = model.predict(features)

    except UserNotFoundException:
        metrics.increase_user_not_found_errors()
        logging.error(f"User {user_id} not found in the feature store")
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    except PredictionException:
        metrics.increase_model_errors()
        logging.error(f"User {user_id}, Prediction not completed")
        raise HTTPException(status_code=500, detail=f"Prediction could not be made")

    except Exception as e:
        metrics.increase_unknown_errors()
        logging.error(f'User {user_id}, unknown exception', exc_info=e)
        raise HTTPException(status_code=500, detail='Unknown exception')

    metrics.observe_predict_duration(start_time)
    return Prediction(predicted_price=predicted_price.mean())


# Metrics
from prometheus_client import core, exposition

@app.get("/metrics")
async def export_metrics():
    return Response(
        content = exposition.generate_latest(core.REGISTRY),
        media_type= 'text/plain'
    )


# This block allows you to run the application using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)