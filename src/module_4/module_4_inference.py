from datetime import datetime
from pathlib import Path
from typing import Dict
from module_4_fit import dafault_path, process_dataframe

import pandas as pd
import logging
import json
import joblib

logger = logging.getLogger(__name__)
logger.level = logging.INFO


def handler_predict(event, _):
    '''
    Receives event dictionary with the following format
    {
        'users': {
            "user_id": {"feature 1": feature value, "feature 2": feature value, ...},
            "user_id2": {"feature 1": feature value, "feature 2": feature value, ...}.
        },

        'model_path': value
    }

    Returns json with predictions from a loaded model. If model not specified in event, model from current date is used.
    '''

    data_to_predict = pd.DataFrame.from_dict(json.loads(event["users"]))
    data_to_predict = process_dataframe(data_to_predict)

    current_date = datetime.now().strftime("%Y_%m_%d")
    model_name = f"push_{current_date}.joblib"

    model_path = event.get('model_path', dafault_path / f'{model_name}')
    model = joblib.load(model_path)

    predictions = model.predict(data_to_predict)

    return {
    "statusCode": "200",
    "body": json.dumps({"prediction": predictions.to_dict()})
}