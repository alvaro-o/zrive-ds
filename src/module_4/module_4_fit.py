from datetime import datetime
from pathlib import Path
from typing import Dict
from push_model import PushModel

import pandas as pd
import logging
import json
import joblib

logger = logging.getLogger(__name__)
logger.level = logging.INFO

default_prediction_threshold = 0.05
default_path = (Path(__file__).parent.parent.parent.resolve() / 'bin/push_models/')

def load_dataframe() -> pd.dataframe:
    '''Loads the dataframe'''

    # read data
    loading_file = (Path(__file__).parent.parent.parent.resolve() / 'groceries/boxbuilder.csv')
    logger.info(f'loading dataset from {loading_file}')
    df = pd.read_csv()

    return df

def process_dataframe(df) -> pd.DataFrame:
    '''Gets dataframe and returns it processed to show only orders with 5 or more products'''

    # transform datetime columns
    df.created_at = pd.to_datetime(df.created_at)
    df.order_date = pd.to_datetime(df.order_date,format='%Y-%m-%d %H:%M:%S')
    # filter to only keep orders with more than 5 bought products
    ids = df[df.outcome == 1].groupby('order_id').variant_id.count() > 4
    df = df[df.order_id.isin(ids[ids == True].index)]

    return df

def handler_fit(event, _):
    '''Receives event dictionary with model parametrisation, prediction threshold and a path.
    Loads the data and fits the model with it. Returns json with the model with its path'''

    # model parameters and path
    model_parametrisation = event["model_parametrisation"]
    classifier_parametrisation = model_parametrisation.get(classifier_parametrisation)
    prediction_threshold = model_parametrisation.get(prediction_threshold, default_prediction_threshold)
    path = model_parametrisation.get(path, default_path)

    # data retrieval
    df = process_dataframe(load_dataframe())

    # model fitting
    model = PushModel(
        classifier_parametrisation,
        prediction_threshold
    )
    model.fit(df)

    # model saving
    current_date = datetime.now().strftime("%Y_%m_%d")
    model_name = f"push_{current_date}"

    model_path = f"{path}{model_name}.joblib"
    joblib.dump(model, model_path)
    
    return {
        "statusCode": "200",
        "body": json.dumps(
        {"model_path": model_path,
        }
    ),
}