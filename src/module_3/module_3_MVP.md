```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.metrics import precision_score, precision_recall_curve, recall_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import logging

logger = logging.getLogger(__name__)
logger.level=logging.INFO
ch = logging.StreamHandler()
logger.addHandler(ch)
```

**TAKEAWAYS**

Steps required:
1. Data loading and validation
2. Pre-processing
3. Model training/selection
- Data is loaded and checked for missing columns, incorrect data types and missing values
- Data is preprocessed to filter it, encode some variables and scale
- A pipeline is built to join the steps
- GridsearchCV is used to tune hiperparameters and select the best model
- The best model is saved and stored

LOADING AND VALIDATION


```python
path = '/home/alvaro/groceries/boxbuilder.csv'

required_columns = ['variant_id', 'product_type', 'order_id', 'user_id', 'created_at',
       'order_date', 'user_order_seq', 'outcome', 'ordered_before',
       'abandoned_before', 'active_snoozed', 'set_as_regular',
       'normalised_price', 'discount_pct', 'vendor', 'global_popularity',
       'count_adults', 'count_children', 'count_babies', 'count_pets',
       'people_ex_baby', 'days_since_purchase_variant_id',
       'avg_days_to_buy_variant_id', 'std_days_to_buy_variant_id',
       'days_since_purchase_product_type', 'avg_days_to_buy_product_type',
       'std_days_to_buy_product_type']

required_datatypes = {'variant_id': 'int64',
 'product_type': 'O',
 'order_id': 'int64',
 'user_id': 'int64',
 'created_at': 'O',
 'order_date': 'O',
 'user_order_seq': 'int64',
 'outcome': 'float64',
 'ordered_before': 'float64',
 'abandoned_before': 'float64',
 'active_snoozed': 'float64',
 'set_as_regular': 'float64',
 'normalised_price': 'float64',
 'discount_pct': 'float64',
 'vendor': 'O',
 'global_popularity': 'float64',
 'count_adults': 'float64',
 'count_children': 'float64',
 'count_babies': 'float64',
 'count_pets': 'float64',
 'people_ex_baby': 'float64',
 'days_since_purchase_variant_id': 'float64',
 'avg_days_to_buy_variant_id': 'float64',
 'std_days_to_buy_variant_id': 'float64',
 'days_since_purchase_product_type': 'float64',
 'avg_days_to_buy_product_type': 'float64',
 'std_days_to_buy_product_type': 'float64'}
```


```python
def load_validate(path: str):
    df = pd.read_csv(path)

    # Check columns
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns: {set(required_columns) - set(df.columns)}")

    # Check nan values
    if df.isnull().any().any():
        logger.info('Missing values found')

    # Check data types
    for col,required_type in required_datatypes.items():
        datatype = df.dtypes[col]
        if required_type != datatype:
            raise TypeError(f"Data type mismatch for column '{col}': Expected '{required_type}', but got '{datatype}'")

    else:
        logger.info('Data loaded correctly')
    
    return df
```

PREPROCESSING


```python
class OrderFilter(BaseEstimator, TransformerMixin):
    ''' filter dataset to only orders with more than 4 products'''
    def fit(self, df):
        return self 
    
    def transform(self, df):
        ids = df[df.outcome == 1].groupby('order_id').variant_id.count() > 4
        df_filtered = df[df.order_id.isin(ids[ids == True].index)]
        
        return df_filtered

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_dict = {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_encoded = X.copy()
        categorical_cols = ['product_type','vendor']
        for col in categorical_cols:
            self.freq_dict[col] = X[col].value_counts().to_dict()
            X_encoded[col] = X[col].map(self.freq_dict[col]).fillna(0)
        return X_encoded
```


```python
def preprocess(df: pd.DataFrame):
    # filter orders with more than 4 products
    df = OrderFilter().fit_transform(df)        

    # prevent info leakage by separating train, validation and test sets by time
    df = df.sort_values(by='order_date')
    X = df.drop(['variant_id',"order_id","user_id","created_at","order_date",'outcome'], axis=1)  
    y = df.outcome

    # train 0,6 and val+test 0,4
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.4, shuffle=False)
    # validation 0,2, test 0,2
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, shuffle=False)

    return X_train, y_train, X_val, y_val, X_test, y_test
```

MODEL OPTIMIZATION AND SAVING


```python
def select_model(X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame):
    pipe = Pipeline(steps=[
            ('freq_encoder', FrequencyEncoder()),
            ('scaling', StandardScaler()),
            ('lr', LogisticRegression())
            ])

    params = {
        'lr__C': [0.0000001, 0.00001, 0.001]
    }


    grid = GridSearchCV(estimator=pipe, param_grid=params, scoring=['precision','recall'],refit='precision', verbose=True)
    grid.fit(X_train,y_train)

    score = grid.score(X_val,y_val)
    best_model = grid.best_estimator_
    best_params = grid.best_params_

    return best_model, best_params, score
```


```python
def save_model(best_model,best_params):
    param = best_params['lr__C']
    joblib.dump(best_model, f'lr_c_{param}.joblib')
```


```python
def main():
    df = load_validate(path)
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess(df)
    best_model, best_params, score = select_model(X_train, y_train, X_val, y_val)

    logger.info(f'Score: {score}')
    logger.info(f'Best hyperparameters: {best_params}')

    save_model(best_model,best_params)
```


```python
if __name__ == "__main__":
    main()
```

    Data loaded correctly


    Fitting 5 folds for each of 3 candidates, totalling 15 fits


    /home/alvaro/.cache/pypoetry/virtualenvs/zrive-ds-HJx0T28e-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1531: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
    /home/alvaro/.cache/pypoetry/virtualenvs/zrive-ds-HJx0T28e-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1531: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
    /home/alvaro/.cache/pypoetry/virtualenvs/zrive-ds-HJx0T28e-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1531: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
    /home/alvaro/.cache/pypoetry/virtualenvs/zrive-ds-HJx0T28e-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1531: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
    /home/alvaro/.cache/pypoetry/virtualenvs/zrive-ds-HJx0T28e-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1531: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
    Score: 0.6844106463878327
    Best hyperparameters: {'lr__C': 1e-05}

