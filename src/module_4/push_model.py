from sklearn.ensemble import GradientBoostingClassifier
from typing import Dict, Tuple

import pandas as pd


class PushModel:

    features = [
    'user_order_seq', 
    'ordered_before',
    'abandoned_before', 
    'active_snoozed', 
    'set_as_regular',
    'normalised_price', 
    'discount_pct', 
    'global_popularity',
    'count_adults', 
    'count_children', 
    'count_babies', 
    'count_pets',
    'people_ex_baby', 
    'days_since_purchase_variant_id',
    'avg_days_to_buy_variant_id', 
    'std_days_to_buy_variant_id',
    'days_since_purchase_product_type', 
    'avg_days_to_buy_product_type',
    'std_days_to_buy_product_type'
    ]

    target = 'outcome' 

    def __init__(
            self,
            classifier_parametrisation: Dict,
            prediction_threshold: int,
    ) -> None:
        
        self.model = GradientBoostingClassifier(**classifier_parametrisation)
        self.prediction_threshold = prediction_threshold

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.features]
        
    def _extract_target(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.target]
    
    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        return self._extract_features(df), self._extract_target(df)
    
    def fit(self, df: pd.DataFrame) -> None:
        features, target = self._split_data(df)
        self.model.fit(features, target)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        features = self._extract_features(df)
        probs = self.model.predict_proba(features)[:, 1]
        predictions = pd.Series(predictions, name='predictions')
        if hasattr(features, 'index'):
            predictions.index = features.index
        predictions = (probs > self.prediction_threshold).astype(int)        
        return predictions