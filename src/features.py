# src/features.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Custom Transformer to combine features
class FeatureCombiner(BaseEstimator, TransformerMixin):
    """
    A custom transformer to combine existing categorical features into new ones.
    """
    def fit(self, X, y=None):
        """
        Fit method. Does nothing for this transformer.
        """
        return self

    def transform(self, X):
        """
        Transforms the input DataFrame by adding combined feature columns.
        """
        X_transformed = X.copy()
        X_transformed['amount_category'] = pd.cut(X_transformed['amount'], bins = [0, X_transformed['amount'].quantile(0.33), X_transformed['amount'].quantile(0.66), np.inf], labels=['low', 'medium', 'high'])


        X_transformed['merchant_device_add'] = (X_transformed['merchant_type'].astype(str) + "_" +
                                                X_transformed['device_type'].astype(str) + "_" +
                                                X_transformed['amount_category'].astype(str))
        X_transformed['merchant_device_only'] = (X_transformed['merchant_type'].astype(str) + "_" +
                                                 X_transformed['device_type'].astype(str))
        return X_transformed

# Custom Transformer to calculate Fraud Rate per Combination
class FraudRateCalculator(BaseEstimator, TransformerMixin):
    """
    A custom transformer to calculate the mean fraud rate for each unique
    'merchant_device_only' combination based on the training data (y).
    """
    def __init__(self):
        """
        Initializes the FraudRateCalculator with an empty fraud rate map.
        """
        self.fraud_rate_map = None

    def fit(self, X, y):
        """
        Calculates the fraud rate for each 'merchant_device_only' combination.
        """
        df_temp = X.copy()
        df_temp['label'] = y
        self.fraud_rate_map = df_temp.groupby('merchant_device_only')['label'].mean().to_dict()
        return self

    def transform(self, X):
        """
        Maps the calculated fraud rates to the 'fraud_rate_combo' column.
        """
        X_transformed = X.copy()
        X_transformed['fraud_rate_combo'] = X_transformed['merchant_device_only'].map(self.fraud_rate_map).fillna(0)
        return X_transformed

# Custom Transformer to create High Risk Combo Indicator
class HighRiskIndicator(BaseEstimator, TransformerMixin):
    """
    A custom transformer to identify high-risk 'merchant_device_only' combinations
    based on a fraud rate quantile threshold and create a binary indicator column.
    """
    def __init__(self, quantile_threshold=0.85):
        """
        Initializes the HighRiskIndicator with a quantile threshold.
        """
        self.quantile_threshold = quantile_threshold
        self.high_risk_combos = None

    def fit(self, X, y=None):
        """
        Identifies the high-risk 'merchant_device_only' combinations.
        """
        # Assuming 'fraud_rate_combo' is already calculated and present in X
        if 'fraud_rate_combo' not in X.columns:
             raise ValueError("'fraud_rate_combo' column is required for HighRiskIndicator.")

        threshold = X['fraud_rate_combo'].quantile(self.quantile_threshold)
        self.high_risk_combos = X[X['fraud_rate_combo'] >= threshold]['merchant_device_only'].unique().tolist()
        return self

    def transform(self, X):
        """
        Creates the 'is_high_risk_combo' indicator column.
        """
        X_transformed = X.copy()
        X_transformed['is_high_risk_combo'] = X_transformed['merchant_device_only'].isin(self.high_risk_combos).astype(int)
        return X_transformed