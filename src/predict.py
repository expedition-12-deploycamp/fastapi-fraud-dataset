# src/predict.py

import joblib
import pandas as pd
import numpy as np

# Define the path where the pipeline and model are saved
PIPELINE_PATH = '../models/full_preprocessing_pipeline.pkl'
MODEL_PATH = '../models/best_xgboost_model.pkl'

# Load the saved pipeline and model (load once when the module is imported)
try:
    loaded_pipeline = joblib.load(PIPELINE_PATH)
    loaded_model = joblib.load(MODEL_PATH)
    print("Preprocessing pipeline and model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Ensure '{PIPELINE_PATH}' and '{MODEL_PATH}' exist.")
    loaded_pipeline = None
    loaded_model = None

def make_prediction(data: dict, threshold: float) -> bool:
    """
    Makes a fraud prediction on new data using the loaded pipeline and model.

    Args:
        data (dict): A dictionary containing the input transaction data.
                     Expected keys: 'amount', 'merchant_type', 'device_type', 'amount_category'.
                     Note: 'fraud_rate_combo' is calculated by the pipeline.
        threshold (float): The prediction probability threshold.

    Returns:
        bool: True if the prediction is fraud, False otherwise.
    """
    if loaded_pipeline is None or loaded_model is None:
        print("Error: Model or pipeline not loaded. Cannot make prediction.")
        return False # Or raise an exception

    try:
        # Convert input data to a pandas DataFrame
        # Ensure the input data matches the structure expected by the pipeline before feature engineering
        # The required columns before the pipeline are: 'amount', 'merchant_type', 'device_type', 'amount_category'
        input_df = pd.DataFrame([data])

        # Apply the full preprocessing pipeline to the input data
        # Note: SMOTE is part of the training pipeline, but the loaded pipeline here
        # is the 'full_preprocessing_pipeline' which does not include SMOTE.
        # The 'full_preprocessing_pipeline' should handle feature engineering and encoding.
        processed_data = loaded_pipeline.transform(input_df)

        # Make prediction using the loaded model
        # The model expects the output of the preprocessing pipeline
        y_proba = loaded_model.predict_proba(processed_data)[:, 1]
        prediction = (y_proba >= threshold).astype(int)[0]

        return bool(prediction)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return False # Or raise the exception

if __name__ == '__main__':
    # Example usage
    # This part will run only if you execute predict.py directly

    # Example data for prediction (replace with actual data structure)
    example_data = {
        'amount': 50.0,
        'merchant_type': 'electronics',
        'device_type': 'mobile',
        'amount_category': 'medium' # Ensure this is included if your pipeline uses it
    }

    # Replace with your best threshold
    prediction_threshold = 0.88

    is_fraud = make_prediction(example_data, prediction_threshold)

    if is_fraud:
        print(f"Transaction is predicted as FRAUD (Threshold: {prediction_threshold})")
    else:
        print(f"Transaction is predicted as NON-FRAUD (Threshold: {prediction_threshold})")