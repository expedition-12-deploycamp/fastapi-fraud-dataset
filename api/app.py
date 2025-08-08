# api/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Add the src directory to the Python path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import make_prediction

# Define the threshold based on your MLflow experiments
# Replace with the actual best threshold found in your experiment
PREDICTION_THRESHOLD = 0.88

app = FastAPI()

# Define the expected structure of the input data using Pydantic
class TransactionInput(BaseModel):
    amount: float
    merchant_type: str
    device_type: str


@app.get("/")
def read_root():
    """
    Basic endpoint to check if the API is running.
    """
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict")
def predict_fraud(transaction: TransactionInput):
    """
    Endpoint to predict if a transaction is fraudulent.

    Args:
        transaction (TransactionInput): The input transaction data.

    Returns:
        dict: A dictionary containing the prediction result.
    """
    try:
        # Convert the Pydantic model to a dictionary
        input_data = transaction.model_dump()

        # Make prediction using the function from predict.py
        is_fraud = make_prediction(input_data, PREDICTION_THRESHOLD)

        return {"is_fraud": is_fraud}

    except Exception as e:
        # Log the error and return an HTTPException
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during prediction")

if __name__ == "__main__":
    import uvicorn
    # To run this from the main project directory, use:
    # uvicorn api.app:app --reload
    # In a Colab environment, you might need to use ngrok or a similar service
    # to expose the local port.
    print("To run the FastAPI application, execute 'uvicorn api.app:app --reload' from the project root directory.")
    # Example of how to run it directly (less common for production):
    # uvicorn.run(app, host="0.0.0.0", port=8000)