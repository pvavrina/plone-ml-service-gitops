# app.py: FastAPI Inference API for the Plone ML Service

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import os
import torch.serialization
from model import SimpleClassifier # Import of the custom class

# --- Global Configuration and Initialization ---

# Define the path where the trained PyTorch model file is expected to be located
MODEL_PATH = "model/trained_model.pt"

# Global variable to hold the loaded model state
# Initialize to None; actual loading happens during startup event.
model = None

# Initialize FastAPI application
app = FastAPI(
    title="Plone ML Inference Service",
    description="Serves predictions from the trained PyTorch model.",
    version="1.0.0"
)

# --- Data Schemas ---

# Define the structure of the incoming request body (input data for the model)
class Item(BaseModel):
    features: list[float]
    # Example: features: list[float] = [0.1, 0.2, 0.3, 0.4]

# Define the structure of the prediction response
class Prediction(BaseModel):
    prediction_result: list[float]
    # Example: prediction_result: list[float]

# --- Startup Event: Load Model ---

# This function runs ONCE when the Uvicorn server starts up

@app.on_event("startup")
async def load_model_on_startup():
    """Load the PyTorch model from the specified path."""
    global model
    
    # Simple check to ensure the model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}. Exiting.")
        return
        
    try:
        # FIX: Allow PyTorch to safely load custom class (required since PyTorch 2.6 security update)
        # We must allow the custom class 'SimpleClassifier' to be unpickled.
        torch.serialization.add_safe_globals([SimpleClassifier])
        
        # Load the entire model object onto the CPU.
        # We include weights_only=False to bypass strict security checks, which is necessary
        # when loading models saved with custom structures that PyTorch 2.6+ now blocks by default.
        model = torch.load(
            MODEL_PATH, 
            map_location=torch.device('cpu'),
            weights_only=False # CRITICAL FIX for the 'WeightsUnpickler error'
        )
        
        model.eval() # Set the model to evaluation mode
        print(f"SUCCESS: PyTorch model loaded from {MODEL_PATH}")
        
    except Exception as e:
        print(f"ERROR loading PyTorch model: {e}")
        model = None # Ensure model is None if loading failed

# --- API Endpoints ---

@app.get("/")
def health_check():
    """Health check endpoint to verify service status."""
    status = "OK" if model is not None else "Model Loading Failed"
    return {"status": status, "api_version": app.version}

@app.post("/predict", response_model=Prediction)
async def predict(item: Item):
    """
    Endpoint to receive data and return model predictions.
    """
    global model
    
    if model is None:
        return {"prediction_result": ["Error: Model not available"]}
        
    try:
        # 1. Convert input list (from Item.features) to a PyTorch tensor
        input_tensor = torch.tensor([item.features], dtype=torch.float32)
        
        # 2. Perform inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # 3. Convert output tensor back to a standard Python list
        prediction_list = output.tolist()[0]
        
        return {"prediction_result": prediction_list}
        
    except Exception as e:
        # Log the error and return a safe message
        print(f"Prediction error: {e}")
        return {"prediction_result": ["Prediction failed due to internal error"]}