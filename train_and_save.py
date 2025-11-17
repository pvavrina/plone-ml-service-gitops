# train_and_save.py: Script to create and save a placeholder PyTorch model

import torch
import os
from model import SimpleClassifier # Import the model definition

# --- Configuration ---
INPUT_SIZE = 4      # The number of features expected by your model (match Item in app.py)
OUTPUT_SIZE = 1     # The number of outputs (e.g., probability, single class)
MODEL_SAVE_PATH = "model/trained_model.pt"

# --- Main Logic ---

# 1. Ensure the directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# 2. Instantiate the model
placeholder_model = SimpleClassifier(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE)
print(f"Placeholder model created (Input: {INPUT_SIZE}, Output: {OUTPUT_SIZE}).")

# 3. Save the model object
try:
    # We save the entire model object for simplicity in MLOps demo
    torch.save(placeholder_model, MODEL_SAVE_PATH)
    print(f"SUCCESS: Placeholder PyTorch model saved to {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"ERROR: Failed to save model: {e}")

# Note: In a real MLOps pipeline, this script would run after full training
# and save the model for versioning.
