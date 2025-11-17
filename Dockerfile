## Dockerfile for plone-ml-service:v1

# Use the Universal Base Image (UBI) 9 with Python 3.9 for maximum compatibility with AlmaLinux
FROM registry.access.redhat.com/ubi9/python-39

# Set the working directory inside the container
WORKDIR /app

# 1. Copy the requirements file
COPY requirements.txt .

# 2. Install PyTorch/FastAPI dependencies
# These will be the CPU versions. Using --no-cache-dir to keep the image size small.
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the application code (model.py, app.py for the API, and your trained model file, e.g., model.pt)
COPY . .

# 4. Expose the API port (8000 is the FastAPI/Uvicorn default)
EXPOSE 8000

# 5. Define the startup command (run the web API using Uvicorn)
# NOTE: Replace 'app:app' with the name of your Python file (e.g., 'inference_api.py') and FastAPI object.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
