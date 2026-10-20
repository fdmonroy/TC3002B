# Linear Regression Model & Serving (TC3002B)

Implementation and deployment of the linear regression model covered in **Class: Advanced Applications. AI.** (October 19, 2026).

## Overview

The class covered the definition, training, export, and deployment of a single-layer neural network with Keras and TensorFlow Serving:
- Learning the linear relation: `y = 2x + 1`
- Model compilation with `SGD` and `MeanSquaredError`
- Model serialization to TensorFlow SavedModel format (`linear_model/1`)
- Deployment to Docker using `tensorflow/serving` on REST port `8501`
- REST API client and shape considerations to avoid HTTP 400 Bad Request

## Structure

```
cw-linear-regression/
├── .gitignore          # Ignores .venv, cache, and logs
├── keras1.py           # Single-layer architecture and initial weight inspection
├── linear_model.py     # Training, evaluation, and SavedModel export
├── client.py           # REST API client querying the model server
├── server_fastapi.py   # FastAPI alternative serving SavedModel directly
├── README.md           # Documentation and run instructions
└── linear_model/       # Exported SavedModel bundle for TF Serving
    └── 1/
        ├── saved_model.pb
        ├── fingerprint.pb
        └── variables/
```

## Running the Code

### 1. Environment Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install tensorflow numpy requests fastapi uvicorn httpx
```

### 2. Exploratory Architecture (`keras1.py`)

Inspects layer definition and randomly initialized weights `W` and `b`:

```bash
python keras1.py
```

### 3. Model Training & Export (`linear_model.py`)

Generates synthetic data, shuffles, splits into training/validation, fits for 20 epochs, and exports the SavedModel bundle:

```bash
python linear_model.py
```

### 4. Docker TensorFlow Serving Deployment

Commands executed from class:

```bash
# Pull official image
docker pull tensorflow/serving

# Create base container
docker run -d --name serving_base tensorflow/serving

# Copy exported model weights
docker cp linear_model serving_base:/models/linear_model

# Commit container with model environment configuration
docker commit --change "ENV MODEL_NAME linear_model" serving_base my_serving_base

# Clean base container and launch production serving container
docker rm -f serving_base
docker run -d --name linear_model_serving -p 8501:8501 my_serving_base
```

### 5. Testing Predictions via Client (`client.py`)

Query the running TensorFlow Serving REST API:

```bash
python client.py
```

Example output:
```
Connecting to TensorFlow Serving at http://localhost:8501/v1/models/linear_model:predict...
Status Code: 200
Response: {'predictions': [[0.315133393], [2.30863214], [8.2891283]]}
Input: 0.0 -> Prediction: 0.3151 (Expected: 1.0)
Input: 1.0 -> Prediction: 2.3086 (Expected: 3.0)
Input: 4.0 -> Prediction: 8.2891 (Expected: 9.0)
```

### 6. Alternative Serving via FastAPI (`server_fastapi.py`)

Lightweight fallback alternative implemented during class by Lu:

```bash
python server_fastapi.py
```

Test with client:
```bash
python client.py 8000
```

## Troubleshooting: 400 Bad Request Root Cause

During class, requests to TensorFlow Serving returned `400 Bad Request`.
- **Cause**: The Dense layer input shape is `[1]` (`(None, 1)`). A flat array `{"instances": [0.0, 1.0, 4.0]}` sends scalar instances of shape `()`.
- **Solution**: Instances must match input rank: `{"instances": [[0.0], [1.0], [4.0]]}`.
