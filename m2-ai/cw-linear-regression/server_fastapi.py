import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf

app = FastAPI(title="Linear Model Serving (FastAPI fallback)")

# Model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "linear_model", "1")
model = None

class PredictRequest(BaseModel):
    instances: list

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            # Load SavedModel
            model = tf.saved_model.load(MODEL_PATH)
            print(f"Loaded SavedModel from {MODEL_PATH}")
        except Exception as e:
            print(f"Could not load SavedModel: {e}")
    else:
        print(f"Model path {MODEL_PATH} not found. Please train the model first.")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
@app.post("/v1/models/linear_model:predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        data = np.array(req.instances, dtype=np.float32)
        # Reshape to (N, 1) if provided as 1D array
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        infer = model.signatures["serving_default"]
        tensor_input = tf.constant(data)
        out = infer(tensor_input)
        # First output tensor
        first_key = list(out.keys())[0]
        preds = out[first_key].numpy().tolist()
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
