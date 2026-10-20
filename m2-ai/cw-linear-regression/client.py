import sys
import requests

def test_tf_serving(url="http://localhost:8501/v1/models/linear_model:predict"):
    print(f"Connecting to TensorFlow Serving at {url}...")
    # TensorFlow Serving requires instances matching the layer input dimension:
    # Each sample is shape (1,), so 3 samples are formatted as a 2D array:
    payload = {
        "instances": [[0.0], [1.0], [4.0]]
    }
    try:
        response = requests.post(url, json=payload, timeout=5)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            results = response.json()
            print("Response:", results)
            predictions = results.get("predictions", [])
            for inp, pred in zip([0.0, 1.0, 4.0], predictions):
                val = pred[0] if isinstance(pred, list) else pred
                print(f"Input: {inp} -> Prediction: {val:.4f} (Expected: {2.0 * inp + 1.0:.1f})")
            return True
        else:
            print(f"Error Response: {response.text}")
            return False
    except Exception as e:
        print(f"Request failed: {e}")
        return False

def test_fastapi(url="http://localhost:8000/predict"):
    print(f"\nTesting FastAPI service at {url}...")
    payload = {
        "instances": [[0.0], [1.0], [4.0]]
    }
    try:
        response = requests.post(url, json=payload, timeout=5)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            results = response.json()
            print("Response:", results)
            return True
        else:
            print(f"Error Response: {response.text}")
            return False
    except Exception as e:
        print(f"FastAPI request failed: {e}")
        return False

if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "8501"
    if port == "8000":
        test_fastapi()
    else:
        success = test_tf_serving(f"http://localhost:{port}/v1/models/linear_model:predict")
        if not success:
            test_fastapi()
