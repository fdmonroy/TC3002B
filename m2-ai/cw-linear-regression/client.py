import requests

URL = "http://localhost:8501/v1/models/linear_model:predict"

# TensorFlow Serving expects instances matching the input tensor rank: [[x]]
payload = {
    "instances": [[0.0], [1.0], [4.0]]
}

response = requests.post(URL, json=payload, timeout=5)
print(f"Status Code: {response.status_code}")

if response.status_code == 200:
    predictions = response.json().get("predictions", [])
    for inp, pred in zip([0.0, 1.0, 4.0], predictions):
        val = pred[0] if isinstance(pred, list) else pred
        print(f"Input: {inp:.1f} -> Predicted: {val:.4f} (Expected: {2.0 * inp + 1.0:.1f})")
else:
    print(f"Error: {response.text}")
