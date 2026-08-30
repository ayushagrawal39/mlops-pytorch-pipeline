import os
from pathlib import Path
import torch
from PIL import Image
from flask import Flask, jsonify, request
from torchvision import transforms
from model import get_model

CHECKPOINT = Path(os.getenv("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt"))
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
try:
    if CHECKPOINT.exists():
        saved = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        model = get_model(saved.get("architecture", "cnn"), saved.get("num_classes", 10)).to(device)
        model.load_state_dict(saved["model_state_dict"])
        model.eval()
except Exception as exc:
    app.logger.error("model load failed: %s", exc)

preprocess = transforms.Compose([
    transforms.Resize((32, 32)), transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


@app.get("/health")
def health():
    return jsonify({"status": "ok" if model is not None else "model_not_loaded"}), 200 if model else 503


@app.post("/predict")
def predict():
    if model is None:
        return jsonify({"error": "model not loaded"}), 503
    if "image" not in request.files:
        return jsonify({"error": "multipart field 'image' is required"}), 400
    x = preprocess(Image.open(request.files["image"].stream).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(x), 1)[0].cpu().tolist()
    return jsonify({"class_probabilities": probabilities, "predicted_class": int(max(range(len(probabilities)), key=probabilities.__getitem__))})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
