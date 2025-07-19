import os
import torch
import torchvision.transforms as transforms
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch.nn as nn

# Initialize Flask App
app = Flask(__name__, template_folder="templates", static_folder="static")

# Model Path
MODEL_PATH = "D:\\Projects\\Brain_tumor\\brain_tumor_classification_model.pth"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

# ✅ Define the Correct Model Structure
class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 18 * 18, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # Output 2 classes (NO, YES)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# ✅ Load Trained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BrainTumorCNN().to(device)

try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Error loading PyTorch model: {e}")

# ✅ Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ✅ Class Labels (Modified for Positive/Negative)
class_labels = ["Brain Tumor Negative", "Brain Tumor Positive"]  # 0 = No, 1 = Yes

# ✅ Route for Home Page
@app.route("/")
def index():
    return render_template("index.html")

# ✅ Route for Prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Load and preprocess the image
        image = Image.open(file).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Debugging: Print Image Shape
        print("Image Shape:", image_tensor.shape)

        # Predict using the model
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)[0]  # Get probabilities
            confidence, predicted = torch.max(probabilities, 0)  # Get highest confidence score

        # Get class label and confidence percentage
        prediction_label = class_labels[predicted.item()]
        confidence_percentage = round(confidence.item() * 100, 2)  # Convert to percentage

        # Debugging: Print Model Output
        print(f"Model Output: {output.tolist()}")
        print(f"Probabilities: {probabilities.tolist()}")
        print(f"Predicted Class: {prediction_label} with {confidence_percentage}% confidence")

        # Return JSON response
        return jsonify({
            "prediction": prediction_label,
            "confidence": confidence_percentage
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(debug=True)