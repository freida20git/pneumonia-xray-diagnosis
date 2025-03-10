import torch
import numpy as np
import cv2
from PIL import Image
import base64
from torchvision import transforms
from model_loader import model  # Import the trained model
from model_loader import GradCAM  # GradCAM class is included in model_loader.py file
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Define image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


# Main processing function
def process_image(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Preprocess input image
    image_tensor = preprocess_image(image).to(device)
    # Initialize GradCAM
    grad_cam = GradCAM(model, target_layer_name='efficientnet.features.7.0.block.3.1')

    # Run forward pass & get prediction
    output = grad_cam.forward(image_tensor)
    print("Raw logits:", output.item())
    probability = torch.sigmoid(output).item()
    print(f"probability:{probability}")
    pred_class = 1 if probability > 0.5 else 0
    confidence = probability if pred_class == 1 else 1 - probability
    diagnosis = "Healthy" if pred_class == 1 else "Pneumonia"

    # Generate GradCAM heatmap
    model.zero_grad()
    output.backward()  # Compute gradients
    heatmap = grad_cam.generate_cam(pred_class)
    grad_cam.remove_hooks()

    # Convert original image to NumPy
    original_image = np.array(image.resize((224, 224)))

    # Overlay heatmap on the image
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

    # Encode the image to return as response
    _, buffer = cv2.imencode('.png', superimposed_img)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    return base64_image, f"{diagnosis} (Confidence: {confidence:.2f})"

@app.route('/process_image', methods=['POST'])
def process_image_route():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'})

        file = request.files['image']
        image = Image.open(file.stream).convert("RGB")

        processed_image, diagnosis_msg = process_image(image)

        return jsonify({
            'success': True,
            'processed_image': processed_image,
            'msg': diagnosis_msg
        })

    except Exception as e:
        print(f"🔥 Error occurred: {e}")  # Debugging print
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)