from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import numpy as np
import cv2
import io
import os
from PIL import Image

app = Flask(__name__)

# Load YOLO custom model
model = YOLO("best1.pt")

@app.route('/')
def index():
    """Home page with webcam interface"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receive uploaded image, run YOLO detection, return annotated image"""
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    image_bytes = file.read()
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Run YOLO detection
    results = model.predict(source=img, save=False, conf=0.3)
    result = results[0]
    annotated_img = result.plot(labels=True, names=CLASS_NAMES)

    # Extract detected class labels
    detected_classes = set()
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        detected_classes.add(CLASS_NAMES[class_id])

    # Encode annotated image to hex string
    _, buffer = cv2.imencode('.jpg', annotated_img)
    image_bytes = buffer.tobytes()

    return jsonify({
        'image': image_bytes.hex(),
        'labels': list(detected_classes)
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
