from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import numpy as np
import cv2
import io
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
    annotated_img = results[0].plot(labels=True)

    # Encode the result back to image format
    _, buffer = cv2.imencode('.jpg', annotated_img)
    return send_file(
        io.BytesIO(buffer),
        mimetype='image/jpeg',
        as_attachment=False,
        download_name='result.jpg'
    )

if __name__ == '__main__':
    app.run(debug=True)
