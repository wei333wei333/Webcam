from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import numpy as np
import cv2
import io
import os

app = Flask(__name__)

# Load YOLO custom model
model = YOLO("best1.pt")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    image_bytes = file.read()
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model.predict(source=img, save=False, conf=0.3)
    annotated_img = results[0].plot()

    _, buffer = cv2.imencode('.jpg', annotated_img)
    return send_file(
        io.BytesIO(buffer),
        mimetype='image/jpeg'
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
