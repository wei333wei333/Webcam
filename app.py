from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import numpy as np
import cv2
import io
from PIL import Image

app = Flask(__name__)

# Load YOLO custom model
model = YOLO("best1.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)  # Open webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Flags for controlling detection
detection_active = False  # Start detection as False
detection_paused = False  # Flag for pausing detection

def detect_objects_in_image(image):
    """Detect objects in a captured image using YOLO"""
    results = model.predict(source=image, save=False, conf=0.3)
    annotated_frame = results[0].plot(labels=True)
    return annotated_frame

def generate():
    """Generate video stream for Flask app"""
    while True:
        if detection_active and not detection_paused:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame with YOLO if detection is active
            annotated_frame = detect_objects_in_image(frame)

            ret, jpeg = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        elif detection_paused:
            # Send webcam feed without detection if paused
            ret, frame = cap.read()
            if not ret:
                break

            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        else:
            # No detection or paused, keep the feed running without processing
            ret, frame = cap.read()
            if not ret:
                break

            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

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
