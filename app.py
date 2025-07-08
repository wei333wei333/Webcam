from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO

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
    """Home route to display the video feed"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video feed route for Flask"""
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_active, detection_paused
    detection_active = True
    detection_paused = False
    return jsonify({"status": "Detection started"})

@app.route('/pause_detection', methods=['POST'])
def pause_detection():
    global detection_paused
    detection_paused = True
    return jsonify({"status": "Detection paused"})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_active, detection_paused
    detection_active = False
    detection_paused = False
    return jsonify({"status": "Detection stopped"})

if __name__ == '__main__':
    app.run(debug=True)
