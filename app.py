# app.py
from flask import Flask, render_template, request, send_file, Response, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
import io
import zipfile
import os
from datetime import datetime
from threading import Thread, Lock
import time

app = Flask(__name__)
model = YOLO("best1.pt")

camera_active = False
detection_active = False
detection_paused = False
cap = None
frame_buffer = None
camera_lock = Lock()
processing_progress = {}

# ========== Camera Handling ==========
def init_camera():
    global cap, camera_active
    with camera_lock:
        if cap is None:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera_active = True
                return True
            cap = None
            return False
        return camera_active

def release_camera():
    global cap, camera_active
    with camera_lock:
        if cap:
            cap.release()
            cap = None
        camera_active = False

# ========== Object Detection ==========
def detect_objects(frame):
    results = model.predict(source=frame, save=False, conf=0.3)
    return results[0].plot()

# ========== Video Streaming ==========
def generate_frames():
    global frame_buffer
    while True:
        with camera_lock:
            if not camera_active or cap is None:
                frame_buffer = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                ret, frame = cap.read()
                if not ret:
                    frame_buffer = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    if detection_active and not detection_paused:
                        frame = detect_objects(frame)
                    frame_buffer = frame

def start_streaming():
    t = Thread(target=generate_frames)
    t.daemon = True
    t.start()

# ========== Routes ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def get_status():
    return jsonify({
        "camera_active": camera_active,
        "detection_active": detection_active,
        "detection_paused": detection_paused
    })

@app.route('/camera_on', methods=['POST'])
def camera_on():
    if init_camera():
        start_streaming()
        return jsonify({"success": True})
    return jsonify({"success": False})

@app.route('/camera_off', methods=['POST'])
def camera_off():
    release_camera()
    return jsonify({"success": True})

@app.route('/start', methods=['POST'])
def start_detection():
    global detection_active, detection_paused
    detection_active = True
    detection_paused = False
    return jsonify({"status": "Detection started"})

@app.route('/pause', methods=['POST'])
def pause_detection():
    global detection_paused
    detection_paused = True
    return jsonify({"status": "Detection paused"})

@app.route('/stop', methods=['POST'])
def stop_detection():
    global detection_active, detection_paused
    detection_active = False
    detection_paused = False
    return jsonify({"status": "Detection stopped"})

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if frame_buffer is not None:
                _, jpeg = cv2.imencode('.jpg', frame_buffer)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_image', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return 'No image uploaded', 400
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    result_img = detect_objects(img)
    _, buffer = cv2.imencode('.jpg', result_img)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/detection_progress')
def get_detection_progress():
    session_id = request.args.get('session')
    return jsonify(processing_progress.get(session_id, {
        "total_files": 0,
        "processed_files": 0,
        "current_file_index": -1
    }))

@app.route('/detect_multiple_images', methods=['POST'])
def detect_multiple_images():
    session_id = request.args.get('session')
    files = request.files.getlist('images')
    if not files:
        return 'No files uploaded', 400

    processing_progress[session_id] = {
        "total_files": len(files),
        "processed_files": 0,
        "current_file_index": -1
    }

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for i, file in enumerate(files):
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            result_img = detect_objects(img)
            _, buffer = cv2.imencode('.jpg', result_img)
            zip_file.writestr(f"detected_{file.filename}", buffer.tobytes())

            processing_progress[session_id]["current_file_index"] = i
            processing_progress[session_id]["processed_files"] = i + 1

    processing_progress.pop(session_id, None)

    zip_buffer.seek(0)
    return Response(zip_buffer.getvalue(), mimetype='application/zip',
                    headers={'Content-Disposition': f'attachment; filename=detected_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
