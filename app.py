from flask import Flask, render_template_string, Response, jsonify, request
import cv2
from ultralytics import YOLO
import threading
import atexit
import logging
import os
import socket
import numpy as np
import zipfile
import io
from datetime import datetime
import time
import json

# --- Utility ---
def find_available_port(start_port=5000):
    port = start_port
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            port += 1

# --- Setup ---
port = find_available_port()
os.environ['FLASK_ENV'] = 'development'
logging.getLogger('werkzeug').setLevel(logging.ERROR)
app = Flask(_name_)
model = YOLO("best1.pt")  # Load your YOLOv8 model

# --- Global States ---
cap = None
camera_active = False
detection_active = False
detection_paused = False
streaming_active = False
streaming_thread = None
frame_buffer = None
camera_lock = threading.Lock()
processing_progress = {}  # Store progress for each session

# --- Camera Handling ---
def init_camera():
    global cap, camera_active
    with camera_lock:
        if cap is None:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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

atexit.register(release_camera)

# --- Object Detection ---
def detect_objects(frame):
    results = model.predict(source=frame, save=False, conf=0.3)
    return results[0].plot()

# --- Video Stream ---
def generate_frames():
    global frame_buffer, streaming_active
    while streaming_active:
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

def start_streaming_thread():
    global streaming_thread, streaming_active
    if not streaming_active:
        streaming_active = True
        streaming_thread = threading.Thread(target=generate_frames)
        streaming_thread.start()

def stop_streaming_thread():
    global streaming_active, streaming_thread
    streaming_active = False
    if streaming_thread:
        streaming_thread.join()
        streaming_thread = None

# --- Routes ---
@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Drowning Detection System</title>
    <style>
        :root {
            --primary-color: #3498db;
            --secondary-color: #2980b9;
            --danger-color: #e74c3c;
            --warning-color: #f39c12;
            --success-color: #2ecc71;
            --light-color: #ecf0f1;
            --dark-color: #2c3e50;
            --gray-color: #95a5a6;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }
        
        .container {
            display: flex;
            min-height: 100vh;
            padding: 20px;
            gap: 20px;
        }
        
        .camera-section {
            flex: 1;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
            display: flex;
            flex-direction: column;
        }
        
        .upload-section {
            width: 400px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
            display: flex;
            flex-direction: column;
        }
        
        h2 {
            color: var(--dark-color);
            margin-bottom: 20px;
            text-align: center;
            font-weight: 600;
        }
        
        h3 {
            color: var(--dark-color);
            margin-bottom: 15px;
            font-weight: 500;
            font-size: 1.2rem;
        }
        
        .status-box {
            background-color: var(--light-color);
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 20px;
            font-size: 0.9rem;
        }
        
        .status-box strong {
            color: var(--dark-color);
        }
        
        .button {
            padding: 10px 15px;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            margin: 5px 0;
            font-size: 16px;
            font-weight: 500;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }
        
        .button:hover {
            transform: translateY(-2px);
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }
        
        .button:active {
            transform: translateY(0);
        }
        
        .button i {
            margin-right: 8px;
        }
        
        .camera-on { background: var(--primary-color); }
        .camera-off { background: var(--gray-color); }
        .start { background: var(--danger-color); }
        .pause { background: var(--warning-color); }
        .stop { background: var(--gray-color); }
        .download { background: var(--success-color); }
        
        #videoFeed {
            border: 2px solid #ddd;
            border-radius: 8px;
            margin: 10px 0;
            max-width: 100%;
            background: #000;
            flex-grow: 1;
        }
        
        .controls {
            display: flex;
            gap: 10px;
            margin: 15px 0;
        }
        
        .controls .button {
            flex: 1;
        }
        
        .tab {
            overflow: hidden;
            border: 1px solid #ccc;
            background-color: #f1f1f1;
            border-radius: 6px 6px 0 0;
            display: flex;
        }
        
        .tab button {
            background-color: inherit;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 12px 16px;
            transition: 0.3s;
            flex: 1;
            text-align: center;
            font-weight: 500;
        }
        
        .tab button:hover {
            background-color: #ddd;
        }
        
        .tab button.active {
            background-color: var(--primary-color);
            color: white;
        }
        
        .tabcontent {
            display: none;
            padding: 15px;
            border: 1px solid #ccc;
            border-top: none;
            border-radius: 0 0 6px 6px;
            background: white;
        }
        
        .file-list {
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 10px;
            margin: 10px 0;
            background: #f9f9f9;
        }
        
        .file-item {
            padding: 8px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .file-item:last-child {
            border-bottom: none;
        }
        
        .progress-text {
            color: #666;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .current-processing {
            background-color: rgba(52, 152, 219, 0.1);
            border-left: 3px solid var(--primary-color);
        }
        
        progress {
            width: 100%;
            height: 10px;
            border-radius: 5px;
            margin: 10px 0;
            display: none;
        }
        
        progress::-webkit-progress-bar {
            background-color: #f0f0f0;
            border-radius: 5px;
        }
        
        progress::-webkit-progress-value {
            background-color: var(--primary-color);
            border-radius: 5px;
        }
        
        .upload-status {
            margin: 10px 0;
            color: var(--success-color);
            font-size: 0.9rem;
            min-height: 20px;
        }
        
        #uploadedResult, #multiDownloadBtn {
            width: 100%;
            margin-top: 10px;
            display: none;
        }
        
        #uploadedResult {
            border: 2px solid #ddd;
            border-radius: 8px;
            max-height: 300px;
            object-fit: contain;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        input[type="file"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            background: #f9f9f9;
        }
        
        hr {
            border: none;
            border-top: 1px solid #eee;
            margin: 20px 0;
        }
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
</head>
<body>
    <div class="container">
        <!-- Left Section - Camera Feed -->
        <div class="camera-section">
            <h2><i class="fas fa-video"></i> Drowning Detection System</h2>
            
            <div class="status-box">
                <strong>Status:</strong> 
                Camera: <span id="cameraStatus">OFF</span> | 
                Detection: <span id="detectionStatus">OFF</span>
            </div>
            
            <button id="cameraToggle" onclick="toggleCamera()" class="button camera-off">
                <i class="fas fa-power-off"></i> Turn On Camera
            </button>
            
            <img id="videoFeed" src="/video_feed" width="640" height="480">
            
            <div class="controls">
                <button id="toggleDetection" onclick="toggleDetection()" class="button start">
                    <i class="fas fa-play"></i> Start Detection
                </button>
                <button onclick="stopDetection()" class="button stop">
                    <i class="fas fa-stop"></i> Stop Detection
                </button>
            </div>
        </div>
        
        <!-- Right Section - Upload -->
        <div class="upload-section">
            <h2><i class="fas fa-upload"></i> Image Detection</h2>
            
            <div class="tab">
                <button class="tablinks active" onclick="openTab(event, 'singleImageTab')">
                    <i class="fas fa-image"></i> Single
                </button>
                <button class="tablinks" onclick="openTab(event, 'multipleImagesTab')">
                    <i class="fas fa-images"></i> Multiple
                </button>
            </div>
            
            <div id="singleImageTab" class="tabcontent" style="display: block;">
                <h3>Upload Single Image</h3>
                <form id="uploadForm" enctype="multipart/form-data">
                    <div class="form-group">
                        <input type="file" name="image" accept="image/*" required>
                    </div>
                    <button type="submit" class="button start">
                        <i class="fas fa-search"></i> Detect Image
                    </button>
                </form>
                <progress id="progressBar" value="0" max="100"></progress>
                <div id="uploadStatus" class="upload-status"></div>
                <img id="uploadedResult" src="">
                <a id="downloadBtn" class="button download">
                    <i class="fas fa-download"></i> Download Result
                </a>
            </div>
            
            <div id="multipleImagesTab" class="tabcontent">
                <h3>Upload Multiple Images</h3>
                <form id="multiUploadForm" enctype="multipart/form-data">
                    <div class="form-group">
                        <input type="file" name="images" accept="image/*" multiple required>
                    </div>
                    <button type="submit" class="button start">
                        <i class="fas fa-search"></i> Detect All Images
                    </button>
                </form>
                <div id="fileList" class="file-list"></div>
                <progress id="multiProgressBar" value="0" max="100"></progress>
                <div id="multiProgressText" class="progress-text" style="display:none;"></div>
                <div id="multiUploadStatus" class="upload-status"></div>
                <a id="multiDownloadBtn" class="button download">
                    <i class="fas fa-file-archive"></i> Download All (ZIP)
                </a>
            </div>
        </div>
    </div>

<script>
function openTab(evt, tabName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}

function updateStatus() {
    fetch('/status')
        .then(res => res.json())
        .then(data => {
            document.getElementById('cameraStatus').textContent = data.camera_active ? 'ON' : 'OFF';
            document.getElementById('detectionStatus').textContent = data.detection_active ? (data.detection_paused ? 'PAUSED' : 'ACTIVE') : 'OFF';
            const camBtn = document.getElementById('cameraToggle');
            camBtn.innerHTML = data.camera_active ? '<i class="fas fa-power-off"></i> Turn Off Camera' : '<i class="fas fa-power-off"></i> Turn On Camera';
            camBtn.className = data.camera_active ? 'button camera-on' : 'button camera-off';
            const detectBtn = document.getElementById('toggleDetection');
            detectBtn.innerHTML = !data.detection_active || data.detection_paused ? '<i class="fas fa-play"></i> Start Detection' : '<i class="fas fa-pause"></i> Pause Detection';
            detectBtn.className = !data.detection_active || data.detection_paused ? 'button start' : 'button pause';
        });
}

function toggleCamera() {
    const action = document.getElementById('cameraStatus').textContent === 'ON' ? '/camera_off' : '/camera_on';
    fetch(action, { method: 'POST' }).then(() => updateStatus());
}

function toggleDetection() {
    fetch('/status')
        .then(res => res.json())
        .then(data => {
            const url = !data.detection_active || data.detection_paused ? '/start' : '/pause';
            fetch(url, { method: 'POST' }).then(() => updateStatus());
        });
}

function stopDetection() {
    fetch('/stop', { method: 'POST' }).then(() => updateStatus());
}

// Single image upload
document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    const statusDiv = document.getElementById('uploadStatus');
    const progressBar = document.getElementById('progressBar');
    const downloadBtn = document.getElementById('downloadBtn');
    
    // Reset UI
    progressBar.style.display = 'block';
    progressBar.value = 0;
    statusDiv.textContent = 'Uploading and detecting...';
    document.getElementById('uploadedResult').style.display = 'none';
    downloadBtn.style.display = 'none';

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/detect_image', true);

    xhr.upload.onprogress = function(e) {
        if (e.lengthComputable) {
            const percent = (e.loaded / e.total) * 100;
            progressBar.value = percent;
        }
    };

    xhr.onload = function() {
        progressBar.style.display = 'none';
        if (xhr.status === 200) {
            const blob = xhr.response;
            const url = URL.createObjectURL(blob);
            const resultImg = document.getElementById('uploadedResult');
            resultImg.src = url;
            resultImg.style.display = 'block';
            downloadBtn.href = url;
            downloadBtn.style.display = 'block';
            statusDiv.textContent = 'Detection completed successfully!';
        } else {
            statusDiv.textContent = 'Error during detection. Please try again.';
        }
    };

    xhr.onerror = function() {
        progressBar.style.display = 'none';
        statusDiv.textContent = 'Upload error. Please check your connection.';
    };

    xhr.responseType = 'blob';
    xhr.send(formData);
});

// Multiple images upload
document.getElementById('multiUploadForm').addEventListener('change', function(e) {
    const fileList = document.getElementById('fileList');
    fileList.innerHTML = '';
    
    Array.from(e.target.files).forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.id = 'file-' + index;
        fileItem.innerHTML = `
            <span>${index + 1}. ${file.name}</span>
            <span class="progress-text" id="progress-${index}">Pending</span>
        `;
        fileList.appendChild(fileItem);
    });
});

let progressInterval;
let sessionId = null;

document.getElementById('multiUploadForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    const statusDiv = document.getElementById('multiUploadStatus');
    const progressBar = document.getElementById('multiProgressBar');
    const progressText = document.getElementById('multiProgressText');
    const fileList = document.getElementById('fileList');
    const downloadBtn = document.getElementById('multiDownloadBtn');
    
    // Generate a unique session ID
    sessionId = 'session-' + Date.now();
    
    // Reset UI
    progressBar.style.display = 'block';
    progressText.style.display = 'block';
    progressBar.value = 0;
    statusDiv.textContent = 'Starting detection process...';
    downloadBtn.style.display = 'none';
    
    // Clear any previous progress highlights
    const fileItems = fileList.querySelectorAll('.file-item');
    fileItems.forEach(item => {
        item.classList.remove('current-processing');
    });
    
    // Start tracking progress
    progressInterval = setInterval(() => {
        fetch('/detection_progress?session=' + sessionId)
            .then(res => res.json())
            .then(data => {
                if (data.total_files > 0) {
                    const percent = (data.processed_files / data.total_files) * 100;
                    progressBar.value = percent;
                    progressText.textContent = Processing: ${data.processed_files} of ${data.total_files} files (${Math.round(percent)}%);
                    
                    // Update individual file status
                    if (data.current_file_index >= 0) {
                        // Clear previous current file highlight
                        fileItems.forEach(item => {
                            item.classList.remove('current-processing');
                        });
                        
                        // Highlight current file
                        if (data.current_file_index < fileItems.length) {
                            const currentFile = document.getElementById('file-' + data.current_file_index);
                            if (currentFile) {
                                currentFile.classList.add('current-processing');
                                const progressSpan = document.getElementById('progress-' + data.current_file_index);
                                if (progressSpan) {
                                    progressSpan.textContent = 'Processing...';
                                    progressSpan.style.color = '#3498db';
                                }
                            }
                        }
                    }
                    
                    // Update completed files
                    for (let i = 0; i < data.processed_files; i++) {
                        const progressSpan = document.getElementById('progress-' + i);
                        if (progressSpan) {
                            progressSpan.textContent = 'Completed';
                            progressSpan.style.color = 'green';
                        }
                    }
                }
            });
    }, 500);  // Update every 500ms
    
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/detect_multiple_images?session=' + sessionId, true);

    xhr.upload.onprogress = function(e) {
        if (e.lengthComputable) {
            const percent = (e.loaded / e.total) * 100;
            progressText.textContent = Uploading: ${Math.round(percent)}%;
        }
    };

    xhr.onload = function() {
        clearInterval(progressInterval);
        progressBar.style.display = 'none';
        if (xhr.status === 200) {
            const blob = xhr.response;
            const url = URL.createObjectURL(blob);
            downloadBtn.href = url;
            downloadBtn.style.display = 'block';
            downloadBtn.download = 'detected_images_' + new Date().toISOString().slice(0, 10) + '.zip';
            statusDiv.textContent = 'Detection completed for all images!';
            progressText.textContent = 'All files processed successfully!';
            
            // Mark all files as completed
            const fileItems = fileList.querySelectorAll('.file-item');
            fileItems.forEach((item, index) => {
                const progressSpan = document.getElementById('progress-' + index);
                if (progressSpan) {
                    progressSpan.textContent = 'Completed';
                    progressSpan.style.color = 'green';
                }
                item.classList.remove('current-processing');
            });
        } else {
            statusDiv.textContent = 'Error during detection. Please try again.';
            progressText.textContent = 'Processing failed';
        }
    };

    xhr.onerror = function() {
        clearInterval(progressInterval);
        progressBar.style.display = 'none';
        statusDiv.textContent = 'Upload error. Please check your connection.';
        progressText.textContent = 'Processing failed';
    };

    xhr.responseType = 'blob';
    xhr.send(formData);
});

updateStatus();
setInterval(updateStatus, 1000);
</script>
</body>
</html>
''')

@app.route('/status')
def get_status():
    return jsonify({
        "camera_active": camera_active,
        "detection_active": detection_active,
        "detection_paused": detection_paused
    })

@app.route('/camera_on', methods=['POST'])
def camera_on():
    try:
        success = init_camera()
        if success:
            start_streaming_thread()
        return jsonify({ "success": success })
    except Exception as e:
        return jsonify({ "success": False, "message": str(e) }), 500

@app.route('/camera_off', methods=['POST'])
def camera_off():
    stop_streaming_thread()
    release_camera()
    return jsonify({ "success": True })

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if frame_buffer is not None:
                _, jpeg = cv2.imencode('.jpg', frame_buffer)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            else:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                _, jpeg = cv2.imencode('.jpg', blank)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

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

@app.route('/detect_image', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return 'No image uploaded', 400
    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    result_img = detect_objects(img)
    _, buffer = cv2.imencode('.jpg', result_img)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/detection_progress')
def get_detection_progress():
    session_id = request.args.get('session')
    if session_id in processing_progress:
        return jsonify(processing_progress[session_id])
    return jsonify({"total_files": 0, "processed_files": 0, "current_file_index": -1})

@app.route('/detect_multiple_images', methods=['POST'])
def detect_multiple_images():
    session_id = request.args.get('session')
    if 'images' not in request.files:
        return 'No images uploaded', 400
    
    files = request.files.getlist('images')
    if not files or files[0].filename == '':
        return 'No selected files', 400
    
    # Initialize progress tracking
    processing_progress[session_id] = {
        "total_files": len(files),
        "processed_files": 0,
        "current_file_index": -1
    }
    
    # Create a zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, file in enumerate(files):
            if file.filename == '':
                continue
                
            # Update progress
            processing_progress[session_id]["current_file_index"] = i
            processing_progress[session_id]["processed_files"] = i
            
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            result_img = detect_objects(img)
            
            # Convert image to bytes
            _, buffer = cv2.imencode('.jpg', result_img)
            img_bytes = buffer.tobytes()
            
            # Save to zip with original filename (prepend "detected_")
            original_filename = file.filename
            detected_filename = f"detected_{original_filename}"
            zip_file.writestr(detected_filename, img_bytes)
            
            # Reset file pointer for safety
            file.seek(0)
            
            # Small delay to simulate processing (remove in production)
            time.sleep(0.5)
    
    # Final progress update
    processing_progress[session_id]["processed_files"] = len(files)
    processing_progress[session_id]["current_file_index"] = -1
    
    # Clean up progress tracking
    del processing_progress[session_id]
    
    # Prepare the response
    zip_buffer.seek(0)
    return Response(
        zip_buffer.getvalue(),
        mimetype='application/zip',
        headers={
            'Content-Disposition': f'attachment; filename=detected_images_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        }
    )

# --- Start Server Thread ---
def run_flask():
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

threading.Thread(target=run_flask, daemon=True).start()

from IPython.display import display, HTML
display(HTML(f'''
<div style="text-align:center; padding:20px;">
    <h3>Detection System Ready</h3>
    <a href="http://127.0.0.1:{port}" target="_blank" 
       style="padding:10px 20px; background:#3498db; color:white; text-decoration:none; border-radius:4px; font-size:16px;">
       Open Camera Interface
    </a>
</div>
'''))

