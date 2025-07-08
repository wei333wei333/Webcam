from flask import Flask, render_template, request, send_file, jsonify 
import torch
import torch.serialization

# ✅ 添加这一段，允许 YOLO 模型结构被反序列化
torch.serialization.add_safe_globals({
    'ultralytics.nn.tasks.DetectionModel': __import__('ultralytics').nn.tasks.DetectionModel
})

from ultralytics import YOLO
import cv2
import os
import uuid
import shutil
import zipfile
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# ✅ 你的模型路径
model = YOLO("best1.pt")

os.makedirs("runs", exist_ok=True)

# 首页
@app.route('/')
def index():
    return render_template('index.html')

# Webcam 推理（返回十六进制图像 + labels）
@app.route('/predict', methods=['POST'])
def predict_webcam():
    file = request.files['image']
    img = Image.open(file.stream).convert("RGB")

    results = model.predict(img)[0]
    result_img = results.plot()
    labels = [model.names[int(cls)] for cls in results.boxes.cls]

    _, img_encoded = cv2.imencode('.jpg', result_img)
    hex_string = img_encoded.tobytes().hex()

    return jsonify({"image": hex_string, "labels": labels})

# 单图检测（返回处理后的图像）
@app.route('/detect_image', methods=['POST'])
def detect_image():
    file = request.files['image']
    img = Image.open(file.stream).convert("RGB")

    results = model.predict(img)[0]
    result_img = results.plot()
    
    _, buffer = cv2.imencode(".jpg", result_img)
    return send_file(BytesIO(buffer.tobytes()), mimetype="image/jpeg")

# 多图检测（返回 ZIP）
@app.route('/detect_multiple_images', methods=['POST'])
def detect_multiple_images():
    files = request.files.getlist('images')
    temp_dir = f"runs/batch_{uuid.uuid4().hex}"
    os.makedirs(temp_dir, exist_ok=True)

    for file in files:
        img = Image.open(file.stream).convert("RGB")
        results = model.predict(img)[0]
        result_img = results.plot()

        save_path = os.path.join(temp_dir, file.filename)
        cv2.imwrite(save_path, result_img)

    # 创建 ZIP 文件
    zip_path = f"{temp_dir}.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for filename in os.listdir(temp_dir):
            zipf.write(os.path.join(temp_dir, filename), arcname=filename)

    shutil.rmtree(temp_dir)  # 清除临时图像

    return send_file(zip_path, mimetype='application/zip', as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
