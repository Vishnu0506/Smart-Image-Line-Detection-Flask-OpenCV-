# server.py
import os
import uuid
import cv2 as cv
import numpy as np
import math
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

# --- FIXED: Use absolute path ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print(f"📁 Upload folder set to: {UPLOAD_FOLDER}")

def find_lines(img):
    blurred = cv.GaussianBlur(img, (5, 5), 5)
    edges = cv.Canny(blurred, 50, 200)
    min_len = img.shape[1] * 0.1
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                           minLineLength=min_len, maxLineGap=10)
    return lines, edges

def line_length(line):
    x1, y1, x2, y2 = line
    return math.hypot(x2 - x1, y2 - y1)

def line_angle(line):
    x1, y1, x2, y2 = line
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

def filter_lines_by_angle(lines, tol=30):
    if lines is None or len(lines) < 2:
        return lines
    angles = [line_angle(l[0]) for l in lines]
    dominant = np.median(angles)
    filtered = [l[0] for i, l in enumerate(lines) if abs(angles[i] - dominant) < tol]
    return filtered if len(filtered) >= 2 else [l[0] for l in lines]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'imageFile' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['imageFile']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.png', '.jpg', '.jpeg']:
        return jsonify({'error': 'Invalid image format. Use PNG or JPG.'}), 400

    unique_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, f"input_{unique_id}{ext}")
    file.save(input_path)

    img_gray = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
    img_color = cv.imread(input_path)
    if img_gray is None or img_color is None:
        return jsonify({'error': 'Could not read image. Invalid file?'}), 400

    lines, edges = find_lines(img_gray)
    lines = filter_lines_by_angle(lines)

    if lines is None or len(lines) < 2:
        h, w = img_color.shape[:2]
        lines = [[0, 0, w-1, h-1], [0, h-1, w-1, 0]]

    lines.sort(key=line_length, reverse=True)
    longest, second = lines[:2]

    cv.line(img_color, (longest[0], longest[1]), (longest[2], longest[3]), (0, 255, 0), 3)
    cv.line(img_color, (second[0], second[1]), (second[2], second[3]), (0, 255, 0), 3)

    overlay = img_color.copy()
    cv.rectangle(overlay, (0, 0), (img_color.shape[1], img_color.shape[0]), (0, 255, 0), 10)
    cv.addWeighted(overlay, 0.05, img_color, 0.95, 0, img_color)

    output_name = f'houghlines_{unique_id}.png'
    edges_name = f'edges_{unique_id}.png'

    output_path = os.path.join(UPLOAD_FOLDER, output_name)
    edges_path = os.path.join(UPLOAD_FOLDER, edges_name)

    cv.imwrite(output_path, img_color)
    cv.imwrite(edges_path, edges)

    # DEBUG: Print actual paths and existence
    print(f"✅ Output saved at: {output_path} → Exists? {os.path.exists(output_path)}")
    print(f"✅ Edges saved at: {edges_path} → Exists? {os.path.exists(edges_path)}")

    return jsonify({
        'output_image': f'/static/uploads/{output_name}',
        'edges_image': f'/static/uploads/{edges_name}'
    })

# Optional: Serve static files explicitly (redundant but safe)
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)