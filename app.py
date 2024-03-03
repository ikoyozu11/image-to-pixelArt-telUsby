from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
from pixel_art import convert_to_pixel_art
import base64
from numba import njit
from io import BytesIO

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@njit(fastmath=True)
def accelerate_conversion(image, width, height, color_coeff, step):
    array_of_values = []
    for x in range(0, width, step):
        for y in range(0, height, step):
            r, g, b = image[x, y] // color_coeff
            if r + g + b:
                array_of_values.append(((r, g, b), (x, y)))
    return array_of_values

def process_image(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    pixel_art_img = convert_to_pixel_art(img)

    _, pixel_art_buffer = cv2.imencode('.jpg', pixel_art_img)
    pixel_art_base64 = base64.b64encode(pixel_art_buffer).decode('utf-8')

    _, original_buffer = cv2.imencode('.jpg', img)
    original_base64 = base64.b64encode(original_buffer).decode('utf-8')

    return original_base64, pixel_art_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error='No file provided')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No file selected')

    if file and allowed_file(file.filename):
        original_image, pixel_art_image = process_image(file)
        return render_template('index.html', original_image=original_image, pixel_art_image=pixel_art_image)

    return render_template('index.html', error='Format tidak sesuai. Upload gambar lain!')

@app.route('/download')
def download():
    pixel_art_base64 = request.args.get('pixel_art_base64', None)
    if pixel_art_base64:
        pixel_art_data = base64.b64decode(pixel_art_base64)
        return send_file(BytesIO(pixel_art_data), mimetype='image/jpg', as_attachment=True, download_name='pixel_art.jpg')
    else:
        return "Error: No pixel art data provided for download."

if __name__ == '__main__':
    app.run(debug=True)
