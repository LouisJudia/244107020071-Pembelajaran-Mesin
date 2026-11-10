import os
import numpy as np
import cv2
import pickle
import tensorflow as tf
from flask import Flask, request, render_template_string
from skimage.feature import hog

app = Flask(__name__)

# Load Model & Scaler
MODEL_PATH = 'day_night_model.h5'
SCALER_PATH = 'scaler.pkl'

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("✅ System Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading system: {e}")

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(8,8),
                   cells_per_block=(2,2), block_norm='L2-Hys',
                   visualize=False, feature_vector=True)
                   
    return scaler.transform(hog_feat.reshape(1, -1))

@app.route('/', methods=['GET'])
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Day vs Night Classifier</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
            .card { background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); text-align: center; max-width: 450px; width: 100%; }
            h1 { color: #2c3e50; margin-bottom: 25px; font-size: 24px; }
            input[type=file] { margin: 20px 0; padding: 10px; border: 1px dashed #3498db; border-radius: 8px; width: 85%; }
            button { background-color: #3498db; color: white; border: none; padding: 12px 28px; font-size: 16px; border-radius: 6px; cursor: pointer; transition: 0.3s; }
            button:hover { background-color: #2980b9; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Day vs Night Classifier</h1>
            <p>Unggah citra untuk mengklasifikasikan apakah suasana siang hari atau malam hari.</p>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required><br>
                <button type="submit">Prediksi Citra</button>
            </form>
        </div>
    </body>
    </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        data = preprocess_image(file.read())
        prediction = model.predict(data, verbose=0)[0][0]
        
        label = "Day (Siang Hari)" if prediction > 0.5 else "Night (Malam Hari)"
        prob_percent = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
        icon = "☀️" if prediction > 0.5 else "🌙"
        
        return render_template_string(f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hasil Prediksi</title>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; background-color: #f4f6f9; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }}
                .card {{ background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); text-align: center; max-width: 450px; width: 100%; }}
                h2 {{ color: #2c3e50; }}
                .badge {{ font-size: 40px; margin: 15px 0; }}
                a {{ text-decoration: none; color: white; background: #3498db; padding: 10px 20px; border-radius: 6px; display: inline-block; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="card">
                <div class="badge">{icon}</div>
                <h2>Hasil Prediksi: {label}</h2>
                <p>Tingkat Keyakinan: <b>{prob_percent:.2f}%</b></p>
                <a href="/">Unggah Citra Lain</a>
            </div>
        </body>
        </html>
        ''')
    except Exception as e:
        return f"Terjadi kesalahan saat memproses citra: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
