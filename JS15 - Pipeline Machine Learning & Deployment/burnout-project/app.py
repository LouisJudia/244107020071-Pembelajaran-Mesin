import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load Model dan Preprocessor
MODEL_PATH = 'burnout_model.h5'
PREPROCESSOR_PATH = 'burnout_preprocessor.pkl'

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(PREPROCESSOR_PATH, 'rb') as f:
        artifacts = pickle.load(f)
    preprocessor = artifacts['preprocessor']
    num_cols = artifacts['feature_names_num']
    cat_cols = artifacts['feature_names_cat']
    risk_inv = artifacts['risk_map_inv']
    print("✅ Burnout Prediction System Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading system: {e}")

@app.route('/', methods=['GET'])
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <title>Employee Burnout Assessment AI</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #eef2f7; margin: 0; padding: 30px; }
            .container { max-width: 750px; background: white; margin: auto; padding: 35px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
            h1 { color: #1e3a8a; text-align: center; margin-bottom: 8px; }
            p.sub { text-align: center; color: #64748b; margin-bottom: 25px; }
            .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
            .form-group { display: flex; flex-direction: column; }
            label { font-size: 13px; font-weight: 600; color: #334155; margin-bottom: 5px; }
            input, select { padding: 9px 12px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 14px; }
            button { grid-column: span 2; background: #2563eb; color: white; padding: 13px; font-size: 16px; font-weight: bold; border: none; border-radius: 6px; cursor: pointer; margin-top: 15px; transition: 0.3s; }
            button:hover { background: #1d4ed8; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Employee Burnout Risk Assessment</h1>
            <p class="sub">Prediksi risiko burnout karyawan berbasis Artificial Neural Network (Deep Learning Pipeline)</p>
            <form action="/predict" method="post">
                <div class="form-grid">
                    <div class="form-group"><label>Usia (Tahun):</label><input type="number" name="Age" value="30" required></div>
                    <div class="form-group"><label>Jenis Kelamin:</label>
                        <select name="Gender"><option value="Male">Male</option><option value="Female">Female</option></select>
                    </div>
                    <div class="form-group"><label>Negara Domisili:</label><input type="text" name="Country" value="Indonesia" required></div>
                    <div class="form-group"><label>Pekerjaan (Occupation):</label><input type="text" name="Occupation" value="Software Engineer" required></div>
                    <div class="form-group"><label>Tingkat Pendidikan:</label>
                        <select name="Education_Level"><option value="Bachelor">Bachelor</option><option value="Master">Master</option><option value="PhD">PhD</option><option value="High School">High School</option></select>
                    </div>
                    <div class="form-group"><label>Status Pekerjaan:</label>
                        <select name="Employment_Status"><option value="Full-time">Full-time</option><option value="Part-time">Part-time</option><option value="Freelance">Freelance</option></select>
                    </div>
                    <div class="form-group"><label>Pendapatan Bulanan (USD):</label><input type="number" name="Monthly_Income_USD" value="3500" required></div>
                    <div class="form-group"><label>Jam Kerja Per Minggu:</label><input type="number" name="Work_Hours_Per_Week" value="48" required></div>
                    <div class="form-group"><label>Sistem Kerja (Remote):</label>
                        <select name="Remote_Work"><option value="Hybrid">Hybrid</option><option value="Yes">Remote</option><option value="No">On-site</option></select>
                    </div>
                    <div class="form-group"><label>Kepuasan Kerja (1-10):</label><input type="number" step="0.1" name="Job_Satisfaction" value="5.5" required></div>
                    <div class="form-group"><label>Work-Life Balance (1-10):</label><input type="number" step="0.1" name="Work_Life_Balance" value="5.0" required></div>
                    <div class="form-group"><label>Durasi Tidur (Jam/Hari):</label><input type="number" step="0.1" name="Sleep_Hours" value="6.0" required></div>
                    <div class="form-group"><label>Kualitas Tidur:</label>
                        <select name="Sleep_Quality"><option value="Fair">Fair</option><option value="Good">Good</option><option value="Poor">Poor</option></select>
                    </div>
                    <div class="form-group"><label>Tingkat Stres:</label>
                        <select name="Stress_Level"><option value="Medium">Medium</option><option value="Low">Low</option><option value="High">High</option></select>
                    </div>
                    <div class="form-group"><label>Skor Kecemasan (1-10):</label><input type="number" step="0.1" name="Anxiety_Score" value="6.0" required></div>
                    <div class="form-group"><label>Skor Depresi (1-10):</label><input type="number" step="0.1" name="Depression_Score" value="5.0" required></div>
                    <div class="form-group"><label>Skor Mood (1-10):</label><input type="number" step="0.1" name="Mood_Score" value="6.0" required></div>
                    <div class="form-group"><label>Stabilitas Emosional (1-10):</label><input type="number" step="0.1" name="Emotional_Stability" value="5.5" required></div>
                    <div class="form-group"><label>Aktivitas Fisik (Jam/Minggu):</label><input type="number" step="0.1" name="Physical_Activity_Hours" value="2.5" required></div>
                    <div class="form-group"><label>Frekuensi Olahraga:</label>
                        <select name="Exercise_Frequency"><option value="Weekly">Weekly</option><option value="Daily">Daily</option><option value="Rarely">Rarely</option></select>
                    </div>
                    <div class="form-group"><label>Waktu Meditasi (Menit/Hari):</label><input type="number" name="Meditation_Minutes" value="10" required></div>
                    <div class="form-group"><label>Screen Time (Jam/Hari):</label><input type="number" step="0.1" name="Screen_Time_Hours" value="8.0" required></div>
                    <div class="form-group"><label>Media Sosial (Jam/Hari):</label><input type="number" step="0.1" name="Social_Media_Hours" value="3.0" required></div>
                    <div class="form-group"><label>Gaming (Jam/Hari):</label><input type="number" step="0.1" name="Gaming_Hours" value="1.0" required></div>
                    <div class="form-group"><label>Konsumsi Kopi (Cangkir/Hari):</label><input type="number" name="Coffee_Cups_Per_Day" value="2" required></div>
                    <div class="form-group"><label>Konsumsi Alkohol:</label>
                        <select name="Alcohol_Consumption"><option value="None">None</option><option value="Moderate">Moderate</option><option value="High">High</option></select>
                    </div>
                    <div class="form-group"><label>Merokok:</label>
                        <select name="Smoking"><option value="No">No</option><option value="Yes">Yes</option></select>
                    </div>
                    <div class="form-group"><label>Pola Makan Sehat:</label>
                        <select name="Healthy_Diet"><option value="Moderate">Moderate</option><option value="Yes">Yes</option><option value="No">No</option></select>
                    </div>
                    <div class="form-group"><label>Stres Kronis:</label>
                        <select name="Chronic_Stress"><option value="No">No</option><option value="Yes">Yes</option></select>
                    </div>
                    <div class="form-group"><label>Riwayat Gangguan Mental Keluarga:</label>
                        <select name="Family_History_Mental_Illness"><option value="No">No</option><option value="Yes">Yes</option></select>
                    </div>
                    <div class="form-group"><label>Mengikuti Terapi Psikologis:</label>
                        <select name="Therapy_Attendance"><option value="No">No</option><option value="Yes">Yes</option></select>
                    </div>
                    <div class="form-group"><label>Sistem Dukungan (Support System):</label>
                        <select name="Support_System"><option value="Moderate">Moderate</option><option value="Strong">Strong</option><option value="Weak">Weak</option></select>
                    </div>
                    <div class="form-group"><label>Kepuasan Hidup (1-10):</label><input type="number" step="0.1" name="Life_Satisfaction" value="6.0" required></div>
                    <div class="form-group"><label>Skor Produktivitas (1-10):</label><input type="number" step="0.1" name="Productivity_Score" value="7.0" required></div>
                    <div class="form-group"><label>Hari Absen Kerja (Hari/Tahun):</label><input type="number" name="Absenteeism_Days" value="3" required></div>
                    <div class="form-group"><label>Status Kesehatan Mental:</label>
                        <select name="Mental_Health_Status"><option value="Stable">Stable</option><option value="Stressed">Stressed</option><option value="Depressed">Depressed</option></select>
                    </div>
                    <button type="submit">Analisis Tingkat Risiko Burnout</button>
                </div>
            </form>
        </div>
    </body>
    </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Kumpulkan input formulir ke dalam DataFrame
        input_data = {}
        for col in num_cols:
            val = request.form.get(col, 0)
            input_data[col] = [float(val) if val else 0.0]
        for col in cat_cols:
            val = request.form.get(col, '')
            input_data[col] = [str(val)]
        
        df_input = pd.DataFrame(input_data)
        
        # Transformasi dengan pipeline preprocessor
        X_trans = preprocessor.transform(df_input)
        
        # Prediksi dengan ANN Classifier
        probs = model.predict(X_trans, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = risk_inv.get(pred_idx, 'Unknown')
        confidence = probs[pred_idx] * 100
        
        # Rekomendasi berdasarkan kelas risiko
        if pred_idx == 0:
            color = "#16a34a"
            recom = "Kondisi kerja Anda sehat dan seimbang. Pertahankan rutinitas istirahat dan aktivitas fisik teratur."
        elif pred_idx == 1:
            color = "#d97706"
            recom = "Terdeteksi gejala awal kelelahan kerja. Kurangi jam lembur, perbaiki kualitas tidur, dan lakukan meditasi singkat."
        else:
            color = "#dc2626"
            recom = "Tingkat risiko burnout tinggi! Segera konsultasikan beban kerja dengan atasan dan manfaatkan layanan konseling/kesehatan mental."

        return render_template_string(f'''
        <!DOCTYPE html>
        <html lang="id">
        <head>
            <title>Hasil Analisis Risiko Burnout</title>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; background: #eef2f7; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; padding: 20px; }}
                .card {{ background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); max-width: 520px; width: 100%; text-align: center; }}
                .badge {{ font-size: 26px; font-weight: bold; color: {color}; padding: 12px; border: 2px solid {color}; border-radius: 8px; margin: 20px 0; background: #f8fafc; }}
                p.recom {{ background: #f1f5f9; padding: 15px; border-radius: 6px; text-align: left; color: #334155; line-height: 1.5; }}
                a {{ text-decoration: none; color: white; background: #2563eb; padding: 11px 22px; border-radius: 6px; display: inline-block; margin-top: 20px; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="card">
                <h2>Hasil Penilaian Risiko Burnout</h2>
                <div class="badge">Risiko: {pred_label}</div>
                <p>Tingkat Keyakinan Model: <b>{confidence:.2f}%</b></p>
                <div style="display:flex; justify-content:space-around; margin:15px 0; font-size:13px; color:#64748b;">
                    <span>Low: {probs[0]*100:.1f}%</span>
                    <span>Moderate: {probs[1]*100:.1f}%</span>
                    <span>High: {probs[2]*100:.1f}%</span>
                </div>
                <h3>Rekomendasi AI Wellness:</h3>
                <p class="recom">{recom}</p>
                <a href="/">Evaluasi Karyawan Lain</a>
            </div>
        </body>
        </html>
        ''')
    except Exception as e:
        return f"Terjadi kesalahan saat inferensi: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
