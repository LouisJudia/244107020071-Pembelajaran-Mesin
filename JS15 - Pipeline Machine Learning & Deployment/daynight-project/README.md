---
title: Day vs Night Classifier
emoji: ☀️🌙
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Day vs Night Classifier (MLOps Deployment)

Aplikasi web klasifikasi citra Siang vs Malam menggunakan ekstraksi fitur HOG (Histogram of Oriented Gradients) dan Artificial Neural Network (TensorFlow/Keras).

## Cara Menjalankan Secara Lokal (Menggunakan Docker)
```bash
docker build -t daynight-classifier .
docker run -p 7860:7860 daynight-classifier
```
Buka browser di `http://localhost:7860`.
