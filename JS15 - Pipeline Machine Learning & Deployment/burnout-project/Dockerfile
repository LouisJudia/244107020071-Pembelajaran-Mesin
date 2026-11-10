# Base Image Python 3.9 Slim
FROM python:3.9-slim

# Setup User Non-Root
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Direktori Kerja
WORKDIR /app

# Install Dependensi
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy Aplikasi & Artefak Model
COPY --chown=user . /app

EXPOSE 7860
CMD ["python", "app.py"]
