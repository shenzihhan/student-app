FROM python:3.10-slim

# Install only what you need for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Use a single pip install to reduce layer size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "app.py", "--server.port=10000", "--server.enableCORS=false"]
