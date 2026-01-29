FROM python:3.11-slim

# Install system dependencies for EasyOCR
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR model during build (not at runtime)
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=True)"

# Copy app code
COPY . .

RUN mkdir -p /tmp/expense_uploads

EXPOSE 10000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--timeout", "120", "--workers", "1"]
