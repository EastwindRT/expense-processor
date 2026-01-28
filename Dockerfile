FROM python:3.11-slim

# Install Tesseract OCR with language data
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/* \
    && tesseract --version

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create uploads directory
RUN mkdir -p /tmp/expense_uploads

# Expose port
EXPOSE 10000

# Health check
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# Run the app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--timeout", "120"]
