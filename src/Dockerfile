# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && \
    apt-get -qq install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libmagic-dev \
    libreoffice \
    pandoc \
    libgl1 \
    libmagic1 && \
    rm -rf /var/lib/apt/lists/*  # Clean up APT when done to reduce image size

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt /app/

# Install dependencies before copying the rest of the code
# This step will be cached as long as requirements.txt doesn't change
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your app code
COPY . /app

# Expose port 5000 for the Flask app
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "server.py"]
