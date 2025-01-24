# Use an official Python runtime as a parent image
FROM python:3.12-slim AS builder

# Set the working directory in the container
WORKDIR /myapp

RUN pip install pyarmor

COPY ./app /myapp/app

# Obfuscate the application source files
RUN pyarmor gen -O dist app

FROM python:3.12-slim AS runtime

WORKDIR /myapp

RUN apt-get update && \
    apt-get -qq install -y --no-install-recommends \
    libmagic-dev \
    ffmpeg \
    poppler-utils \
    tesseract-ocr \
    libmagic1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*  # Clean up APT when done to reduce image size

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt /myapp/

# Install dependencies before copying the rest of the code
# This step will be cached as long as requirements.txt doesn't change
RUN pip install --no-cache-dir -r requirements.txt

# Copy obfuscated files from the builder stage
COPY --from=builder /myapp/dist /myapp/

# Copy the modified opik library into site-packages
COPY opik /usr/local/lib/python3.12/site-packages/opik

COPY celery_worker.py /myapp/

ENV NLTK_DATA=/usr/share/nltk_data
RUN mkdir -p $NLTK_DATA && python -m nltk.downloader -d $NLTK_DATA punkt punkt_tab averaged_perceptron_tagger_eng

# Expose port 5000 for the Flask app
EXPOSE 5000

# Run app.py when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]
