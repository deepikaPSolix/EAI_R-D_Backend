FROM python:3.11-slim AS base

# System packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libmagic-dev ffmpeg poppler-utils tesseract-ocr \
      libreoffice \
      libmagic1 libatk1.0-0 libatk-bridge2.0-0 libatspi2.0-0 \
      libxcomposite1 libxdamage1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /myapp

# Copy only requirements.txt first
COPY requirements.txt /myapp/

RUN pip install --upgrade pip setuptools wheel
# Install requirementsa
RUN pip install --no-cache-dir -r requirements.txt

# Install other tooling that you need for final runtime
RUN playwright install-deps
RUN playwright install
RUN python -m spacy download en_core_web_sm

#Builder docker

FROM python:3.11-slim AS pyarmor_builder
RUN pip install pyarmor

WORKDIR /myapp

# Copy your app source
COPY ./app /myapp/app

# Obfuscate
RUN pyarmor gen -O dist app

#Runtime dockr
FROM base AS runtime

WORKDIR /myapp

# Copy the obfuscated code from builder
COPY --from=pyarmor_builder /myapp/dist /myapp/

# If you have extra custom code (like opik library) or config:
COPY opik /usr/local/lib/python3.11/site-packages/opik
COPY celery_worker.py /myapp/

ENV NLTK_DATA=/usr/share/nltk_data
RUN mkdir -p $NLTK_DATA && \
    python -m nltk.downloader -d $NLTK_DATA punkt punkt_tab averaged_perceptron_tagger_eng

EXPOSE 5000
CMD ["flask", "run", "--host=0.0.0.0", "--with-threads"]