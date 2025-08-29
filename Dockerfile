# 1) Base image
FROM python:3.11-slim-bookworm

# 2) System deps (we pull in sed so we can strip opik later)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      sed \
      libmagic-dev ffmpeg poppler-utils tesseract-ocr \
      libreoffice \
      libmagic1 libatk1.0-0 libatk-bridge2.0-0 libatspi2.0-0 \
      libxcomposite1 libxdamage1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /myapp

# 3) Copy your requirements
COPY requirements.txt .

# 4) Pre-install pip tools, then:
#    a) install httpx==0.28.1 (satisfies chromadb & biomcp-python)
#    b) install opik w/o deps (so it won’t pull in httpx<0.28.0)
#    c) filter out the opik line and install everything else
RUN pip install --upgrade pip setuptools wheel && \
    pip install httpx==0.28.1 && \
    pip install opik==1.3.5 --no-deps && \
    sed '/^opik==.*$/d' requirements.txt > rest.txt && \
    pip install --no-cache-dir -r rest.txt

# 5) Any additional CLI tools
RUN playwright install-deps && \
    playwright install && \
    python -m spacy download en_core_web_sm && \
    crawl4ai-setup

# 6) NLTK data
ENV NLTK_DATA=/usr/share/nltk_data
RUN mkdir -p $NLTK_DATA && \
    python -m nltk.downloader -d $NLTK_DATA punkt averaged_perceptron_tagger

# 7) Copy your app
COPY ./app /myapp/app
COPY celery_worker.py /myapp/

# 8) (Optional) If you have a custom opik fork, you could COPY it here instead of --no-deps:
#    COPY vendor/opik /usr/local/lib/python3.11/site-packages/opik

EXPOSE 5000
CMD ["flask", "run", "--host=0.0.0.0", "--with-threads"]
