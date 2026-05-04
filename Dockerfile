FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
		PYTHONDONTWRITEBYTECODE=1 \
		PIP_NO_CACHE_DIR=1 \
		PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Create a non-root user for production safety
RUN useradd -m -u 10001 appuser

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and other required directories
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser data/ /app/data/
COPY --chown=appuser:appuser models/ /app/models/

# Expose API port
EXPOSE 8000

# Set python path
ENV PYTHONPATH=/app

# Healthcheck for Railway and container orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
	CMD python -c "import os,urllib.request; url='http://127.0.0.1:{port}/health'.format(port=os.getenv('PORT','8000')); urllib.request.urlopen(url, timeout=2).read()"

USER appuser

# Command to run the application
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
