FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and other required directories
COPY src/ /app/src/
COPY data/ /app/data/
COPY models/ /app/models/

# Expose API port
EXPOSE 8000

# Set python path
ENV PYTHONPATH=/app

# Command to run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
