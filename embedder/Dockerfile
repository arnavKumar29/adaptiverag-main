FROM python:3.11-slim

WORKDIR /app

# Install only what the embedder needs
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001
CMD ["python", "main.py"]
