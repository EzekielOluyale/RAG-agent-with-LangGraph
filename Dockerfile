# 1. Use a highly optimized lightweight base Python image
FROM python:3.11-slim

# 2. Prevent Python from writing cache files and force instant real-time logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Establish our working directory inside the container
WORKDIR /app

# 4. Install basic system tool build-requirements and curl for health checking
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy and install dependencies first (Leverages Docker caching layers to build faster)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 6. Create a non-privileged system user for maximum enterprise security
RUN groupadd -r appuser && useradd -r -g appuser -s /bin/false appuser

# 7. Copy your code into the container
COPY main.py .
COPY src/ ./src/

# 8. Change ownership of the app files to the secure user
RUN chown -R appuser:appuser /app

# 9. Switch context to the secure low-privilege user
USER appuser

# 10. Open up port 8000 for network communication
EXPOSE 8000

# 11. Healthcheck probe hitting your app's docs path every 30 seconds
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/docs || exit 1

# 12. Run multi-worker production configuration without --reload
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]