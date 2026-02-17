FROM python:3.13-slim
LABEL "language"="python"

WORKDIR /app

# Install system dependencies for numpy/pandas compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Copy application code
COPY . .

# Entrypoint seeds references.json and creates directories at runtime
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]

# Environment variables
ENV DATA_ROOT=/data
ENV TZ=Asia/Taipei
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

# Default: run bot + scheduler + web dashboard
CMD ["python", "main.py", "serve", "--components", "all"]
