# Dockerfile
# ===========
# Packages the FastAPI server into a container image.
#
# WHAT IS DOCKER? (Kid-friendly explanation)
# Think of Docker like a lunchbox.
# Your app (food) + everything it needs (utensils, napkin) goes inside.
# Anyone can open the same lunchbox and get the EXACT same experience —
# doesn't matter if they're on Mac, Windows, Linux, or a cloud server.
#
# Without Docker: "It works on my machine!" (nightmare)
# With Docker: "It works in the container!" (consistent everywhere)
#
# HOW TO BUILD AND RUN:
#   docker build -t fraud-api .
#   docker run -p 8000:8000 -v $(pwd)/models:/app/models fraud-api
#
# THEN VISIT:
#   http://localhost:8000/docs   → Swagger UI
#   http://localhost:8000/health → health check

# ── Base image ────────────────────────────────────────────
# python:3.10-slim = Python 3.10 on Debian with minimal extras
# WHY SLIM? Full python:3.10 is ~900MB. Slim is ~120MB.
# We don't need compilers, dev tools, etc. in production.
FROM python:3.10-slim

# ── Set working directory inside the container ────────────
WORKDIR /app

# ── System dependencies ───────────────────────────────────
# libgomp1 = OpenMP library, required by XGBoost for parallel tree building
# curl     = used by Docker healthcheck to probe /health endpoint
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*
    # ↑ Always clean apt cache — keeps image size small

# ── Install Python dependencies ───────────────────────────
# WHY COPY requirements.txt FIRST (before copying app code)?
# Docker caches layers. If requirements.txt hasn't changed,
# Docker skips the pip install step entirely → much faster rebuilds.
# This is a key Docker best practice.
COPY requirements.txt .

# Install only what the API needs (skip Airflow, Jupyter, etc.)
# We use a slimmer API-specific requirements here
RUN pip install --no-cache-dir \
    xgboost==2.0.3 \
    scikit-learn==1.4.2 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    joblib==1.4.2 \
    shap==0.45.0 \
    fastapi==0.111.0 \
    uvicorn[standard]==0.30.1 \
    pydantic==2.7.1 \
    prometheus-client==0.20.0 \
    python-dotenv==1.0.1

# ── Copy application code ─────────────────────────────────
# We copy AFTER pip install so code changes don't invalidate
# the expensive pip cache layer
COPY src/ ./src/
COPY models/ ./models/

# Create data directory (needed for drift detection in Week 4)
RUN mkdir -p data/raw data/processed

# ── Non-root user (security best practice) ───────────────
# Never run containers as root in production.
# If someone hacks in, they get "appuser" not "root".
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# ── Port ──────────────────────────────────────────────────
# EXPOSE documents which port the app listens on
# It doesn't actually open the port — that's done by docker run -p
EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────
# Docker will call this every 30s.
# If it fails 3 times in a row, Docker marks container as "unhealthy"
# and Kubernetes/Render will restart it.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Startup command ───────────────────────────────────────
# uvicorn = ASGI server that runs FastAPI
# --workers 2 = 2 processes (good for free-tier CPU)
# --host 0.0.0.0 = listen on all interfaces (needed inside Docker)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
