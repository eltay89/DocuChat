# DocuChat v2.0 - Multi-stage Docker build
# Optimized for production deployment with security best practices

# Builder stage for dependencies
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ANONYMIZED_TELEMETRY=False \
    LLAMA_CPP_LOG_LEVEL=2 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    liblapack3 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r docuchat && useradd -r -g docuchat -s /bin/bash docuchat

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application directories
RUN mkdir -p /app/documents /app/models /app/vector_store /app/config /app/logs && \
    chown -R docuchat:docuchat /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=docuchat:docuchat src/ ./src/
COPY --chown=docuchat:docuchat tools/ ./tools/
COPY --chown=docuchat:docuchat config/ ./config/
COPY --chown=docuchat:docuchat setup.py requirements.txt ./

# Install application in development mode
RUN pip install -e .

# Switch to non-root user
USER docuchat

# Create volume mount points
VOLUME ["/app/documents", "/app/models", "/app/vector_store", "/app/config", "/app/logs"]

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app/src'); from docuchat.core.config import ConfigManager; print('OK')" || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "docuchat.cli.main"]