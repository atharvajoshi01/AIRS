# AIRS Dockerfile - Multi-stage build for different targets

# =============================================================================
# Base Stage: Common dependencies
# =============================================================================
FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY airs/ ./airs/

# Install base dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e .

# =============================================================================
# API Stage: FastAPI service
# =============================================================================
FROM base as api

# Install additional API dependencies if needed
RUN pip install gunicorn

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the API
CMD ["uvicorn", "airs.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Airflow Stage: Airflow workers and schedulers
# =============================================================================
FROM base as airflow

# Install Airflow
RUN pip install apache-airflow[postgres]==2.7.3 \
    apache-airflow-providers-postgres \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.3/constraints-3.11.txt"

# Set Airflow home
ENV AIRFLOW_HOME=/opt/airflow

# Create Airflow directories
RUN mkdir -p ${AIRFLOW_HOME}/dags ${AIRFLOW_HOME}/logs ${AIRFLOW_HOME}/plugins

# Copy DAGs
COPY dags/ ${AIRFLOW_HOME}/dags/

# Initialize Airflow database (will be overridden by entrypoint in production)
RUN airflow db init || true

# Expose Airflow webserver port
EXPOSE 8080

# Default command (can be overridden)
CMD ["airflow", "webserver", "--port", "8080"]

# =============================================================================
# Development Stage: Full dev environment
# =============================================================================
FROM base as dev

# Install all optional dependencies
RUN pip install -e ".[all]"

# Install development tools
RUN pip install ipython jupyter

# Create non-root user
RUN useradd --create-home --shell /bin/bash devuser && \
    chown -R devuser:devuser /app
USER devuser

# Expose ports for API and Jupyter
EXPOSE 8000 8888

# Default to bash
CMD ["/bin/bash"]

# =============================================================================
# Test Stage: For running tests
# =============================================================================
FROM base as test

# Install test dependencies
RUN pip install -e ".[dev]"

# Copy test files
COPY tests/ ./tests/

# Run tests by default
CMD ["pytest", "-v", "--cov=airs", "--cov-report=term-missing"]
