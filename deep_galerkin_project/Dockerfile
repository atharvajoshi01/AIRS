FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Install the package
RUN pip install -e .

# Create directories for outputs
RUN mkdir -p checkpoints logs evaluation_results

# Expose ports for API and Streamlit
EXPOSE 8000 8501

# Default command
CMD ["python", "scripts/train.py", "--config", "dgmlib/configs/bs_european.yaml"]
