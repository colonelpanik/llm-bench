# Here is the entire file
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by optional libraries
# (e.g., build-essential for some packages, although maybe not strictly needed here)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#    build-essential \
#    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install core and *all* optional dependencies from requirements.txt
# This makes the image self-contained for all features.
# Ensure requirements.txt lists all needed packages (requests, PyYAML, psutil, pynvml, matplotlib, sentence-transformers)
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir psutil pynvml matplotlib sentence-transformers

# Copy the rest of the application code into the container
COPY . .

# Make port 11434 available to the world outside this container (optional, for info)
# The benchmark tool itself doesn't listen, but Ollama does.
# EXPOSE 11434

# Define the entrypoint to run the application
ENTRYPOINT ["python", "-m", "benchmark_cli"]

# Default command (e.g., show help)
CMD ["--help"]