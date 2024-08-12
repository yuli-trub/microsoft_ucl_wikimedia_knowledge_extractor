# Use an official Python runtime as the base image
FROM python:3.11.9

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for Cairo
RUN apt-get update && apt-get install -y \
    libcairo2-dev \
    pkg-config \
    libgirepository1.0-dev \
    build-essential \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

