FROM python:3.11.9

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libcairo2-dev \
    pkg-config \
    libgirepository1.0-dev \
    build-essential \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Command to run your application
CMD ["python", "main.py"]