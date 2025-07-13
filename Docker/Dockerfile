# Use an official Python runtime as a parent image
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code
COPY . .



CMD ["python3", "app.py"]