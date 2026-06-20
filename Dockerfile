# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the worker script
COPY worker.py .

# Command to run the FastAPI server
CMD ["uvicorn", "worker:app", "--host", "0.0.0.0", "--port", "8000"]