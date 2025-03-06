# Use official Python image
FROM python:3.10

# Set work directory
WORKDIR /app

# Install system dependencies (including OpenGL for OpenCV)
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
