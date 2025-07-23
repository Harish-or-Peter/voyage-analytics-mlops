# Base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy all project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir flask pandas scikit-learn==1.7.0 joblib

# Expose port
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]
