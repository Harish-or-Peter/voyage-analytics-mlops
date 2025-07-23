# ---- Base image ----
FROM python:3.11-slim

# ---- Metadata (optional but nice) ----
LABEL maintainer="Your Name <you@example.com>" \
      project="Voyage Analytics – Flight Price API"

# ---- Set working dir ----
WORKDIR /app

# ---- Install Python deps first (leverages build‑cache) ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy project files ----
COPY app.py .
COPY flight_price_model.pkl .

# ---- Expose port used by Gunicorn ----
EXPOSE 5000

# ---- Production WSGI server ----
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5000", "--workers", "2", "--threads", "4"]
