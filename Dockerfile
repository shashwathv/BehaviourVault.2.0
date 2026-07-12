FROM python:3.11-slim

WORKDIR /app

# PYTHONUNBUFFERED ensures rich dashboard output streams live to docker logs
# instead of being buffered. Important for real-time monitoring.
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# uvicorn replaces gunicorn — required for FastAPI (ASGI, not WSGI).
# --log-level warning suppresses uvicorn's per-request logs since our
# rich dashboard logs everything we want to see.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:5000/health || exit 1

CMD ["uvicorn", "api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--log-level", "warning"]