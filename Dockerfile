FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

# Dependencies first so this layer is cached until requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The model arrives through the build context (CI downloads it into models/).
# models/.gitkeep keeps this COPY working when no model is present.
COPY config.yaml .
COPY models/ ./models/
COPY app/ ./app/

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

# exec replaces the shell so uvicorn receives SIGTERM directly
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
