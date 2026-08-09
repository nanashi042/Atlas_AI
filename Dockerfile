FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN useradd --create-home --uid 10001 atlas \
    && mkdir -p /data \
    && chown -R atlas:atlas /app /data
USER atlas

# Supply secrets and DATABASE_URL at runtime; for SQLite mount /data as a volume.
CMD ["python", "-m", "app.run_bot"]
