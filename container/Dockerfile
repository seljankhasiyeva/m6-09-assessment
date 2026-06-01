# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System deps (OpenCV headless needs libgl) ──────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
WORKDIR /app
COPY container/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ── App code ──────────────────────────────────────────────────────────────────
COPY container/app /app/app
COPY container/models /app/models
COPY container/STUDENT.json /app/STUDENT.json

# ── Sanity check: confirm model file is present at build time ─────────────────
RUN test -f /app/models/best.onnx || \
    (echo "ERROR: best.onnx not found in container/models/" && exit 1)

# ── Entrypoint ────────────────────────────────────────────────────────────────
ENTRYPOINT ["python", "/app/app/cli.py"]