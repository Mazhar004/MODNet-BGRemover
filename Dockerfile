# CPU image. The CUDA variant lives in Dockerfile.cuda.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODNET_DATA_DIR=/data \
    MODNET_WEIGHTS_DIR=/weights

# curl is for HEALTHCHECK; ffmpeg comes from the imageio-ffmpeg wheel.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The CPU torch wheel is roughly a quarter the size of the CUDA one. Install it
# first so the slow layer is cached independently of the application code.
# The python base image ships setuptools 78.1.0, which carries a
# PackageIndex path traversal (PYSEC-2025-49) and a MANIFEST.in exclusion
# bypass (PYSEC-2026-3447). Nothing here needs setuptools at runtime, but a
# vulnerable copy should not sit in the shipped image.
RUN pip install --upgrade pip "setuptools>=83"

COPY pyproject.toml ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.14.0 torchvision==0.29.0

COPY modnet_bg ./modnet_bg
RUN pip install .

RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /data /weights \
 && chown -R appuser:appuser /data
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# One worker, many threads. The job registry lives in process memory, so a
# second worker would not see the first worker's jobs and status polls would
# intermittently 404. --timeout 0 because a long video legitimately blocks a
# worker thread well past gunicorn's default.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "8", \
     "--timeout", "0", "modnet_bg.web:create_app()"]
