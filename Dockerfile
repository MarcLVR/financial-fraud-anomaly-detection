# Stage 1: build dependencies in an isolated layer.
# Using a separate builder stage keeps the final image small —
# build tools and pip cache are not copied to the runtime image.
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements-api.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements-api.txt


# Stage 2: runtime image.
# Only the installed packages and the application code are copied.
FROM python:3.11-slim AS runtime

# Run as a non-root user. Running containers as root is a security
# risk — if the process is compromised, the attacker has root access
# to the container filesystem.
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from the builder stage.
COPY --from=builder /install /usr/local

# Copy application source.
COPY src/ ./src/

# Copy model artefacts.
# In production these would be mounted as a volume or pulled from
# a model registry rather than baked into the image.
COPY src/models/ ./src/models/

USER appuser

EXPOSE 8000

# LOG_LEVEL and MODELS_DIR can be overridden at runtime via
# docker run -e LOG_LEVEL=DEBUG or in a docker-compose file.
ENV LOG_LEVEL=INFO
ENV MODELS_DIR=src/models

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
