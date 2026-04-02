# photic forecast-gui Dockerfile
#
# Multi-stage build following Astrodash pattern, adapted for Python 3.13 and uv.
#
FROM python:3.13-slim AS deps

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv for dependency management
RUN pip install --no-cache-dir uv

# Install CPU-only PyTorch first (largest dependency)
RUN uv pip install --system --no-cache \
    torch --index-url https://download.pytorch.org/whl/cpu

# Copy project files for dependency resolution
COPY pyproject.toml uv.lock README.md ./

# Install photic and all remaining dependencies
# torch is already installed above, so uv will satisfy it from the existing install
COPY src/ ./src/
RUN uv pip install --system --no-cache .

FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1

COPY --from=deps /usr/local/lib/python3.13/site-packages/ /usr/local/lib/python3.13/site-packages/
COPY --from=deps /usr/local/bin/ /usr/local/bin/

RUN mkdir -p /app /data/forecasts
COPY src/ /app/src/
COPY pyproject.toml README.md /app/
WORKDIR /app

EXPOSE 8080

CMD ["photic", "forecast-gui", "--host", "0.0.0.0", "--port", "8080", "--forecast-dir", "/data/forecasts"]
