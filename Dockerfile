# CPU/reference environment for pytest and CLI (no MUSA driver or TorchMUSA).
FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY musa_aiter ./musa_aiter
COPY op_tests ./op_tests
COPY benchmarks ./benchmarks
COPY examples ./examples
COPY csrc ./csrc
COPY configs ./configs
COPY scripts ./scripts
COPY docs ./docs

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e ".[dev]"

CMD ["python", "-m", "pytest", "op_tests", "-q"]
