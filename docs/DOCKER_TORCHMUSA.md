# Docker: TorchMUSA-oriented image (template)

The default **`Dockerfile`** in this repository targets **CPU PyTorch** for portable CI (see **[Docker](DOCKER.md)**). This page provides a template for an image that includes **TorchMUSA** and optional server stacks. Such images require Moore Threads’ version-pinned wheels or build instructions for the target driver and GPU.

Use this page as a **checklist**. Pin versions to match **[torch_musa](https://github.com/MooreThreads/torch_musa)** and, if applicable, **[vllm-musa](https://github.com/MooreThreads/vllm-musa)**.

---

## 1. Preconditions

- Base image **supported** by your TorchMUSA build (often a vendor-provided CUDA-like or minimal Ubuntu base, per Moore Threads docs).
- **MUSA driver** on the **host** when running GPU containers (`--gpus all` or equivalent).
- A vendor wheel source or build procedure for the selected TorchMUSA combination.

---

## 2. Example multi-stage outline

The following is **illustrative**; replace `ARG` values with your approved versions.

```dockerfile
# Example only — do not use unmodified in production.
ARG PY_VER=3.12
ARG TORCH_MUSA_INDEX=https://example.invalid/torch-musa-wheels/

FROM ubuntu:22.04 AS musa_stack
ARG PY_VER
ARG TORCH_MUSA_INDEX

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PY_VER} python${PY_VER}-venv python3-pip ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/srv
# COPY in vendor wheels or use the vendor installation command:
# RUN pip install --no-cache-dir torch torch_musa -i ${TORCH_MUSA_INDEX}

COPY . /opt/srv/musa-aiter
WORKDIR /opt/srv/musa-aiter
RUN pip install --no-cache-dir -e .

CMD ["python3", "-m", "pytest", "op_tests", "-q", "-m", "musa"]
```

**Notes**

- Prefer **`musa-aiter check`** and **`scripts/validate_native_musa_env.py`** (repository root) inside the container after install.
- For **SGLang** or **vllm-musa**, add layers **after** TorchMUSA stabilizes, following their Docker fragments.
- CI that cannot access GPU may still **`docker build`** a **CPU-only** stage; keep GPU stages in a separate target (`--target`).

---

## 3. Relationship to the CPU image

| Image | Purpose |
| --- | --- |
| Root **`Dockerfile`** | Fast **CPU** reference tests and MkDocs-free smoke. |
| Template on this page | **Lab / deployment** images with **TorchMUSA** and optional frameworks. |
