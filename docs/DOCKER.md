# Docker (development image)

The default image provides a **CPU/reference** container for running tests and CLI checks without the MUSA driver stack.

## Building

From the repository root:

```bash
docker build -t musa-aiter:dev .
```

## Running tests

```bash
docker run --rm musa-aiter:dev
```

The default container command runs:

```bash
python -m pytest op_tests -q
```

## Interactive shell

```bash
docker run --rm -it --entrypoint bash musa-aiter:dev
```

Inside the container:

```bash
musa-aiter check --soft
musa-aiter list-ops
```

## Limitations

- **TorchMUSA-style lab image (template)** — **[Docker + TorchMUSA](DOCKER_TORCHMUSA.md)** (vendor wheels required).
- **TorchMUSA** and native MUSA execution are not included. Use a host or image with the vendor stack for native validation.
- PyTorch is installed from the **CPU** wheel index to keep the image reasonably portable for CI-style checks.
