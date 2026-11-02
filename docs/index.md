# MUSA AITER Documentation

Welcome to the MUSA AITER documentation. These pages cover the public Python API, installation, native MUSA validation, framework integration, and examples.

## Get started

- [Architecture](ARCHITECTURE.md)
- [API policy](API_POLICY.md)
- [Package and installation](deploy/package-install.md)
- [Docker (CPU/reference)](DOCKER.md)
- [Docker with TorchMUSA](DOCKER_TORCHMUSA.md)

## Integration

- [vLLM on MUSA](VLLM_0_20_INTEGRATION_GUIDE.md)
- [SGLang on MUSA](SGLANG_INTEGRATION_GUIDE.md)
- [Examples](examples_index.md)

## Reference

- [Security for artifacts and traces](SECURITY_ARTIFACTS.md)
- [Benchmarks](benchmarks/index.md)

## Build the documentation site

```bash
python3 -m pip install -e ".[docs]"
mkdocs serve
```

The static site is generated in `site/` (ignored by Git).
