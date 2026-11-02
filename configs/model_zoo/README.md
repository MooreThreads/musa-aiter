# Model zoo (shape templates)

This directory provides a **curated manifest** of illustrative model geometry for **tuning**, **benchmarks**, and **integration tests**—not a download of weights. Hugging Face or other hubs remain the source of checkpoints; this repository stores **numbers and IDs** only.

## Versioning policy

| Artifact | Rule |
| --- | --- |
| **`manifest.v1.json`** | Monotonic **`schema`** / document **`version`** fields. Bump **`version`** when entries change incompatibly; append-only preferred for new models. |
| **`revision_date`** | ISO date when the manifest last changed (human audit). |
| **Per-model `id`** | Stable snake_case slug; never rename—add a new `id` if dimensions change. |
| **Tuned kernels** | Rows in **`configs/tuned/`** reference **`arch`**, **`dtype`**, and shape strings; link benchmark logs in your org, not in this repo unless scrubbed. |

## Files

- **`manifest.v1.json`** — machine-readable list (`schema: musa_aiter.model_zoo.v1`).
- **`phase2_reference_configs.json`** (under **`configs/model_configs/`**) — operator-level placeholder tuning rows; distinct from zoo entries.

## Adding a model

1. Add an object to **`models`** with unique **`id`** and numeric fields your schedulers need.
2. Document **`notes`** if the row is synthetic (no public weights) or license-scoped.
3. When real latency data exists, add or update **`configs/tuned/*.csv`** and point **`select_config`** callers at the new rows.

## Relation to frameworks

- **vLLM** — `docs/VLLM_0_20_INTEGRATION_GUIDE.md` (repository root).
- **SGLang** — `docs/SGLANG_INTEGRATION_GUIDE.md`.
