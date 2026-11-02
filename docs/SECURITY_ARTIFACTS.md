# Security: artifacts, caches, and traces

This page documents how **JIT caches**, **build outputs**, **dispatch traces**, and optional **kernel dumps** should be handled so operators do not leak secrets or unstable paths into shared artifacts.

This is **operational policy**, not a product warranty. Adapt to your organization's security review.

## 1. JIT / native build cache

| Location | Default |
| --- | --- |
| musa-aiter JIT / smoke builds | Typically under **`~/.cache/musa_aiter`** (see environment checks in **`musa-aiter check`**) |

**Recommendations**

- Treat compiled **`.so`** and intermediate object files as **sensitive build outputs** when they embed absolute paths from the build host.
- Do not upload raw cache directories to public bug reports; share **hashes** and **manifest** snippets from **`musa_aiter`** JIT metadata when debugging.

## 2. Dispatch traces (`MUSA_AITER_TRACE_DISPATCH=1`)

Traces include **operator names**, **backend labels**, **device**, **dtype**, and a short **reason** string. They do **not** intentionally print tensor values, but logs can still reveal **model topology** if operator names mirror layer structure.

**Recommendations**

- Redact or disable traces before sharing logs externally.
- Prefer **`musa-aiter trace -- your_command`** for a scoped subprocess instead of leaving trace enabled globally in production.

## 3. Kernel dumps and replay artifacts

If you capture **memory dumps**, **NSight**-style profiles, or **replay bundles** while developing native kernels:

- Exclude **API keys**, **tokens**, **SSH keys**, and **home directory paths** from archives.
- Label bundles with **toolkit version**, **driver version**, and **GPU SKU** instead of only git SHA when communicating with vendors.

## 4. Documentation and CI

- **MkDocs** output under **`site/`** should not be committed with private data; regenerate from sanitized sources.
- **Benchmark JSON** committed under **`docs/benchmarks/`** is intended to be **numeric and non-identifying**; review before extending fields.

## 5. Related controls

- **Stable vs experimental APIs** — **`docs/API_POLICY.md`**.
- **Package installation** — **`docs/deploy/package-install.md`** (build and verification commands).
