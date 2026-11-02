"""JIT / smoke helpers: ``mcc`` shared-object builds, manifests, ctypes loading.

Smoke operators (e.g. :func:`~musa_aiter.softmax_last_dim`) import from
``musa_aiter.jit.core`` and ``musa_aiter.jit.ctypes_loader`` — not ``musa_aiter.ops.jit``.
"""
