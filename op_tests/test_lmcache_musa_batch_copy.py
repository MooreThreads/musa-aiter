from __future__ import annotations

from pathlib import Path

import pytest
import torch

import musa_aiter


def _batch_module():
    from musa_aiter.ops import lmcache_musa_batch_copy

    return lmcache_musa_batch_copy


def test_lmcache_musa_batch_copy_public_api_is_exported() -> None:
    from musa_aiter.registry import get

    assert musa_aiter.native_lmcache_musa_batch_copy_abi_version() == 1
    assert callable(musa_aiter.is_native_lmcache_musa_batch_copy_available)
    assert callable(musa_aiter.lmcache_musa_memcpy_batch_async)
    assert callable(musa_aiter.native_lmcache_musa_batch_copy_runtime_versions)
    capability = get("lmcache_musa_memcpy_batch_async")
    assert capability.family == "kvcache"
    assert capability.backends == ("musa-sdk-5.1-batch-copy",)


def test_batch_copy_is_opt_in_and_empty_batch_is_a_noop(monkeypatch) -> None:
    batch = _batch_module()
    monkeypatch.delenv("MUSA_AITER_LMCACHE_BATCH_COPY", raising=False)
    monkeypatch.delenv("LMCACHE_MUSA_DIRECT_COPY", raising=False)
    monkeypatch.setattr(batch, "load_lmcache_musa_batch_copy_ext", lambda: pytest.fail("extension must stay lazy"))

    assert musa_aiter.is_native_lmcache_musa_batch_copy_available() is False
    assert musa_aiter.lmcache_musa_memcpy_batch_async([], [], []) is True


def test_downstream_feature_gate_alias_enables_capability_probe(monkeypatch) -> None:
    batch = _batch_module()
    monkeypatch.delenv("MUSA_AITER_LMCACHE_BATCH_COPY", raising=False)
    monkeypatch.setenv("LMCACHE_MUSA_DIRECT_COPY", "1")

    class FakeExtension:
        def abi_version(self) -> int:
            return 1

        def batch_memcpy_supported(self) -> bool:
            return True

    monkeypatch.setattr(batch, "load_lmcache_musa_batch_copy_ext", lambda: FakeExtension())

    assert batch.is_native_lmcache_musa_batch_copy_available() is True


def test_runtime_version_diagnostic_is_exposed_from_native_extension(monkeypatch) -> None:
    batch = _batch_module()
    monkeypatch.setenv("MUSA_AITER_LMCACHE_BATCH_COPY", "1")

    class FakeExtension:
        def abi_version(self) -> int:
            return 1

        def runtime_version(self) -> int:
            return 5010

        def driver_version(self) -> int:
            return 5010

    monkeypatch.setattr(batch, "load_lmcache_musa_batch_copy_ext", lambda: FakeExtension())

    assert batch.native_lmcache_musa_batch_copy_runtime_versions() == (5010, 5010)


def test_plan_splits_entries_at_host_registration_boundaries() -> None:
    batch = _batch_module()
    plan = batch._build_batch_copy_plan(
        direction="h2d",
        sizes=[10],
        source_capacities=[32],
        destination_capacities=[32],
        source_offsets=[4],
        destination_offsets=[8],
        host_registration_offsets=[6],
        pin_chunk_size=8,
    )

    assert plan.direction == "h2d"
    assert plan.total_bytes == 10
    assert plan.input_entry_count == 1
    assert [(entry.source_offset, entry.destination_offset, entry.size) for entry in plan.entries] == [
        (4, 8, 2),
        (6, 10, 8),
    ]


def test_plan_preserves_entry_order_and_d2h_host_offsets() -> None:
    batch = _batch_module()
    plan = batch._build_batch_copy_plan(
        direction="d2h",
        sizes=[3, 5],
        source_capacities=[16, 16],
        destination_capacities=[16, 16],
        source_offsets=[1, 2],
        destination_offsets=[3, 4],
        host_registration_offsets=[5, 0],
        pin_chunk_size=8,
    )

    assert plan.total_bytes == 8
    assert [(entry.input_index, entry.size) for entry in plan.entries] == [(0, 3), (1, 5)]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"direction": "sideways"}, "direction"),
        ({"sizes": [4, 5]}, "same length"),
        ({"sizes": [0]}, "positive"),
        ({"source_offsets": [-1]}, "non-negative"),
        ({"destination_capacities": [3]}, "destination range"),
        ({"source_capacities": [3]}, "source range"),
        ({"pin_chunk_size": 0}, "pin_chunk_size"),
    ],
)
def test_plan_rejects_invalid_contract(kwargs, message: str) -> None:
    batch = _batch_module()
    values = {
        "direction": "h2d",
        "sizes": [4],
        "source_capacities": [8],
        "destination_capacities": [8],
        "source_offsets": [0],
        "destination_offsets": [0],
        "host_registration_offsets": [0],
        "pin_chunk_size": 8,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        batch._build_batch_copy_plan(**values)


def test_plan_rejects_overlapping_destinations_in_one_unordered_batch() -> None:
    batch = _batch_module()
    with pytest.raises(ValueError, match="overlapping destination"):
        batch._build_batch_copy_plan(
            direction="h2d",
            sizes=[6, 4],
            source_capacities=[16, 16],
            destination_capacities=[16, 16],
            source_offsets=[0, 0],
            destination_offsets=[0, 4],
            destination_keys=[7, 7],
        )


def test_plan_chunks_entries_without_changing_order() -> None:
    batch = _batch_module()
    plan = batch._build_batch_copy_plan(
        direction="h2d",
        sizes=[2, 3, 4, 5, 6],
        source_capacities=[20] * 5,
        destination_capacities=[20] * 5,
        source_offsets=[0, 0, 0, 0, 0],
        destination_offsets=[0, 0, 0, 0, 0],
        destination_keys=list(range(5)),
    )

    chunks = batch._chunk_batch_copy_plan(plan, 2)

    assert [[entry.input_index for entry in chunk] for chunk in chunks] == [[0, 1], [2, 3], [4]]
    assert sum(entry.size for chunk in chunks for entry in chunk) == plan.total_bytes


def test_plan_rejects_nonpositive_batch_limit() -> None:
    batch = _batch_module()
    plan = batch._build_batch_copy_plan(
        direction="h2d",
        sizes=[2],
        source_capacities=[2],
        destination_capacities=[2],
    )

    with pytest.raises(ValueError, match="max_batch_entries"):
        batch._chunk_batch_copy_plan(plan, 0)


def test_batch_limit_uses_environment_override(monkeypatch) -> None:
    batch = _batch_module()
    monkeypatch.setenv("MUSA_AITER_LMCACHE_BATCH_MAX_ENTRIES", "17")
    assert batch._resolve_max_batch_entries(None) == 17

    monkeypatch.setenv("MUSA_AITER_LMCACHE_BATCH_MAX_ENTRIES", "0")
    with pytest.raises(ValueError, match="must be positive"):
        batch._resolve_max_batch_entries(None)


def test_execute_submits_validated_split_plan_to_native_backend(monkeypatch) -> None:
    batch = _batch_module()
    monkeypatch.setenv("MUSA_AITER_LMCACHE_BATCH_COPY", "1")

    source = torch.arange(16, dtype=torch.uint8)
    destination = torch.zeros(16, dtype=torch.uint8)
    device_kinds = {id(source): "cpu", id(destination): "musa"}
    monkeypatch.setattr(batch, "_tensor_device_kind", lambda tensor: device_kinds[id(tensor)])

    class FakeExtension:
        def __init__(self) -> None:
            self.calls = []

        def abi_version(self) -> int:
            return 1

        def batch_memcpy_supported(self) -> bool:
            return True

        def execute_batch_copy(self, *args):
            self.calls.append(args)
            return True

    extension = FakeExtension()
    monkeypatch.setattr(batch, "load_lmcache_musa_batch_copy_ext", lambda: extension)

    assert batch.lmcache_musa_memcpy_batch_async(
        [destination],
        [source],
        [10],
        destination_offsets=[2],
        source_offsets=[1],
        host_registration_offsets=[6],
        pin_chunk_size=8,
        require_pinned_host=False,
        prefer_overlap_with_compute=True,
    )

    assert len(extension.calls) == 1
    destinations, sources, destination_offsets, source_offsets, sizes, prefer_overlap = extension.calls[0]
    assert destinations == [destination, destination]
    assert sources == [source, source]
    assert destination_offsets == [2, 4]
    assert source_offsets == [1, 3]
    assert sizes == [2, 8]
    assert prefer_overlap is True


def test_execute_chunks_native_submissions_at_configured_limit(monkeypatch) -> None:
    batch = _batch_module()
    monkeypatch.setenv("MUSA_AITER_LMCACHE_BATCH_COPY", "1")

    source = torch.arange(10, dtype=torch.uint8)
    destination = torch.zeros(10, dtype=torch.uint8)
    device_kinds = {id(source): "cpu", id(destination): "musa"}
    monkeypatch.setattr(batch, "_tensor_device_kind", lambda tensor: device_kinds[id(tensor)])

    class FakeExtension:
        def __init__(self) -> None:
            self.calls = []

        def abi_version(self) -> int:
            return 1

        def batch_memcpy_supported(self) -> bool:
            return True

        def execute_batch_copy(self, *args):
            self.calls.append(args)
            return True

    extension = FakeExtension()
    monkeypatch.setattr(batch, "load_lmcache_musa_batch_copy_ext", lambda: extension)

    assert batch.lmcache_musa_memcpy_batch_async(
        [destination] * 5,
        [source] * 5,
        [2] * 5,
        destination_offsets=[0, 2, 4, 6, 8],
        source_offsets=[0, 2, 4, 6, 8],
        require_pinned_host=False,
        max_batch_entries=2,
    )

    assert [call[4] for call in extension.calls] == [[2, 2], [2, 2], [2]]
    assert all(call[5] is False for call in extension.calls)


def test_execute_does_not_fallback_after_a_partial_chunk_submission(monkeypatch) -> None:
    batch = _batch_module()
    monkeypatch.setenv("MUSA_AITER_LMCACHE_BATCH_COPY", "1")

    source = torch.zeros(4, dtype=torch.uint8)
    destination = torch.zeros(4, dtype=torch.uint8)
    device_kinds = {id(source): "cpu", id(destination): "musa"}
    monkeypatch.setattr(batch, "_tensor_device_kind", lambda tensor: device_kinds[id(tensor)])

    class PartialExtension:
        def __init__(self) -> None:
            self.calls = 0

        def abi_version(self) -> int:
            return 1

        def batch_memcpy_supported(self) -> bool:
            return True

        def execute_batch_copy(self, *args):
            self.calls += 1
            return self.calls == 1

    extension = PartialExtension()
    monkeypatch.setattr(batch, "load_lmcache_musa_batch_copy_ext", lambda: extension)

    with pytest.raises(RuntimeError, match="earlier chunk"):
        batch.lmcache_musa_memcpy_batch_async(
            [destination] * 4,
            [source] * 4,
            [1] * 4,
            destination_offsets=[0, 1, 2, 3],
            source_offsets=[0, 1, 2, 3],
            require_pinned_host=False,
            max_batch_entries=2,
        )


def test_execute_disables_compute_overlap_hint_by_default(monkeypatch) -> None:
    batch = _batch_module()
    monkeypatch.setenv("MUSA_AITER_LMCACHE_BATCH_COPY", "1")

    source = torch.zeros(8, dtype=torch.uint8)
    destination = torch.zeros(8, dtype=torch.uint8)
    device_kinds = {id(source): "cpu", id(destination): "musa"}
    monkeypatch.setattr(batch, "_tensor_device_kind", lambda tensor: device_kinds[id(tensor)])

    class FakeExtension:
        def abi_version(self) -> int:
            return 1

        def batch_memcpy_supported(self) -> bool:
            return True

        def execute_batch_copy(self, *args):
            assert args[-1] is False
            return True

    monkeypatch.setattr(batch, "load_lmcache_musa_batch_copy_ext", lambda: FakeExtension())
    assert batch.lmcache_musa_memcpy_batch_async(
        [destination], [source], [8], require_pinned_host=False
    )


def test_lmcache_internal_api_requires_host_registration_boundary() -> None:
    batch = _batch_module()

    with pytest.raises(ValueError, match="host registration boundary"):
        batch.lmcache_musa_memcpy_batch_async_internal(
            [],
            [],
            [],
            host_registration_offsets=None,
            pin_chunk_size=8,
        )

    with pytest.raises(ValueError, match="host registration boundary"):
        batch.lmcache_musa_memcpy_batch_async_internal(
            [],
            [],
            [],
            host_registration_offsets=[],
            pin_chunk_size=None,
        )


def test_execute_fails_closed_without_native_backend(monkeypatch) -> None:
    batch = _batch_module()
    monkeypatch.setenv("MUSA_AITER_LMCACHE_BATCH_COPY", "1")
    source = torch.zeros(8, dtype=torch.uint8)
    destination = torch.zeros(8, dtype=torch.uint8)
    device_kinds = {id(source): "cpu", id(destination): "musa"}
    monkeypatch.setattr(batch, "_tensor_device_kind", lambda tensor: device_kinds[id(tensor)])
    monkeypatch.setattr(batch, "load_lmcache_musa_batch_copy_ext", lambda: None)

    assert (
        batch.lmcache_musa_memcpy_batch_async(
            [destination],
            [source],
            [8],
            require_pinned_host=False,
        )
        is False
    )


def test_execute_requires_pinned_host_memory_by_default(monkeypatch) -> None:
    batch = _batch_module()
    monkeypatch.setenv("MUSA_AITER_LMCACHE_BATCH_COPY", "1")
    source = torch.zeros(8, dtype=torch.uint8)
    destination = torch.zeros(8, dtype=torch.uint8)
    device_kinds = {id(source): "cpu", id(destination): "musa"}
    monkeypatch.setattr(batch, "_tensor_device_kind", lambda tensor: device_kinds[id(tensor)])
    monkeypatch.setattr(
        batch,
        "load_lmcache_musa_batch_copy_ext",
        lambda: pytest.fail("pageable host memory must be rejected before loading the extension"),
    )

    with pytest.raises(ValueError, match="pinned memory"):
        batch.lmcache_musa_memcpy_batch_async([destination], [source], [8])


def test_execute_fails_closed_for_an_incompatible_native_abi(monkeypatch) -> None:
    batch = _batch_module()
    monkeypatch.setenv("MUSA_AITER_LMCACHE_BATCH_COPY", "1")
    source = torch.zeros(8, dtype=torch.uint8)
    destination = torch.zeros(8, dtype=torch.uint8)
    device_kinds = {id(source): "cpu", id(destination): "musa"}
    monkeypatch.setattr(batch, "_tensor_device_kind", lambda tensor: device_kinds[id(tensor)])

    class IncompatibleExtension:
        def abi_version(self) -> int:
            return 2

        def batch_memcpy_supported(self) -> bool:
            return True

    monkeypatch.setattr(batch, "load_lmcache_musa_batch_copy_ext", lambda: IncompatibleExtension())

    assert batch.is_native_lmcache_musa_batch_copy_available() is False
    assert (
        batch.lmcache_musa_memcpy_batch_async(
            [destination],
            [source],
            [8],
            require_pinned_host=False,
        )
        is False
    )


def test_execute_rejects_invalid_plan_before_native_submission(monkeypatch) -> None:
    batch = _batch_module()
    monkeypatch.setenv("MUSA_AITER_LMCACHE_BATCH_COPY", "1")
    source = torch.zeros(8, dtype=torch.uint8)
    destination = torch.zeros(8, dtype=torch.uint8)
    device_kinds = {id(source): "cpu", id(destination): "musa"}
    monkeypatch.setattr(batch, "_tensor_device_kind", lambda tensor: device_kinds[id(tensor)])

    class FakeExtension:
        def batch_memcpy_supported(self) -> bool:
            return True

        def execute_batch_copy(self, *args):
            raise AssertionError("invalid plans must not reach the native backend")

    monkeypatch.setattr(batch, "load_lmcache_musa_batch_copy_ext", lambda: FakeExtension())

    with pytest.raises(ValueError, match="destination range"):
        batch.lmcache_musa_memcpy_batch_async(
            [destination],
            [source],
            [9],
            require_pinned_host=False,
        )


def test_native_source_uses_sdk_51_batch_api_without_synchronizing() -> None:
    source = Path("csrc/kvcache/lmcache_musa_batch_copy_ext.cpp").read_text(encoding="utf-8")

    assert "REAL_MUSA_VERSION >= 5010" in source
    assert "musaMemcpyBatchAsync(" in source
    assert "musaMemcpySrcAccessOrderStream" in source
    assert "musaMemcpyFlagPreferOverlapWithCompute" in source
    assert "musaRuntimeGetVersion" in source
    assert "musaDriverGetVersion" in source
    assert "std::sort" in source
    assert "record_event" in source
    assert "recordStream" in source
    assert "execute_copy_loop" in source
    assert "musaMemcpyAsync(" in source
    assert "musaStreamSynchronize(" not in source
