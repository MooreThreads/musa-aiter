from __future__ import annotations

import pytest
import torch

from musa_aiter.ops.kv_layout import KVLayoutKind, KVOrganization
from musa_aiter.ops.lmcache_format import (
    LMCacheFormat,
    describe_lmcache_format,
    describe_sglang_split_list,
)


def test_kv_2ltd_format_is_identified_from_memory_shape():
    facts = describe_lmcache_format(torch.zeros(2, 3, 5, 16))

    assert facts.format is LMCacheFormat.KV_2LTD
    assert facts.spec.kind is KVLayoutKind.KV_2LTD
    assert facts.spec.num_layers == 3
    assert facts.spec.num_tokens == 5


def test_kv_t2d_and_kv_2td_are_not_ambiguous():
    t2d = describe_lmcache_format(torch.zeros(5, 2, 16))
    two_td = describe_lmcache_format(torch.zeros(2, 5, 16))

    assert t2d.format is LMCacheFormat.KV_T2D
    assert t2d.spec.organization is KVOrganization.MERGED
    assert two_td.format is LMCacheFormat.KV_2TD
    assert two_td.spec.organization is KVOrganization.SEPARATE


def test_sglang_split_list_format_preserves_layer_count_and_geometry():
    caches = [
        [torch.zeros(4, 16, 2, 8) for _ in range(3)],
        [torch.zeros(4, 16, 2, 8) for _ in range(3)],
    ]

    facts = describe_sglang_split_list(caches)

    assert facts.format is LMCacheFormat.SGLANG_SPLIT_LIST
    assert facts.spec.kind is KVLayoutKind.TWO_X_NL
    assert facts.spec.num_layers == 3
    assert facts.spec.block_size == 16


def test_format_rejects_wrong_rank_and_empty_split_list():
    with pytest.raises(ValueError, match="rank"):
        describe_lmcache_format(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="non-empty"):
        describe_sglang_split_list([[], []])


def test_format_facts_are_exported_from_package():
    import musa_aiter

    assert musa_aiter.LMCacheFormat is LMCacheFormat
    assert musa_aiter.describe_lmcache_format is describe_lmcache_format
