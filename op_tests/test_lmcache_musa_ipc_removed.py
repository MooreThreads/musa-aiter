import importlib.util

import musa_aiter


REMOVED_IPC_SYMBOLS = (
    "native_lmcache_musa_ipc_abi_version",
    "is_native_lmcache_musa_ipc_available",
    "lmcache_musa_ipc_get_mem_handle",
    "lmcache_musa_ipc_open_mem_handle",
    "lmcache_musa_ipc_close_mem_handle",
    "lmcache_musa_ipc_get_event_handle",
    "lmcache_musa_ipc_open_event_handle",
    "lmcache_musa_ipc_query_event",
    "lmcache_musa_ipc_wait_event",
    "lmcache_musa_ipc_record_event",
)


def test_lmcache_musa_ipc_symbols_are_not_exported_from_aiter():
    for symbol in REMOVED_IPC_SYMBOLS:
        assert not hasattr(musa_aiter, symbol), symbol
        assert symbol not in musa_aiter.__all__, symbol


def test_lmcache_musa_ipc_modules_are_not_packaged_in_aiter():
    assert importlib.util.find_spec("musa_aiter.ops.lmcache_musa_ipc") is None
    assert importlib.util.find_spec("musa_aiter.ops.lmcache_musa_ipc_ext") is None
