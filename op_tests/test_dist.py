import musa_aiter


def test_tensor_parallel_group():
    group = musa_aiter.initialize_model_parallel(tensor_model_parallel_size=2, rank=1)
    assert group.size == 2
    assert group.rank == 1
    assert not group.is_first_rank()
    assert group.is_last_rank()
    assert musa_aiter.get_tensor_model_parallel_rank(group) == 1
    assert musa_aiter.get_tensor_model_parallel_world_size(group) == 2


def test_default_parallel_helpers():
    assert musa_aiter.get_tensor_model_parallel_rank() == 0
    assert musa_aiter.get_tensor_model_parallel_world_size() == 1
