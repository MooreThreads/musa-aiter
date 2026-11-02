from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TensorParallelGroup:
    size: int = 1
    rank: int = 0

    def is_first_rank(self) -> bool:
        return self.rank == 0

    def is_last_rank(self) -> bool:
        return self.rank == self.size - 1


def initialize_model_parallel(tensor_model_parallel_size: int = 1, rank: int = 0) -> TensorParallelGroup:
    if tensor_model_parallel_size < 1:
        raise ValueError("tensor_model_parallel_size must be >= 1")
    if rank < 0 or rank >= tensor_model_parallel_size:
        raise ValueError("rank out of range")
    return TensorParallelGroup(size=tensor_model_parallel_size, rank=rank)


def destroy_model_parallel(group: TensorParallelGroup | None = None) -> None:
    return None


def get_tensor_model_parallel_rank(group: TensorParallelGroup | None = None) -> int:
    return 0 if group is None else group.rank


def get_tensor_model_parallel_world_size(group: TensorParallelGroup | None = None) -> int:
    return 1 if group is None else group.size
