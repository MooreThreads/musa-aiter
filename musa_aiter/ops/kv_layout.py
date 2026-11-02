from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class KVLayoutKind(str, Enum):
    STANDARD_PAGED = "standard_paged"
    LAYER_FIRST = "layer_first"
    PAGE_FIRST = "page_first"
    PAGE_FIRST_DIRECT = "page_first_direct"
    PAGE_HEAD = "page_head"
    MERGED_LINEAR = "merged_linear"
    TWO_X_NL = "two_x_nl"
    KV_2LTD = "kv_2ltd"
    KV_T2D = "kv_t2d"
    KV_2TD = "kv_2td"
    MLA_FLAT = "mla_flat"
    DSV4_STRIDED = "dsv4_strided"
    DSV4_BLOCKED_SCALE = "dsv4_blocked_scale"


class KVOrganization(str, Enum):
    SEPARATE = "separate"
    MERGED = "merged"
    MLA = "mla"
    SPLIT_LIST = "split_list"


class KVScaleKind(str, Enum):
    NONE = "none"
    SCALAR = "scalar"
    PER_HEAD = "per_head"
    PER_LAYER_HEAD = "per_layer_head"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class KVFormatFacts:
    kind: KVLayoutKind
    dtype_name: str
    row_bytes: int
    block_stride_bytes: int
    organization: KVOrganization
    scale_kind: KVScaleKind
    value_bytes: int
    scale_bytes: int
    is_contiguous: bool
    requires_transform: bool
    pointer_table_length: int

    @property
    def supports_direct_batch(self) -> bool:
        return (
            self.is_contiguous
            and not self.requires_transform
            and self.scale_kind is KVScaleKind.NONE
        )


@dataclass(frozen=True)
class KVLayoutSpec:
    kind: KVLayoutKind
    num_layers: int
    num_blocks: int
    block_size: int
    num_heads: int
    head_size: int
    row_bytes: int
    block_stride_bytes: int
    dtype_name: str
    organization: KVOrganization = KVOrganization.SEPARATE
    scale_kind: KVScaleKind = KVScaleKind.NONE
    value_bytes: int = 0
    scale_bytes: int = 0
    num_tokens: int = 0
    is_contiguous: bool = True
    pointer_table_length: int = 0
    device_kind: str = "any"

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", KVLayoutKind(self.kind))
        object.__setattr__(self, "organization", KVOrganization(self.organization))
        object.__setattr__(self, "scale_kind", KVScaleKind(self.scale_kind))

        for name in ("num_layers", "num_heads", "head_size", "row_bytes"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")

        if self.is_paged:
            if self.num_blocks <= 0:
                raise ValueError("num_blocks must be positive for paged layouts")
            if self.block_size <= 0:
                raise ValueError("block_size must be positive for paged layouts")
            if self.block_stride_bytes < self.row_bytes:
                raise ValueError("block_stride_bytes must be >= row_bytes")
        elif self.num_blocks < 0 or self.block_size < 0:
            raise ValueError("linear layout block geometry must be non-negative")

        if not self.dtype_name:
            raise ValueError("dtype_name must be non-empty")
        if self.pointer_table_length < 0:
            raise ValueError("pointer_table_length must be non-negative")
        if self.scale_kind is KVScaleKind.BLOCKED:
            if self.value_bytes <= 0:
                raise ValueError("value_bytes must be positive for blocked scale")
            if self.scale_bytes <= 0:
                raise ValueError("scale_bytes must be positive for blocked scale")
            if self.value_bytes + self.scale_bytes > self.row_bytes:
                raise ValueError("value_bytes + scale_bytes must be <= row_bytes")
        elif self.value_bytes < 0 or self.scale_bytes < 0:
            raise ValueError("scale byte widths must be non-negative")

    @property
    def hidden_size(self) -> int:
        return self.num_heads * self.head_size

    @property
    def is_paged(self) -> bool:
        return self.kind in {
            KVLayoutKind.STANDARD_PAGED,
            KVLayoutKind.PAGE_FIRST,
            KVLayoutKind.PAGE_FIRST_DIRECT,
            KVLayoutKind.PAGE_HEAD,
            KVLayoutKind.TWO_X_NL,
            KVLayoutKind.MLA_FLAT,
            KVLayoutKind.DSV4_STRIDED,
            KVLayoutKind.DSV4_BLOCKED_SCALE,
        }

    @property
    def requires_layout_transform(self) -> bool:
        return self.kind in {
            KVLayoutKind.PAGE_FIRST,
            KVLayoutKind.PAGE_FIRST_DIRECT,
            KVLayoutKind.PAGE_HEAD,
            KVLayoutKind.DSV4_STRIDED,
            KVLayoutKind.DSV4_BLOCKED_SCALE,
        } or not self.is_contiguous

    def format_facts(self) -> KVFormatFacts:
        return KVFormatFacts(
            kind=self.kind,
            dtype_name=self.dtype_name,
            row_bytes=self.row_bytes,
            block_stride_bytes=self.block_stride_bytes,
            organization=self.organization,
            scale_kind=self.scale_kind,
            value_bytes=self.value_bytes,
            scale_bytes=self.scale_bytes,
            is_contiguous=self.is_contiguous,
            requires_transform=self.requires_layout_transform,
            pointer_table_length=self.pointer_table_length,
        )

    @classmethod
    def standard_paged(
        cls,
        *,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype_name: str,
        row_bytes: int,
        block_stride_bytes: int,
        organization: KVOrganization = KVOrganization.SEPARATE,
        scale_kind: KVScaleKind = KVScaleKind.NONE,
        value_bytes: int = 0,
        scale_bytes: int = 0,
        pointer_table_length: int = 0,
        device_kind: str = "any",
    ) -> "KVLayoutSpec":
        return cls(
            kind=KVLayoutKind.STANDARD_PAGED,
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_size=head_size,
            row_bytes=row_bytes,
            block_stride_bytes=block_stride_bytes,
            dtype_name=dtype_name,
            organization=organization,
            scale_kind=scale_kind,
            value_bytes=value_bytes,
            scale_bytes=scale_bytes,
            pointer_table_length=pointer_table_length,
            device_kind=device_kind,
        )

    @classmethod
    def page_first(
        cls,
        *,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype_name: str,
        row_bytes: int,
        block_stride_bytes: int,
        organization: KVOrganization = KVOrganization.SEPARATE,
        pointer_table_length: int = 0,
        device_kind: str = "any",
    ) -> "KVLayoutSpec":
        return cls(
            kind=KVLayoutKind.PAGE_FIRST,
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_size=head_size,
            row_bytes=row_bytes,
            block_stride_bytes=block_stride_bytes,
            dtype_name=dtype_name,
            organization=organization,
            is_contiguous=False,
            pointer_table_length=pointer_table_length,
            device_kind=device_kind,
        )

    @classmethod
    def linear(
        cls,
        *,
        num_layers: int,
        num_tokens: int,
        num_heads: int,
        head_size: int,
        dtype_name: str,
        organization: KVOrganization = KVOrganization.SEPARATE,
        kind: KVLayoutKind = KVLayoutKind.LAYER_FIRST,
        is_contiguous: bool = True,
        device_kind: str = "any",
    ) -> "KVLayoutSpec":
        if kind not in {
            KVLayoutKind.LAYER_FIRST,
            KVLayoutKind.MERGED_LINEAR,
            KVLayoutKind.KV_2LTD,
            KVLayoutKind.KV_T2D,
            KVLayoutKind.KV_2TD,
        }:
            raise ValueError("linear kind must describe a linear layout")
        itemsize = {
            "float32": 4,
            "float16": 2,
            "bfloat16": 2,
            "uint8": 1,
        }.get(dtype_name)
        if itemsize is None:
            raise ValueError(f"unsupported dtype_name={dtype_name!r}")
        row_bytes = num_heads * head_size * itemsize
        return cls(
            kind=kind,
            num_layers=num_layers,
            num_blocks=0,
            block_size=0,
            num_heads=num_heads,
            head_size=head_size,
            row_bytes=row_bytes,
            block_stride_bytes=row_bytes,
            dtype_name=dtype_name,
            organization=organization,
            num_tokens=num_tokens,
            is_contiguous=is_contiguous,
            device_kind=device_kind,
        )

    @classmethod
    def dsv4(
        cls,
        *,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        row_bytes: int,
        block_stride_bytes: int,
        dtype_name: str,
        scale_kind: KVScaleKind = KVScaleKind.NONE,
        value_bytes: int = 0,
        scale_bytes: int = 0,
        device_kind: str = "any",
    ) -> "KVLayoutSpec":
        kind = (
            KVLayoutKind.DSV4_BLOCKED_SCALE
            if scale_kind is KVScaleKind.BLOCKED
            else KVLayoutKind.DSV4_STRIDED
        )
        return cls(
            kind=kind,
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=1,
            head_size=row_bytes,
            row_bytes=row_bytes,
            block_stride_bytes=block_stride_bytes,
            dtype_name=dtype_name,
            scale_kind=scale_kind,
            value_bytes=value_bytes,
            scale_bytes=scale_bytes,
            is_contiguous=False,
            device_kind=device_kind,
        )
