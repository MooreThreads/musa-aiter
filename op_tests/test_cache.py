import torch

import musa_aiter


def test_apply_rope_non_interleaved():
    x=torch.tensor([[[[1.,2.,3.,4.]]]])
    cos=torch.ones(1,1,1,2)
    sin=torch.zeros(1,1,1,2)
    out=musa_aiter.apply_rope(x,cos,sin)
    torch.testing.assert_close(out,x)


def test_apply_rope_interleaved_identity():
    x=torch.randn(2,3,4,8)
    cos=torch.ones(2,3,4,4)
    sin=torch.zeros(2,3,4,4)
    out=musa_aiter.apply_rope(x,cos,sin,interleaved=True)
    torch.testing.assert_close(out,x)


def test_apply_rope_to_key_cache_selected_slots():
    kc = torch.zeros(2, 1, 4, 4)
    kc[0, :, :, 0] = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    kc[1, :, :, 2] = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
    original = kc.clone()
    slots = torch.tensor([0, 6, -1], dtype=torch.int64)
    positions = torch.tensor([0, 1, 2], dtype=torch.int64)
    cos = torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
    sin = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])

    out = musa_aiter.apply_rope_to_key_cache(kc, slots, positions, cos, sin)

    assert out.data_ptr() == kc.data_ptr()
    torch.testing.assert_close(kc[0, :, :, 0], original[0, :, :, 0])
    torch.testing.assert_close(kc[1, :, :, 2], torch.tensor([[-7.0, -8.0, 5.0, 6.0]]))
    torch.testing.assert_close(kc[0, :, :, 1], original[0, :, :, 1])


def test_apply_rope_to_key_cache_rejects_invalid_position():
    kc = torch.zeros(1, 1, 4, 2)
    slots = torch.tensor([0], dtype=torch.int64)
    positions = torch.tensor([2], dtype=torch.int64)
    cos = torch.ones(2, 2)
    sin = torch.zeros(2, 2)

    try:
        musa_aiter.apply_rope_to_key_cache(kc, slots, positions, cos, sin)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for position outside cos/sin table")


def test_copy_to_kv_cache():
    k=torch.randn(3,2,4)
    v=torch.randn(3,2,4)
    kc=torch.zeros(2,2,4,2)
    vc=torch.zeros(2,2,4,2)
    slots=torch.tensor([0,2,3], dtype=torch.int64)
    musa_aiter.copy_to_kv_cache(k,v,kc,vc,slots)
    torch.testing.assert_close(kc[0,:,:,0], k[0])
    torch.testing.assert_close(kc[1,:,:,0], k[1])
    torch.testing.assert_close(kc[1,:,:,1], k[2])
    torch.testing.assert_close(vc[0,:,:,0], v[0])
