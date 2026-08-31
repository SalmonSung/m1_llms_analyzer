"""Pooling math, verified against hand-computed values.

The critical property under test: padding position must not change the result.
"""

import pytest

torch = pytest.importorskip("torch")

from m1_analyzer.utils.pooling import (  # noqa: E402
    apply_pooling,
    cls_pool,
    last_token_pool,
    mean_pool,
    unpad_sequence,
)


@pytest.fixture
def right_padded():
    # Two sequences of length 3 and 1, padded on the right to 3.
    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]])
    return hidden, mask


def test_last_token_uses_last_real_token(right_padded):
    hidden, mask = right_padded
    out = last_token_pool(hidden, mask)
    assert torch.allclose(out, torch.tensor([[5.0, 6.0], [7.0, 8.0]]))


def test_last_token_ignores_padding_side():
    """Right- and left-padded versions of the same batch must agree."""
    right_h = torch.tensor([[[1.0], [2.0], [0.0]]])
    right_m = torch.tensor([[1, 1, 0]])
    left_h = torch.tensor([[[0.0], [1.0], [2.0]]])
    left_m = torch.tensor([[0, 1, 1]])
    assert torch.allclose(last_token_pool(right_h, right_m), last_token_pool(left_h, left_m))
    assert torch.allclose(mean_pool(right_h, right_m), mean_pool(left_h, left_m))
    assert torch.allclose(cls_pool(right_h, right_m), cls_pool(left_h, left_m))


def test_naive_last_index_would_be_wrong(right_padded):
    """Guards the bug this code exists to avoid: hidden[:, -1] pools a PAD token."""
    hidden, mask = right_padded
    naive = hidden[:, -1]
    assert not torch.allclose(naive, last_token_pool(hidden, mask))


def test_mean_excludes_padding(right_padded):
    hidden, mask = right_padded
    out = mean_pool(hidden, mask)
    assert torch.allclose(out, torch.tensor([[3.0, 4.0], [7.0, 8.0]]))


def test_cls_takes_first_real_token(right_padded):
    hidden, mask = right_padded
    assert torch.allclose(cls_pool(hidden, mask), torch.tensor([[1.0, 2.0], [7.0, 8.0]]))


def test_unpad_strips_padding_rows(right_padded):
    hidden, mask = right_padded
    parts = unpad_sequence(hidden, mask)
    assert [tuple(p.shape) for p in parts] == [(3, 2), (1, 2)]
    assert torch.allclose(parts[1], torch.tensor([[7.0, 8.0]]))


def test_zero_length_sequence_raises(right_padded):
    hidden, _ = right_padded
    empty_mask = torch.tensor([[1, 1, 1], [0, 0, 0]])
    with pytest.raises(ValueError, match="zero tokens"):
        last_token_pool(hidden, empty_mask)


def test_apply_pooling_rejects_unknown_mode(right_padded):
    hidden, mask = right_padded
    with pytest.raises(ValueError, match="Unknown pooling mode"):
        apply_pooling("median", hidden, mask)
