import numpy as np
import pytest

from binsmooth import BinSmooth

spline_types = ["PCHIP", "HYMAN", "LINEAR"]


@pytest.mark.parametrize("spline_type", spline_types)
def test_bounds_one_val(spline_type):
    bin_edges = np.array([0, 18200, 37000, 87000, 180000])
    counts = np.array([0, 7527, 13797, 75481, 50646, 803])

    bs = BinSmooth()

    with pytest.raises(
        ValueError,
        match="tail_bounds must only contain an upper and lower bound\\.",
    ):
        bs.fit(
            bin_edges,
            counts,
            spline_type=spline_type,
            includes_tail=False,
            tail_bounds=[1],
        )


@pytest.mark.parametrize("spline_type", spline_types)
def test_bounds_extra_val(spline_type):
    bin_edges = np.array([0, 18200, 37000, 87000, 180000])
    counts = np.array([0, 7527, 13797, 75481, 50646, 803])

    bs = BinSmooth()

    with pytest.raises(
        ValueError,
        match="tail_bounds must only contain an upper and lower bound\\.",
    ):
        bs.fit(
            bin_edges,
            counts,
            spline_type=spline_type,
            includes_tail=False,
            tail_bounds=[1, 2, 3],
        )


@pytest.mark.parametrize("spline_type", spline_types)
def test_bounds_low_val(spline_type):
    bin_edges = np.array([0, 18200, 37000, 87000, 180000])
    counts = np.array([0, 7527, 13797, 75481, 50646, 803])

    bs = BinSmooth()

    with pytest.raises(
        ValueError,
        match=(
            "The lower bound of tail_bounds must be greater than the last"
            " value in x\\."
        ),
    ):
        bs.fit(
            bin_edges,
            counts,
            spline_type=spline_type,
            includes_tail=False,
            tail_bounds=[bin_edges[-1], None],
        )


@pytest.mark.parametrize("spline_type", spline_types)
def test_bounds_high_val(spline_type):
    bin_edges = np.array([0, 18200, 37000, 87000, 180000])
    counts = np.array([0, 7527, 13797, 75481, 50646, 803])

    bs = BinSmooth()

    with pytest.raises(
        ValueError,
        match=(
            "The upper bound of tail_bounds must be greater the lower bound\\."
        ),
    ):
        bs.fit(
            bin_edges,
            counts,
            spline_type=spline_type,
            includes_tail=False,
            tail_bounds=[None, bin_edges[-1]],
        )
