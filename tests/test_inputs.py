import numpy as np
import pytest

from binsmooth import BinSmooth


def test_include_tail():
    bin_edges = np.array([0, 18200, 37000, 87000, 180000])
    counts = np.array([0, 7527, 13797, 75481, 50646, 803])

    bs = BinSmooth()

    with pytest.raises(
        ValueError,
        match="Length of x and y must match when tail is included\\.",
    ):
        bs.fit(bin_edges, counts, includes_tail=True)


def test_not_include_tail():
    bin_edges = np.array([0, 18200, 37000, 87000, 180000, 360000])
    counts = np.array([0, 7527, 13797, 75481, 50646, 803])

    bs = BinSmooth()

    with pytest.raises(
        ValueError,
        match="Length of x must be N-1 when tail is not included\\.",
    ):
        bs.fit(bin_edges, counts, includes_tail=False)


def test_y_zero():
    bin_edges = np.array([0, 18200, 37000, 87000, 180000])
    counts = np.array([10, 7527, 13797, 75481, 50646, 803])

    bs = BinSmooth()

    with pytest.raises(ValueError, match="y must begin with 0\\."):
        bs.fit(bin_edges, counts)


def test_mean_warning():
    with pytest.warns(
        UserWarning, match="No mean provided, results may be innacurate\\."
    ) as record:
        # Positive
        bin_edges = np.array([0, 18200, 37000, 87000, 180000])
        counts = np.array([0, 7527, 13797, 75481, 50646, 803])
        bs = BinSmooth()
        bs.fit(bin_edges, counts, includes_tail=False)

        # Negative
        bin_edges = np.array([0, 18200, 37000, 87000, 180000, 360000])
        counts = np.array([0, 7527, 13797, 75481, 50646, 803])
        bs = BinSmooth()
        bs.fit(bin_edges, counts, includes_tail=True)

    # Should only be 1 matching warning
    assert len(record) == 1


def test_edges_increasing():
    bin_edges = np.array([0, 18200, 37000, 180000, 180000])
    counts = np.array([0, 7527, 13797, 75481, 50646, 803])
    bs = BinSmooth()

    with pytest.raises(ValueError, match="x must be strictly increasing\\."):
        bs.fit(bin_edges, counts)


def test_counts_negative():
    bin_edges = np.array([0, 18200, 37000, 180000, 360000])
    counts = np.array([0, 7527, 13797, -5, 50646, 803])
    bs = BinSmooth()

    with pytest.raises(ValueError, match="y contains negative values\\."):
        bs.fit(bin_edges, counts)


def test_counts_last_zero():
    with pytest.warns(
        UserWarning,
        match="x and y have been trimmed to remove trailing zeros in y.",
    ):
        bin_edges = np.array([0, 18200, 37000, 87000, 180000, 360000])
        counts = np.array([0, 7527, 13797, 75481, 50646, 0])
        bs = BinSmooth()
        bs.fit(bin_edges, counts, includes_tail=True)

        bin_edges = np.array([0, 18200, 37000, 87000, 180000])
        counts = np.array([0, 7527, 13797, 75481, 50646, 0])
        bs = BinSmooth()
        bs.fit(bin_edges, counts, includes_tail=False)
