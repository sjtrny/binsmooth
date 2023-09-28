"""binsmooth - Better Estimates from Binned Income Data.

This module is a re-implementation of the R binsmooth package.

"""

# Author: Stephen Tierney <sjtrny@gmail.com>
#
# License: MIT

__version__ = "2023.9.0"

import warnings

import numdifftools as nd
import numpy as np
import splines
from scipy.integrate import cumtrapz, romb
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.optimize import minimize


def estimate_mean(x, cdf_fn, integral_pow=10):
    """Estimate the mean from a CDF.

    Parameters
    ----------
    x: ndarray
        Values from the original horizontal axis. Only x[0] and x[-1] are used.
    cdf_fn: function
        The CDF function
    integral_pow: int
        The value of `k` for the `2**k + 1` equally-spaced samples of the CDF
        that are used by the Romberg integrator.

    Returns
    -------
    mean: float
        The estimated mean
    """
    integral_num = 1 + 2**integral_pow

    x_integral, dx = np.linspace(
        x[0], x[-1], integral_num, endpoint=True, retstep=True
    )

    return romb(1 - cdf_fn(x_integral), dx=dx)


def estimate_auc(x, cdf_fn, integral_pow=5):
    """Estimate the AUC of a pdf from a CDF.

    Parameters
    ----------
    x: ndarray
        Values from the original horizontal axis. Only x[0] and x[-1] are used.
    cdf_fn: function
        The CDF function
    integral_pow: int
        The value of `k` for the `2**k + 1` equally-spaced samples of the CDF
        that are used by the Romberg integrator.

    Returns
    -------
    mean: float
        The estimated mean
    """
    integral_num = 1 + 2**integral_pow

    x_integral, dx = np.linspace(
        x[0], x[-1], integral_num, endpoint=True, retstep=True
    )

    df = nd.Derivative(cdf_fn, n=1, method="backward")
    dxs = df(x_integral)

    return romb(dxs, dx=dx)


def interpolated_inverse(x, y, num, endpoint=True):
    """Generate samples from an interpolated inverse.

    Parameters
    ----------
    x: ndarray
        Values from the original horizontal axis
    y: ndarray
        Values from the original vertical axis
    num: int
        Number of samples to generate
    endpoint: bool
        If True, `stop` is the last sample. Otherwise, it is not included.

    Returns
    -------
    samples: ndarray
        Values that are draw sequentially over the range of the inverse
    """
    intfunc = interp1d(y, x, kind=1)

    return intfunc(np.linspace(0, 1, num, endpoint=endpoint))


def cumdensityspace(
    start,
    stop,
    cdf_fn,
    num=50,
    interp_num=50,
    endpoint=True,
):
    """Return numbers over an interval spaced by a given CDF.

    Spacing is determined by the gradient of the CDF. Steeper regions will have
    smaller spacing between points while flatter regions will have large
    spacing between points.

    Parameters
    ----------
    start: float
        The starting value of the sequence.
    stop: float
        The end value of the sequence, unless `endpoint` is set to False.
    density_fn: func
        The function specifying the relative density along the axis.
    num: int
        Number of samples to generate
    endpoint: bool
        If True, `stop` is the last sample. Otherwise, it is not included.
    integral_num: int
        The number of points to evaluate the integral of the density with. A
        larger value will yield a more accurate estimate and vice versa.

    Returns
    -------
    samples: ndarray
        The point are `num` equally spaced samples in the closed interval
    """
    xs = np.linspace(start, stop, interp_num)
    cps = cdf_fn(xs)
    cps[-1] = 1

    return interpolated_inverse(xs, cps, num=num, endpoint=endpoint)


def densityspace(
    start,
    stop,
    density_fn,
    num=50,
    interp_num=50,
    endpoint=True,
):
    """Return numbers over an interval spaced by a given density.

    Spacing is determined by the local density. In regions of higher density
    the spacing will be lower, while in areas of low density the spacing
    will be larger.

    Based on https://stackoverflow.com/a/62740029/922745

    Parameters
    ----------
    start: float
        The starting value of the sequence.
    stop: float
        The end value of the sequence, unless `endpoint` is set to False.
    density_fn: func
        The function specifying the relative density along the axis.
    num: int
        Number of samples to generate
    endpoint: bool
        If True, `stop` is the last sample. Otherwise, it is not included.
    integral_num: int
        The number of points to evaluate the integral of the density with. A
        larger value will yield a more accurate estimate and vice versa.

    Returns
    -------
    samples: ndarray
        The point are `num` equally spaced samples in the closed interval
    """
    xs = np.linspace(start, stop, interp_num)
    ps = density_fn(xs)

    cps = cumtrapz(ps, xs, initial=0)
    cps /= cps[-1]

    return interpolated_inverse(xs, cps, num=num, endpoint=endpoint)


class Hyman:
    """A wrapper for the Hyman spline provided by the splines package.

    Attributes
    ----------
    x_min: float
        The minimum value along the x-axis
    x_max: float
        The maximum value along the x-axis
    obj: splines.PiecewiseMonotoneCubic
        An instance of the Hyman spline object that is being wrapped
    """

    def __init__(self, x, y):
        self.x_min = x[0]
        self.x_max = x[-1]
        self.obj = splines.PiecewiseMonotoneCubic(y, grid=x, closed=False)

    def __call__(self, x):
        """Evaluate the spline at x or for every element of x.

        Parameters
        ----------
        x: float, ndarray
            The position or positions to evaluate the spline at.

        Returns
        -------
        est : float, ndarray
            Value of the spline
        """
        return self.obj.evaluate(np.clip(x, self.x_min, self.x_max), 0)


class BinSmooth:
    """A binned data smoother.

    This class implements the method outlined in [1]. It proceeds by fitting a
    cubic spline to the empirical distribution function of the binned data.

    Attributes
    ----------
    min_x_: float
        The minimum value of the empirical distribution function
    tail_: float
        The maximum value of the estimated CDF
    mean_est_: float
        The estimated mean of the estimated distribution
    cdf_cs_: func
        The estimated CDF function
    inv_cdf_cs_: func
        The estimated inverse CDF function

    References
    ----------
    .. [1] P. von Hippel, D. Hunter, M. Drown "Better Estimates from Binned
           Income Data: Interpolated CDFs and Mean-Matching", 2017.

    Examples
    --------

    >>> from binsmooth import BinSmooth

    >>>  bin_edges = np.array([0, 18200, 37000, 87000, 180000])
    >>> counts = np.array([0, 7527, 13797, 75481, 50646, 803])

    >>>  bs = BinSmooth()

    >>> bs.fit(bin_edges, counts)

    >>> print(bs.inv_cdf(0.5))
    70120.071...
    """

    def fit(
        self,
        x,
        y,
        spline_type="PCHIP",
        includes_tail=False,
        tail_method="mean",
        tail_bounds=None,
        m=None,
    ):
        """Fit the cubic spline to the data.

        Parameters
        ----------
        x: ndarray
            The bin edges
        y: ndarray
            The values for each bin
        spline_type: str, default=`PCHIP`
            The type of spline to use. Either:
            - `PCHIP`: use scipy.interpolate.PchipInterpolator;
            - `HYMAN`: use splines.PiecewiseMonotoneCubic;
            - `LINEAR`: use scipy.interpolate.interp1d
        includes_tail: bool, default=False
            If True then it is assumed that the last value in x is the upper
            bound of the distribution, otherwise the upper bound is estimated
        tail_method: str, default=`mean`
            The method used to estimate tail value of the distribution when
            `includes_tail` is False. Either:
            - `mean`: tail is selected so mean of distribution matches the
                given mean value `m`;
            - `auc`: tail is selected so that area under the curve of the PDF
                is 1;
        tail_bounds: tuple, default=None
            Constrain the search of the tail point to this range. Either:
            - `None`: search is unrestricted;
            - `(lb, ub)`: search is limited between lb (lower bound) and
                ub (upper bound);
        m: float, default=None
            The mean of the distribution, used to estimate the tail value when
            `includes_tail` is False and `tail_method` is `mean`.
        tail_slope: float, default=None
            The maximum slope value in the tail when `includes_tail` is False
            and `tail_method` is `slope`.


        Returns
        -------
        self : object
            Fitted estimator.
        """
        spline_function_map = {
            "PCHIP": PchipInterpolator,
            "HYMAN": Hyman,
            "LINEAR": interp1d,
        }
        if spline_type not in spline_function_map:
            raise ValueError(
                "Invalid spline type. Must be one of"
                f" {spline_function_map.keys()}."
            )

        self.f = spline_function_map[spline_type]

        if includes_tail and len(x) != len(y):
            raise ValueError(
                "Length of x and y must match when tail is included."
            )

        if not includes_tail and len(x) != len(y) - 1:
            raise ValueError(
                "Length of x must be N-1 when tail is not included."
            )

        x = x.astype(np.float64)
        y = y.astype(np.float64)

        if y[0] != 0:
            raise ValueError("y must begin with 0.")

        dx = np.diff(x)
        if np.any(dx <= 0):
            raise ValueError("x must be strictly increasing.")

        self.min_x_ = x[0]

        # Check for negative values in y
        if np.any(y < 0):
            raise ValueError("y contains negative values.")

        # Check if last value of y is zero
        # which would cause inverse CDF to fail
        if np.isclose(y[-1], 0):
            warnings.warn(
                "x and y have been trimmed to remove trailing zeros in y."
            )
            y_len = len(y)
            y = np.trim_zeros(y, "b")
            n_trailing_zeros = y_len - len(y)
            x = x[:-n_trailing_zeros]

        y_ecdf = np.cumsum(y)
        y_ecdf_normed = y_ecdf / np.max(y_ecdf)

        if includes_tail:
            self.tail_ = x[-1]
            x_wtail = x
        else:
            # Temporarily set the tail bounds to slightly above x[-1] and
            # largest possible float value
            tail_0 = (
                x[-1] + np.finfo(np.float32).eps
            )  # float64.eps is too small
            tail_1 = np.finfo(np.float64).max
            # Override tail bounds if supplied
            if tail_bounds:
                if len(tail_bounds) != 2:
                    raise ValueError(
                        "tail_bounds must only contain an upper and lower"
                        " bound."
                    )

                if tail_bounds[0]:
                    if tail_bounds[0] <= x[-1]:
                        raise ValueError(
                            "The lower bound of tail_bounds must be greater"
                            " than the last value in x."
                        )
                    tail_0 = tail_bounds[0]

                if tail_bounds[1]:
                    if tail_bounds[1] <= tail_0:
                        raise ValueError(
                            "The upper bound of tail_bounds must be greater"
                            " the lower bound."
                        )
                    tail_1 = tail_bounds[1]

            x_wtail = np.concatenate([x, [tail_0]])

            if tail_method == "mean":
                if m is None:
                    raise ValueError(
                        "A value for m must be provided when tail_method is"
                        " 'mean'."
                    )

                estimator = estimate_mean
                target = m

            elif tail_method == "auc":
                if m:
                    warnings.warn(
                        "A value for the mean has been provided. If this value"
                        " is the true mean then using the 'mean' tail_method"
                        " is preferred over 'auc'."
                    )

                estimator = estimate_auc
                target = 1
            else:
                raise ValueError("Invalid tail_method selected.")

            def evaluate_tail(tail, x, y, cdf_f, estimator, target):
                # Use current tail guess
                x[-1] = tail
                # Fit spline
                cdf = cdf_f(x, y)
                # Estimate mean
                est_val = estimator(x, cdf)
                # Calculate loss
                return (target - est_val) ** 2

            self.tail_ = minimize(
                evaluate_tail,
                tail_0,
                args=(
                    x_wtail.copy(),
                    y_ecdf_normed,
                    self.f,
                    estimator,
                    target,
                ),
                bounds=[(tail_0, tail_1)],
                method="Nelder-Mead",
            ).x[0]

            x_wtail[-1] = self.tail_

        # Estimate the CDF by fitting a spline
        spline_fit_param_map = {
            "PCHIP": {},
            "HYMAN": {},
            "LINEAR": dict(
                fill_value=(0, y_ecdf_normed[-1]), bounds_error=False
            ),
        }
        self.cdf_cs_ = self.f(
            x_wtail, y_ecdf_normed, **spline_fit_param_map[spline_type]
        )

        self.mean_est_ = estimate_mean(x_wtail, self.cdf_cs_)

        # Approximate inverse CDF by sampling the CDF
        x_cs = cumdensityspace(
            self.min_x_, self.tail_, self.cdf_cs_, interp_num=1000
        )
        y_cs = self.cdf_cs_(x_cs)

        self.inv_cdf_cs_ = interp1d(y_cs, x_cs)

        return self

    def pdf(self, x):
        """Estimated PDF.

        Parameters
        ----------
        x: ndarray
            Values to calculate the PDF of.

        Returns
        -------
        pdf : ndarray
            Estimated PDF values
        """

        def f(x):
            # Compute foward derivative below last bin
            # Otherwise compute backward
            if x <= self.x[-1]:
                method = "backward"
            else:
                method = "forward"

            df = nd.Derivative(self.cdf, n=1, method=method)

            return df(np.clip(x, self.min_x_, self.tail_))

        return np.clip(np.vectorize(f)(x), 0, None)

    def cdf(self, x):
        """Estimated CDF.

        Parameters
        ----------
        x: ndarray
            Values to calculate the CDF of.

        Returns
        -------
        cdf : ndarray
            Estimated CDF values
        """
        return np.clip(self.cdf_cs_(np.clip(x, self.min_x_, self.tail_)), 0, 1)

    def inv_cdf(self, percentile):
        """Estimated inverse CDF.

        Parameters
        ----------
        percentile: ndarray
            Values to calculate the inverse CDF of

        Returns
        -------
        inverse_cdf : ndarray
            Estimated inverse CDF values
        """
        return self.inv_cdf_cs_(np.clip(percentile, 0, 1))
