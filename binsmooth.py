import numpy as np
from scipy.integrate import cumtrapz, trapz
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.optimize import fmin


def estimate_mean(lb, ub, cdf_fn, integral_num=50):
    x_integral = np.linspace(lb, ub, integral_num)

    return ub - trapz(cdf_fn(x_integral), x_integral)


def loss(mean_t, mean_e):
    return (mean_t - mean_e) ** 2


def optim(tail, x, y, mean):
    # Use current tail guess
    x[-1] = tail
    # Fit spline
    cdf = PchipInterpolator(x, y)
    # Estimate mean
    est_mean = estimate_mean(x[0], x[-1], cdf)
    # Calculate loss
    return loss(mean, est_mean)


def densityspace(
    start,
    stop,
    density_fn,
    num=50,
    normalize=True,
    endpoint=True,
    integral_num=50,
):
    # Based on https://stackoverflow.com/a/62740029/922745

    xs = np.linspace(start, stop, integral_num)
    ps = density_fn(xs)

    cps = cumtrapz(ps, xs, initial=0)
    cps /= cps[-1]

    intfunc = interp1d(cps, xs, kind=1)

    if normalize:
        start = 0
        stop = 1

    return intfunc(np.linspace(start, stop, num, endpoint=endpoint))


class BinSmooth:
    def fit(self, x, y, m=None, includes_tail=False):

        if includes_tail and len(x) != len(y):
            raise ValueError(
                "Length of x and y must match when tail is included"
            )

        if not includes_tail and len(x) != len(y) - 1:
            raise ValueError(
                "Length of x must be N-1 when tail is not included"
            )

        if m is None:
            # Adhoc mean estimate if none supplied
            if includes_tail:
                bin_edges = x
            else:
                bin_edges = np.concatenate([x[:-1] / 2, [x[-1], x[-1] * 2]])

            m = np.average(bin_edges, weights=y / np.sum(y),)

        self.min_x_ = x[0]

        y_ecdf = np.cumsum(y)
        y_ecdf_normed = y_ecdf / np.max(y_ecdf)

        if includes_tail is False:
            # Temporarily set the tail value
            tail_0 = x[-1] * 2
            x_wtail = np.concatenate([x, [tail_0]])

            # Search for a tail
            self.tail_ = fmin(
                optim,
                tail_0,
                args=(x_wtail.copy(), y_ecdf_normed, m),
                maxiter=16,
                disp=False,
            )[0]

            x_wtail[-1] = self.tail_
        else:
            self.tail_ = x[-1]
            x_wtail = x

        # Estimate the CDF by fitting a spline
        self.cdf_cs_ = PchipInterpolator(x_wtail, y_ecdf_normed)

        self.mean_est_ = estimate_mean(x_wtail[0], x_wtail[-1], self.cdf_cs_)

        # Approximate inverse CDF by sampling the CDF
        # Sample with higher density in steeper areas of the CDF
        # and lower density in flatter areas of the CDF
        # Density is given by the derivative of the CDF
        x_cs = densityspace(self.min_x_, self.tail_, self.cdf_cs_.derivative())
        y_cs = self.cdf_cs_(x_cs)
        self.inv_cdf_cs_ = PchipInterpolator(y_cs, x_cs)

        return self

    def pdf(self, x_val):
        return self.cdf_cs_.derivative()(
            np.clip(x_val, self.min_x_, self.tail_)
        )

    def cdf(self, x_val):
        return self.cdf_cs_(np.clip(x_val, self.min_x_, self.tail_))

    def inv_cdf(self, percentile):
        return self.inv_cdf_cs_(np.clip(percentile, 0, 1))
