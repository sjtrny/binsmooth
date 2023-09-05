# binsmooth

[![PyPI version](https://badge.fury.io/py/binsmooth.svg)](https://badge.fury.io/py/binsmooth)
![Build Status](https://github.com/sjtrny/binsmooth/actions/workflows/ci.yml/badge.svg)

Python implementation of "[Better Estimates from Binned Income Data][1]"

	Better Estimates from Binned Income Data: Interpolated CDFs and Mean-Matching
	Paul T. von Hippel, David J. Hunter, McKalie Drown
	Sociological Science
	Volume 4, Number 26, Pages 641-655
	2017

Originally implemented in the R package [`binsmooth`][2].

## Usage

    from binsmooth import BinSmooth

    bin_edges = np.array([0, 18200, 37000, 87000, 180000])
    counts = np.array([0, 7527, 13797, 75481, 50646, 803])

    bs = BinSmooth()
    bs.fit(bin_edges, counts)

    # Print median estimate
    print(bs.inv_cdf(0.5))

## Installation

Install via pip

    pip install binsmooth

pypi page [https://pypi.org/project/binsmooth/](https://pypi.org/project/binsmooth/)

## Improvements

**Better tail estimate** by bounded optimisation rather than the adhoc search
method found in the R implementation.

**More precise inverse CDF** by dynamically sampling the CDF. This is done
by sampling proportional to the steepness of the CDF i.e. sampling more
in areas where the CDF is steeper.

## Warnings

**Results** may not exactly match R `binsmooth` because of a different approach
to estimating the tail (upper bound).

Furthermore the `fit` method uses `spline_type="HYMAN"` as a default, which is
scipy's `PchipInterpolator` \[1\]. While the R implementation uses the spline
from \[2\], which can be mimiced by setting `spline_type="HYMAN"`.

## References

\[1\]: Fritsch, F. N. and Carlson, R. E. (1980). [Monotone piecewise cubic interpolation][3]. SIAM Journal on Numerical Analysis  
\[2\]: Hyman, J. M. (1983). [Accurate monotonicity preserving cubic interpolation][4]. SIAM Journal on Scientific and Statistical Computing

[1]: https://sociologicalscience.com/download/vol-4/november/SocSci_v4_641to655.pdf
[2]: https://cran.r-project.org/web/packages/binsmooth/
[3]: http://www.ams.sunysb.edu/~jiao/teaching/ams527_spring13/lectures/SNA000238.pdf
[4]: https://www.osti.gov/servlets/purl/5328033
