# binsmooth

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

## Improvements

**Better tail estimate** by using scipy's fmin to perform automatic optimisation
rather than the adhoc search method found in the R implementation.

**More precise inverse CDF** by dynamically sampling the CDF. This is done
by sampling more densely in areas where the CDF is steeper and less in flatter
areas, rather than evenly spaced sampling.

## Warnings

**Results** will be different to the original R implementation due to
differences in spline implementation between R's splinefun and scipy's
PchipInterpolator.

**Accuracy** is highly dependent on the mean of the distribution. If you do
not supply a mean, then one will be estimated in an adhoc manner and the accuracy
of estimates may be poor.

[1]: https://sociologicalscience.com/download/vol-4/november/SocSci_v4_641to655.pdf
[2]: https://cran.r-project.org/web/packages/binsmooth/binsmooth.pdf
