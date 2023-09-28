from binsmooth import BinSmooth
import numpy as np
from scipy import stats

np.random.seed(0)

mu = 5
sigma = 10
s = np.sqrt(np.log(1 + (sigma/mu)**2))
scale = mu / np.sqrt(1 + (sigma/mu)**2)
dist = stats.lognorm(s=s, scale=scale)
sample_size = 10000

sample = dist.rvs(sample_size)
percentiles = np.linspace(0, 100, 11)  # +1 to get the right number of bin edges
bin_edges = np.percentile(sample, percentiles)
hist, bin_edges = np.histogram(sample, bins=bin_edges) # get binned data
counts = [0] + hist.tolist()
counts_l = counts
counts = np.array(counts)


bin_edges[0] = 0
# print(bin_edges)

bs = BinSmooth()
bs.fit(bin_edges[:-1], counts, includes_tail=False, spline_type='PCHIP', tail_method='auc')
# bs.fit(bin_edges[:-1], counts, m=mu, includes_tail=False, spline_type='LINEAR', tail_method='auc')


# Print median estimate
print(bs.cdf(0.5))
print(bs.tail_, np.max(sample))
