import numpy as np

from binsmooth import BinSmooth

# Income of all Secondary School Teachers in Australia
# Table 14A https://data.gov.au/data/dataset/taxation-statistics-2016-17
bin_edges = np.array([0, 18200, 37000, 87000, 180000])
counts = np.array([0, 7527, 13797, 75481, 50646, 803])

bs = BinSmooth()

bs.fit(bin_edges, counts)

# Print median estimate
print(bs.inv_cdf([0.5, 0.8]))
