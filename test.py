import numpy as np


def gaussian(size, sig, max_val=1):
    mu = 0
    x = np.linspace(mu - 0.15, 2, size)
    dist = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) * max_val

    return dist

ggg = gaussian(5, 1)

print(ggg, sum(ggg))