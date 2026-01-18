import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    # Write code here
    cdf = 0.0
    for i in range(k + 1):
        pdf = comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        cdf += pdf
    return pdf, cdf
