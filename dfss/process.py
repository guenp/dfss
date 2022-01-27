from typing import Iterable
import numpy as np
import scipy.stats as stats
import pandas as pd

def calculate_d2(n: int) -> float:
    """Calculate the expectation value (d2) of the sample range n
    of a normal population with standard deviation = 1.

    :param n: Number of samples
    :type n: int
    :return: d2
    :rtype: float
    """
    standard_normal = stats.norm()
    approx_x, dx = np.linspace(-20.0, 20.0, 1001, retstep=True)
    sample_vals = (
        1.0
        - (1.0 - standard_normal.cdf(approx_x)) ** n
        - standard_normal.cdf(approx_x) ** n
    )
    return np.trapz(sample_vals, dx=dx)

def calculate_process(samples: Iterable[float], LSL: float, USL: float, window: int=8) -> pd.Series:
    """Calculate process capability values such as Cpk and Ppk for a given set of samples

    :param samples: Samples or measurements
    :type samples: Iterable[float]
    :param LSL: Lower spec limit
    :type LSL: float
    :param USL: Upper spec limit
    :type USL: float
    :param window: Rolling window size, defaults to 8
    :type window: int, optional
    :return: Process capability values
    :rtype: pd.Series
    """
    mu = samples.mean()
    std = samples.std()
    r = samples.rolling(window)
    Rbar = r.max() - r.min()
    std_within = (Rbar/calculate_d2(window)).mean()

    ppk = np.min([
        (USL-mu)/(3*std),
        (mu-LSL)/(3*std)
    ])
    
    cpk = np.min([
        (USL-mu)/(3*std_within),
        (mu-LSL)/(3*std_within)
    ])

    d = {
        "sample size": len(samples),
        "mean": mu,
        "std": std,
        "std_within": std_within,
        "ppk": ppk,
        "cpk": cpk,
        "min": samples.min(),
        "max": samples.max()
    }
    return pd.Series(d, index=['sample size', 'mean', 'std', 'std_within', 'ppk', 'cpk', 'min', 'max'])

def calculate_p_value(mean: float, std: float, samples: Iterable[float]) -> float:
    """Calculate the p value of a given set of samples

    :param mean: Mean value of dataset
    :type mean: float
    :param std: Standard deviation of dataset
    :type std: float
    :param samples: Samples or measurements
    :type samples: Iterable[float]
    :return: Mean P value
    :rtype: float
    """
    normal = stats.norm(loc=mean, scale=std)
    p_value = np.mean(
        [2. * min([1 - normal.cdf(_s), normal.cdf(_s)]) for _s in samples]
    )
    return p_value
