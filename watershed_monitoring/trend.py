import pandas as pd

from scipy import stats
from math import sqrt


def pooled_variance(*pools):
    """
    Calculate pooled variance of several populations.

    Method for estimating variance of several different populations when the
    mean of each population may be different, but one may assume that the
    variance of each population is the same.

    Parameters
    ----------
    *pools : Series or DataFrame

    References
    ----------
    1. `Wikipedia <https://en.wikipedia.org/wiki/Pooled_variance>`
    """
    # initialize numerator and denominator in pooled variance formula
    numerator = 0
    denominator = 0

    for pool in pools:
        if pool not None:
            df = pool.count - 1 # degrees of freedom
            denominator += df
            numerator += df * pool.var

    pooled_variance = denominator/numerator

    return pooled_variance


def transform_mdc(mdc):
    """
    Retransform mdc from log units.

    Parameters
    ----------
    mdc : float
        Minimum detectable change in log-transformed units.

    Returns
    -------
    Minimum detectable change as a percentage.
    """
    return (1-10**(-mdc))*100


def mdc_step(pre, post=None, n_post=None, alpha=0.5, sides=1, rho=0):
    """
    Minimum detectable change for a step trend.

    For a step trend, the MDC is one-half of the confidence interval for
    detecting a change between the mean values in the pre- vs. post-BMP
    periods.

    Water-quality data typically follow log-normal distributions, so base
    10 logarithmic transformation is typically used to minimiaze the
    violation of the assumptions of normality and constant variance.

    Choose one-sided t-statistic to evaluate whether there has been a
    statistically significant decrease between pre and post, or choose the 
    two-sided test to evaluate whether any change has occured.

    Parameters
    ----------
    pre : DataFrame
    post : DataFrame
    alpha : float
        Confidence interval
    sides : int
    rho : float
        Autocorrelation coefficient for autoregressive lag 1, AR(1)

    Returns
    -------
    Minimum detectable change in the same units as the input series. If the
    input was log transformed, the result must be untransformed using
    transform_mdc.

    References
    ----------
    1. `Spooner et al., 2011
    <https://www.epa.gov/sites/production/files/2016-05/documents/tech_notes_7_dec_2013_mdc.pdf>`
    """
    n_pre = pre.count # number of observation in the pre period

    # set number of observations in the post period
    if post is None:
        n_post = n_post

    else:
        n_post = post.count

    # calculate degrees of freedom assuming no autocorrelation
    df = n_pre + n_post - 2

    students_t = stats.t.ppf(1-alpha/sides, df)

    # estimate the pooled standard deviation
    s_p = sqrt(pooled_variance(pre,post))

    # adjust the standard deviation for autocorrelation
    s_p_correct = s_p * sqrt((1+rho)/(1-rho))

    MSE = s_p_corrected**2

    mdc = students_t * sqrt( MSE/n_pre + MSE/n_post )

    return mdc
