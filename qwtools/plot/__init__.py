"""
Module for timeseries analysis.
"""
import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.tsa.api as smt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_mean
"""
Return means for each period in x. freq is an int that gives the
number of periods per cycle. E.g., 12 for monthly. NaNs are ignored
in the mean.

return np.array([pd_nanmean(x[i::freq], axis=0) for i in range(freq)])
"""
def qc_plot(df, q_col, c_col,
            q_label=None,
            c_label=None,
            normalized=False, 
            ax=None):
    """Plot flow versus concentration.

    Parameters
    ----------
    df : DataFrame

    q : string
        Flow column
    c : string
        Concentration column name
    """
    if q_label is None:
        q_label = q_col
    
    if c_label is None:
        c_label = c_col

    if ax is not None:
        plt.sca(ax)
        fig = plt.gca().figure
    else:
        fig, ax = plt.subplots()

    df = df.copy().dropna(subset=[q_col,c_col])

    ax.plot(df[q_col], df[c_col], lw=0.3, color='k', alpha=0.8, zorder=1)
    cs = ax.scatter(df[q_col], df[c_col] , c=df.index, cmap='magma', alpha=1,
        zorder=2)

    ax.set_xlabel(q_label)
    ax.set_ylabel(c_label)
    cbar = fig.colorbar(cs)
    cbar.set_label(r'Time $\longrightarrow$')
    cbar.set_ticks([])

    plt.sci(cs)
    return

def ts_plot(y, lags=None, title=''):
    '''
    Calculate acf, pacf, histogram, and qq plot for a given time series.

	Use ts_plot to quickly evaluate statistical and distributional phenomena
	of a given time-series process.

	Parameters
	----------

	Source
	------
	http://www.blackarbs.com/blog/time-series-analysis-in-python-linear-models-to-garch/11/1/2016
    '''
    # if time series is not a Series object, make it so
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # initialize figure and axes
    fig = plt.figure(figsize=(14, 12))
    layout = (3, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    qq_ax = plt.subplot2grid(layout, (2, 0))
    hist_ax = plt.subplot2grid(layout, (2, 1))

    # time series plot
    y.plot(ax=ts_ax)
    plt.legend(loc='best')
    ts_ax.set_title(title);

    # acf and pacf
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5) 

    # qq plot
    sm.qqplot(y, line='s', ax=qq_ax)
    qq_ax.set_title('Normal QQ Plot')

    # hist plot
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    plt.tight_layout()
    plt.show()
    return
