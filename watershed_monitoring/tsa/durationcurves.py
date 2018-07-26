import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import rankdata

def label_dc_flow_conditions(ax):
    """Label flow ranges on a duration curve.
    """
    label_y = ax.get_ylim()[1]*.95
    fontsize=6
    conditions = {
        # Label : Range
        'High\nFlows' : [0,10],
        'Moist\nConditions': [10,40],
        'Mid-range\nFlows' : [40,60],
        'Dry\nConditions': [60,90],
        'Low\nFlows': [90,100]
    }

    for key in conditions.keys():
        plt.axvline(conditions[key][1], color='black')
        label_x = np.mean(conditions[key])

        ax.text(label_x, label_y, key,
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=fontsize,
                style='italic')

def load_mean_concentration(concentration, discharge):
    """Calculate the mean concentration of a load.

    Parameters
    ----------
    discharge : array_like
    concentration : array_like
    """
    return (discharge * concentration).mean()/discharge.mean()

def target_ldc(concentration, discharge, conversion_factor, ax=None):
    """
    concentration : float
        Target concentration for watershed.
    discharge : array_like
    conversion_factor : float
        0.002695 * cfs * mg/L = tons/day
    """
    discharge = np.sort(discharge)
    rank = rankdata(discharge)
    target_load = discharge * concentration * conversion_factor
    prob = (1-(rank/(len(rank)+1)))*100
    if ax is None:
        fig, ax = plt.subplots(1)

    ax.plot(prob, target_load, color='black', linestyle=':')

    #ax.set_yscale('log')


def ldc(concentration, discharge,
        conversion_factor,
        target_concentration=None,
        y_label='Load',
        yscale='log',
        ylim=None,
        ax=None,
        label_conditions=False):
    """Plot a load duration curve.

    TODO: conversion factor

    Paramters
    ---------
    concentration : array_like

    disharge : array_like

    conversion_factor : float
        0.002695 * cfs * mg/L = tons/day

    target_concentration : float
        Target concentration of watershed. If None, defaults to mean
        conentration of load.


    ax : axes

    label_conditions : bool, optional, default is *False*
        if label_conditions is *True*, label the flow conditions.

    yscale : {'linear', 'log'}, default is 'log'
        'log' or 'linear'

    units : string
        Unit of flow, e.g. cfs.

    """
    rank = rankdata(discharge)
    # calculate probability of each rank
    prob = (1-(rank/(len(rank)+1)))*100
    load = concentration * discharge * conversion_factor

    if ax is None:
        fig, ax = plt.subplots(1)

    # plot target curve
    if target_concentration is None:
        target_concentration = load_mean_concentration(concentration, discharge)

    hb_extent = None
    if ylim is not None:
        plot_ylim = ylim
        hb_extent=(1,100, ylim[0], ylim[1])

        if yscale is 'log':
            plot_ylim = (10**ylim[0], 10**ylim[1])



    # plot data via matplotlib
    hb = ax.hexbin(prob, load, gridsize=50,
                   yscale=yscale,
                   cmap='pink_r',
                   mincnt=1,
                   extent=hb_extent,
                   linewidths=0.2)

    target_ldc(target_concentration, discharge, conversion_factor, ax=ax)

    cb = plt.colorbar(hb, ax=ax, orientation='horizontal')
    cb.set_label('Count')
    #ax.set_yscale('log')
    ax.set_ylabel(y_label)
    ax.set_xlabel('Flow Duration Interval (%)')
    ax.grid(alpha=0.2, which='both', linestyle='--', linewidth=0.5)
    #ax.grid(alpha=0.2, which='major', linestyle='--', linewidth=0.5)
    ax.set_xlim(0,100)

    if ylim is not None:
        ax.set_ylim(plot_ylim)

    if label_conditions==True:
        label_dc_flow_conditions(ax)



def fdc(discharge, ax=None, label_conditions=False, flow_unit='cfs'):
    """Plot flow duration curve.

    Paramters
    ---------
    discharge : array_like
    ax : axes
    label_conditions : bool, optional, default is *False*
    units : str, default is 'cfs'
        unit of flow
    """
    data = np.sort(discharge)
    rank = rankdata(data, method='average')
    prob = (1 - rank/(len(rank)+1))*100

    if ax is None:
        fig, ax = plt.subplots(1)

    # plot data via matplotlib

    #plot load obs, consider heatmap
    ax.plot(prob,data)
    ax.set_yscale('log')
    ax.set_ylabel('Discharge in {}'.format(flow_unit))
    ax.set_xlabel('Flow Duration Interval (%)')
    ax.grid(alpha=0.2, which='both', linestyle='--', linewidth=0.5)
    #ax.grid(alpha=0.2, which='major', linestyle='--', linewidth=0.5)
    ax.set_xlim(0,100)

    if label_conditions==True:
        label_dc_flow_conditions(ax)
