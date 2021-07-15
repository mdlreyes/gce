"""
plot_sfh.py

Makes plot of SFH
"""

import numpy as np
from tqdm import tqdm
from numpy.random import default_rng
rng = default_rng()

# Backend for matplotlib on mahler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Import other modules
from astropy.io import ascii  # only needed for SFH test stuff
from astropy.cosmology import FlatLambdaCDM  # needed to compute redshifts
cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)  # using Planck (2018) params
import astropy.units as u
from astropy.cosmology import z_at_value
import gce_fast as gce

# Do some formatting stuff with matplotlib
from matplotlib import rc
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, MaxNLocator
rc('font', family='serif')
rc('axes', labelsize=14) 
rc('xtick', labelsize=12)
rc('ytick', labelsize=12)
rc('xtick.major', size=10)
rc('ytick.major', size=10)
rc('legend', fontsize=12, frameon=False)
rc('text',usetex=True)
rc('xtick',direction='in')
rc('ytick',direction='in')

# Some other stuff to make cool colors
import cycler
import cmasher as cmr

def plotsfh(cumulativeSFH=True, chainfile=None, plot_path='plots/', burnin=5000, Niter=100):
    """Generates plots: [X/Fe] vs [Fe/H], [X/Fe] vs [time], model parameters vs time

    Args:
        cumulativeSFH (bool): if True (default), plot cumulative SFH
        chainfile (str): if not None, use chain file to plot errors 
        plot_path (str): path to store plots
        burnin (int): number of steps to burn in
        Niter (int): number of samples to use when computing errors
    """
    # Plot SFH
    cwheelsize = 3
    color = cmr.bubblegum(np.linspace(0,1,cwheelsize,endpoint=True))
    matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
    cwheel = [np.array(matplotlib.rcParams['axes.prop_cycle'])[x]['color'] for x in range(cwheelsize)]
    if cumulativeSFH:
        fig = plt.figure(figsize=(8,6))
    else:
        fig = plt.figure(figsize=(8,2))
    ax = plt.subplot()
    #plt.title(title, fontsize=16)
    plt.xlabel('Lookback time (Gyr)')
    plt.setp([a.minorticks_on() for a in fig.axes[:]])
    if cumulativeSFH:
        plt.ylabel('Cumulative SFH')
        plt.ylim([0,1.])
    else:
        plt.ylabel('SFR ($10^{-4}$ M$_\odot$ yr$^{-1})$')
        plt.ylim([0,30])
        plt.yticks([0, 10, 20, 30])
    #plt.xlim([0,1.7])
    #plt.xticks([0.0,0.5,1.0,1.5])
    #plt.suptitle("Final mass: %.1f x $10^6$ M$_\odot$"%(model['mgal'][-1]/1e6), y=0.97)

    # Add redshift axis on top
    ages = np.array([13, 9, 6, 4, 3, 2, 1.5, 1.2, 1, 0.8, 0.6])*u.Gyr
    ageticks = [13.791 - z_at_value(cosmo.age, age) for age in ages]
    ax2 = ax.twiny()
    ax2.set_xticks(ageticks)
    ax2.set_xticklabels(['{:g}'.format(age) for age in ages.value])
    zmin, zmax = 14.0, 0
    ax.set_xlim(zmin, zmax)
    ax2.set_xlim(zmin, zmax)

    # Get samples from file
    chain = np.load(chainfile)
    nwalkers, nsteps, ndim = np.shape(chain)
    samples = chain[:,burnin:, :].reshape((-1, ndim))
    print(samples.shape)

    sfrs = np.zeros((Niter,1150))
    finalt = np.arange(0,1.150,0.001)
    idxs = rng.choice(len(samples[:,0]),size=Niter)
    print(idxs)
    for i, idx in tqdm(enumerate(idxs)):
        currentmodel, _, _ = gce.runmodel(samples[idx,:], plot=False, title="Sculptor dSph", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False)
        if len(currentmodel['mdot']) > 1150:
            sfrs[i,:] = currentmodel['mdot'][:1150]
        else:
            sfrs[i,:len(currentmodel['mdot'])] = currentmodel['mdot']

    # Plot SFH from 50th percentile model
    finalsfr = np.percentile(sfrs,50,axis=0)/1e5 # 1e-4 Mdot/yr
    finalsfr_lo = np.percentile(sfrs,16,axis=0)/1e5
    finalsfr_hi = np.percentile(sfrs,84,axis=0)/1e5

    print(finalsfr, finalsfr_lo, finalsfr_hi)

    handles, labels = [], []
    if cumulativeSFH:
        # Compute cumulative SFH
        cumsfr = np.cumsum(finalsfr)/np.nansum(finalsfr) # Compute normalized SFH
        for i in range(len(idxs)):
            cumsfr_current = np.cumsum(sfrs[i,:])/np.nansum(sfrs[i,:])
            plt.plot(13.791 - finalt - 0.5, cumsfr_current, ls='-', lw=0.5, color='k', alpha=0.1)
        #cumsfr_lo = np.cumsum(finalsfr_lo)/np.nansum(finalsfr_lo)
        #cumsfr_hi = np.cumsum(finalsfr_hi)/np.nansum(finalsfr_hi)
        p1, = plt.plot(13.791 - finalt - 0.5, cumsfr, ls='-', lw=2, color='k')
        #p1a = plt.fill_between(13.791 - finalt, cumsfr_lo, cumsfr_hi, color='k', alpha=0.3)
    else:
        p1, = plt.plot(13.791 - finalt, finalsfr, ls='-', lw=2, color='k')
        p1a = plt.fill_between(13.791 - finalt, finalsfr_lo, finalsfr_hi, color='k', alpha=0.3)
    #handles.append((p1,p1a))
    handles.append(p1)
    labels.append('This work')

    # Also from other models
    bettinelli = ascii.read('data/sfhtest/bettinelli19.dat')
    bettinelli = [bettinelli['Lookback time (Gyr)'], bettinelli['SFR (1e-4 Msun/y)']]
    deboer = ascii.read('data/sfhtest/deboer12.dat')
    deboer = [deboer['Age (Gyr)'], deboer['SFR (1e-4 Msun/y)']]
    titles = ['Bettinelli et al. (2019)', 'de Boer et al. (2012)']
    linestyles = ['--', ':']
    if cumulativeSFH:
        #bettinelli[0] = np.concatenate(([13.4],bettinelli[0]))
        bettinelli[1] = np.cumsum(bettinelli[1])/np.nansum(bettinelli[1])
        deboer[1] = np.cumsum(deboer[1])/np.nansum(deboer[1])
    for idx, data in enumerate([bettinelli, deboer]):
        p2, = plt.plot(data[0], data[1], ls=linestyles[idx], lw=2, color=plt.cm.Set2(idx))
        handles.append(p2)
        labels.append(titles[idx])

    # Add Weisz+14 SFH
    weisz = ascii.read('data/sfhtest/weisz14.dat')
    scl_idx = np.where(weisz['ID']=='Sculptor')
    colnames = [name for name in weisz.colnames if name.startswith('f')]
    t = [(10.**10.15)/1e9]+[10.**float(name[1:])/1e9 for name in colnames] # Lookback in Gyr
    cumsfh = np.asarray([0]+[float(weisz[name][scl_idx]) for name in colnames])
    cumsfh_uplim = np.asarray([0]+[float(weisz['Ut'+name][scl_idx]) for name in colnames])
    cumsfh_lolim = np.asarray([0]+[float(weisz['Lt'+name][scl_idx]) for name in colnames])
    if cumulativeSFH:
        p3, = plt.plot(t, cumsfh, ls='dashdot', lw=2, color=plt.cm.Set2(2))
        p4 = plt.fill_between(t, cumsfh-cumsfh_lolim, cumsfh+cumsfh_uplim, color=plt.cm.Set2(2), alpha=0.3)
        handles.append((p3,p4))
        labels.append('Weisz et al. (2014)')

    plt.legend(handles=handles, labels=labels, loc='best')
    if cumulativeSFH:
        plt.savefig((plot_path+'cumsfh.png').replace(' ',''), bbox_inches='tight')
    else:
        plt.savefig((plot_path+'sfh.png').replace(' ',''), bbox_inches='tight')
    plt.show()

    return

if __name__=="__main__":
    plotsfh(chainfile='output/empiricaltest_bothba_ba.npy', burnin=10000, Niter=500, cumulativeSFH=True)