"""
gce_plot.py

Makes plots (stored in plots/ folder)
"""

import numpy as np
import scipy.ndimage
import sys

# Backend for matplotlib on mahler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Import other modules
from getdata import getdata

# Systematic errors
# Fe/H and alpha/Fe calculated by Evan on 12/28/17
# C/Fe from Kirby+15, Ba/Fe from Duggan+18, Mn/Fe from de los Reyes+20
syserr = {'Fe':0.10103081, 'alpha':0.084143983, 'Mg':0.076933658,
        'Si':0.099193360, 'Ca':0.11088295, 'Ti':0.10586739,
        'C':0.100, 'Ba':0.100, 'Mn':0.100}

accuracy_cutoff = 0.28 #dex cut for alpha/Fe error, Ba/Fe error, and Mg/Fe error

# Do some formatting stuff with matplotlib
from matplotlib import rc
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, MaxNLocator
rc('font', family='serif')
rc('axes', labelsize=14) 
rc('xtick', labelsize=12)
rc('ytick', labelsize=12)
rc('xtick.major', size=12)
rc('ytick.major', size=12)
rc('legend', fontsize=12, frameon=False)
rc('text',usetex=True)
rc('xtick',direction='in')
rc('ytick',direction='in')

def makeplots(model, atomic, title, plot=False, dsph='Scl', skip_end_dots=-1, NSM=False, plot_path='plots/',
            abunds=True, time=True, params=True): 
    """Generates plots: [X/Fe] vs [Fe/H], [X/Fe] vs [time], model parameters vs time

    Args:
        model (array): Output array from gce_fast
        atomic (array): Output SN_yield['atomic'] array from gce_fast
        title (str): Title of plots
        plot (bool): If 'True', show plots
        dsph (str): Name of galaxy to get observed data (options: 'Scl')
        skip_end_dots (int): Remove last n dots (useful if model becomes erratic)
        plot_path (str): Path to store plots
        abunds, time, params (bool): Types of plots
    """

    single_plot = False

    # Remove last dots if needed
    model = model[:skip_end_dots]
    
    # Get list of atomic numbers
    plot_atomic = atomic

    # Create labels
    elem_names = {1:'H', 2:'He', 6:'C', 8:'O', 12:'Mg', 14:'Si', 20:'Ca', 22:'Ti', 25:'Mn', 26:'Fe', 56:'Ba'}
    snindex = {}
    labels = []
    labels_h = []
    for idx, elem in enumerate(atomic):
        if elem not in [26, 1, 2, 22]:
            labels.append(elem_names[elem])
        if elem != 1:
            labels_h.append(elem_names[elem])
        snindex[elem_names[elem]] = idx

    # Open observed data
    elem_data, delem_data = getdata(galaxy='Scl', c=True, ba=True, mn=True)
    obs_idx = {'Fe':0, 'Mg':1, 'Si':2, 'Ca':3, 'C':4, 'Ba':5, 'Mn':6}  # Maps content of observed elem_data to index

    print(elem_names, snindex, labels, obs_idx)

    if abunds:
        # Plot abundances vs Fe
        fig, axs = plt.subplots(len(labels)+1, figsize=(5,11),sharex=True)
        fig.subplots_adjust(bottom=0.06,top=0.97, left = 0.2,wspace=0.29,hspace=0)
        plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune=None)) for a in [fig.axes[0]]])
        plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='upper')) for a in fig.axes[1:]])
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        plt.setp([a.minorticks_on() for a in fig.axes[:]])
        axs = axs.ravel()

        # Get [Fe/H] info from model to plot MDF
        feh = model['eps'][:,snindex['Fe']]
        feh_step = 0.1
        feh_plot = np.arange(-3.5,feh_step,feh_step)
        mdot = []
        for i in range(len(feh_plot)):
            # Add up all star formation that happened in this [Fe/H] step
            mask = np.where((feh_plot[i] - feh_step/2. < feh) & (feh < feh_plot[i] + feh_step/2.))[0]
            mdot.append(np.sum(model['mdot'][mask]))
        mdot = np.array(mdot)
        gauss_sigma = 1

        # Get [Fe/H] observed data to plot
        feh_obs = elem_data[0,:]
        obsmask = np.where((feh_obs > -3.5) & (feh_obs < 0.))[0]
        feh_obs = feh_obs[obsmask]

        # In top panel, plot metallicity distribution function
        axs[0].plot(feh_plot,scipy.ndimage.filters.gaussian_filter(mdot/np.sum(mdot)*len(feh_obs),gauss_sigma),'k-')
        axs[0].hist(feh_obs, bins=20)
        axs[0].set_ylabel('dN/d[Fe/H]') 
        
        # Plot other elements as a function of [Fe/H]
        for i, label in enumerate(labels):

            # Plot observed [X/Fe]
            if label in obs_idx:
                obs_data = elem_data[obs_idx[label],obsmask]
                obs_errs = np.sqrt(delem_data[obs_idx[label],obsmask]**2. + syserr['Fe']**2. + syserr[label]**2.)
                goodidx = np.where((feh_obs > -990) & (obs_data > -990))[0]
                axs[i+1].scatter(feh_obs[goodidx], obs_data[goodidx], c='r', s=1/(obs_errs[goodidx]), alpha=0.5)
            
            # Plot model [X/Fe]
            modeldata = model['eps'][:,snindex[label]] - model['eps'][:,snindex['Fe']]
            if i == 4:
                # Add 0.2 to [Mg/Fe]
                modeldata += 0.2
            axs[i+1].plot(feh,modeldata,'k.-')
            
            # Add title and labels
            if i == 0:
                axs[0].set_title(title)
            axs[i+1].set_ylabel('['+label+'/Fe]')
            axs[i+1].plot([-6,0],[0,0],':k')

            # Set different limits/ticks for H and He
            if label in [elem_names[1],elem_names[2]]:
                axs[i+1].set_yticks([0.5,1.5,2.5,3.5])
                axs[i+1].set_ylim([0,3.5])

        plt.xlim([-3.5,0])
        plt.xlabel('[Fe/H]')
        plt.savefig((plot_path+title+'_feh.png').replace(' ',''))
        if plot==True: 
            plt.show()
        
    # Plot abundances vs time
    if time:
        fig, axs = plt.subplots(len(labels_h), figsize=(5,11),sharex=True)
        fig.subplots_adjust(bottom=0.06,top=0.97, left = 0.2,wspace=0.29,hspace=0)
        plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune=None)) for a in [fig.axes[0]]])
        plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='upper')) for a in fig.axes[1:]])
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        plt.setp([a.minorticks_on() for a in fig.axes[:]])
        axs = axs.ravel()

        for i, label in enumerate(labels_h):

            # Plot model [X/H]
            modeldata = model['eps'][:,snindex[label]]
            axs[i].plot(model['t'],modeldata,'k.-')

            # Add title and labels
            axs[i].set_ylabel(labels_h[i])
            if i == 0:
                axs[0].set_title(title)

            # Set different tickmarks for He, Mn, Ba
            if label in [elem_names[2],elem_names[25],elem_names[56]]:
                axs[i].locator_params(axis='y',nbins=4)
            else:                        
                axs[i].set_yticks([-3,-2,-1])
                axs[i].set_ylim([-3.2,0])

        plt.xlabel('Time (Gyr)')
        plt.xlim([0,1.7])
        plt.xticks([0.0,0.5,1.0,1.5])
        plt.savefig((plot_path+title+'_time_abund.png').replace(' ',''))
        if plot==True: 
            plt.show()

    # Plot model parameters
    if params:
        nplots = 6
        fig, axs = plt.subplots(nplots, figsize=(5,11),sharex=True)#, sharey=True)
        fig.subplots_adjust(bottom=0.06,top=0.95, left = 0.2,wspace=0.29,hspace=0)
        plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune=None)) for a in [fig.axes[0]]])
        plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='upper')) for a in fig.axes[1:]])
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        plt.setp([a.minorticks_on() for a in fig.axes[:]])
        plt.suptitle("Final mass: %.1f x $10^6$ M$_\odot$"%(model['mgal'][-1]/1e6), y=0.97)
        axs[0].set_title(title, y = 1.15)
        axs = axs.ravel()    
        
        axs[0].plot(model['t'],model['f_in']/1e6,'k-')
        axs[0].set_ylabel('$f_{in}$ $(10^6$ M$_\odot$ Gyr$^{-1})$')
        axs[1].plot(model['t'],model['mgas']/1e6,'b-',label='gas')
        axs[1].plot(model['t'],model['mstar']/1e6,'r-',label='stars')
        axs[1].plot(model['t'],model['mgal']/1e6,'k-',label='total')
        axs[1].set_ylabel('M $(10^6$ M$_\odot)$')
        axs[2].plot(model['t'],model['mdot']/1e6,'k-')
        axs[2].set_ylabel('SFR $(10^6$ M$_\odot$Gyr$^{-1})$')
        axs[3].plot(model['t'],-1*np.sum(model['de_dt']/1e6,1),'b-',label='gas')
        axs[3].plot(model['t'],-1*np.sum(model['dstar_dt']/1e6,1),'r-',label='stars')
        axs[3].plot(model['t'],-1*np.sum(model['mout']/1e6,1),'k-',label='out')
        axs[3].plot(model['t'],model['f_in']/1e6,'g-',label='in')
        axs[3].set_ylabel('$\dot{M} (10^6$ M$_\odot$Gyr$^{-1})$')
        axs[4].plot(model['t'],model['z']*1e3,'k-')
        axs[4].set_ylabel('Z $(10^{-3})$')
        axs[5].plot(model['t'],model['Ia_rate']*1e-3,'k-',label="SN Ia $(10^3)$")
        axs[5].plot(model['t'],model['II_rate']*1e-3,'r-',label="SN II $(10^3)$")
        axs[5].plot(model['t'],model['AGB_rate']*1e-4,'b-',label="AGB $(10^4)$")
        axs[5].set_ylabel('Rate $($Gyr$^{-1})$')
        
        # Add legends                                          
        axs[1].legend()
        axs[3].legend()
        axs[5].legend()    

        plt.xlabel('time (Gyr)')
        plt.xlim([0,1.7])
        plt.xticks([0.0,0.5,1.0,1.5])
        plt.savefig((plot_path+title+'_time_param.png').replace(' ',''))
        if plot==True: 
            plt.show()
    
    return