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
import params as paramfile
from astropy.io import ascii  # only needed for SFH test stuff
from astropy.cosmology import FlatLambdaCDM  # needed to compute redshifts
cosmo = FlatLambdaCDM(H0=67.7, Om0=0.310)
import astropy.units as u
from astropy.cosmology import z_at_value

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
rc('xtick.major', size=10)
rc('ytick.major', size=10)
rc('legend', fontsize=12, frameon=False)
rc('text',usetex=True)
rc('xtick',direction='in')
rc('ytick',direction='in')

# Some other stuff to make cool colors
import cycler
import cmasher as cmr

def makeplots(model, atomic, title, plot=False, datasource='deimos', dsph='Scl', skip_end_dots=-1, NSM=False, 
            plot_path='plots/', abunds=True, time=True, params=True, feh=True): 
    """Generates plots: [X/Fe] vs [Fe/H], [X/Fe] vs [time], model parameters vs time

    Args:
        model (array): Output array from gce_fast
        atomic (array): Output SN_yield['atomic'] array from gce_fast
        title (str): Title of plots
        plot (bool): If 'True', show plots
        datasource (str): Source of observed data (options: 'deimos', 'dart', 'both')
        dsph (str): Name of galaxy to get observed data (options: 'Scl')
        skip_end_dots (int): Remove last n dots (useful if model becomes erratic)
        plot_path (str): Path to store plots
        abunds, time, params (bool): Types of plots
        feh (bool): If 'True', plot abundances as a function of [Fe/H]; else, as a function of [Mg/H]
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
        if elem == 63:
            snindex['Eu'] = idx
            continue
        if elem not in [26, 1, 2, 22]:
            labels.append(elem_names[elem])
        if elem != 1:
            labels_h.append(elem_names[elem])
        snindex[elem_names[elem]] = idx

    if feh==False:
        labels.remove('Mg')
        labels.append('Fe')

    # Open observed data
    elem_data, delem_data, _ = getdata(galaxy='Scl', source='deimos', c=True, ba=True, mn=True, eu=True, removerprocess='statistical', feh_denom=feh)
    if datasource=='dart' or datasource=='both':
        elem_data_dart, delem_data_dart, _ = getdata(galaxy='Scl', source='dart', c=True, ba=True, mn=True, eu=True, removerprocess='statistical', feh_denom=feh)
    
    # Map content of observed elem_data to index
    if feh:
        obs_idx = {'Fe':0, 'Mg':1, 'Si':2, 'Ca':3, 'C':4, 'Ba':5, 'Mn':6, 'Eu':7}
    else:
        obs_idx = {'Mg':0, 'Si':1, 'Ca':2, 'C':3, 'Ba':4, 'Mn':5, 'Eu':6, 'Fe':7}
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

        # Get x-axis model data ([Fe/H] or [Mg/H]) to plot MDF
        if feh:
            x = model['eps'][:,snindex['Fe']]
            x_step = 0.1
            x_plot = np.arange(-3.5,x_step,x_step)
        else:
            x = model['eps'][:,snindex['Mg']] + 0.2
            x_step = 0.1
            x_plot = np.arange(-2.5,1.5,x_step)
            
        mdot = []
        for i in range(len(x_plot)):
            # Add up all star formation that happened in this [Fe/H] step
            mask = np.where((x_plot[i] - x_step/2. < x) & (x < x_plot[i] + x_step/2.))[0]
            mdot.append(np.sum(model['mdot'][mask]))
        mdot = np.array(mdot)
        gauss_sigma = 1

        # Get x-axis observed data ([Fe/H] or [Mg/H]) to plot
        x_obs = elem_data[0,:]
        if feh:
            obsmask = np.where((x_obs > -3.5) & (x_obs < 0.) & (delem_data[0,:] < 0.4))[0]
        else:
            obsmask = np.where((x_obs > -2.5) & (x_obs < 0.) & (delem_data[0,:] < 0.4))[0]
        x_obs = x_obs[obsmask]

        # In top panel, plot model MDF
        axs[0].plot(x_plot,scipy.ndimage.filters.gaussian_filter(mdot/np.sum(mdot)/x_step,gauss_sigma),'k-')
            
        if datasource=='dart' or datasource=='both':
            # Get x-axis observed data from DART to plot
            x_obs_dart = elem_data_dart[0,:]
            if feh:
                obsmask_dart = np.where((x_obs_dart > -3.5) & (x_obs_dart < 0.) & (delem_data_dart[0,:] < 0.4))[0]
            else:
                obsmask_dart = np.where((x_obs_dart > -2.5) & (x_obs_dart < 0.) & (delem_data_dart[0,:] < 0.4))[0]
            x_obs_dart = x_obs_dart[obsmask_dart]

            # In top panel, plot metallicity distribution function
            axs[0].hist(x_obs_dart, bins=20, fill=False, histtype='step', edgecolor=plt.cm.Set3(3), linewidth=1.5, density=True, label='DART')
            label='[Fe/H]'
            if feh==False: label='[Mg/H]'
            axs[0].set_ylabel('dN/d'+label) 

        if datasource=='deimos' or datasource=='both':
            # Plot observed DEIMOS MDF    
            axs[0].hist(x_obs, bins=20, color=plt.cm.Set3(0), alpha=0.7, density=True, label='DEIMOS')
            label='[Fe/H]'
            if feh==False: label='[Mg/H]'
            axs[0].set_ylabel('dN/d'+label) 

        if datasource=='both':
            axs[0].legend(loc='best', fontsize=8)
        
        # Plot other elements as a function of [Fe/H]
        for i, label in enumerate(labels):

            # Plot observed [X/Fe]
            if label in obs_idx:
                if datasource=='deimos' or datasource=='both': # and label!='Ba':
                    obs_data = elem_data[obs_idx[label],obsmask]
                    obs_errs = delem_data[obs_idx[label],obsmask]
                    if feh:
                        x_errs = delem_data[obs_idx['Fe'],obsmask]
                    else:
                        x_errs = delem_data[obs_idx['Mg'],obsmask]
                    totalerrs = np.sqrt(delem_data[obs_idx[label],obsmask]**2. + x_errs**2.)
                    goodidx = np.where((x_obs > -990) & (obs_data > -990) & 
                                    (np.abs(obs_errs) < 0.4) & (np.abs(x_errs) < 0.4))[0]
                    print(label, obs_idx[label], len(obs_data), len(obs_data[goodidx]))
                    #axs[i+1].scatter(feh_obs[goodidx], obs_data[goodidx], c='r', s=0.8/(totalerrs[goodidx])**2., alpha=0.5)
                    axs[i+1].errorbar(x_obs[goodidx], obs_data[goodidx], xerr=x_errs[goodidx], yerr=obs_errs[goodidx], 
                                    color=plt.cm.Set3(0), linestyle='None', marker='o', markersize=3, alpha=0.7, linewidth=0.5)

                if datasource=='dart' or datasource=='both':
                    obs_data = elem_data_dart[obs_idx[label],obsmask_dart]
                    obs_errs = delem_data_dart[obs_idx[label],obsmask_dart]
                    if feh:
                        x_errs = delem_data_dart[obs_idx['Fe'],obsmask_dart]
                    else:
                        x_errs = delem_data_dart[obs_idx['Mg'],obsmask_dart]
                    totalerrs = np.sqrt(delem_data_dart[obs_idx[label],obsmask_dart]**2. + x_errs**2.)
                    goodidx = np.where((x_obs_dart > -990) & (obs_data > -990) & 
                                    (np.abs(obs_errs) < 0.4) & (np.abs(x_errs) < 0.4))[0]
                    #axs[i+1].scatter(feh_obs[goodidx], obs_data[goodidx], c='r', s=0.8/(totalerrs[goodidx])**2., alpha=0.5)
                    axs[i+1].errorbar(x_obs_dart[goodidx], obs_data[goodidx], xerr=x_errs[goodidx], yerr=obs_errs[goodidx], 
                                    mfc='white', mec=plt.cm.Set3(3), ecolor=plt.cm.Set3(3), linestyle='None', marker='o', markersize=3, linewidth=0.5)
            
            # Plot model [X/Fe] (or [X/Mg])
            if feh:
                modeldata = model['eps'][:,snindex[label]] - model['eps'][:,snindex['Fe']]
                if label == 'Mg':
                    # Add 0.2 to [Mg/Fe]
                    modeldata += 0.2
            else:
                modeldata = model['eps'][:,snindex[label]] - (model['eps'][:,snindex['Mg']] + 0.2)
                
            axs[i+1].plot(x,modeldata,'k.-', zorder=100)
            
            # Add title and labels
            if i == 0:
                axs[0].set_title(title)
            if feh:
                denom_label = '/Fe]'
            else:
                denom_label = '/Mg]'
            if label == 'Ba':
                denom_label = denom_label + r'$_{s}$'
            axs[i+1].set_ylabel('['+label+denom_label)
            axs[i+1].plot([-6,0],[0,0],':k')

            # Set different limits/ticks for H and He
            if feh:
                if label in [elem_names[1],elem_names[2]]:
                    axs[i+1].set_yticks([0.5,1.5,2.5,3.5])
                    axs[i+1].set_ylim([0,3.5])

            if label == elem_names[56]:
                axs[i+1].set_yticks([-2,-1,0,1])
                if feh:
                    axs[i+1].set_ylim([-2.5,2])
                else:
                    axs[i+1].set_ylim([-3,2])

        if feh:
            plt.xlim([-3.5,0])
            plt.xlabel('[Fe/H]')
            filelabel = '_feh.png'
        else:
            plt.xlim([-2.5,0.])
            plt.xlabel('[Mg/H]')
            filelabel = '_mgh.png'

        plt.savefig((plot_path+title+filelabel).replace(' ',''))
        if plot==True: 
            plt.show()
        else:
            plt.close()

        # Plot [Ba/Eu] vs [Fe/H] separately
        if 63 in atomic:
            fig = plt.figure(figsize=(5,3))
            ax = plt.subplot()

            # Plot observed data from DART
            if datasource=='dart' or datasource=='both':
                ba_data = elem_data_dart[obs_idx['Ba'],obsmask_dart]
                eu_data = elem_data_dart[obs_idx['Eu'],obsmask_dart]
                obs_data = ba_data - eu_data
                obs_errs = np.sqrt(delem_data_dart[obs_idx['Ba'],obsmask_dart]**2. + delem_data_dart[obs_idx['Eu'],obsmask_dart]**2.)
                if feh:
                    x_errs = delem_data_dart[obs_idx['Fe'],obsmask_dart]
                else:
                    x_errs = delem_data_dart[obs_idx['Mg'],obsmask_dart]
                goodidx = np.where((x_obs_dart > -990) & (ba_data > -990) & (eu_data > -990) & 
                                (np.abs(obs_errs) < 0.4) & (np.abs(x_errs) < 0.4))[0]
                ax.errorbar(x_obs_dart[goodidx], obs_data[goodidx], xerr=x_errs[goodidx], yerr=obs_errs[goodidx], 
                            mfc='white', mec=plt.cm.Set3(3), ecolor=plt.cm.Set3(3), linestyle='None', marker='o', markersize=3, linewidth=0.5)

            # Plot model [Ba/Eu] vs [Fe/H]
            modeldata = model['eps'][:,snindex['Ba']] - model['eps'][:,snindex['Eu']]
            plt.plot(x,modeldata,'k.-', zorder=100)
            
            # Add title and labels
            plt.title(title)
            plt.ylabel('[Ba/Eu]')
            if feh:
                plt.xlabel('[Fe/H]')
                plt.xlim([-3.5,0])
            else:
                plt.xlabel('[Mg/H]')
            plt.plot([-6,0],[0,0],':k')
            plt.ylim([-2.5,2.5])
            #agbtitle = {'kar':'Karakas+18', 'cri15':'FRUITY'}
            #ax.text(0.05, 0.9, agbtitle[paramfile.AGB_source], transform=ax.transAxes, fontsize=12)

            plt.savefig((plot_path+title+'_baeu.png').replace(' ',''), bbox_inches='tight')
            if plot==True: 
                plt.show()
            else:
                plt.close()
        
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
        plt.suptitle("Final mass: %.2f x $10^6$ M$_\odot$"%(model['mgal'][-1]/1e6), y=0.97)
        print('test', model['mgal'][-1]/1e6, model['mstar'][-1]/1e6)
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
        axs[4].plot(model['t'],model['eps'][:,snindex['Fe']],'k-')
        axs[4].set_ylabel('[Fe/H]')
        #axs[4].set_ylabel('Z $(10^{-3})$')
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
        else:
            plt.close()

        # Plot SFH
        cwheelsize = 3
        color = cmr.bubblegum(np.linspace(0,1,cwheelsize,endpoint=True))
        matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
        cwheel = [np.array(matplotlib.rcParams['axes.prop_cycle'])[x]['color'] for x in range(cwheelsize)]

        fig = plt.figure(figsize=(8,2))
        ax = plt.subplot()
        plt.title(title, fontsize=16)
        plt.xlabel('Time (Gyr)')
        plt.setp([a.minorticks_on() for a in fig.axes[:]])
        plt.ylabel('SFR ($10^{-4}$ M$_\odot$ yr$^{-1})$')
        plt.ylim([0,30])
        plt.yticks([0, 10, 20, 30])
        #plt.xlim([0,1.7])
        #plt.xticks([0.0,0.5,1.0,1.5])
        #plt.suptitle("Final mass: %.1f x $10^6$ M$_\odot$"%(model['mgal'][-1]/1e6), y=0.97)

        # Add redshift axis on top
        ages = np.array([13, 10, 8, 6, 4, 3, 2, 1.5, 1.2, 1, 0.8, 0.6])*u.Gyr
        ageticks = [z_at_value(cosmo.age, age) for age in ages]
        ax2 = ax.twiny()
        ax2.set_xticks(ageticks)
        ax2.set_xticklabels(['{:g}'.format(age) for age in ages.value])
        zmin, zmax = 0.0, 8.5
        ax.set_xlim(zmin, zmax)
        ax2.set_xlim(zmin, zmax)
        
        # Plot SFH from this model
        plt.plot(model['t'],model['mdot']/1e5, ls='-', lw=2, color=plt.cm.Set2(0), label='This work')

        # Also from other models
        bettinelli = ascii.read('data/sfhtest/bettinelli19.dat')
        bettinelli = [13.792 - bettinelli['Lookback time (Gyr)'], bettinelli['SFR (1e-4 Msun/y)']]
        deboer = ascii.read('data/sfhtest/deboer12.dat')
        deboer = [13.792 - deboer['Age (Gyr)'], deboer['SFR (1e-4 Msun/y)']]
        titles = ['Bettinelli et al. (2019)', 'de Boer et al. (2012)']
        linestyles = ['--', ':']
        for idx, data in enumerate([bettinelli, deboer]):
            plt.plot(data[0], data[1], ls=linestyles[idx], lw=2, label=titles[idx], color=plt.cm.Set2(idx+1))

        plt.legend(loc='best')
        plt.savefig((plot_path+title+'_sfh.png').replace(' ',''), bbox_inches='tight')
        if plot==True: 
            plt.show()
        else:
            plt.close() 
    
    return