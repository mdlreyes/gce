import numpy as np
import scipy.ndimage
import sys

# Backend for matplotlib on mahler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, MaxNLocator

# Import other modules
import gce_fast as gce

#   using names used in the paper:  pars = [A_in/1e6,  tau_in,   A_out/1000,    A_star/1e6, alpha,    M_gas_0]
scl_pars = [701.57967, 0.26730922, 5.3575732, 0.47251228, 0.82681450, 0.49710685] #result of "restore, 'scl_sfr-law.sav'" in idl
umi_pars = [1470.5896, 0.16522838, 11.038576, 1.2065735, 0.26234735, 0.53814755] #result of "restore, 'umi_sfr-law.sav'" in idl
for_pars = [2464.2364, 0.30686976, 1.5054730, 5.0189799, 0.98204341, 14.575519] #'old/for_sfr-law.sav'
dra_pars = [1272.6409, 0.21561223, 9.5079511, 0.87843537, 0.34350762, 2.3213345]

# systematic errors calculated by Evan on 12/28/17
fehsyserr = 0.10103081
alphasyserr = 0.084143983
mgfesyserr = 0.076933658
sifesyserr = 0.099193360
cafesyserr = 0.11088295
tifesyserr = 0.10586739
bafesyserr = 0.100

mg_solar = 7.58
fe_solar = 7.52
ba_solar = 2.13
eu_solar = 0.51

accuracy_cutoff = 0.28 #dex cut for alpha/Fe error, Ba/Fe error, and Mg/Fe error

plot_path = '/raid/madlr/gce/plots/'

rc('font', family='serif')
#rc('font', weight='medium')
#rc('mathtext', default='sf')
#rc("lines", markeredgewidth=0.7)
rc('axes', labelsize=14) #24
#rc("axes", linewidth=2)
rc('xtick', labelsize=12)
rc('ytick', labelsize=12)
rc('xtick.major', size=12)
rc('ytick.major', size=12)
#rc('xtick.minor', size=7)
#rc('ytick.minor', size=7)
rc('legend', fontsize=12, frameon=False) #16
rc('text',usetex=True)
#rc('text.latex', preamble='\usepackage{color}')
rc('xtick',direction='in')
rc('ytick',direction='in')
#rc('xtick.major', pad=8)
#rc('ytick.major', pad=8)


def plotting_compare(model, atomic, title1, model2=[], atomic2=[], title2='', plot=False, 
                     skip_end_dots = -1, skip_end_dots2=-1, eu_estimate=False, NSM=False): 
    # generates three figures: elements/H vs. Fe/H, elements/Fe vs. time, other model parameters vs. time
    # note: the code will break if plot_atomic contains elements that are not included in atomic and atomic2 
        
    #dsph = ['For', 'Scl', 'UMi', 'Dra']
    #skip_end_dots = [-30,-10,-5,-1] #remove dots once the model becomes erratic
    
    if (model2 == []) & (atomic2 == []) & (title2 == ''):
        single_plot = True
    elif (model2 != []) & (atomic2 != []) & (title2 != ''):
        single_plot = False
    else: 
        print("ERROR: Must define all or none of the following: model2, atomic2, and title2")
        return

    model = model[:skip_end_dots]
    if single_plot == False:
        model2 = model[:skip_end_dots2]
    
    #atomic = np.array([1,2,6,8,12,14,20,22,26,56])
    #plot_atomic = np.array([1,2,6,8,12,14,20,22,26])
    #plot_atomic = np.sort(np.unique(np.concatenate((atomic,atomic2))))
    plot_atomic = atomic

    if len(plot_atomic) == 9:
        labels = ['[H/Fe]', '[He/Fe]', '[C/Fe]', '[O/Fe]', '[Mg/Fe]', '[Si/Fe]', '[Ca/Fe]', '[Ti/Fe]']
        labels_h = ['[He/H]', '[C/H]', '[O/H]', '[Mg/H]', '[Si/H]', '[Ca/H]', '[Ti/H]', '[Fe/H]']
    elif len(plot_atomic) == 10:
        labels = ['[H/Fe]', '[He/Fe]', '[C/Fe]', '[O/Fe]', '[Mg/Fe]', '[Si/Fe]', '[Ca/Fe]', '[Ti/Fe]', '[Ba/Fe]']
        labels_h = ['[He/H]', '[C/H]', '[O/H]', '[Mg/H]', '[Si/H]', '[Ca/H]', '[Ti/H]', '[Fe/H]', '[Ba/H]']  
        if eu_estimate == True:
            labels = ['[He/H]', '[He/Fe]', '[Ba/Eu]', '[O/Fe]', '[Mg/Fe]', '[Si/Fe]', '[Ca/Fe]', '[Ti/Fe]', '[Ba/Fe]']
    else: 
        print("Need to state plot labels!")
        return
    
    fe_plot_index = np.where(plot_atomic == 26)[0][0]
    h_plot_index = np.where(plot_atomic == 1)[0][0]
    map_data_index_to_plot = np.zeros(len(plot_atomic),dtype=int)    
    for i in range(len(map_data_index_to_plot)):
        map_data_index_to_plot[i] = np.where(atomic == plot_atomic[i])[0]

    feh = model['eps'][:,map_data_index_to_plot[fe_plot_index]]
    feh_mask = feh > -3.5
    indexfeh = np.argsort(feh[feh_mask])
    dfeh = feh[feh_mask][indexfeh][1:]-feh[feh_mask][indexfeh][:-1]
    dstar = np.sum(model['dstar_dt'],1)
    print(("Min and Max SFR:", min(dstar),max(dstar)))#,dstar.shape,feh.shape,dstar[feh_mask],dstar[feh_mask][indexfeh],dstar[feh_mask][indexfeh][1:]
    normalize = np.ma.masked_invalid(np.sum(model['dstar_dt'],1)[feh_mask][indexfeh][1:]*dfeh).sum()
    #print normalize#, np.sum(model['dstar_dt'],1)[feh_mask][indexfeh],dfeh

    if single_plot == False:
        map_data_index_to_plot2 = np.zeros(len(plot_atomic),dtype=int)    
        for i in range(len(map_data_index_to_plot2)):
            map_data_index_to_plot2[i] = np.where(atomic2 == plot_atomic[i])[0]
        feh2 = model2['eps'][:,map_data_index_to_plot2[fe_plot_index]]
        feh_mask2 = feh2 > -3.5
        indexfeh2 = np.argsort(feh2[feh_mask2])
        dfeh2 = feh2[feh_mask2][indexfeh2][1:]-feh2[feh_mask2][indexfeh2][:-1]
        dstar2 = np.sum(model2['dstar_dt'],1)
        print(("Min and Max SFR:", min(dstar2),max(dstar2)))#,dstar.shape,feh.shape,dstar[feh_mask],dstar[feh_mask][indexfeh],dstar[feh_mask][indexfeh][1:]
        normalize2 = np.ma.masked_invalid(np.sum(model2['dstar_dt'],1)[feh_mask2][indexfeh2][1:]*dfeh2).sum()
        #print normalize#, np.sum(model['dstar_dt'],1)[feh_mask][indexfeh],dfeh
        
    # plot abundances vs Fe
    fig, axs = plt.subplots(len(labels)+1, figsize=(5,11),sharex=True)#, sharey=True)
    fig.subplots_adjust(bottom=0.06,top=0.97, left = 0.2,wspace=0.29,hspace=0)
    plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune=None)) for a in [fig.axes[0]]])
    plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='upper')) for a in fig.axes[1:]])
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.setp([a.minorticks_on() for a in fig.axes[:]])
    axs = axs.ravel()
    #axs[0].set_title(title1 + ' vs. ' +'\textcolor{red}{%s}'%title2)
    str= '''
        if ~mw then plothist, moogify.feh, xhist, yhist, bin=0.1, /noplot else plothist, feh_venn, xhist, yhist, bin=0.1, /noplot
        yrange = 1.1*max(yhist)
        plot, model[7:maxt].png[7], dmd, xrange=[-3.5, 0.0], yrange=[0, yrange], ytitle='!7d!19N!7 / d[Fe/H]', charsize=1.5, xstyle=1, ystyle=1, psym=-4, /nodata
        feh_x = dindgen(1000)/999.*3.5-3.5
    '''
    #dmd = (model.mdot*((model.de_dt[8]/model.abund[8] - model.de_dt[0]/model.abund[0])/alog(10.0))^(-1.0)) / int_tabulated(model.t, model.mdot > 0.0) * 0.1
    feh_step = 0.1
    feh_plot = np.arange(-3.5,0+feh_step,feh_step)
    mdot = []
    if single_plot == False: mdot2 = []
    for i in range(len(feh_plot)):
        mask = (feh_plot[i] - feh_step/2. < feh) & (feh < feh_plot[i] + feh_step/2.)
        mdot.append(sum(model['mdot'][mask]))
        if single_plot == False:
            mask2 = (feh_plot[i] - feh_step/2. < feh2) & (feh2 < feh_plot[i] + feh_step/2.)
            mdot2.append(sum(model2['mdot'][mask2]))
    mdot = np.array(mdot)
    if single_plot == False: mdot2 = np.array(mdot2)
    gauss_sigma = 1
    #axs[0].plot(feh_plot,mdot/np.sum(mdot)*100,'k.-')
    axs[0].plot(feh_plot,scipy.ndimage.filters.gaussian_filter(mdot/np.sum(mdot)*100,gauss_sigma),'k-')
    #axs[0].plot(feh,model['mdot']/np.max(model['mdot']),'k--')
    if single_plot == False:
        #axs[0].plot(feh_plot,mdot2/np.sum(mdot2)*100,'r.-')
        axs[0].plot(feh_plot,scipy.ndimage.filters.gaussian_filter(mdot2/np.sum(mdot2)*100,gauss_sigma),'r-')
        #axs[0].plot(feh2,model2['mdot']/np.max(model2['mdot']),'r--')
    axs[0].set_ylabel('dN/d[Fe/H]') 
    #plt.sca(axs[0])
    #plt.yticks([0,20,40,60,80])
    #print "plot index, atomic number, label"
    element_mask = (plot_atomic != 26) #& (atomic != 1)
    for i in np.arange(len(labels)):
        if eu_estimate == True and i == 2:
            # ignores gas outflows and inflows!!!!
            elem_index = -1
            if NSM==True:
                nsm_ba_mass_per_Ia = 1.4e-05 #Msun
            else:
                nsm_ba_mass_per_Ia = 0
            eu_add = (model['II_rate']*3.21e-09+model['Ia_rate']*nsm_ba_mass_per_Ia*1e-1)/152.
            ba_add = (model['II_rate']*3.21e-08 + model['AGB_rate']*2.5e-7+model['Ia_rate']*nsm_ba_mass_per_Ia)/137.565      #gas phase abundance (number of atoms in units of M_sun/amu = 1.20d57)
            eu_abund = eu_add
            ba_abund = ba_add
            for j in np.arange(1,len(eu_add)):
                eu_abund[j] = eu_add[j-1]+eu_add[j] 
                ba_abund[j] = ba_add[j-1]+ba_add[j] 
            eu_abund = np.log10(eu_abund) - eu_solar
            ba_abund = np.log10(ba_abund) - ba_solar
                    
            axs[i+1].plot(feh,ba_abund-eu_abund,'k.-',label=title1)
            #axs[i].plot(feh,ba_abund + 12 - model['eps'][:,0] -feh) #check failed
            if single_plot == False:
                eu_abund2 = np.log10((model2['II_rate']*3.21e-09)/152) - eu_solar
                ba_abund2 = np.log10((model2['II_rate']*3.21e-08 + model2['AGB_rate']*2.5e-7)/137.565) -ba_solar     #gas phase abundance (number of atoms in units of M_sun/amu = 1.20d57)
                axs[i+1].plot(feh2,ba_abund2-eu_abund2,'r.--',label=title2)
            plot_index = len(labels)-1
        else:
            plot_index = np.arange(len(plot_atomic))[element_mask][i]
            #print i+1, plot_atomic[plot_index], labels[i], np.ma.masked_invalid(model['eps'][:,map_data_index_to_plot[plot_index]]-model['eps'][:,map_data_index_to_plot[fe_plot_index]]).sum()#, atomic[map_data_index_to_plot[plot_index]]
            axs[i+1].plot(feh,model['eps'][:,map_data_index_to_plot[plot_index]]-model['eps'][:,map_data_index_to_plot[fe_plot_index]],'k.-',label=title1)
            if single_plot == False:
                axs[i+1].plot(feh2,model2['eps'][:,map_data_index_to_plot2[plot_index]]-model2['eps'][:,map_data_index_to_plot2[fe_plot_index]],'r.--',label=title2)
            elif i==0:
                #plt.title(title1)
                axs[0].set_title(title1)
        axs[i+1].set_ylabel(labels[i])
        axs[i+1].plot([-6,0],[0,0],':k')
        plt.sca(axs[i+1])
        if eu_estimate == True and i == 2:
            pass
        elif plot_atomic[plot_index] in [1,2]: #if it is hydrogen or helium
            plt.yticks([0.5,1.5,2.5,3.5])
            plt.ylim([0,3.5])
        elif plot_atomic[plot_index] in [56]:
            pass
        else:                        
            plt.yticks([-0.5,0.0,0.5,1])
            plt.ylim([-0.8,1.2])
        if (single_plot==False) & (i == len(plot_atomic)-2):
            axs[i+1].legend()
    plt.xlim([-3.5,0])
    plt.xlabel('[Fe/H]')
    plt.savefig((plot_path+title1+title2+'_feh.png').replace(' ',''))
    if plot==True: 
        plt.show()
        
    # plot abundances vs time
    fig, axs = plt.subplots(len(plot_atomic)-1, figsize=(5,11),sharex=True)#, sharey=True)
    fig.subplots_adjust(bottom=0.06,top=0.97, left = 0.2,wspace=0.29,hspace=0)
    plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune=None)) for a in [fig.axes[0]]])
    plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='upper')) for a in fig.axes[1:]])
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.setp([a.minorticks_on() for a in fig.axes[:]])
    axs = axs.ravel()
    #axs[0].set_title(title1 + ' vs. ' +r'\textcolor{red}{%s}'%title2)
    #print "plot index, atomic number, label"
    element_mask = (plot_atomic != 1)
    for i in np.arange(len(plot_atomic)-1):
        plot_index = np.arange(len(plot_atomic))[element_mask][i]
        #print i, plot_atomic[plot_index], labels_h[i], np.ma.masked_invalid(model['eps'][:,map_data_index_to_plot[plot_index]]-model['eps'][:,map_data_index_to_plot[h_plot_index]]).sum()#, atomic[map_data_index_to_plot[plot_index]]
        axs[i].plot(model['t'],model['eps'][:,map_data_index_to_plot[plot_index]]-model['eps'][:,map_data_index_to_plot[h_plot_index]],'k.-',label=title1)
        if single_plot == False:
            axs[i].plot(model2['t'],model2['eps'][:,map_data_index_to_plot2[plot_index]]-model2['eps'][:,map_data_index_to_plot2[h_plot_index]],'r.--',label=title2)
        axs[i].set_ylabel(labels_h[i])
        if i == 0:
            if single_plot == False:
                axs[0].legend(loc=4)
            else:
                axs[0].set_title(title1)
        plt.sca(axs[i])
        if plot_atomic[plot_index] in [2,56]: #if it is  helium or barium
            plt.locator_params(axis='y',nbins=4)
        else:                        
            plt.yticks([-3,-2,-1])
            plt.ylim([-3.2,0])
    plt.xlabel('time (Gyr)')
    plt.xlim([0,1.7])
    plt.xticks([0.0,0.5,1.0,1.5])
    plt.savefig((plot_path+title1+title2+'_time_abund.png').replace(' ',''))
    if plot==True: 
        plt.show()

    nplots = 6
    fig, axs = plt.subplots(nplots, figsize=(5,11),sharex=True)#, sharey=True)
    fig.subplots_adjust(bottom=0.06,top=0.95, left = 0.2,wspace=0.29,hspace=0)
    plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune=None)) for a in [fig.axes[0]]])
    plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='upper')) for a in fig.axes[1:]])
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.setp([a.minorticks_on() for a in fig.axes[:]])
    #plt.title(title1 + ' vs. ' + title2 +"(dashed)", y = 1.1)
    if single_plot == False:
        plt.suptitle("Final mass: %.1f (or %.1f) x $10^6$ M$_\odot$"%(model['mgal'][-1]/1e6,model2['mgal'][-1]/1e6), y=0.97)
        axs[0].set_title(title1 + ' vs. ' + title2 +" (dashed)", y = 1.15)
    else: 
        plt.suptitle("Final mass: %.1f x $10^6$ M$_\odot$"%(model['mgal'][-1]/1e6), y=0.97)
        axs[0].set_title(title1, y = 1.15)
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
    try:
        axs[5].plot(model['t'],model['AGB_rate']*1e-4,'b-',label="AGB $(10^4)$")
    except(ValueError):
        pass
    axs[5].set_ylabel('Rate $($Gyr$^{-1})$')
    
    if single_plot == False:
        axs[0].plot(model2['t'],model2['f_in']/1e6,'k--')
        axs[1].plot(model2['t'],model2['mgas']/1e6,'b--')#,label='gas')
        axs[1].plot(model2['t'],model2['mstar']/1e6,'r--')#label='star')
        axs[1].plot(model2['t'],model2['mgal']/1e6,'k--')#label='total')
        axs[2].plot(model2['t'],model2['mdot']/1e6,'k--')
        axs[3].plot(model2['t'],-1*np.sum(model2['de_dt']/1e6,1),'b--')#,label='gas')
        axs[3].plot(model2['t'],-1*np.sum(model2['dstar_dt']/1e6,1),'r--')#label='star')
        axs[3].plot(model2['t'],-1*np.sum(model2['mout']/1e6,1),'k--')#label='out')
        axs[3].plot(model2['t'],model2['f_in']/1e6,'g--')#label='in')
        axs[4].plot(model2['t'],model2['z']*1e3,'k--')
        axs[5].plot(model2['t'],model2['Ia_rate']*1e-3,'k--')#label="Ia")
        axs[5].plot(model2['t'],model2['II_rate']*1e-3,'r--')#label="II")
        axs[5].plot(model2['t'],model2['AGB_rate']*1e-4,'b--')
                                              
    axs[1].legend()
    axs[3].legend()
    axs[5].legend()    

    plt.xlabel('time (Gyr)')
    plt.xlim([0,1.7])
    plt.xticks([0.0,0.5,1.0,1.5])
    plt.savefig((plot_path+title1 + title2+'_time_param.png').replace(' ',''))
    if plot==True: 
        plt.show()
    
    #axs[i].locator_params(axis='y',nbins=6)
    #plt.suptitle(subtitle, y = 0.955)
    #xminorLocator = plt. MultipleLocator (1)
    #xmajorLocator = plt. MultipleLocator (10)
    #ymajorLocator = plt. MultipleLocator (0.25)
    #plt.setp([a.yaxis.set_minor_locator(yminorLocator) for a in fig.axes[:]])
    #yminorLocator = plt. MultipleLocator (0.1)
    #plt.setp([a.yaxis.set_major_locator(MaxNLocator(prune='both')) for a in fig.axes[:]])
    #plt.setp([a.get_yticklabels()[0].set_visible(False) for a in fig.axes[:]])
    #plt.setp([a.get_yticklabels()[-1].set_visible(False) for a in fig.axes[1:]])
    
    return
    

