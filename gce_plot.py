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
from data_load import load_ba_results

#   using names used in the paper:  pars = [A_in/1e6,  tau_in,   A_out/1000,    A_star/1e6, alpha,    M_gas_0]
scl_pars = [ 701.57967, 0.26730922, 5.3575732, 0.47251228, 0.82681450, 0.49710685] #result of "restore, 'scl_sfr-law.sav'" in idl
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
    #fe_plot_index = np.where(atomic == 26)[0][0]
    #h_plot_index = np.where(atomic == 1)[0][0]
    # plot_atomic = atomic[map_data_index_to_plot]
    # fe_plot_index = map_data_index_to_plot[fe_plot_index]
    # atomic[map_data_index_to_plot[fe_plot_index]] = plot_atomic[fe_plot_index]

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
    
def plot_compare_element_mult(num_models_to_compare = 2, name_fragment ='',title = [''],
                              r_process_keyword = ['typical_SN_only'], imf_model = ['kroupa93'], 
                              AGB_yield_mult = [1],AGB_source=['kar16'],SNII_yield_mult =[1],
                              NSM_yield_mult =[1] ,element_Z = 56, data=True, 
                              plot=True, ylim=True,model=True,Keck_proposal=False,show_legend=True,color=''):
    
    num_models_to_compare = int(num_models_to_compare)

    # set parameters to be the same for every parameter that isn't the varied parameter
    if len(title) != num_models_to_compare:
        print(("ERROR: please set title = [label_1, label_2, ...] equal to the number of models, which is", num_models_to_compare))
        title = [title[0]]*num_models_to_compare    
    if len(r_process_keyword) != num_models_to_compare:
        r_process_keyword = [r_process_keyword[0]]*num_models_to_compare
    if len(imf_model) != num_models_to_compare:
        imf_model = [imf_model[0]]*num_models_to_compare
    if len(AGB_yield_mult) != num_models_to_compare:
        AGB_yield_mult = [AGB_yield_mult[0]]*num_models_to_compare
    if len(AGB_source) != num_models_to_compare:
        AGB_source = [AGB_source[0]]*num_models_to_compare
    if len(SNII_yield_mult) != num_models_to_compare:
        SNII_yield_mult = [SNII_yield_mult[0]]*num_models_to_compare
    if len(NSM_yield_mult) != num_models_to_compare:
        NSM_yield_mult = [NSM_yield_mult[0]]*num_models_to_compare        
    
    models = []
    for model_index in range(num_models_to_compare):    
        model1, atomic1 = gce.gce_model_ba(for_pars, r_process_keyword = r_process_keyword[model_index], 
                                       imf_model = imf_model[model_index], AGB_yield_mult = AGB_yield_mult[model_index], 
                                       AGB_source = AGB_source[model_index], 
                                       SNII_yield_mult = SNII_yield_mult[model_index], 
                                       NSM_yield_mult = NSM_yield_mult[model_index])
        model2, atomic2 = gce.gce_model_ba(scl_pars, r_process_keyword = r_process_keyword[model_index], 
                                       imf_model = imf_model[model_index], AGB_yield_mult = AGB_yield_mult[model_index], 
                                       AGB_source = AGB_source[model_index], 
                                       SNII_yield_mult = SNII_yield_mult[model_index], 
                                       NSM_yield_mult = NSM_yield_mult[model_index])
        model3, atomic3 = gce.gce_model_ba(umi_pars, r_process_keyword = r_process_keyword[model_index], 
                                       imf_model = imf_model[model_index], AGB_yield_mult = AGB_yield_mult[model_index], 
                                       AGB_source = AGB_source[model_index], 
                                       SNII_yield_mult = SNII_yield_mult[model_index], 
                                       NSM_yield_mult = NSM_yield_mult[model_index])
        model4, atomic4 = gce.gce_model_ba(dra_pars, r_process_keyword = r_process_keyword[model_index], 
                                       imf_model = imf_model[model_index], AGB_yield_mult = AGB_yield_mult[model_index], 
                                       AGB_source = AGB_source[model_index], 
                                       SNII_yield_mult = SNII_yield_mult[model_index], 
                                       NSM_yield_mult = NSM_yield_mult[model_index])
    
        models.append([model1, model2, model3, model4])
#    axs[5].plot(model['t'],model['Ia_rate']*1e-3,'k-',label="SN Ia $(10^3)$")
# scipy.integrate.simps(np.sum(model['dstar_dt'],1),model['t'])

    labels = ['Fornax', 'Sculptor', 'Draco', 'Ursa Minor']
    dsph = ['For', 'Scl', 'Dra', 'UMi']
    skip_end_dots = [-30,-10,-5,-1] #remove dots once the model becomes erratic
    
    element_index = np.where(atomic1 == element_Z)[0][0] #Barium
    if element_Z == 56:
        label = '[Ba/Fe]'
        data_element = 'BAFE'
        data_err = 'BAFEERR'
        data_feh = 'FEH'
        data_feh_err = 'FEHERR'
        dot_size = 0.4
    elif element_Z == 12:
        label = '[Mg/Fe]'
        data_element = 'MgFe'
        data_err = 'Mgerr'
        data_feh = 'FeH'
        data_feh_err = 'Feerr'
        dot_size = 0.4
    else: print("No label specified")
        
    fe_index = np.where(atomic1 == 26)[0][0]

    filename = name_fragment+''.join(title)+'_'+label
    filename = (((((filename.replace('[','')).replace('/','')).replace(']','')).replace("\\",'')).replace('$','')).replace(' ','')
    
    
    # plot Ba abundances vs Fe
    fig, axs = plt.subplots(4, figsize=(5,11),sharex=True)#, sharey=True)
    fig.subplots_adjust(bottom=0.06,top=0.97, left = 0.2,wspace=0.29,hspace=0)
    plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune=None)) for a in [fig.axes[0]]])
    plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='upper')) for a in fig.axes[1:]])
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.setp([a.minorticks_on() for a in fig.axes[:]])
    axs = axs.ravel()
    #axs[0].set_title(title1 + ' vs. ' +'\textcolor{red}{%s}'%title2)

    for i in np.arange(4):
        if model==True:
            for model_index, model_group in enumerate(models):
                if (num_models_to_compare == 1) and (len(color)!= 0):
                    axs[i].plot(model_group[i]['eps'][:,fe_index][:skip_end_dots[i]],
                            model_group[i]['eps'][:,element_index][:skip_end_dots[i]]
                            -model_group[i]['eps'][:,fe_index][:skip_end_dots[i]],
                            '-',label=title[model_index],color=color) 
                else:                   
                    axs[i].plot(model_group[i]['eps'][:,fe_index][:skip_end_dots[i]],
                            model_group[i]['eps'][:,element_index][:skip_end_dots[i]]
                            -model_group[i]['eps'][:,fe_index][:skip_end_dots[i]],
                            '-',label=title[model_index])

        if data==True:        
            if i==0: filename = filename + '_data'
            if element_Z == 56:
                results, mask, ba_final = load_ba_results(dsph[i])
            elif element_Z == 12:
                # load Kirby 2010 Mg measurements
                filename_member = '/raid/m31/dsph/alldsph/dsph_catalog.dat'
                members = np.genfromtxt(filename_member, skip_header=1,
                                        dtype=[('dSph', 'S9'), ('objname', 'S11'), ('RAh', '<i8'), ('RAm', '<i8'), ('RAs', '<f8'), ('Decd', '<i8'), 
                                               ('Decm', '<i8'), ('Decs', '<f8'), ('v_raw', '<f8'), ('Teff', '<f8'), ('logg', '<f8'), ('vt', '<f8'), 
                                               ('FeH', '<f8'), ('Feerr', '<f8'), ('AlphaFe', '<f8'), ('Alphaerr', '<f8'), ('MgFe', '<f8'), ('Mgerr', '<f8'),
                                               ('SiFe', '<f8'), ('Sierr', '<f8'), ('CaFe', '<f8'), ('Caerr', '<f8'), ('TiFe', '<f8'), ('Tierr', '<f8')],
                                        delimiter=(9,11,3,3,6,3,3,6,7,5,5,6,6,5,6,5,6,5,6,5,6,5,6,5))
                members['dSph']=np.core.defchararray.strip(members['dSph'])
                members['objname']=np.core.defchararray.strip(members['objname'])
                #members = np.genfromtxt(filename_member, skip_header=1,delimiter=(9,11,3,3,6,3,3,6,7,5,5,6,6,5,6,5,6,5,6,5,6,5,6,5), names=True, dtype=None) #Use to get dtype list
                # AlphaFe and Alphaerr are not present in all objects, so can't read them in. 
                # Need to consider replacing spaces with negative value so error isn't thrown.
                mask_dsph = (dsph[i]==members['dSph'])
                results = np.copy(members[mask_dsph])
                results['Feerr'] = np.sqrt(results['Feerr']**2+fehsyserr**2)
                results['Alphaerr'] = np.sqrt(results['Alphaerr']**2+alphasyserr**2)
                results['Mgerr'] = np.sqrt(results['Mgerr']**2+mgfesyserr**2)
                    
                mask = (results['Mgerr']<=accuracy_cutoff)&(results['Mgerr']>mgfesyserr
                        )&(results['Alphaerr']<=accuracy_cutoff)&(results['Alphaerr']>alphasyserr
                        )&(results['Feerr']<=accuracy_cutoff)&(results['Feerr']>fehsyserr)

            err = np.reciprocal(results[data_feh_err][mask]**2+results[data_err][mask]**2)
            axs[i].scatter(results[data_feh][mask],results[data_element][mask],
                                   marker='o',c='k',s=err*1.3+dot_size, alpha=0.7,linewidth=0)#,capsize=3))   


            #axs[i].plot(ba_results['FeH'][masklower],ba_results['BaFe'][masklower],'^',color='0.6',markeredgewidth=0.5,markeredgecolor='k')
            #axs[i].plot(ba_results['FeH'][mask][maskupper],ba_results['BaFe'][mask][maskupper],'v',color='0.6',markeredgewidth=0.5,markeredgecolor='k')

        if (num_models_to_compare == 1) and (i==0):
            #plt.title(title1)
            axs[0].set_title(title[0])
        axs[i].set_ylabel(labels[i]+'\n'+label)
        axs[i].plot([-6,0],[0,0],':k')
        if num_models_to_compare != 1 and show_legend == True: #plot legend
            if element_Z == 56:
                axs[i].legend(loc=2)
            else: axs[i].legend(loc=3)
        if (i == 0) or Keck_proposal==True: #plot representative errorbar
            if Keck_proposal == True:
                rc('legend', fontsize=10, frameon=False) #16
                if 'Scl' in name_fragment:
                    axs[i].set_ylim([-1.8,1.1])
                    x_errorbar_loc = -0.8
                elif 'UMi' in name_fragment:
                    axs[i].set_ylim([-1.2,1.2])
                    x_errorbar_loc = -1.35
                elif 'Dra' in name_fragment:
                    axs[i].set_ylim([-1.6,1.45])
                    x_errorbar_loc = -1.3
                else:
                    x_errorbar_loc = -0.3                    
            else:
                x_errorbar_loc = -0.5
                if element_Z == 12:
                    x_errorbar_loc = -3
            #axs[0].scatter([-3.2],[-.3],marker='o',c='k',s=np.median(err)*0.9-5,alpha=0.7)
            min_model = np.nanmin(model_group[i]['eps'][:,element_index][:skip_end_dots[i]]-model_group[i]['eps'][:,fe_index][:skip_end_dots[i]])
            min_data = np.nanmin(results[data_element][mask])
            min = np.nanmin([min_model,min_data])+0.3
            if ylim == True:
                min = -1
            if (Keck_proposal==True) and ('Dra' in name_fragment):
                min=-1.0
            print(("y location of error bar",min))               
            axs[i].errorbar([x_errorbar_loc], [min],color='k',xerr=[np.mean(results[data_feh_err][mask])],yerr=[np.nanmean(results[data_err][mask])])
        if ylim==True:
            axs[i].set_ylim([-1.8,1.7])
            #axs[i].set_ylim([-2.3,1.7])
            #axs[i].set_ylim([-2,1.2])
    if Keck_proposal == True: 
        if 'Scl' in name_fragment:
            plt.xlim([-4,-0.5])  
        elif 'UMi' in name_fragment:
            plt.xlim([-3.6,-1.2]) 
        elif 'Dra' in name_fragment:
            plt.xlim([-3.8,-1.0])                           
    else:
        plt.xlim([-3.5,-0.2])
    plt.xlabel('[Fe/H]')
    print(filename)
    plt.savefig(plot_path+filename+'.png')
    if plot==True: plt.show()

def plot_imf():
    #m_int2 = np.logspace(np.log10(0.865),np.log10(10),150)                                                                            #dummy integration array (M_sun), 50 steps between 0.865 and 10
    m_int2 = np.logspace(np.log10(0.08),np.log10(120),120)                                                                            #dummy integration array (M_sun), 50 steps between 0.865 and 10
    #print m_int2
    dm = m_int2[1:]-m_int2[:-1]
    print(('Kroupa', np.sum((m_int2[:-1]+dm/2.)*(IMFkroupa93(m_int2)[:-1]+IMFkroupa93(m_int2)[1:])/2.*dm)))
    print(('Chabrier', np.sum((m_int2[:-1]+dm/2.)*(IMFchabrier03(m_int2)[:-1]+IMFchabrier03(m_int2)[1:])/2.*dm)))
    print(('Salpeter', np.sum((m_int2[:-1]+dm/2.)*(IMFsalpeter55(m_int2)[:-1]+IMFsalpeter55(m_int2)[1:])/2.*dm)))
    mask = np.where((m_int2 > 1.3) & (m_int2 < 4))[0]
    mask_wd = np.where((m_int2 > 2.5) & (m_int2 < 10))[0]
    mask_ns = np.where((m_int2 > 10) & (m_int2 < 25))[0]
    dm = m_int2[mask][1:]-m_int2[mask][:-1]
    dm_wd = m_int2[mask_wd][1:]-m_int2[mask_wd][:-1]
    dm_ns = m_int2[mask_ns][1:]-m_int2[mask_ns][:-1]
    print(('Kroupa M(1.3<M<4) = ', np.sum((m_int2[mask][:-1]+dm/2.)*(IMFkroupa93(m_int2[mask])[:-1]+IMFkroupa93(m_int2[mask])[1:])/2.*dm)))
    print(('Chabrier M(1.3<M<4)', np.sum((m_int2[mask][:-1]+dm/2.)*(IMFchabrier03(m_int2[mask])[:-1]+IMFchabrier03(m_int2[mask])[1:])/2.*dm)))    
    print(('Salpeter M(1.3<M<4)', np.sum((m_int2[mask][:-1]+dm/2.)*(IMFsalpeter55(m_int2[mask])[:-1]+IMFsalpeter55(m_int2[mask])[1:])/2.*dm)))   
    print(('Kroupa WD = ', np.sum((m_int2[mask_wd][:-1]+dm_wd/2.)*(IMFkroupa93(m_int2[mask_wd])[:-1]+IMFkroupa93(m_int2[mask_wd])[1:])/2.*dm_wd)))
    print(('Chabrier WD', np.sum((m_int2[mask_wd][:-1]+dm_wd/2.)*(IMFchabrier03(m_int2[mask_wd])[:-1]+IMFchabrier03(m_int2[mask_wd])[1:])/2.*dm_wd)))    
    print(('Salpeter WD', np.sum((m_int2[mask_wd][:-1]+dm_wd/2.)*(IMFsalpeter55(m_int2[mask_wd])[:-1]+IMFsalpeter55(m_int2[mask_wd])[1:])/2.*dm_wd)))    
    print(('Kroupa NS = ', np.sum((m_int2[mask_ns][:-1]+dm_ns/2.)*(IMFkroupa93(m_int2[mask_ns])[:-1]+IMFkroupa93(m_int2[mask_ns])[1:])/2.*dm_ns)))
    print(('Chabrier NS', np.sum((m_int2[mask_ns][:-1]+dm_ns/2.)*(IMFchabrier03(m_int2[mask_ns])[:-1]+IMFchabrier03(m_int2[mask_ns])[1:])/2.*dm_ns)))    
    print(('Salpeter NS', np.sum((m_int2[mask_ns][:-1]+dm_ns/2.)*(IMFsalpeter55(m_int2[mask_ns])[:-1]+IMFsalpeter55(m_int2[mask_ns])[1:])/2.*dm_ns)))    
    #print 'Kroupa old', np.sum((IMFkroupa93_old(m_int2)[:-1]+IMFkroupa93_old(m_int2)[1:])/2.*dm)
    plt.loglog(m_int2, IMFkroupa93(m_int2), label="kroupa93")
    plt.loglog(m_int2, IMFchabrier03(m_int2), label="chabrier03")
    plt.loglog(m_int2, IMFsalpeter55(m_int2), label="salpeter55")
    plt.xlabel('Mass $($M$_\odot)$')
    plt.ylabel('dN/dM')
    plt.legend()
    plt.show()

def plot_alpha_vs_barium_fit(dsph_name, model, atomic, title1, model2=[], atomic2=[], title2='', plot=False, data=True): 
    # generates three figures: elements/H vs. Fe/H, elements/Fe vs. time, other model parameters vs. time
    # note: the code will break if plot_atomic contains elements that are not included in atomic and atomic2 

    if data == True:
        ba_results, mask, ba_final = load_ba_results(dsph_name)
        
    if (model2 == []) & (atomic2 == []) & (title2 == ''):
        single_plot = True
    elif (model2 != []) & (atomic2 != []) & (title2 != ''):
        single_plot = False
    else: 
        print("ERROR: Must define all or none of the following: model2, atomic2, and title2")
        return
    
    plot_atomic = atomic

    if len(plot_atomic) == 9:
        labels = ['[H/Fe]', '[He/Fe]', '[C/Fe]', '[O/Fe]', '[Mg/Fe]', '[Si/Fe]', '[Ca/Fe]', '[Ti/Fe]']
        labels_h = ['[He/H]', '[C/H]', '[O/H]', '[Mg/H]', '[Si/H]', '[Ca/H]', '[Ti/H]', '[Fe/H]']
    elif len(plot_atomic) == 10:
        labels = ['[H/Fe]', '[He/Fe]', '[C/Fe]', '[O/Fe]', '[Mg/Fe]', '[Si/Fe]', '[Ca/Fe]', '[Ti/Fe]', '[Ba/Fe]']
        labels_h = ['[He/H]', '[C/H]', '[O/H]', '[Mg/H]', '[Si/H]', '[Ca/H]', '[Ti/H]', '[Fe/H]', '[Ba/H]']        
    else: 
        print("Need to state plot labels!")
        return
    
    fe_plot_index = np.where(plot_atomic == 26)[0][0]
    h_plot_index = np.where(plot_atomic == 1)[0][0]
    map_data_index_to_plot = np.zeros(len(plot_atomic),dtype=int)    
    for i in range(len(map_data_index_to_plot)):
        map_data_index_to_plot[i] = np.where(atomic == plot_atomic[i])[0]
    #fe_plot_index = np.where(atomic == 26)[0][0]
    #h_plot_index = np.where(atomic == 1)[0][0]
    # plot_atomic = atomic[map_data_index_to_plot]
    # fe_plot_index = map_data_index_to_plot[fe_plot_index]
    # atomic[map_data_index_to_plot[fe_plot_index]] = plot_atomic[fe_plot_index]

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
    fig, axs = plt.subplots(3, figsize=(5,5),sharex=True)#, sharey=True)
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
        mask_feh = (feh_plot[i] - feh_step/2. < feh) & (feh < feh_plot[i] + feh_step/2.)
        mdot.append(sum(model['mdot'][mask_feh]))
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
    axs[0].set_ylabel('dN / d[Fe/H]') 
    #plt.sca(axs[0])
    #plt.yticks([0,20,40,60,80])
    #print "plot index, atomic number, label"
    element_mask = (plot_atomic != 26) #& (atomic != 1)
 
    if data == True:
    #element_index = np.where(atomic1 == element_Z)[0][0] #Barium
        i=1
        dot_size = 0.4
        data_element = 'MgFe'
        data_err = 'Mgerr'
        # load Kirby 2010 Mg measurements
        filename_member = '/raid/m31/dsph/alldsph/dsph_catalog.dat'
        members = np.genfromtxt(filename_member, skip_header=1,
                                dtype=[('dSph', 'S9'), ('objname', 'S11'), ('RAh', '<i8'), ('RAm', '<i8'), ('RAs', '<f8'), ('Decd', '<i8'), 
                                       ('Decm', '<i8'), ('Decs', '<f8'), ('v_raw', '<f8'), ('Teff', '<f8'), ('logg', '<f8'), ('vt', '<f8'), 
                                       ('FeH', '<f8'), ('Feerr', '<f8'), ('AlphaFe', '<f8'), ('Alphaerr', '<f8'), ('MgFe', '<f8'), ('Mgerr', '<f8'),
                                       ('SiFe', '<f8'), ('Sierr', '<f8'), ('CaFe', '<f8'), ('Caerr', '<f8'), ('TiFe', '<f8'), ('Tierr', '<f8')],
                                delimiter=(9,11,3,3,6,3,3,6,7,5,5,6,6,5,6,5,6,5,6,5,6,5,6,5))
        members['dSph']=np.core.defchararray.strip(members['dSph'])
        members['objname']=np.core.defchararray.strip(members['objname'])
        #members = np.genfromtxt(filename_member, skip_header=1,delimiter=(9,11,3,3,6,3,3,6,7,5,5,6,6,5,6,5,6,5,6,5,6,5,6,5), names=True, dtype=None) #Use to get dtype list
        # AlphaFe and Alphaerr are not present in all objects, so can't read them in. 
        # Need to consider replacing spaces with negative value so error isn't thrown.
        mask_dsph = (dsph_name==members['dSph'])
        results = np.copy(members[mask_dsph])
        results['Feerr'] = np.sqrt(results['Feerr']**2+fehsyserr**2)
        results['Alphaerr'] = np.sqrt(results['Alphaerr']**2+alphasyserr**2)
        results['Mgerr'] = np.sqrt(results['Mgerr']**2+mgfesyserr**2)
            
        mask_mgfe = (results['Mgerr']<=accuracy_cutoff)&(results['Mgerr']>mgfesyserr
                )&(results['Alphaerr']<=accuracy_cutoff)&(results['Alphaerr']>alphasyserr
                )&(results['Feerr']<=accuracy_cutoff)&(results['Feerr']>fehsyserr)
        
        #mask_mgfe = (ba_results['MGFEERR']<=accuracy_cutoff)&(np.around(ba_results['MGFEERR'],3)>mgfesyserr)&(ba_results['BAFEERR']<0.3)&(np.around(ba_results['BAFEERR'],3)>0.0)
        err = np.reciprocal(results['Feerr'][mask_mgfe]**2+results[data_err][mask_mgfe]**2)
        axs[i].scatter(results['FeH'][mask_mgfe],results[data_element][mask_mgfe],
                       marker='o',c='k',s=err*1.3+dot_size, alpha=0.7,linewidth=0)#,capsize=3))
        print((np.nanmax(err*0.9),np.nanmin(err*0.9)))
        axs[i].errorbar([-.25], [-1.2],color='k',
                        xerr=[np.nanmedian(results['Feerr'][mask_mgfe])],
                        yerr=[np.nanmedian(results[data_err][mask_mgfe])])
        print((np.nanmedian(results[data_err][mask_mgfe])))
        i=2
        dot_size = 0.4
        data_element = 'BAFE'
        data_err = 'BAFEERR'
        err = np.reciprocal(ba_results['FEHERR'][mask]**2+ba_results[data_err][mask]**2)
        axs[i].scatter(ba_results['FEH'][mask],ba_results[data_element][mask],
                       marker='o',c='k',s=err*1.3+dot_size, alpha=0.7,linewidth=0)#,capsize=3))
        axs[i].errorbar([-.25], [-1.2],color='k',
                        xerr=[np.nanmedian(ba_results['FEHERR'][mask])],
                        yerr=[np.nanmedian(ba_results[data_err][mask])])
        print((np.nanmedian(ba_results[data_err][mask])))

    for i,j in enumerate([4,8]):
        print((i,j))
        plot_index = np.arange(len(plot_atomic))[element_mask][j]
        #print i+1, plot_atomic[plot_index], labels[i], np.ma.masked_invalid(model['eps'][:,map_data_index_to_plot[plot_index]]-model['eps'][:,map_data_index_to_plot[fe_plot_index]]).sum()#, atomic[map_data_index_to_plot[plot_index]]
        axs[i+1].plot(feh,model['eps'][:,map_data_index_to_plot[plot_index]]-model['eps'][:,map_data_index_to_plot[fe_plot_index]],'k-',label=title1)
        plt.ylim([-1.8,1.2])
        if single_plot == False:
            axs[i+1].plot(feh2,model2['eps'][:,map_data_index_to_plot2[plot_index]]-model2['eps'][:,map_data_index_to_plot2[fe_plot_index]],'r--',label=title2)
        elif i==0:
            #plt.title(title1)
            axs[0].set_title(title1)
        axs[i+1].set_ylabel(labels[j])
        axs[i+1].plot([-6,0],[0,0],':k')
        plt.sca(axs[i+1])
        if plot_atomic[plot_index] in [1,2]: #if it is hydrogen or helium
            plt.yticks([0.5,1.5,2.5,3.5])
            plt.ylim([0,3.5])
        elif plot_atomic[plot_index] in [56]:
            pass
        #else:                        
            #plt.yticks([-0.5,0.0,0.5,1])
            #plt.ylim([-0.8,1.2])
        #if (single_plot==False) & (i == len(plot_atomic)-2):
            #axs[i+1].legend()
    plt.xlim([-3.5,0])
    plt.xlabel('[Fe/H]')
    plt.savefig((plot_path+title1+title2+'_feh_refine.png').replace(' ',''))
    if plot==True: plt.show()
        
    nplots = 1
    fig, axs = plt.subplots(nplots, figsize=(5,2),sharex=True)#, sharey=True)
    #fig.subplots_adjust(bottom=0.24,top=0.95, left = 0.2,wspace=0.29,hspace=0)
    plt.setp([a.minorticks_on() for a in fig.axes[:]])
    #plt.title(title1 + ' vs. ' + title2 +"(dashed)", y = 1.1)
    if single_plot == False:
        plt.title("Final mass: %.1f (or %.1f) x $10^6$ M$_\odot$"%(model['mgal'][-1]/1e6,model2['mgal'][-1]/1e6))
        #plt.title(title1 + ' vs. ' + title2 +" (dashed)")
    else: 
        plt.title("Final mass: %.1f x $10^6$ M$_\odot$"%(model['mgal'][-1]/1e6), y=0.97)
        #plt.title(title1, y = 1.15)
    #axs = axs.ravel()    
    
    axs.plot(model['t'],model['mdot']/1e6,'k-')
    axs.set_ylabel('SFR $(10^6$ M$_\odot$Gyr$^{-1})$')
    
    if single_plot == False:
        axs.plot(model2['t'],model2['mdot']/1e6,'r--')                                        

    plt.xlabel('time (Gyr)')
    plt.xlim([0,1.7])
    plt.xticks([0.0,0.5,1.0,1.5])
    plt.tight_layout
    plt.savefig((plot_path+title1 + title2+'_sfh.png').replace(' ',''))
    if plot==True: plt.show()
    
def parameter_test(r_process_keyword = 'typical_SN',NSM_yield_mult=[0], plot=False,data=True):
    model16, atomic16 = gce.gce_model_ba(scl_pars,r_process_keyword = r_process_keyword, 
                                     NSM_yield_mult=NSM_yield_mult,imf_model = 'chabrier03', 
                                     AGB_source='kar16')
    num = 5
    par0 = np.linspace(500, 2500, num)
    par1 = np.linspace(0.17, 0.42, num)
    par2 = np.linspace(1.5, 11, num)
    par3 = np.linspace(0.4, 5, num)
    par4 = np.linspace(0.25, 1, num)
    par5 = np.linspace(0.05, 15, num)
    pars = [par0, par1, par2, par3, par4, par5]
    test_pars = scl_pars
    
    for i in range(len(scl_pars)):
        for j in range(num):
            test_pars[i] = pars[i][j]
            model15, atomic15 = gce.gce_model_ba(test_pars,r_process_keyword = r_process_keyword, 
                                             NSM_yield_mult=NSM_yield_mult, 
                                             imf_model = 'chabrier03', AGB_source='kar16')
            plot_alpha_vs_barium_fit('Scl',model16, atomic16, "Sculptor K15 parameters", 
                                     model15, atomic15, title2="test%i%i"%(i,pars[i][j])+r_process_keyword, 
                                     plot=plot, data=data)
            print(("test_%i%i"%(i,pars[i][j])))

def plots_abundance_slope():
    # These plots are to prove that the abundance slope would be the same between
    #    elements created solely by the same source 
    #    (regardless of the amount released per event)
    ###
    # short hand: CCSNe rate: y=10^4.8*x*e^(-x/0.3), SNIa rate: y=10^4*x*e^(-x/0.8)
    # n(x) = number density of element X
    # log(epsilon(x)) = log(n(x)/n(H))+12
    # [X/H] = log(epsilon(x))_obj - log(epsilon(x))_sun
    # [X/Y] = log(epsilon(x))_obj - log(epsilon(x))_sun - log(epsilon(y))_obj + log(epsilon(y))_sun
    fe_h = np.log10(n_fe/n_h) - feh_solar + 12
    mg_fe = np.log10(n_mg/n_fe) - mg_solar + fe_solar
    ba_fe = np.log10(n_ba/n_fe) - ba_solar + fe_solar
    

