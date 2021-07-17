"""
gce_comparison.py

This program runs multiple GCE models and plots their outputs on the same axes.
"""

# Backend for matplotlib on mahler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Import other packages
import numpy as np
import gce_fast as gce
from getdata import getdata
from astropy.cosmology import FlatLambdaCDM  # needed to compute redshifts
cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)  # using Planck (2018) params
import astropy.units as u
from astropy.cosmology import z_at_value

# Do some formatting stuff with matplotlib
from matplotlib import rc
rc('font', family='serif')
rc('axes', labelsize=14) 
rc('xtick', labelsize=12)
rc('ytick', labelsize=12)
rc('xtick.major', size=8)
rc('ytick.major', size=8)
rc('legend', fontsize=12, frameon=False)
rc('text',usetex=True)
rc('xtick',direction='in')
rc('ytick',direction='in')
import cycler
import cmasher as cmr

# Parameters for GCE models
scl_test_bothba_ba = [0.43936705, 0.30507695, 4.94212835, 0.49191929, 0.83307305, 0.40318864, 0.5627217, 1.291076, 0.85956343, 0.28562448, 1.56742018, 0.89342641, 0.01507226, 0.]
scl_delaySFtest = [0.36963815, 0.34288084, 5.07381209, 0.73231527, 0.62864803, 0.29277844, 0.57543231, 1.2897485, 0.84635716, 0.30015869, 1.63283893, 0.93347269, 0.01408449, 0.]
scl_reioniz = [0.6411130691937341,0.24774922551128908,4.643962259917035,0.780729799230917,0.8778813431231577,0.612699567249839,0.6460839575200857,1.325818097299844,0.7336606535374587,0.26137519407263077,2.7368268789441252,0.9896010405595509,0.0056036256827435346, 0.]
scl_rampressure = [0.40324345287403596,0.31993145962170993,4.678875833236568,0.4311305388839332,0.901133475075236,0.3874332761124177,0.5687645229241343,1.2899619752803073,0.8487435966713881,0.2857591805747674,1.5867499800816725,0.9139277884487471,0.013695390962180884,2.761601093904625]
scl_iadtd_maoz17 = [0.5164266933312606,0.22668646095205333,5.083938866111591,0.6708444072138402,0.7705726151153531,0.5831520287559541,0.8952553665964144,1.3892756558045938,0.47734137612923366,0.14861562077682866,0.435341510493796,0.33493152648882035,0.013300279965722219, 0.]
scl_iadtd_himin = [0.12655011964708354,0.48083816401816953,4.774355073300842,1.0142068952827288,0.13937182878786417,0.9673484547698562,0.8931868300415441,1.548247398565755,0.40718356024183644,0.037801676437251096,0.5965616973892593,0.8185482795511301,1.3007176610686921, 0.]
scl_iadtd_lomin = [0.5671755376691066,0.29287428128668863,5.015868834444396,0.30612159610634737,1.0034407937884338,0.4612588576531018,0.44599230385432126,1.161552777383641,1.2779361777765668,0.43394265803765714,1.326460414915983,0.9806089731602307,0.0059330597053563775, 0.]
scl_iadtd_medhimin = [0.2581362038956129,0.3671543880935386,4.884509919096489,0.519627246584796,0.6921089016677752,0.7093941886457517,0.667125036335381,1.4048235865635883,0.6442822761890015,0.17279690215969257,1.076567643613428,0.5756914070867104,0.0789755064127836, 0.]
scl_iadtd_loindex = [0.31309403734878677,0.32698844029546187,5.119860962340789,0.5074794913085319,0.7677725611456582,0.27817942445348165,0.7747072609145225,1.3579266977743019,0.6409945773107304,0.24639512831333843,0.8835860105097602,0.5557520537168783,0.023556275510243575, 0.]
scl_iadtd_hiindex = [0.5446693466306317,0.3092340505074539,4.662881112688744,0.6610461169621856,0.6648004259776421,0.22834192428764163,0.434048932393723,1.2372641358088885,1.21868854143266,0.30455907377622926,2.5503064633438433,0.9921019155833941,0.00552116094663595, 0.]
scl_iadtd_cutoff = [0.3907399651848807,0.31789100855381613,4.976079316209285,0.4695236246906028,0.846066267741512,0.3848772970464857,0.5875359459715601,1.301460415128067,0.8259997983101177,0.28742136661443407,1.3484797631127226,0.7983782066064008,0.017047139327600602]
scl_imf_chabrier03 = [1.0680354182219103,0.29087824048307825,5.785175190841888,0.32614582504078626,0.6880109337422085,0.47419668814764776,0.8347392374670606,1.3517172298659013,0.5890139428180761,0.2739631120786506,1.7994398252473034,0.9807143044292836,0.011967114634611836]
scl_imf_salpeter55 = [0.5089476125938007,0.32350548351207437,5.391102320123509,0.4003999995632118,0.7799460946453387,0.5549969145097149,0.6164885754010938,1.3299979696872426,0.7198534106528632,0.25245975628500583,2.182828951358294,0.9847378266515173,0.00954264476609045]

def compare_sfh(models, title, fiducialtitle='Fiducial'):
    """Compare SFH outputs from multiple GCE models."""

    # Run GCE models
    model, _, ll = gce.runmodel(scl_test_bothba_ba, plot=False, title="Sculptor dSph", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False)
    model_reioniz, _, ll_reioniz = gce.runmodel(scl_reioniz, plot=False, title="Sculptor dSph (reionization)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=True)
    model_delaysf, _, ll_delaysf = gce.runmodel(scl_delaySFtest, plot=False, title="Sculptor dSph (delayed SF)", empirical=True, empiricalfit=True, feh_denom=True, delay=True)
    model_rampressure, _, ll_rampressure = gce.runmodel(scl_rampressure, plot=False, title="Sculptor dSph (ram pressure)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False, rampressure=True)
    model_maoz17, _, ll_maoz17 = gce.runmodel(scl_iadtd_maoz17, plot=False, title="Sculptor dSph (Maoz+17 field Ia DTD)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, ia_dtd='maoz17')
    model_lomin, _, ll_lomin = gce.runmodel(scl_iadtd_lomin, plot=False, title="Sculptor dSph (min Ia delay time = 50 Myr)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, ia_dtd='lowmindelay')
    model_himin, _, ll_himin = gce.runmodel(scl_iadtd_himin, plot=False, title="Sculptor dSph (min Ia delay time = 500 Myr)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, ia_dtd='highmindelay')
    model_medhimin, _, ll_medhimin = gce.runmodel(scl_iadtd_medhimin, plot=False, title="Sculptor dSph (min Ia delay time = 200 Myr)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, ia_dtd='medhidelay')
    model_loindex, _, ll_loindex = gce.runmodel(scl_iadtd_loindex, plot=False, title="Sculptor dSph (Ia DTD index = -0.5)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, ia_dtd='index05')
    model_hiindex, _, ll_hiindex = gce.runmodel(scl_iadtd_hiindex, plot=False, title="Sculptor dSph (Ia DTD index = -1.5)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, ia_dtd='index15')
    model_cutoff, _, ll_cutoff = gce.runmodel(scl_iadtd_cutoff, plot=False, title="Sculptor dSph (Ia DTD with cutoff)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, ia_dtd='cutoff')
    model_chabrier, _, ll_chabrier = gce.runmodel(scl_imf_chabrier03, plot=False, title="Sculptor dSph (Chabrier+03 IMF)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, imf='chabrier03')
    model_salpeter, _, ll_salpeter = gce.runmodel(scl_imf_salpeter55, plot=False, title="Sculptor dSph (Salpeter+55 IMF)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, imf='salpeter55')

    # Get model names
    modelnames = {'fiducial':model, 'reioniz':model_reioniz, 'delaysf':model_delaysf, 'rampressure':model_rampressure,
                'iadtd_maoz17':model_maoz17, 'iadtd_lomin':model_lomin, 'iadtd_himin':model_himin, 'iadtd_medhimin':model_medhimin, 'iadtd_index05':model_loindex, 'iadtd_index15':model_hiindex, 'iadtd_cutoff':model_cutoff,
                'imf_chabrier':model_chabrier, 'imf_salpeter':model_salpeter}
    llnames = {'fiducial':ll, 'reioniz':ll_reioniz, 'delaysf':ll_delaysf, 'rampressure':ll_rampressure,
                'iadtd_maoz17':ll_maoz17,'iadtd_lomin':ll_lomin,'iadtd_himin':ll_himin, 'iadtd_medhimin':ll_medhimin, 'iadtd_index05':ll_loindex, 'iadtd_index15':ll_hiindex, 'iadtd_cutoff':ll_cutoff,
                'imf_chabrier':ll_chabrier, 'imf_salpeter':ll_salpeter}
    titles = {'fiducial':'Fiducial', 'reioniz':'With reionization', 'delaysf':'Delayed SF', 'rampressure':'Ram pressure',
                'iadtd_maoz17':'Maoz+17 Ia DTD', 'iadtd_lomin':'Ia DTD: '+r'$t_{\mathrm{min}}=50$Myr', 'iadtd_himin':'Ia DTD: '+r'$t_{\mathrm{min}}=500$Myr', 'iadtd_medhimin':'Ia DTD: '+r'$t_{\mathrm{min}}=200$Myr', 
                'iadtd_index05':'Ia DTD: '+r'$t^{-0.5}$', 'iadtd_index15':'Ia DTD: '+r'$t^{-1.5}$', 'iadtd_cutoff':'Ia DTD with cutoff',
                'imf_chabrier':'Chabrier (2003) IMF', 'imf_salpeter':'Salpeter (1955) IMF'}

    # Create figure
    fig = plt.figure(figsize=(6,4))
    ax = plt.subplot()
    #plt.title(title, fontsize=16)
    plt.xlabel('Time (Gyr)')
    plt.setp([a.minorticks_on() for a in fig.axes[:]])
    plt.ylabel('SFR ($10^{-4}$ M$_\odot$ yr$^{-1})$')

    # Add redshift axis on top
    ages = np.array([13, 10, 8, 6, 4, 3, 2, 1.5, 1.2, 1, 0.8, 0.6])*u.Gyr
    ageticks = [z_at_value(cosmo.age, age) for age in ages]
    ax2 = ax.twiny()
    ax2.set_xticks(ageticks)
    ax2.set_xticklabels(['{:g}'.format(age) for age in ages.value])
    zmin, zmax = 0.0, 2
    ax.set_xlim(zmin, zmax)
    ax2.set_xlim(zmin, zmax)
    plt.ylim(0,40)

    colors = ['k', plt.cm.Set2(0), plt.cm.Set2(1), plt.cm.Set2(2)]
    linestyles = ['-','--',':','dashdot']

    # For Ia DTD plot
    #colors = ['k', plt.cm.tab10(0), plt.cm.tab20(2), plt.cm.tab20(3), plt.cm.tab20(4), plt.cm.tab20(5), plt.cm.tab20(6)]
    #linestyles = ['-','--',':',':','dashdot','dashdot',(0,(5,10))]
    
    # Plot SFH from different models
    for i, modelname in enumerate(models):
        model = modelnames[modelname]
        if modelname=='fiducial':
            plt.plot(model['t'],model['mdot']/1e5, color=colors[i], ls=linestyles[i], lw=2, label=fiducialtitle) #, label=titles[modelname]) #+r', $-\log(L)=$'+' '+str(-int(llnames[modelname])))
        else:
            plt.plot(model['t'],model['mdot']/1e5, color=colors[i], ls=linestyles[i], lw=2, label=titles[modelname]+r', $\Delta(\mathrm{AIC})=$'+' '+str(int(ll-llnames[modelname])))

    plt.legend(loc='best')
    plt.savefig('plots/'+title.replace(' ','')+'_sfhcompare.png', bbox_inches='tight')
    plt.show()

    return

def compare_yields(plottype, feh_denom=True):
    """Compare abundance trends for elem (options: 'mn', 'ni', 'ba')."""

    # Set keyword to remove r-process yields
    if plottype in ['rprocess','rprocess_bestfit']:
        removerprocess=None  # If plotting r-process test, keep both r- and s-process Ba
    else: 
        removerprocess='statistical'

    # Open observed data
    elem_data, delem_data, elems = getdata(galaxy='Scl', source='deimos', c=True, ba=True, mn=True, eu=True, ni=True, removerprocess=removerprocess) #, feh_denom=feh_denom)
    elem_data_dart, delem_data_dart, _ = getdata(galaxy='Scl', source='dart', c=True, ba=True, mn=True, eu=True, ni=True, removerprocess=removerprocess) #, feh_denom=feh_denom)
    print('test', elems)

    # Get [Fe/H] DEIMOS data
    x_obs = elem_data[0,:]
    obsmask = np.where((x_obs > -3.5) & (x_obs < 0.) & (delem_data[0,:] < 0.4))[0]
    x_obs = x_obs[obsmask]
    x_errs = delem_data[0,obsmask]

    # Get [Fe/H] DART data
    x_obs_dart = elem_data_dart[0,:]
    obsmask_dart = np.where((x_obs_dart > -3.5) & (x_obs_dart < 0.) & (delem_data_dart[0,:] < 0.4))[0]
    x_obs_dart = x_obs_dart[obsmask_dart]
    x_errs_dart = delem_data_dart[0,obsmask_dart]

    if plottype=='IaSN':
        # Make plot
        fig, axs = plt.subplots(2, figsize=(6,4), sharex=True)
        fig.subplots_adjust(bottom=0.06,top=0.97, left = 0.2,wspace=0.29,hspace=0)
        plt.xlabel('[Fe/H]')
        plt.xlim([-3.5,0])

        # Titles
        titles = {'MCh_DDT':'Leung \& Nomoto (2018) '+r'$M_{\mathrm{Ch}}$'+' DDT',
                    'MCh_def':'Leung \& Nomoto (2018) '+r'$M_{\mathrm{Ch}}$'+' pure def',
                    'subMCh_He':'Leung \& Nomoto (2020) sub-'+r'$M_{\mathrm{Ch}}$'+' He shell',
                    'subMCh_bare':'Shen et al. (2018) sub-'+r'$M_{\mathrm{Ch}}$'+' bare'}

        labels = ['Mn','Ni']
        elemdicts = [{'MCh_DDT':7.25e-3, 'MCh_def':8.21e-3, 'subMCh_He':1.79e-3, 'subMCh_bare':0.1383e-3},{'MCh_DDT':5.65e-2, 'MCh_def':5.89e-2, 'subMCh_He':1.53e-2, 'subMCh_bare':2.19e-2}]
        elem_idx = [6,8]

        for idx, label in enumerate(labels):
            axs[idx].set_ylabel('['+label+'/Fe]')
            axs[idx].plot([-3.5,0],[0,0],':k')

            # Plot observed DEIMOS data
            obs_data = elem_data[elem_idx[idx],obsmask]
            obs_errs = delem_data[elem_idx[idx],obsmask]
            goodidx = np.where((x_obs > -990) & (obs_data > -990) & (np.abs(obs_errs) < 0.4) & (np.abs(x_errs) < 0.4))[0]
            axs[idx].errorbar(x_obs[goodidx], obs_data[goodidx], xerr=x_errs[goodidx], yerr=obs_errs[goodidx], 
                                        color=plt.cm.Set3(0), linestyle='None', marker='o', markersize=3, alpha=0.7, linewidth=0.5)

            # Repeat for DART data
            obs_data = elem_data_dart[elem_idx[idx],obsmask_dart]
            obs_errs = delem_data_dart[elem_idx[idx],obsmask_dart]
            goodidx = np.where((x_obs_dart > -990) & (obs_data > -990) & (np.abs(obs_errs) < 0.4) & (np.abs(x_errs_dart) < 0.4))[0]
            axs[idx].errorbar(x_obs_dart[goodidx], obs_data[goodidx], xerr=x_errs_dart[goodidx], yerr=obs_errs[goodidx], 
                            mfc='white', mec=plt.cm.Set3(3), ecolor=plt.cm.Set3(3), linestyle='None', marker='o', markersize=3, linewidth=0.5)

            # Set colorwheel
            cwheelsize = len(titles)
            color = cmr.bubblegum(np.linspace(0,1,cwheelsize,endpoint=True))
            matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
            cwheel = [np.array(matplotlib.rcParams['axes.prop_cycle'])[x]['color'] for x in range(cwheelsize)]

            for line_idx, iamodel in enumerate(elemdicts[idx]):
                if label=='Mn':
                    model, atomic, ll = gce.runmodel(scl_test_bothba_ba, plot=False, title="Sculptor dSph", empirical=True, empiricalfit=True, mn=elemdicts[idx][iamodel])
                elif label=='Ni':
                    model, atomic, ll = gce.runmodel(scl_test_bothba_ba, plot=False, title="Sculptor dSph", empirical=True, empiricalfit=True, ni=elemdicts[idx][iamodel])

                # Get indexes to access model
                elem_names = {1:'H', 2:'He', 6:'C', 8:'O', 12:'Mg', 14:'Si', 20:'Ca', 22:'Ti', 25:'Mn', 26:'Fe', 28:'Ni', 56:'Ba', 63:'Eu'}
                snindex = {}
                for snindex_idx, snindex_elem in enumerate(atomic):
                    snindex[elem_names[snindex_elem]] = snindex_idx

                x = model['eps'][:,snindex['Fe']]
                y = model['eps'][:,snindex[label]] - model['eps'][:,snindex['Fe']]
                axs[idx].plot(x, y, color=cwheel[line_idx], linestyle='-', lw=4, label=titles[iamodel], zorder=100)

            # Create legend
            if idx==0:
                axs[idx].legend(fontsize=10, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left")
        plt.savefig('plots/'+'iasn_compare.png', bbox_inches='tight')
        plt.show()

    if plottype=='rprocess':
        # Make plot
        fig, axs = plt.subplots(2, figsize=(6,4), sharex=True)
        fig.subplots_adjust(bottom=0.06,top=0.97, left = 0.2,wspace=0.29,hspace=0)
        plt.xlabel('[Fe/H]')
        plt.xlim([-3.5,0])

        # Titles
        titles = ['No r-process','Prompt r-process (typical CCSNe only)','Delayed r-process ('+r'$t^{-1.5}$ DTD, '+r'$t_{\mathrm{min}}=10$ Myr)','Prompt+delayed']
        rprocesskeywords = ['none','typical_SN_only','rare_event_only','both']

        labels = ['Ba','Eu']
        elem_idx = [5,7]

        for idx, label in enumerate(labels):
            if feh_denom:
                axs[idx].set_ylabel('['+label+'/Fe]')
                axs[idx].plot([-3.5,0],[0,0],':k')
                axs[idx].set_ylim([-2,2])
            else:
                axs[idx].set_ylabel('['+label+'/Mg]')
                axs[idx].plot([-3.5,0],[0,0],':k')
                axs[idx].set_ylim([-3,2])

            # Plot observed DEIMOS data
            if feh_denom:
                obs_data = elem_data[elem_idx[idx],obsmask]
                obs_errs = delem_data[elem_idx[idx],obsmask]
            else:
                obs_data = elem_data[elem_idx[idx],obsmask] - elem_data[1,obsmask]
                obs_errs = np.sqrt(delem_data[elem_idx[idx],obsmask]**2. + delem_data[1,obsmask]**2.)
            goodidx = np.where((x_obs > -990) & (obs_data > -990) & (np.abs(obs_errs) < 0.4) & (np.abs(x_errs) < 0.4))[0]
            axs[idx].errorbar(x_obs[goodidx], obs_data[goodidx], xerr=x_errs[goodidx], yerr=obs_errs[goodidx], 
                                        color=plt.cm.Set3(0), linestyle='None', marker='o', markersize=3, alpha=0.7, linewidth=0.5)

            # Repeat for DART data
            if feh_denom:
                obs_data = elem_data_dart[elem_idx[idx],obsmask_dart]
                obs_errs = delem_data_dart[elem_idx[idx],obsmask_dart]
            else:
                obs_data = elem_data_dart[elem_idx[idx],obsmask_dart] - elem_data_dart[1,obsmask_dart]
                obs_errs = np.sqrt(delem_data_dart[elem_idx[idx],obsmask_dart]**2. + delem_data_dart[1,obsmask_dart]**2.)
            goodidx = np.where((x_obs_dart > -990) & (obs_data > -990) & (np.abs(obs_errs) < 0.4) & (np.abs(x_errs_dart) < 0.4))[0]
            axs[idx].errorbar(x_obs_dart[goodidx], obs_data[goodidx], xerr=x_errs_dart[goodidx], yerr=obs_errs[goodidx], 
                            mfc='white', mec=plt.cm.Set3(3), ecolor=plt.cm.Set3(3), linestyle='None', marker='o', markersize=3, linewidth=0.5)

            # Set colorwheel
            cwheelsize = len(titles)
            color = cmr.cosmic_r(np.linspace(0,1,cwheelsize,endpoint=True))
            matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
            cwheel = [np.array(matplotlib.rcParams['axes.prop_cycle'])[x]['color'] for x in range(cwheelsize)]

            for line_idx, keyword in enumerate(rprocesskeywords):
                model, atomic, ll = gce.runmodel(scl_test_bothba_ba, plot=False, title="Sculptor dSph", empirical=True, empiricalfit=True, rprocess=keyword)
                
                # Get indexes to access model
                elem_names = {1:'H', 2:'He', 6:'C', 8:'O', 12:'Mg', 14:'Si', 20:'Ca', 22:'Ti', 25:'Mn', 26:'Fe', 28:'Ni', 56:'Ba', 63:'Eu'}
                snindex = {}
                for snindex_idx, snindex_elem in enumerate(atomic):
                    snindex[elem_names[snindex_elem]] = snindex_idx
            
                if feh_denom:
                    x = model['eps'][:,snindex['Fe']]
                    y = model['eps'][:,snindex[label]] - model['eps'][:,snindex['Fe']]
                else:
                    x = model['eps'][:,snindex['Fe']]
                    y = model['eps'][:,snindex[label]] - (model['eps'][:,snindex['Mg']] + 0.2)
                axs[idx].plot(x, y, color=cwheel[line_idx], linestyle='-', lw=4, label=titles[line_idx], zorder=100)

            # Create legend
            if idx==0:
                axs[idx].legend(fontsize=10, bbox_to_anchor=(-0.03,1.02,1,0.2), loc="lower left")

        #plt.legend(fontsize=10) #, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left")
        plt.savefig('plots/'+'rprocess_compare.png', bbox_inches='tight')
        plt.show()

    if plottype=='rprocess_bestfit':
        # Make plot
        fig, axs = plt.subplots(2, figsize=(6,4), sharex=True)
        fig.subplots_adjust(bottom=0.06,top=0.97, left = 0.2,wspace=0.29,hspace=0)
        plt.xlabel('[Fe/H]')
        plt.xlim([-3.5,0])

        # Titles
        titles = ['Prompt r-process (20x Ba yield, 5x Eu yield)','Prompt (20x Ba yield, 5x Eu yield) + delayed','Delayed r-process ('+r'$t_{\mathrm{min}}=50$ Myr, 5x normalization)','Prompt + delayed ('+r'$t_{\mathrm{min}}=50$ Myr, 5x normalization)']
        rprocesskeywords = ['typical_SN_only','both','rare_event_only','both']
        specialkeywords = ['enhancedprompt','enhancedprompt','enhanceddelay','enhanceddelay']
        ls = [':',':','--','--']
        # Set colorwheel
        cwheelsize = len(titles)
        color = cmr.cosmic_r(np.linspace(0,1,cwheelsize,endpoint=True))
        cwheel = [color[1],color[3],color[2],color[3]]

        labels = ['Ba','Eu']
        elem_idx = [5,7]

        for idx, label in enumerate(labels):
            if feh_denom:
                axs[idx].set_ylabel('['+label+'/Fe]')
                axs[idx].plot([-3.5,0],[0,0],':k')
                axs[idx].set_ylim([-2,2])
            else:
                axs[idx].set_ylabel('['+label+'/Mg]')
                axs[idx].plot([-3.5,0],[0,0],':k')
                axs[idx].set_ylim([-3,2])

            # Plot observed DEIMOS data
            if feh_denom:
                obs_data = elem_data[elem_idx[idx],obsmask]
                obs_errs = delem_data[elem_idx[idx],obsmask]
            else:
                obs_data = elem_data[elem_idx[idx],obsmask] - elem_data[1,obsmask]
                obs_errs = np.sqrt(delem_data[elem_idx[idx],obsmask]**2. + delem_data[1,obsmask]**2.)
            goodidx = np.where((x_obs > -990) & (obs_data > -990) & (np.abs(obs_errs) < 0.4) & (np.abs(x_errs) < 0.4))[0]
            axs[idx].errorbar(x_obs[goodidx], obs_data[goodidx], xerr=x_errs[goodidx], yerr=obs_errs[goodidx], 
                                        color=plt.cm.Set3(0), linestyle='None', marker='o', markersize=3, alpha=0.7, linewidth=0.5)

            # Repeat for DART data
            if feh_denom:
                obs_data = elem_data_dart[elem_idx[idx],obsmask_dart]
                obs_errs = delem_data_dart[elem_idx[idx],obsmask_dart]
            else:
                obs_data = elem_data_dart[elem_idx[idx],obsmask_dart] - elem_data_dart[1,obsmask_dart]
                obs_errs = np.sqrt(delem_data_dart[elem_idx[idx],obsmask_dart]**2. + delem_data_dart[1,obsmask_dart]**2.)
            goodidx = np.where((x_obs_dart > -990) & (obs_data > -990) & (np.abs(obs_errs) < 0.4) & (np.abs(x_errs_dart) < 0.4))[0]
            axs[idx].errorbar(x_obs_dart[goodidx], obs_data[goodidx], xerr=x_errs_dart[goodidx], yerr=obs_errs[goodidx], 
                            mfc='white', mec=plt.cm.Set3(3), ecolor=plt.cm.Set3(3), linestyle='None', marker='o', markersize=3, linewidth=0.5)

            for line_idx, keyword in enumerate(rprocesskeywords):
                model, atomic, ll = gce.runmodel(scl_test_bothba_ba, plot=False, title="Sculptor dSph", empirical=True, empiricalfit=True, rprocess=keyword, specialrprocess=specialkeywords[line_idx])
                
                # Get indexes to access model
                elem_names = {1:'H', 2:'He', 6:'C', 8:'O', 12:'Mg', 14:'Si', 20:'Ca', 22:'Ti', 25:'Mn', 26:'Fe', 28:'Ni', 56:'Ba', 63:'Eu'}
                snindex = {}
                for snindex_idx, snindex_elem in enumerate(atomic):
                    snindex[elem_names[snindex_elem]] = snindex_idx
            
                if feh_denom:
                    x = model['eps'][:,snindex['Fe']]
                    y = model['eps'][:,snindex[label]] - model['eps'][:,snindex['Fe']]
                else:
                    x = model['eps'][:,snindex['Fe']]
                    y = model['eps'][:,snindex[label]] - (model['eps'][:,snindex['Mg']] + 0.2)
                axs[idx].plot(x, y, color=cwheel[line_idx], linestyle=ls[line_idx], lw=4, label=titles[line_idx], zorder=100)

            # Create legend
            if idx==0:
                axs[idx].legend(fontsize=10, bbox_to_anchor=(-0.03,1.02,1,0.2), loc="lower left")

        #plt.legend(fontsize=10) #, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left")
        plt.savefig('plots/'+'rprocess_compare_bestfit.png', bbox_inches='tight')
        plt.show()

    return

if __name__=="__main__":
    #compare_sfh(['fiducial','delaysf','reioniz','rampressure'], 'modelconstruction')
    #compare_sfh(['fiducial','iadtd_medhimin','iadtd_index05','iadtd_cutoff'], 'iadtd', fiducialtitle='Fiducial: '+r'$t^{-1.1}$, '+r'$t_{\mathrm{min}}=100$Myr')
    #compare_sfh(['fiducial','imf_chabrier','imf_salpeter'], 'imf', fiducialtitle='Fiducial: Kroupa et al. (1993) IMF')
    compare_yields('IaSN')
    #compare_yields('rprocess', feh_denom=False)
    #compare_yields('rprocess_bestfit', feh_denom=False)