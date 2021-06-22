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
from astropy.cosmology import FlatLambdaCDM  # needed to compute redshifts
cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)  # using Planck (2018) params
import astropy.units as u
from astropy.cosmology import z_at_value

# Do some formatting stuff with matplotlib
from matplotlib import rc
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, MaxNLocator
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

# Parameters for GCE models
scl_test_bothba_ba = [0.43936705, 0.30507695, 4.94212835, 0.49191929, 0.83307305, 0.40318864, 0.5627217, 1.291076, 0.85956343, 0.28562448, 1.56742018, 0.89342641, 0.01507226]
scl_delaySFtest = [0.36963815, 0.34288084, 5.07381209, 0.73231527, 0.62864803, 0.29277844, 0.57543231, 1.2897485, 0.84635716, 0.30015869, 1.63283893, 0.93347269, 0.01408449]
scl_reioniz = [0.6411130691937341,0.24774922551128908,4.643962259917035,0.780729799230917,0.8778813431231577,0.612699567249839,0.6460839575200857,1.325818097299844,0.7336606535374587,0.26137519407263077,2.7368268789441252,0.9896010405595509,0.0056036256827435346]
scl_iadtd_maoz17 = [0.5164266933312606,0.22668646095205333,5.083938866111591,0.6708444072138402,0.7705726151153531,0.5831520287559541,0.8952553665964144,1.3892756558045938,0.47734137612923366,0.14861562077682866,0.435341510493796,0.33493152648882035,0.013300279965722219]
scl_iadtd_himin = [0.12655011964708354,0.48083816401816953,4.774355073300842,1.0142068952827288,0.13937182878786417,0.9673484547698562,0.8931868300415441,1.548247398565755,0.40718356024183644,0.037801676437251096,0.5965616973892593,0.8185482795511301,1.3007176610686921]
scl_iadtd_lomin = [0.5671755376691066,0.29287428128668863,5.015868834444396,0.30612159610634737,1.0034407937884338,0.4612588576531018,0.44599230385432126,1.161552777383641,1.2779361777765668,0.43394265803765714,1.326460414915983,0.9806089731602307,0.0059330597053563775]
scl_iadtd_medhimin = [0.2581362038956129,0.3671543880935386,4.884509919096489,0.519627246584796,0.6921089016677752,0.7093941886457517,0.667125036335381,1.4048235865635883,0.6442822761890015,0.17279690215969257,1.076567643613428,0.5756914070867104,0.0789755064127836]

# Run GCE models
model, _, ll = gce.runmodel(scl_test_bothba_ba, plot=False, title="Sculptor dSph", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False)
model_reioniz, _, ll_reioniz = gce.runmodel(scl_reioniz, plot=False, title="Sculptor dSph (reionization)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=True)
model_delaysf, _, ll_delaysf = gce.runmodel(scl_delaySFtest, plot=False, title="Sculptor dSph (delayed SF)", empirical=True, empiricalfit=True, feh_denom=True, delay=True)
model_maoz17, _, ll_maoz17 = gce.runmodel(scl_iadtd_maoz17, plot=False, title="Sculptor dSph (Maoz+17 field Ia DTD)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, ia_dtd='maoz17')
model_lomin, _, ll_lomin = gce.runmodel(scl_iadtd_lomin, plot=False, title="Sculptor dSph (min Ia delay time = 50 Myr)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, ia_dtd='lowmindelay')
model_himin, _, ll_himin = gce.runmodel(scl_iadtd_himin, plot=False, title="Sculptor dSph (min Ia delay time = 500 Myr)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, ia_dtd='highmindelay')
model_medhimin, _, ll_medhimin = gce.runmodel(scl_iadtd_medhimin, plot=False, title="Sculptor dSph (min Ia delay time = 200 Myr)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, ia_dtd='medhidelay')

# Get model names
modelnames = {'fiducial':model, 'reioniz':model_reioniz, 'delaysf':model_delaysf, 
            'iadtd_maoz17':model_maoz17, 'iadtd_lomin':model_lomin, 'iadtd_himin':model_himin, 'iadtd_medhimin':model_medhimin}
llnames = {'fiducial':ll, 'reioniz':ll_reioniz, 'delaysf':ll_delaysf, 
            'iadtd_maoz17':ll_maoz17,'iadtd_lomin':ll_lomin,'iadtd_himin':ll_himin, 'iadtd_medhimin':ll_medhimin}
titles = {'fiducial':'Fiducial', 'reioniz':'With reionization', 'delaysf':'Delayed SF',
            'iadtd_maoz17':'Maoz+17 Ia DTD', 'iadtd_lomin':'Ia DTD: '+r'$t_{\mathrm{min}}=50$Myr', 'iadtd_himin':'Ia DTD: '+r'$t_{\mathrm{min}}=500$Myr', 'iadtd_medhimin':'Ia DTD: '+r'$t_{\mathrm{min}}=200$Myr'}

colors = ['k', plt.cm.Set2(0), plt.cm.Set2(1), plt.cm.Set2(2)]
linestyles = ['-','--',':','dashdot']

def compare_sfh(models, title):
    """Compare SFH outputs from multiple GCE models."""

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
    
    # Plot SFH from different models
    for i, modelname in enumerate(models):
        model = modelnames[modelname]
        plt.plot(model['t'],model['mdot']/1e5, color=colors[i], ls=linestyles[i], lw=2, label=titles[modelname]+r', $-\log(L)=$'+' '+str(-int(llnames[modelname])))

    plt.legend(loc='best')
    plt.savefig('plots/'+title.replace(' ','')+'_sfhcompare.png', bbox_inches='tight')
    plt.show()

    return

if __name__=="__main__":
    compare_sfh(['fiducial','delaysf','reioniz'], 'modelconstruction')
    compare_sfh(['fiducial','iadtd_maoz17','iadtd_lomin','iadtd_medhimin'], 'iadtd')