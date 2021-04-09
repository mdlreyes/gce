"""
plot_yields.py

Plot yields as a function of mass, metallicity
"""

# Backend for matplotlib on mahler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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
import cycler
import cmasher as cmr

# Import other packages
import numpy as np
from gce_yields import *
from scipy import interpolate
import imf
import params

# Define list of yield sources
titles = {'nom06':'Nomoto et al. (2006)','nom13':'Nomoto et al. (2013)','lim18':'Limongi \& Chieffi (2018)',
        'leu20':'Leung \& Nomoto (2020)','cri15':'FRUITY (Cristallos et al. 2015)','kar':'Karakas et al. (2016, 2018)'}

def plotyields(yieldsource, yield_path='yields/'):
    """Plot yields as a function of mass, metallicity

    Args:
        yieldsource (str): which yields to use
    """

    # Figure out which yield is being altered
    if yieldsource in ['nom06','nom13','lim18']:
        yields, M, loadZ = load_II(yieldsource, yield_path, nel, atomic_names, atomic_num)
        yields = yields['II']
        yieldtype='CCSN'

    elif yieldsource in ['leu20']:
        _, _, loadZ = load_II('nom13', yield_path, nel, atomic_names, atomic_num)
        yields = load_Ia(yieldsource, yield_path, SN_yield, atomic_names, loadZ)   
        yieldtype='IaSN'

    elif yieldsource in ['cri15','kar']:
        M, loadZ, yields = load_AGB(AGB_source, yield_path, atomic_num, atomic_names, atomic_weight)
        yieldtype='AGB'

    else:
        raise ValueError('yieldsource is not valid!')

    # If needed, extrapolate yields to Z=0
    if ~np.isclose(loadZ[0],0.):
        if yieldtype in ['CCSN','AGB']:
            yields_z0 = yields[:,0,:]+(0-loadZ[0])*(yields[:,1,:]-yields[:,0,:])/(loadZ[1]-loadZ[0])
            yields = np.concatenate((yields_z0[:,None,:], yields), axis=1)   # Concatenate yield tables

        else:
            yields_z0 = yields[:,0]+(0-loadZ[0])*(yields[:,1]-yields[:,0])/(loadZ[1]-loadZ[0])
            yield_ia = np.concatenate((yields_z0[:,None], yields), axis=1)   # Concatenate yield tables

        loadZ = np.concatenate(([0],loadZ))

    # Interpolate yields to a common metallicity scale
    Z = np.linspace(0,2e-3,5)
    f_interp = interpolate.interp1d(loadZ, yields, axis=1)
    yields = f_interp(Z)

    # Weight yields by IMF
    dN_dM = imf.imf(M, params.imf_model)
    print(yields.shape, dN_dM.shape)
    yields = yields * dN_dM

    # Make plots!

    # First, get color wheel
    color = cmr.heat(np.linspace(0,1,len(Z),endpoint=False))
    matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
    cwheel = [np.array(matplotlib.rcParams['axes.prop_cycle'])[x]['color'] for x in range(len(Z))]

    # Create labels
    elem_atomic = [1, 2, 6, 12, 14, 20, 22, 25, 26]
    elem_names = {1:'H', 2:'He', 6:'C', 8:'O', 12:'Mg', 14:'Si', 20:'Ca', 22:'Ti', 25:'Mn', 26:'Fe'}
    # Add other yields if needed
    if yieldtype=='AGB':
        elem_names[56]='Ba'
        elem_names[63]='Eu'
        elem_atomic.append(56)
        elem_atomic.append(63)
    
    # Create and format plot
    fig, axs = plt.subplots(len(elem_atomic), figsize=(5,12),sharex=True)
    fig.subplots_adjust(bottom=0.06,top=0.96,left=0.2,wspace=0.29,hspace=0)
    plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune=None)) for a in [fig.axes[0]]])
    plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='upper')) for a in fig.axes[1:]])
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.setp([a.minorticks_on() for a in fig.axes[:]])
    axs = axs.ravel()

    # Create each subplot
    for idx_elem, elem in enumerate(elem_atomic):
        axs[idx_elem].set_ylabel(elem_names[elem]+r'($M_{\odot}$)')

        # Plot abundances as function of mass, colored by metallicity
        for idx_Z, metal in enumerate(Z):
            axs[idx_elem].plot(M, yields[idx_elem, idx_Z, :], linestyle='-', marker='None', 
                                color=cwheel[idx_Z], label=r'$Z=$'+str(metal))

    # Final plot formatting
    fig.suptitle(titles[yieldsource], fontsize=16)
    plt.legend(loc='center left', fontsize=10,
                bbox_to_anchor=(0.92, 0.5), bbox_transform=plt.gcf().transFigure)
    plt.xlabel(r'Mass ($M_{\odot}$)')
    plt.savefig('plots/'+'yieldtest_'+yieldsource+'.png', bbox_inches='tight')
    plt.show()

    return

if __name__ == "__main__":
    plotyields('lim18')