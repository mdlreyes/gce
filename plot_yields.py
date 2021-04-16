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
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, MaxNLocator, FormatStrFormatter
rc('font', family='serif')
rc('axes', labelsize=14) 
rc('xtick', labelsize=10)
rc('ytick', labelsize=10)
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
from scipy.optimize import curve_fit
import imf

# Define list of yield sources
titles = {'nom06':'Nomoto et al. (2006)','nom13':'Nomoto et al. (2013)','lim18':'Limongi \& Chieffi (2018)',
        'leu20':'Leung \& Nomoto (2020)','cri15':'FRUITY (Cristallos et al. 2015)','kar':'Karakas et al. (2016, 2018)'}

def getyields(yieldsource, yield_path='yields/', imfweight=None):
    """ Get yields as a function of mass, metallicity 
        and prep for plotting

    Args:
        yieldsource (str): which yields to use
        imfweight (str): if 'True', weight the yields by IMF
                        (options: 'kroupa93', 'kroupa01', 'chabrier03', 'salpeter55')
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
        M, loadZ, yields = load_AGB(yieldsource, yield_path, atomic_num, atomic_names, atomic_weight)
        yieldtype='AGB'
        yields = yields['AGB']

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
    if imfweight is not None:
        dN_dM = imf.imf(M, imfweight)
        yields = yields * dN_dM

    return yields, M, Z

# Some test functions
def exp(x, a, b, c):
    return a * np.exp(-b * x) + c

def plotyields(yieldtype, fit=None):
    """ Plot yields as a function of mass, metallicity

    Args: 
        yieldtype (str): type of yields to plot (options: 'CCSN', 'IaSN', 'AGB')
        fit (str list): if 'True', try fitting abundances with functions for each 
    """

    # Get yields
    yieldlist = []
    masslist = []
    yieldtitles = []

    if yieldtype=='CCSN':
        nom13yields, nom13M, Z = getyields('nom13', imfweight='kroupa93')
        lim18yields, lim18M, _ = getyields('lim18', imfweight='kroupa93')
        
        yieldlist = [nom13yields, lim18yields]
        masslist = [nom13M, lim18M]
        yieldtitles = ['nom13', 'lim18']

    if yieldtype=='AGB':
        cri15yields, cri15M, Z = getyields('cri15', imfweight='kroupa93')
        karyields, karM, _ = getyields('kar', imfweight='kroupa93')
        
        yieldlist = [cri15yields, karyields]
        masslist = [cri15M, karM]
        yieldtitles = ['cri15', 'kar']

    # First, get colors and linestyles
    color = cmr.bubblegum(np.linspace(0,1,len(Z),endpoint=True))
    matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
    cwheel = [np.array(matplotlib.rcParams['axes.prop_cycle'])[x]['color'] for x in range(len(Z))]
    lswheel = ['solid','dotted','dashed','dashdot']

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
    figsizes = {'CCSN':(5,12), 'AGB':(5,13), 'IaSN':(5,12)}
    fig, axs = plt.subplots(len(elem_atomic), figsize=figsizes[yieldtype],sharex=True)
    fig.subplots_adjust(bottom=0.06,top=0.96,left=0.2,wspace=0.29,hspace=0)
    plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='upper')) for a in [fig.axes[-1]]])
    plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='both')) for a in fig.axes[1:-1]])
    plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='lower')) for a in [fig.axes[0]]])
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.setp([a.minorticks_on() for a in fig.axes[:]])
    axs = axs.ravel()

    # Create each subplot
    for idx_elem, elem in enumerate(elem_atomic):

        # Compute normalization order of magnitude
        maxyield = [np.max(yieldlist[i][idx_elem, :, :]) for i in range(len(yieldlist))]
        maxyield = np.max(maxyield)
        if ~np.isclose(maxyield, 0., atol=1e-16):
            base10 = np.floor(np.log10(maxyield))
            # Increase by 1 if we're close to the next magnitude
            if str(maxyield)[0] == '9':
                base10 += 1       
        else:
            base10 = 1.

        # Loop over all yields in set
        for idx_yields, yields in enumerate(yieldlist):

            handles = []

            if yieldtype=='CCSN' or yieldtype=='AGB':
                # Plot abundances as function of mass, colored by metallicity
                allyields = []  # list to store all yields in
                for idx_Z, metal in enumerate(Z):
                    allyields.append(yields[idx_elem, idx_Z, :])
                    line, = axs[idx_elem].plot(masslist[idx_yields], yields[idx_elem, idx_Z, :]/(10**base10), 
                                        linestyle=lswheel[idx_yields], marker='None', 
                                        color=cwheel[idx_Z], label=r'$Z=$'+str(metal))
                    handles.append(line)

                # If needed, try fitting the Nom+13 data
                if fit and idx_yields==0:
                    popt, pcov = curve_fit(exp, masslist[idx_yields], np.average(np.asarray(allyields),axis=0)/(10**base10), p0=[10, 0.5, 0])
                    print(popt)
                    fitline, = axs[idx_elem].plot(masslist[idx_yields], exp(masslist[idx_yields], *popt), ls='-', marker='None', color='c', 
                                lw=2, label=r"$y = {:.1e}e^{{-{:.2f}x}} + {:.2f}$".format(*popt), zorder=100)
                    legend = axs[idx_elem].legend(handles=[fitline], loc='upper right', fontsize=10)
                    axs[idx_elem].add_artist(legend)

                # Create legends
                legend = plt.legend(handles=handles, loc='upper left', fontsize=10, title=titles[yieldtitles[idx_yields]],
                    bbox_to_anchor=(0.92, 0.9 - idx_yields*0.12), bbox_transform=plt.gcf().transFigure)
                legend._legend_box.align = "left"
                plt.gca().add_artist(legend)
  
        # Do other formatting
        axs[idx_elem].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axs[idx_elem].set_ylabel(r'$M$('+elem_names[elem]+') ($10^{'+str(int(base10))+'}M_{\odot}$)', fontsize=10)
        axs[idx_elem].set_ylim(ymin=0)

    # Final plot formatting
    if yieldtype=='CCSN' or yieldtype=='AGB':
        plt.xlabel(r'Mass ($M_{\odot}$)', fontsize=14)

    plt.savefig('plots/'+'yieldtest_'+yieldtype+'.pdf', bbox_inches='tight')
    plt.show()

    return

if __name__ == "__main__":
    plotyields('CCSN', fit=True)