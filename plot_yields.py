"""
plot_yields.py

Plot yields as a function of mass, metallicity
"""

# Backend for matplotlib on mahler
from re import T
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
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

# Import other packages
import numpy as np
from gce_yields import *
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.stats import lognorm, norm
import imf
import dtd
import params

# Define list of yield sources
titles = {'nom06':'Nomoto et al. (2006)','nom13':'Nomoto et al. (2013)','lim18':'Limongi \& Chieffi (2018)',
        'leu18_ddt':'Leung \& Nomoto (2018) MCh DDT','leu18_def':'Leung \& Nomoto (2018) MCh pure deflagration','shen18':'Shen et al. (2018) sub-MCh bare WD','leu20':'Leung \& Nomoto (2020) sub-MCh with He shell',
        'cri15':'FRUITY (Cristallos et al. 2015)','kar':'Karakas et al. (2016, 2018)'}

# Some test functions
def exp(x, a, b, c):
    return a * np.exp(-b * x) + c
def powerlaw(x, a, b, c):
    return a * x**(-b) + c
def lognormal(x, amp, s, loc):
    return amp*lognorm.pdf(x, s, loc=loc)
def normal(x, amp, loc, s):
    return amp*norm.pdf(x, loc=loc, scale=s)
def powernorm(x, a, b, c, amp, loc, s):
    return a * x**(-b) + c + amp*norm.pdf(x, loc=loc, scale=s)
def linearnorm(x, m, b, amp, s, loc):
    return m * x + b + amp*norm.pdf(x, loc=loc, scale=s)

def getyields(yieldsource, yield_path='yields/', imfweight=None, empirical=False, fit=None, weakrprocess=False):
    """ Get yields as a function of mass, metallicity 
        and prep for plotting

    Args:
        yieldsource (str): which yields to use
        imfweight (str): if 'True', weight the yields by IMF
                        (options: 'kroupa93', 'kroupa01', 'chabrier03', 'salpeter55')
        empirical (bool): if 'True', plot empirical parameterizations of yields
        fit (float list): if not None, these are the parameters for the empirical yields
        weakrprocess (bool): if 'True', include r-process contributions to CCSN yields
    """

    # Set metallicity scale
    Z = np.linspace(0,2e-3,5)

    # Get model yield sets
    if empirical==False:

        # Figure out which yield to load
        if yieldsource in ['nom06','nom13','lim18']:
            yields, M, loadZ = load_II(yieldsource, yield_path, nel, atomic_names, atomic_num)
            yieldtype='CCSN'
            yields = yields['II']

            # Add r-process yields if needed
            if weakrprocess:

                # Barium yield for weak r-process event as a function of mass (Li+2014)
                M_li14 = np.array([13.0, 15.0, 18.0, 20.0, 25.0, 30.0, 40.0])
                ba_li14_weakr = np.asarray([1.38e-8, 2.83e-8, 5.38e-8, 6.84e-8, 9.42e-8, 0, 0])

                # Eu yield for weak r-process event as a function of mass (Cescutti+2006)
                M_ces06 = np.array([12.0, 15.0, 30.0, 40.0])
                eu_ces06_weakr = np.array([4.5e-8, 3.0e-8, 5.0e-10, 0.])

                # Linearly interpolate to high-mass end
                ba_li14_weakr[-2] = ba_li14_weakr[-3]*M_li14[-2]/M_li14[-3]
                ba_li14_weakr[-1] = ba_li14_weakr[-2]*M_li14[-1]/M_li14[-2]
                eu_ces06_weakr[-1] = eu_ces06_weakr[-2]*M_ces06[-1]/M_ces06[-2]

                # Interpolate to match SN mass array
                ba_li14_weakr = np.interp(M, M_li14, ba_li14_weakr)
                eu_ces06_weakr = np.interp(M, M_ces06, eu_ces06_weakr)

                # Add to SNII yield arrays, assuming no Z dependence
                for i in range(len(loadZ)):
                    yields[-2][i,:] = ba_li14_weakr  # Ba
                    yields[-1][i,:] = eu_ces06_weakr # Eu

            yields=np.delete(yields,6,axis=0) # Don't include Ti

        elif yieldsource in ['leu18_ddt','leu18_def','shen18','leu20']:
            SN_yield, M, loadZ = load_II('nom06', yield_path, nel, atomic_names, atomic_num)
            yields = load_Ia(yieldsource, yield_path, SN_yield, atomic_names, loadZ)   
            yieldtype='IaSN'
            #yields=yields['Ia'][2:,:] 
            yields=np.delete(yields['Ia'],[0,1,6],axis=0) # Don't include H, He, Ti

        elif yieldsource in ['cri15','kar']:
            M, loadZ, yields = load_AGB(yieldsource, yield_path, atomic_num, atomic_names, atomic_weight)
            yieldtype='AGB'
            yields = yields['AGB']

            yields=np.delete(yields,6,axis=0) # Don't include Ti

        else:
            raise ValueError('yieldsource is not valid!')

        # If needed, extrapolate yields to Z=0
        if ~np.isclose(loadZ[0],0.):
            if yieldtype in ['CCSN','AGB']:
                yields_z0 = yields[:,0,:]+(0-loadZ[0])*(yields[:,1,:]-yields[:,0,:])/(loadZ[1]-loadZ[0])
                yields = np.concatenate((yields_z0[:,None,:], yields), axis=1)   # Concatenate yield tables

            else:
                yields_z0 = yields[:,0]+(0-loadZ[0])*(yields[:,1]-yields[:,0])/(loadZ[1]-loadZ[0])
                yields = np.concatenate((yields_z0[:,None], yields), axis=1)   # Concatenate yield tables

            loadZ = np.concatenate(([0],loadZ))

        # Interpolate yields to a common metallicity scale
        f_interp = interpolate.interp1d(loadZ, yields, axis=1)
        yields = f_interp(Z)

        # Weight yields by IMF
        if imfweight is not None:
            dN_dM = imf.imf(M, imfweight)
            yields = yields * dN_dM

    # Return empirical yields
    if empirical:

        # Ia yields
        if yieldsource in ['leu18_ddt','leu18_def','shen18','leu20','fit_ia']:
            M = None
            yields = np.array([1.e-3, 1.e-2, 0.15, 2.e-2, 1.e-3, 1., 0.8, 1.5e-3]) # Default array (no changes for Mn)
            
            # Put in Mn yields
            mnyields = {'leu18_ddt':7.25e-3, 'leu18_def':8.21e-3, 'leu20':1.79e-3, 'shen18':0.14e-3, 'fit_ia':1.79e-3}
            yields[5] = mnyields[yieldsource]

            # Put in Ni yields
            niyields = {'leu18_ddt':5.65e-2, 'leu18_def':5.89e-2, 'leu20':1.53e-2, 'shen18':2.19e-2, 'fit_ia':1.53e-2}
            yields[7] = niyields[yieldsource]

            # Pure deflagration yields
            if yieldsource=='leu18_def':
                yields[0] = 0.36   # C
                yields[3] = 0.15e-2  # Ca
                yields[6] = 0.45   # Fe

            elif yieldsource=='fit_ia':
                # Default constant for now
                #c1 = 2e-3
                #yields[5] = c1  # Mn
                yields[6] = fit[0]  # Fe

            # Drop Ti
            yields = np.delete(yields,4)

            # Create final yield table
            print(yields.shape, len(Z))
            yields = np.tile(yields, (len(Z),1)).T
            yieldtype='IaSN'

        # Core-collapse yields
        elif yieldsource in ['nom13','lim18','fit_ii']:
            M = np.linspace(13,40,20)
            yields = np.zeros((12, len(Z), len(M)))
            yieldtype='CCSN'

            # Common yields
            yields[0,:,:] = np.tile(1e-3 * (255*M**(-1.88) - 0.1), (len(Z),1)) # H
            yields[1,:,:] = np.tile(1e-3 * (45*M**(-1.35) - 0.2), (len(Z),1)) # He
            yields[4,:,:] = np.tile(1e-5 * (2260*M**(-2.83) + 0.8), (len(Z),1)) # Si
            yields[6,:,:] = np.tile(1e-8 * (1000*M**(-2.3)), (len(Z),1)) # Ti
            yields[7,:,:] = np.tile(1e-7 * (30*M**(-1.32) - 0.25), (len(Z),1)) # Mn
            yields[8,:,:] = np.tile(1e-5 * (2722*M**(-2.77)), (len(Z),1)) # Fe

            if yieldsource=='nom13':
                yields[2,:,:] = np.tile(1e-5 * (100*M**(-1.35)), (len(Z),1)) # C
                yields[3,:,:] = np.tile(1e-5 * (261*M**(-1.8) + 0.33), (len(Z),1)) # Mg
                #yields[4,:,:] = np.tile(1e-5 * (2260*M**(-2.83) + 0.8), (len(Z),1)) # Si
                yields[5,:,:] = np.tile(1e-6 * (15.4*M**(-1) + 0.06), (len(Z),1)) # Ca
                yields[9,:,:] = np.tile(1e-6 * (8000*M**(-3.6)), (len(Z),1)) #Ni
            elif yieldsource=='lim18':
                Cyields = 1e-5 * (100*M**(-1))
                Cyields[np.where(M>30)] = 0.
                yields[2,:,:] = np.tile(Cyields, (len(Z),1)) # C
                yields[3,:,:] = np.tile(1e-5 * normal(M, 13, 19, 6.24), (len(Z),1)) # Mg
                #yields[4,:,:] = np.tile(1e-5 * (28*M**(-0.34) - 8.38), (len(Z),1)) # Si
                yields[5,:,:] = np.array([[1e-6 * normal(mass, 40, 17.5-3000*metal, 3) for metal in Z] for mass in M]).T # Ca
                yields[9,:,:] = np.tile(1e-6 * (8000*M**(-3.2)), (len(Z),1)) #Ni
            elif yieldsource=='fit_ii':
                yields[2,:,:] = np.tile(1e-5 * (100*M**(-fit[1])), (len(Z),1)) # C
                yields[3,:,:] = np.tile(1e-5 * (fit[2] + normal(M, 13, 19, 6.24)), (len(Z),1)) # Mg
                yields[5,:,:] = np.tile(1e-6 * (15.4*M**(-1) + 0.06), (len(Z),1)) + np.array([[fit[3] * 1e-6 * normal(mass, 40-10000*metal, 15, 3) for metal in Z] for mass in M]).T # Ca
                yields[9,:,:] = np.tile(1e-6 * (8000*M**(-3.2)), (len(Z),1)) #Ni (use lim18 for now)

            if weakrprocess:
                # Ba from Li+14
                yields[10,:,:] = np.tile(1e-12 * (1560*M**(-1.80) + 0.14 - normal(M,480,5,5.5)), (len(Z),1))

                # Eu from Cescutti+06
                yields[11,:,:] = np.tile(1e-11 * (77600*M**(-4.31)), (len(Z),1))

            # Drop Ti
            yields = np.delete(yields,6,axis=0)

        # AGB yields
        elif yieldsource in ['cri15','kar','fit_agb']:
            M = np.linspace(1,7,20)
            yields = np.zeros((12, len(Z), len(M)))
            yieldtype='AGB'

            # Common yields
            yields[0,:,:] = np.tile(1e-1 * (1.1*M**(-0.9) - 0.15), (len(Z),1)) # H
            yields[1,:,:] = np.tile(1e-2 * (4*M**(-1.07) - 0.22), (len(Z),1)) # He
            yields[3,:,:] = np.array([[1e-5*((400*metal + 1.1)*mass**(0.08 - 340*metal) + (360*metal - 1.27)) for metal in Z] for mass in M]).T # Mg
            yields[4,:,:] = np.array([[1.e-5*((800*metal)*mass**(-0.9) - (0.03 + 80*metal)) for metal in Z] for mass in M]).T  # Si
            yields[5,:,:] = np.array([[1.e-6*((-0.1 + 800*metal)*mass**(-0.96) - (80*metal)) for metal in Z] for mass in M]).T  # Ca
            yields[6,:,:] = np.array([[1.e-8*((3400*metal)*mass**(-0.88) - (480*metal)) for metal in Z] for mass in M]).T  # Ti
            yields[7,:,:] = np.array([[1.e-7*((1500*metal)*mass**(-0.95) - (160*metal)) for metal in Z] for mass in M]).T  # Mn
            yields[8,:,:] = np.array([[1.e-5*((1500*metal)*mass**(-0.95) - (160*metal)) for metal in Z] for mass in M]).T  # Fe
            yields[9,:,:] = np.array([[1.e-6*((840*metal)*mass**(-0.92) - (80*metal + 0.04)) for metal in Z] for mass in M]).T  # Ni 

            if yieldsource=='cri15':
                yields[2,:,:] = np.tile((1e-3 * normal(M, 0.89, 1.9, 0.58)), (len(Z),1)) # C
                yields[10,:,:] = np.array([[1e-8 * normal(mass, (400*metal - 0.1), 2, 0.5) for metal in Z] for mass in M]).T # Ba
                yields[11,:,:] = np.array([[1e-11 * normal(mass, (2000*metal - 0.6), 2, 0.65) for metal in Z] for mass in M]).T # Eu
            elif yieldsource=='kar':
                yields[2,:,:] = np.array([[1e-3 * normal(mass, (1.68-220*metal), 2, 0.6) for metal in Z] for mass in M]).T # C
                yields[3,:,:] += np.array([[1e-5 * normal(mass, (0.78-300*metal), 2.3, 0.14) for metal in Z] for mass in M]).T # Mg
                yields[10,:,:] = np.array([[1e-8 * normal(mass, (1000*metal + 0.2), 2.3, (0.75-100*metal)) for metal in Z] for mass in M]).T # Ba
                yields[11,:,:] = np.array([[1e-11 * normal(mass, (3400*metal + 0.4), 2.2, 0.65) for metal in Z] for mass in M]).T # Eu
            elif yieldsource=='fit_agb':
                # Default constants for now
                c2 = fit[5] #0.33
                c3 = fit[6] #1.0
                c4 = 0.5
                c5 = 0.2
                yields[2,:,:] = fit[4]*np.array([[1e-3 * normal(mass, (1.68-220*metal), 2, 0.6) for metal in Z] for mass in M]).T # C
                yields[10,:,:] = c2*np.array([[1e-8 * normal(mass, (1000*metal + 0.2), c3, (0.75-100*metal)) for metal in Z] for mass in M]).T # Ba
                yields[11,:,:] = c4*np.array([[1e-11 * normal(mass, (3400*metal + 0.4), 2.2-c5, 0.65) for metal in Z] for mass in M]).T # Eu

            # Drop Ti
            yields = np.delete(yields,6,axis=0)

        else:
            raise ValueError('yieldsource is not valid!')

    print(yields.shape) # Shape: elements, metallicity, (mass)
    return yields, M, Z

def plotyields(yieldtype, fit=None, func=None, empirical=False, empiricalfit=None, weakrprocess=False, imfweight='kroupa93'):
    """ Plot yields as a function of mass, metallicity

    Args: 
        yieldtype (str): type of yields to plot (options: 'CCSN', 'IaSN', 'AGB')
        fit (str list): if 'True', try fitting abundances with functions for each (options: 'exp', 'powerlaw', 'lognorm')
        func (list of float lists): if not None, manually plot fits using function listed in fit
        empirical (bool): if 'True', use empirical versions of literature yield sets
        empiricalfit (float list): if not None, plot empirical fit parameters as well as literature values
    """

    # Get yields
    yieldlist = []
    masslist = []
    yieldtitles = []

    if yieldtype=='CCSN':
        nom13yields, nom13M, Z = getyields('nom13', imfweight=imfweight, empirical=empirical, weakrprocess=weakrprocess)
        lim18yields, lim18M, _ = getyields('lim18', imfweight=imfweight, empirical=empirical, weakrprocess=weakrprocess)
        
        yieldlist = [nom13yields, lim18yields]
        masslist = [nom13M, lim18M]
        yieldtitles = ['nom13', 'lim18']
        if weakrprocess:
            yieldlist = [nom13yields[9:,:,:]]
            masslist = [nom13M]

        if empiricalfit is not None:
            fityields, fitM, _ = getyields('fit_ii', imfweight=imfweight, empirical=True, fit=empiricalfit, weakrprocess=weakrprocess)
            if weakrprocess:
                fityields = fityields[9:,:,:]

    if yieldtype=='AGB':
        cri15yields, cri15M, Z = getyields('cri15', imfweight=imfweight, empirical=empirical)
        karyields, karM, _ = getyields('kar', imfweight=imfweight, empirical=empirical)
        
        yieldlist = [cri15yields, karyields]
        masslist = [cri15M, karM]
        yieldtitles = ['cri15', 'kar']

        if empiricalfit is not None:
            fityields, fitM, _ = getyields('fit_agb', imfweight=imfweight, empirical=True, fit=empiricalfit)

    if yieldtype=='IaSN':
        leu18ddt_yields, _, Z = getyields('leu18_ddt', empirical=empirical)
        shen18_yields, _, _ = getyields('shen18', empirical=empirical)
        leu20_yields, _, _ = getyields('leu20', empirical=empirical)
        leu18def_yields, _, _ = getyields('leu18_def', empirical=empirical)
        
        yieldlist = [leu18ddt_yields, leu18def_yields, leu20_yields, shen18_yields]
        yieldtitles = ['leu18_ddt', 'leu18_def', 'leu20', 'shen18']

        if empiricalfit is not None:
            fityields, _, _ = getyields('fit_ia', imfweight=imfweight, empirical=True, fit=empiricalfit)

    # First, get colors and linestyles
    if yieldtype in ['CCSN','AGB']:
        cwheelsize = len(Z)
    else:
        cwheelsize = len(yieldtitles)

    color = cmr.bubblegum(np.linspace(0,1,cwheelsize,endpoint=True))
    matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
    cwheel = [np.array(matplotlib.rcParams['axes.prop_cycle'])[x]['color'] for x in range(cwheelsize)]
    lswheel = ['solid','dotted','dashed','dashdot']

    colorfit = cmr.cosmic(np.linspace(0.5,1,cwheelsize,endpoint=True))
    matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler('color', colorfit)
    cwheelfit = [np.array(matplotlib.rcParams['axes.prop_cycle'])[x]['color'] for x in range(cwheelsize)]

    # Create labels
    elem_atomic = [6, 12, 14, 20, 25, 26, 28]
    elem_names = {1:'H', 2:'He', 6:'C', 8:'O', 12:'Mg', 14:'Si', 20:'Ca', 22:'Ti', 25:'Mn', 26:'Fe', 28:'Ni', 56:'Ba', 63:'Eu'}
    # Add other yields if needed
    if yieldtype in ['AGB', 'CCSN']:
        elem_atomic = [1,2] + elem_atomic
    if yieldtype=='AGB':
        elem_atomic = elem_atomic + [56] #, 63]

    if yieldtype in ['CCSN'] and weakrprocess:
        elem_atomic = [56, 63]
    
    # Create and format plot
    if yieldtype=='CCSN' or yieldtype=='AGB':
        if weakrprocess:
            #fig, axs = plt.subplots(len(elem_atomic), figsize=(5,4),sharex=True)
            fig, axs = plt.subplots(1, figsize=(4,4))
        else:
            figsizes = {'CCSN':(5,13), 'AGB':(5,14)}
            fig, axs = plt.subplots(len(elem_atomic), figsize=figsizes[yieldtype],sharex=True)
        fig.subplots_adjust(bottom=0.06,top=0.96,left=0.2,wspace=0.29,hspace=0)
        plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='upper')) for a in [fig.axes[-1]]])
        plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='both')) for a in fig.axes[1:-1]])
        plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='lower')) for a in [fig.axes[0]]])
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        plt.setp([a.minorticks_on() for a in fig.axes[:]])
        axs = axs.ravel()

    else:
        fig = plt.figure(figsize=(5,4))
        ax = plt.subplot()

        dy = 0.02  # distance between lines
        nelem = len(elem_atomic)  # total number of elements
        print(nelem)

        ax.set_ylim(0.5, nelem+0.5)
        plt.tick_params(axis='y', which='both', left=False, right=False)
        ticklabels = []

    # Create each subplot
    for idx_elem, elem in enumerate(elem_atomic):
        print(idx_elem, elem)

        # Compute normalization order of magnitude
        if yieldtype=='CCSN' or yieldtype=='AGB':
            maxyield = [np.max(yieldlist[i][idx_elem, :, :]) for i in range(len(yieldlist))]
        else:
            maxyield = [np.max(yieldlist[i][idx_elem, :]) for i in range(len(yieldlist))]
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
                    if weakrprocess:
                        labels = ['Li et al. (2014)','Cescutti et al. (2006)']
                        line, = axs[idx_elem].plot(masslist[idx_yields], yields[idx_elem, idx_Z, :]/(10**base10), 
                                        linestyle='-', marker='None', 
                                        color='k', label=labels[idx_elem])
                        if idx_Z == 0:
                            handles.append(line)
                    else:
                        line, = axs[idx_elem].plot(masslist[idx_yields], yields[idx_elem, idx_Z, :]/(10**base10), 
                                            linestyle=lswheel[idx_yields], marker='None', 
                                            color=cwheel[idx_Z], label=r'$Z=$'+str(metal))
                        handles.append(line)

                '''
                # If needed, put in manual power law fits
                if func is not None and idx_yields==1 and elem in [28]:
                    popt = func
                    if np.all(np.isclose(popt, 0.)):
                        continue
                    elif fit=='powerlaw':
                        fitline, = axs[idx_elem].plot(masslist[idx_yields], powerlaw(masslist[idx_yields], *popt), ls='-', marker='None', color='c', 
                                    lw=2, label=r"$y = {:.1f}x^{{-{:.2f}}} + {:.2f}$".format(*popt), zorder=100)

                    elif fit=='lognorm':
                        fitline, = axs[idx_elem].plot(masslist[idx_yields], lognormal(masslist[idx_yields], *popt), ls='-', marker='None', color='c', 
                                    lw=2, label=r"$y = {:.2f}$LogNorm$(x,\sigma={:.2f})$, loc={:.2f}".format(*popt), zorder=100)

                    elif fit=='norm':
                        fitline, = axs[idx_elem].plot(masslist[idx_yields], normal(masslist[idx_yields], *popt), ls='-', marker='None', color='c', 
                                    lw=2, label=r"$y = {:.2f}$N$(\mu={:.2f},\sigma={:.2f})$".format(*popt), zorder=100)

                    elif fit=='powernorm':
                        fitline, = axs[idx_elem].plot(masslist[idx_yields], powernorm(masslist[idx_yields], *popt), ls='-', marker='None', color='c', 
                                lw=2, label=r"$y = {:.1f}x^{{-{:.2f}}} + {:.2f} + {:.2f}$N$({:.2f},{:.2f})$".format(*popt), zorder=100)

                    legend = axs[idx_elem].legend(handles=[fitline], loc='upper right', fontsize=10)
                    axs[idx_elem].add_artist(legend)

                # If needed, try fitting the Nom+13 data
                if func is None and fit in ['exp', 'powerlaw', 'lognorm', 'norm', 'powernorm', 'linearnorm'] and idx_yields==0 and elem in [25,28]:
                    try:
                        if fit=='exp':
                            popt, pcov = curve_fit(exp, masslist[idx_yields], yields[idx_elem, 3, :]/(10**base10), p0=[10, 0.1, 0])
                            fitline, = axs[idx_elem].plot(masslist[idx_yields], exp(masslist[idx_yields], *popt), ls='-', marker='None', color='c', 
                                    lw=2, label=r"$y = {:.1f}e^{{-{:.2f}x}} + {:.2f}$".format(*popt), zorder=100)
                        elif fit=='powerlaw':
                            popt, pcov = curve_fit(powerlaw, masslist[idx_yields], yields[idx_elem, 4, :]/(10**base10), p0=[1, 0, 0])
                            print(elem, popt)
                            fitline, = axs[idx_elem].plot(masslist[idx_yields], powerlaw(masslist[idx_yields], *popt), ls='-', marker='None', color='c', 
                                    lw=2, label=r"$y = {:.1f}x^{{-{:.2f}}} + {:.2f}$".format(*popt), zorder=100)
                        elif fit=='lognorm':
                            popt, pcov = curve_fit(lognormal, masslist[idx_yields], yields[idx_elem, 1, :]/(10**base10), p0=[0.5,10,15])
                            print(elem, popt)
                            fitline, = axs[idx_elem].plot(masslist[idx_yields], lognormal(masslist[idx_yields], *popt), ls='-', marker='None', color='c', 
                                    lw=2, label=r"$y = {:.2f}$LogNorm$(x,\sigma={:.2f})$, loc={:.2f}".format(*popt), zorder=100)
                        elif fit=='norm':
                            popt, pcov = curve_fit(normal, masslist[idx_yields], yields[idx_elem, 4, :]/(10**base10), p0=[10,10,1])
                            print(elem, popt)
                            fitline, = axs[idx_elem].plot(masslist[idx_yields], normal(masslist[idx_yields], *popt), ls='-', marker='None', color='c', 
                                    lw=2, label=r"$y = {:.2f}$N$(\mu={:.2f},\sigma={:.2f})$".format(*popt), zorder=100)
                        elif fit=='powernorm':
                            popt, pcov = curve_fit(powernorm, masslist[idx_yields], yields[idx_elem, 1, :]/(10**base10), p0=[20,1,-3,1,15,1])
                            print(elem, popt)
                            fitline, = axs[idx_elem].plot(masslist[idx_yields], powernorm(masslist[idx_yields], *popt), ls='-', marker='None', color='c', 
                                    lw=2, label=r"$y = {:.1f}x^{{-{:.2f}}} + {:.2f} + {:.2f}$N$({:.2f},{:.2f})$".format(*popt), zorder=100)
                        elif fit=='linearnorm':
                            popt, pcov = curve_fit(linearnorm, masslist[idx_yields], yields[idx_elem, 2, :]/(10**base10), p0=[1,0.1,1,1,2])
                            print(elem, popt)
                            masses = np.linspace(1,7,20)
                            fitline, = axs[idx_elem].plot(masses, linearnorm(masses, *popt), ls='-', marker='None', color='c', 
                                    lw=2, label=r"$y = {:.1f}x + {:.2f} + {:.2f}$N$({:.2f},{:.2f})$".format(*popt), zorder=100)

                        legend = axs[idx_elem].legend(handles=[fitline], loc='upper right', fontsize=10)
                        axs[idx_elem].add_artist(legend)
                        
                    except:
                        pass
                '''

                if weakrprocess:
                    topdx = 0.98
                    dx = 0.31
                    textdy = 0.85
                else:
                    topdx = 0.9
                    dx = 0.12
                    textdy = 0.75

                # Create legends
                if weakrprocess == False:
                    legend = plt.legend(handles=handles, loc='upper left', fontsize=10, title=titles[yieldtitles[idx_yields]],
                        bbox_to_anchor=(0.92, topdx - idx_yields*dx), bbox_transform=plt.gcf().transFigure)
                    legend._legend_box.align = "left"
                    plt.gca().add_artist(legend)

            if yieldtype=='IaSN':
                # Plot abundances
                line = ax.axvline(yields[idx_elem, 0]/(10**base10), ymin = (nelem-(idx_elem+1))/nelem + dy, ymax = (nelem-idx_elem)/nelem - dy, 
                        color=cwheel[idx_yields], label=titles[yieldtitles[idx_yields]], linewidth=2)
                handles.append(line)

                if idx_elem==0:
                    #handles.append(line)
                    legend = plt.legend(handles=handles, loc='upper left', fontsize=10,
                        bbox_to_anchor=(0.92, 0.9 - idx_yields*0.05), bbox_transform=plt.gcf().transFigure)
                    legend._legend_box.align = "left"
                    plt.gca().add_artist(legend)

                if elem==28:
                    print(titles[yieldtitles[idx_yields]], yields[idx_elem, 0]/(10**base10))

                if idx_yields==0:
                    ticklabels.append(elem_names[elem]+' ($10^{'+str(int(base10))+'}M_{\odot}$)')

        # Plot empirical fits if needed
        if empiricalfit is not None:
            if weakrprocess:
                if yieldtype=='CCSN' or yieldtype=='AGB':
                    for idx_Z, metal in enumerate(Z):
                        line, = axs[idx_elem].plot(fitM, fityields[idx_elem, idx_Z, :]/(10**base10), 
                                            linestyle='--', marker='None', lw=1,
                                            color=cwheelfit[1], label='Best fit')
                        if idx_Z == 0:
                            handles.append(line)

                # Create legends
                locs = ['lower left','center right']
                legend = axs[idx_elem].legend(handles=handles, loc=locs[idx_elem], fontsize=10)
                #legend._legend_box.align = "left"
                #plt.gca().add_artist(legend)

            else:
                handles = []
                if yieldtype=='CCSN' or yieldtype=='AGB':
                    for idx_Z, metal in enumerate(Z):
                        line, = axs[idx_elem].plot(fitM, fityields[idx_elem, idx_Z, :]/(10**base10), 
                                            linestyle='--', marker='None', lw=1,
                                            color=cwheelfit[idx_Z], label=r'$Z=$'+str(metal))
                        handles.append(line)

                    # Create legends
                    legend = plt.legend(handles=handles, loc='upper left', fontsize=10, title="Best-fit yields",
                        bbox_to_anchor=(0.92, topdx - 2*dx), bbox_transform=plt.gcf().transFigure)
                    legend._legend_box.align = "left"
                    plt.gca().add_artist(legend)

                if yieldtype=='IaSN':
                    # Plot abundances
                    line = ax.axvline(fityields[idx_elem, 0]/(10**base10), ymin = (nelem-(idx_elem+1))/nelem + dy, ymax = (nelem-idx_elem)/nelem - dy, 
                            color='C1', label="Best-fit yields", linewidth=2, linestyle=':')
                    if idx_elem==0:
                        handles.append(line)

                        legend = plt.legend(handles=handles, loc='upper left', fontsize=10,
                            bbox_to_anchor=(0.92, 0.9 - 4*0.05), bbox_transform=plt.gcf().transFigure)
                        legend._legend_box.align = "left"
                        plt.gca().add_artist(legend)
  
        # Do other formatting
        if yieldtype in ['CCSN', 'AGB']:
            axs[idx_elem].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            if (yieldtype=='CCSN' and elem_names[elem] in ['C','Mg','Ca']) or (yieldtype=='AGB' and elem_names[elem] in ['C','Ba']):
                axs[idx_elem].text(0.95, textdy, r'\textbf{'+elem_names[elem]+'}', transform=axs[idx_elem].transAxes, fontsize=12, color='C1', horizontalalignment='right') #, bbox=dict(fc=cwheelfit[-1], ec='k', linewidth=0.5))
            else:
                axs[idx_elem].text(0.95, textdy, elem_names[elem], transform=axs[idx_elem].transAxes, fontsize=12, horizontalalignment='right') #, bbox=dict(fc='None', ec='k', linewidth=0.5))
            axs[idx_elem].set_ylabel(r'$10^{'+str(int(base10))+'}M_{\odot}$', fontsize=10)
            axs[idx_elem].set_ylim(ymin=0)

    if yieldtype=='IaSN':
        # Fix y-axis labels
        ticklabels.append('')
        print(ticklabels)
        ax.yaxis.set_ticklabels(ticklabels[::-1])

        plt.gca().get_yticklabels()[2].set_color('C1') 
    
        # Create legends
        #legend = plt.legend(handles=handles, loc='upper left', fontsize=10, 
        #        bbox_to_anchor=(0.92, 0.9 - 0.12), bbox_transform=plt.gcf().transFigure)
        #legend._legend_box.align = "left"

    # Final plot formatting
    if yieldtype=='CCSN' or yieldtype=='AGB':
        plt.xlabel(r'Mass ($M_{\odot}$)', fontsize=14)
    else:
        plt.xlabel(r'Yield mass', fontsize=14)

    # Output figure
    if weakrprocess:
        yieldtype += '_weakr'
    if empirical:
        outputname = 'plots/yieldtest_'+yieldtype+'_empirical.pdf'
    elif empiricalfit is not None:
        outputname = 'plots/yieldtest_'+yieldtype+'_empiricalfit.pdf'
    else:
        outputname = 'plots/yieldtest_'+yieldtype+'.pdf'
    plt.savefig(outputname, bbox_inches='tight')
    plt.show()

    return

if __name__ == "__main__":

    # Plot yields
    #getyields('nom13', empirical=False, weakrprocess=True)
    #plotyields('CCSN', empirical=False, empiricalfit=[0.8, 1., 1., 0., 0.6], weakrprocess=True)
    #[1.07, 0.16, 4.01, 0.89, 0.82, 0.59, 0.8, 1., 1., 0., 0.6, 0.33, 1.0] 
    #plotyields('AGB', empirical=False, empiricalfit=[0.563019743600889,1.2909839533334972,0.8604762167017103,0.2864776957718226,1.5645763678916176,0.8939183631841486,3-0.014997329848299233], imfweight='kroupa93')
    plotyields('CCSN', empirical=False, empiricalfit=[0.54901945, 1.31771318, 0.81434372, 0.22611351, 1.64741211, 0.93501212, 0.034125], weakrprocess=True)
    #plotyields('AGB', empirical=False, empiricalfit=[0.5394764248347781,1.3204750735928574,1.359457919837111,0.13139217074287995,1.612782258893653,0.36473661768759114,4.402225273291386])
    #plotyields('CCSN', empirical=False, empiricalfit=[0.563019743600889,1.2909839533334972,0.8604762167017103,0.2864776957718226,1.5645763678916176,0.8939183631841486,3-0.014997329848299233], imfweight='kroupa93')
