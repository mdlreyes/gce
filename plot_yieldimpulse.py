"""
plot_yieldimpulse.py

Plot yields as a response to a burst of SF
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
import dtd
import params
import gce_yields

# Define list of yield sources
titles = {'nom06':'Nomoto et al. (2006)','nom13':'Nomoto et al. (2013)','lim18':'Limongi \& Chieffi (2018)',
        'leu18_ddt':'Leung \& Nomoto (2018) MCh DDT','leu18_def':'Leung \& Nomoto (2018) MCh pure deflagration','shen18':'Shen et al. (2018) sub-MCh bare WD','leu20':'Leung \& Nomoto (2020) sub-MCh with He shell',
        'cri15':'FRUITY (Cristallos et al. 2015)','kar':'Karakas et al. (2016, 2018)'}

def getyields(yieldsource, m_himass, m_intmass, fit=None, weakrprocess=False, imfweight='kroupa93'):
    """ Get yields and prep for plotting

    Args:
        yieldsource (str): which yields to use
        m_himass, m_intmass (arrays): arrays of stellar masses
        fit (float list): if not None, these are the parameters for the empirical yields
        imfweight (str): if not None, weight all yields by input IMF 
    """

    # Load all sources of chemical yields
    if yieldsource in ['nom06','nom13','lim18']:
        nel, eps_sun, SN_yield, AGB_yield, M_SN, z_II, M_AGB, z_AGB = gce_yields.initialize_yields(
            Ia_source=params.Ia_source, II_source=yieldsource, 
            AGB_source=params.AGB_source, r_process_keyword=params.r_process_keyword)

        # Linearly extrapolate supernova yields to min/max progenitor masses
        sn_min = SN_yield['II'][:,:,0] * params.M_SN_min/M_SN[0]             # Extrapolate yields to min progenitor mass
        sn_max = SN_yield['II'][:,:,-1] * params.M_SN_max/M_SN[-1]           # Extrapolate yields to max progenitor mass
        yield_ii = np.concatenate((sn_min[...,None], SN_yield['II'], sn_max[...,None]), axis=2)   # Concatenate yield tables
        M_SN = np.concatenate(([params.M_SN_min], M_SN, [params.M_SN_max]))     # Concatenate mass list

        # Weight yields by IMF
        #if imfweight is not None:
        #    dN_dM = imf.imf(M_SN, imfweight)
        #    yield_ii = yield_ii * dN_dM

        # If needed, linearly extrapolate SN yields to Z=0
        if ~np.isclose(z_II[0],0.):
            ii_z0 = yield_ii[:,0,:]+(0-z_II[0])*(yield_ii[:,1,:]-yield_ii[:,0,:])/(z_II[1]-z_II[0])
            yield_ii = np.concatenate((ii_z0[:,None,:], yield_ii), axis=1)   # Concatenate yield tables

            ia_z0 = SN_yield['Ia'][:,0]+(0-z_II[0])*(SN_yield['Ia'][:,1]-SN_yield['Ia'][:,0])/(z_II[1]-z_II[0])
            yield_ia = np.concatenate((ia_z0[:,None], SN_yield['Ia']), axis=1)   # Concatenate yield tables
            
            z_II = np.concatenate(([0],z_II))
        
        else:
            yield_ia = SN_yield['Ia']
            weight_ii = SN_yield['weight_II']

        # Interpolate yield tables over mass
        f_ii_mass = interp1d(M_SN, yield_ii, axis=2, bounds_error=False, copy=False, assume_sorted=True)
        ii_yield_mass = f_ii_mass(m_himass) # Compute yields of masses of stars that will explode
        f_ii_metallicity = interp1d(z_II, ii_yield_mass, axis=1, bounds_error=False, copy=False, assume_sorted=True)

        return f_ii_metallicity

    elif yieldsource in ['cri15','kar']:
        nel, eps_sun, SN_yield, AGB_yield, M_SN, z_II, M_AGB, z_AGB = gce_yields.initialize_yields(
            Ia_source=params.Ia_source, II_source=params.II_source, 
            AGB_source=yieldsource, r_process_keyword=params.r_process_keyword)

        # Linearly extrapolate AGB yields to min/max progenitor masses
        agb_min = AGB_yield['AGB'][:,:,0] * params.M_AGB_min/M_AGB[0]        # Extrapolate yields to min progenitor mass
        agb_max = AGB_yield['AGB'][:,:,-1] * params.M_AGB_max/M_AGB[-1]      # Extrapolate yields to max progenitor mass
        yield_agb = np.concatenate((agb_min[...,None], AGB_yield['AGB'], agb_max[...,None]), axis=2)   # Concatenate yield tables
        M_AGB = np.concatenate(([params.M_AGB_min], M_AGB, [params.M_AGB_max])) # Concatenate mass list 

        # Weight yields by IMF
        #if imfweight is not None:
        #    dN_dM = imf.imf(M_AGB, imfweight)
        #    yield_agb = yield_agb * dN_dM

        # Linearly extrapolate AGB yields to Z = 0
        agb_z0 = yield_agb[:,0,:]+(0-z_AGB[0])*(yield_agb[:,1,:]-yield_agb[:,0,:])/(z_AGB[1]-z_AGB[0])
        yield_agb = np.concatenate((agb_z0[:,None,:], yield_agb), axis=1)   # Concatenate yield tables
        z_AGB = np.concatenate(([0],z_AGB))

        f_agb_mass = interp1d(M_AGB, yield_agb, axis=2, bounds_error=False, copy=False, assume_sorted=True)
        agb_yield_mass = f_agb_mass(m_intmass) # Compute yields of masses of stars that will produce AGB winds
        f_agb_metallicity = interp1d(z_AGB, agb_yield_mass, axis=1, bounds_error=False, copy=False, assume_sorted=True) 
        
        return f_agb_metallicity

    if yieldsource in ['leu18_ddt','leu18_def','shen18','leu20']:
        nel, eps_sun, SN_yield, AGB_yield, M_SN, z_II, M_AGB, z_AGB = gce_yields.initialize_yields(
            Ia_source=yieldsource, II_source=params.II_source, 
            AGB_source=params.AGB_source, r_process_keyword=params.r_process_keyword)

        # Chop off extraneous elements
        yield_ia = SN_yield['Ia'][2:-2,:]

        # If needed, linearly extrapolate SN yields to Z=0
        if ~np.isclose(z_II[0],0.):

            ia_z0 = yield_ia[:,0]+(0-z_II[0])*(yield_ia[:,1]-yield_ia[:,0])/(z_II[1]-z_II[0])
            yield_ia = np.concatenate((ia_z0[:,None], yield_ia), axis=1)   # Concatenate yield tables

            z_II = np.concatenate(([0],z_II))

        f_ia_metallicity = interp1d(z_II, yield_ia, axis=1, bounds_error=False, copy=False, assume_sorted=True) 
        
        return f_ia_metallicity
        

    elif fit is not None:
        nel, eps_sun, atomic, weight, f_ia_metallicity, f_ii_metallicity, f_agb_metallicity, _ = gce_yields.initialize_empirical(
            Ia_source=params.Ia_source, II_source=params.II_source, AGB_source=params.AGB_source, 
            r_process_keyword=params.r_process_keyword,
            II_mass=m_himass, AGB_mass=m_intmass, fit=True)

        return f_ia_metallicity, f_ii_metallicity, f_agb_metallicity

def plotyieldimpulse(yieldtype, empiricalfit=None, weakrprocess=False, ia_dtd='maoz10', imfweight='kroupa93'):
    """ Plot yields as a function of time after a single burst of SF.

    Args: 
        yieldtype (str): type of yields to plot (options: 'CCSN', 'IaSN', 'AGB')
        empiricalfit (float list): if not None, plot empirical fit parameters as well as literature values
        weakrprocess (bool): if True, plot Ba and Eu from CCSNe
    """

    # Set metallicity scale
    Z = np.linspace(0,2e-3,5)

    # Get numbers of events
    delta_t = 0.001     # time step (Gyr)
    n = int(1.36/delta_t)     # number of timesteps in the model 
    t = np.arange(n)*delta_t    # time passed in model array -- age universe (Gyr)
    sfburst = 100   # assume stellar mass of 100 Msun formed in instantaneous burst

    n_wd = dtd.dtd_ia(t, ia_dtd) * delta_t  # Number of SNe/Gyr/Msun * Gyr = SNe/Msun
    n_ia = sfburst * n_wd  # SNe that formed after SF burst

    m_himass, n_himass = dtd.dtd_ii(t, imfweight)       # Mass and fraction of stars that will explode in the future
    goodidx = np.where((m_himass > params.M_SN_min) & (m_himass < params.M_SN_max))[0]  # Limit to timesteps where stars will explode as CCSN
    m_himass = m_himass[goodidx]    
    n_himass = n_himass[goodidx]
    n_ii = sfburst * n_himass

    m_intmass, n_intmass = dtd.dtd_agb(t, imfweight)    # Mass and fraction of stars that become AGBs in the future
    goodidx_agb = np.where((m_intmass > params.M_AGB_min) & (m_intmass < params.M_AGB_max))[0] # Limit to timesteps where stars between 0.865-10 M_sun will become AGB stars
    m_intmass = m_intmass[goodidx_agb]
    n_intmass = n_intmass[goodidx_agb]
    n_agb = sfburst * n_intmass

    # Save event numbers in a dictionary
    n_event = {'CCSN':n_ii, 'AGB':n_agb} #, 'IaSN':n_ia}
    goodidxs = {'CCSN':goodidx, 'AGB':goodidx_agb}

    # Get yields
    yieldlist = []
    yieldtitles = []

    if yieldtype=='CCSN':
        f_ii_nom13 = getyields('nom13', m_himass, m_intmass, weakrprocess=weakrprocess)
        f_ii_lim18 = getyields('lim18', m_himass, m_intmass, weakrprocess=weakrprocess)
        
        yieldlist = [f_ii_nom13, f_ii_lim18]
        yieldtitles = ['nom13', 'lim18']

        if empiricalfit is not None:
            _, fityields, _ = getyields('fit_ii', m_himass, m_intmass, fit=empiricalfit, weakrprocess=weakrprocess)

    if yieldtype=='AGB':
        f_agb_cri15 = getyields('cri15', m_himass, m_intmass, weakrprocess=weakrprocess)
        f_agb_kar = getyields('kar', m_himass, m_intmass, weakrprocess=weakrprocess)
        
        yieldlist = [f_agb_cri15, f_agb_kar]
        yieldtitles = ['cri15', 'kar']

        if empiricalfit is not None:
            _, _, fityields = getyields('fit_agb', m_himass, m_intmass, fit=empiricalfit, weakrprocess=weakrprocess)

    if yieldtype=='IaSN':
        f_ia_leu18ddt = getyields('leu18_ddt', m_himass, m_intmass, weakrprocess=weakrprocess)
        f_ia_shen18 = getyields('shen18', m_himass, m_intmass, weakrprocess=weakrprocess)
        f_ia_leu20 = getyields('leu20', m_himass, m_intmass, weakrprocess=weakrprocess)
        f_ia_leu18def = getyields('leu18_def', m_himass, m_intmass, weakrprocess=weakrprocess)
        
        yieldlist = [f_ia_leu18ddt, f_ia_leu18def, f_ia_leu20, f_ia_shen18]
        yieldtitles = ['leu18_ddt', 'leu18_def', 'leu20', 'shen18']

        if empiricalfit is not None:
            fityields, _, _ = getyields('fit_ia', m_himass, m_intmass, fit=empiricalfit, weakrprocess=weakrprocess)

    if empiricalfit is not None:
        fe_ia = empiricalfit[6]         # Fe yield from IaSNe
        cexp_ii = empiricalfit[7]       # C exponent for CCSN yields
        cnorm_agb = empiricalfit[10]    # C normalization for AGB yields
        banorm_agb = empiricalfit[11]   # Ba normalization for AGB yields
        bamean_agb = empiricalfit[12]   # Ba mean for AGB yields
        mgnorm_ii = empiricalfit[8]     # Mg normalization for CCSN yields
        canorm_ii = empiricalfit[9]     # Ca normalization for CCSN yields

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
    elem_atomic = [6, 12, 14, 20, 22, 25, 26, 28]
    elem_names = {1:'H', 2:'He', 6:'C', 8:'O', 12:'Mg', 14:'Si', 20:'Ca', 22:'Ti', 25:'Mn', 26:'Fe', 28:'Ni', 56:'Ba', 63:'Eu'}
    # Add other yields if needed
    if yieldtype=='CCSN':
        elem_atomic = [1,2] + elem_atomic
    if yieldtype=='AGB':
        elem_atomic = [1,2] + elem_atomic + [56] #, 63]

    if yieldtype in ['CCSN'] and weakrprocess:
        elem_atomic = [56, 63]
    
    # Create and format plot
    if weakrprocess:
        fig, axs = plt.subplots(len(elem_atomic), figsize=(5,4),sharex=True)
    else:
        figsizes = {'CCSN':(5,13), 'AGB':(5,14), 'IaSN':(5,9)}
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
        print(idx_elem, elem)

        # Compute normalization order of magnitude
        if yieldtype=='CCSN' or yieldtype=='AGB':
            maxyield = [np.max(n_event[yieldtype] * yields(Z)[idx_elem,:]) for yields in yieldlist]
        else:
            maxyield = [np.max(n_ia * yields(0)[idx_elem]) for yields in yieldlist]
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
                for idx_Z, metal in enumerate(Z):
                    if weakrprocess:
                        labels = ['Li et al. (2014)','Cescutti et al. (2006)']
                        line, = axs[idx_elem].plot(t[goodidxs[yieldtype]], n_event[yieldtype] * yields(metal)[idx_elem,:]/(10**base10), 
                                        linestyle='-', marker='None', 
                                        color='k', label=labels[idx_elem])
                        if idx_Z == 0:
                            handles.append(line)
                    else:
                        line, = axs[idx_elem].plot(t[goodidxs[yieldtype]], n_event[yieldtype] * yields(metal)[idx_elem,:]/(10**base10), 
                                            linestyle=lswheel[idx_yields], marker='None', 
                                            color=cwheel[idx_Z], label=r'$Z=$'+str(metal))
                        handles.append(line)

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
                # Plot abundances as function of mass, colored by model
                line, = axs[idx_elem].loglog(t, n_ia * yields(0)[idx_elem]/(10**base10), 
                        linestyle='-', marker='None', 
                        color=cwheel[idx_yields], label=titles[yieldtitles[idx_yields]], linewidth=1)
                handles.append(line)

                topdx = 0.9
                dx = 0.12
                textdy = 0.75

        # Plot empirical fits if needed
        if empiricalfit is not None:
            handles = []

            if yieldtype=='CCSN':
                for idx_Z, metal in enumerate(Z):
                    # Weight yields by IMF
                    #dN_dM = imf.imf(m_himass, imfweight)
                    #finalyields = fityields(metal, cexp_ii=cexp_ii, mgnorm_ii=mgnorm_ii, canorm_ii=canorm_ii)[idx_elem,:] * dN_dM
                    finalyields = fityields(metal, cexp_ii=cexp_ii, mgnorm_ii=mgnorm_ii, canorm_ii=canorm_ii)[idx_elem,:]

                    line, = axs[idx_elem].plot(t[goodidxs[yieldtype]], n_event[yieldtype] * finalyields/(10**base10), 
                                        linestyle='--', marker='None', lw=1,
                                        color=cwheelfit[idx_Z], label=r'$Z=$'+str(metal))
                    handles.append(line)

            if yieldtype=='AGB':                
                for idx_Z, metal in enumerate(Z):
                    # Weight yields by IMF
                    #dN_dM = imf.imf(m_intmass, imfweight)
                    #finalyields = fityields(metal, cnorm_agb=cnorm_agb, banorm_agb=banorm_agb, bamean_agb=bamean_agb)[idx_elem,:] * dN_dM
                    finalyields = fityields(metal, cnorm_agb=cnorm_agb, banorm_agb=banorm_agb, bamean_agb=bamean_agb)[idx_elem,:]
                    
                    line, = axs[idx_elem].plot(t[goodidxs[yieldtype]], n_event[yieldtype] * finalyields/(10**base10), 
                                        linestyle='--', marker='None', lw=1,
                                        color=cwheelfit[idx_Z], label=r'$Z=$'+str(metal))
                    handles.append(line)

            if yieldtype=='IaSN':                
                # Chop off extraneous elements
                yield_ia = fityields(0, fe_ia=fe_ia)[2:-2]
                finalyields = yield_ia[idx_elem]
                
                line, = axs[idx_elem].loglog(t, n_ia * finalyields/(10**base10), 
                                    linestyle='--', marker='None', lw=2,
                                    color='C1', label=r'Best-fit yields')
                handles.append(line)

            # Create legends
            if yieldtype in ['CCSN','AGB']:
                legend = plt.legend(handles=handles, loc='upper left', fontsize=10, title="Best-fit yields",
                    bbox_to_anchor=(0.92, topdx - 2*dx), bbox_transform=plt.gcf().transFigure)
                legend._legend_box.align = "left"
                plt.gca().add_artist(legend)
  
        # Do other formatting
        #axs[idx_elem].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if (yieldtype=='CCSN' and elem_names[elem] in ['C','Mg','Ca']) or (yieldtype=='AGB' and elem_names[elem] in ['C','Ba']) or (yieldtype=='IaSN' and elem_names[elem] in ['Fe']):
            axs[idx_elem].text(0.95, textdy, r'\textbf{'+elem_names[elem]+'}', transform=axs[idx_elem].transAxes, fontsize=12, color='C1', horizontalalignment='right') #, bbox=dict(fc=cwheelfit[-1], ec='k', linewidth=0.5))
        else:
            axs[idx_elem].text(0.95, textdy, elem_names[elem], transform=axs[idx_elem].transAxes, fontsize=12, horizontalalignment='right') #, bbox=dict(fc='None', ec='k', linewidth=0.5))
        #axs[idx_elem].set_ylabel(r'$M_{\odot}$', fontsize=10)
        axs[idx_elem].set_ylabel(r'$10^{'+str(int(base10))+'}M_{\odot}$', fontsize=10)
        axs[idx_elem].set_ylim(ymin=0)

    # Create legend for IaSNe
    if yieldtype=='IaSN':
        legend = plt.legend(handles=handles, loc='upper left', fontsize=10, 
                    bbox_to_anchor=(0.92, 0.9 - 0.12), bbox_transform=plt.gcf().transFigure)
        legend._legend_box.align = "left"

    # Final plot formatting
    plt.xlabel(r'$t$ (Gyr)', fontsize=14)

    # Output figure
    if weakrprocess:
        yieldtype += '_weakr'
    outputname = 'plots/yieldimpulse_'+yieldtype+'_empiricalfit.pdf'
    plt.savefig(outputname, bbox_inches='tight')
    plt.show()

    return

if __name__ == "__main__":

    # Plot yields
    plotyieldimpulse('IaSN', empiricalfit=[0.4389863146518289,0.305259626216913,4.942444967900384,0.4925229043278246,0.8329968649356562,0.40094641862489994,0.563019743600889,1.2909839533334972,0.8604762167017103,0.2864776957718226,1.5645763678916176,0.8939183631841486,3-0.014997329848299233])