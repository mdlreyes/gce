"""
gce_fast.py

This program is based on Gina Duggan's and Ivanna Escala's GCE model code. 
The original version of the code (described in n Kirby+2011b) was written by E.N.K in IDL.

Rather than computing integrals at each time step, this code tracks a "forward-looking" array.
"""

# For testing purposes
import time

# Import packages
import numpy as np
from numpy.ma.core import masked_not_equal
from scipy.interpolate import interp1d
import sys

# Import other files
import params
import dtd
import gce_yields as gce_yields
import gce_plot
from mcmc_mpi_empirical import neglnlike

def runmodel(pars, plot=False, title="", amr=None, sfh=None, empirical=False, empiricalfit=False, 
            feh_denom=True, delay=False, reioniz=False, ia_dtd='maoz10', rampressure=False,
            mn=None, ni=None, rprocess='none', specialrprocess='none', imf='kroupa93',
            inflow='none', outflow='none', mgenhance=True, nomgas0=False):
    """Galactic chemical evolution model.

    Takes in additional parameters from params.py, reads yields using gce_yields.py, 
    and computes delay-time distributions using dtd.py

    Args:
        pars (list): Parameters as described in paper = 
                    [A_in/1e6, tau_in, A_out/1000, A_star/1e6, alpha, M_gas_0]
        plot (bool): If True, run gce_plot function to make plots
        title (str): Title of output plots (if plot==True)
        amr (str): If not None, title of file to save age-metallicity relation
        sfh (str): If not None, title of file to save SFH
        empirical (bool): If True, use empirical parameterizations of yields 
                        rather than literature yield sets
        empiricalfit (bool): If True, use additional parameters to set yields
                        (note: pars must include these additional parameters!)
        feh_denom (bool): If True, use [Fe/H] as x-axis of plots; otherwise use [Mg/H]
        delay (bool): If True, delay SF by 50 Myr
        reioniz (bool): If True, simulate reionization by stopping gas infall at z~8.8 (0.6 Gyr)
        inflow (str): If not 'none', use different default inflow parameterizations ('const', 'gausslike')
        outflow (str): If not 'none', use different default outflow parameterizations ('inflow')
        mgenhance (bool): If True (default), add 0.2 dex to Mg/Fe

    Returns:
        model (array): All outputs of model.
        atomic (array): Atomic numbers of all tracked elements.
    """  

    # Integration parameters
    delta_t = 0.001     # time step (Gyr)
    n = int(1.36/delta_t)     # number of timesteps in the model 
    t = np.arange(n)*delta_t    # time passed in model array -- age universe (Gyr)

    # Prepare arrays for SNe and AGB calculations
    n_wd = dtd.dtd_ia(t, ia_dtd) * delta_t      # Fraction of stars that will explode as Type Ia SNe in future
    #print('test', n_wd)

    m_himass, n_himass = dtd.dtd_ii(t, imf)       # Mass and fraction of stars that will explode in the future
    goodidx = np.where((m_himass > params.M_SN_min) & (m_himass < params.M_SN_max))[0]  # Limit to timesteps where stars will explode as CCSN
    m_himass = m_himass[goodidx]    
    n_himass = n_himass[goodidx]

    m_intmass, n_intmass = dtd.dtd_agb(t, imf)    # Mass and fraction of stars that become AGBs in the future
    goodidx_agb = np.where((m_intmass > params.M_AGB_min) & (m_intmass < params.M_AGB_max))[0] # Limit to timesteps where stars between 0.865-10 M_sun will become AGB stars
    m_intmass = m_intmass[goodidx_agb]
    n_intmass = n_intmass[goodidx_agb]

    if rprocess in ['rare_event_only', 'both']:
        if specialrprocess=='enhanceddelay':
            n_nsm = dtd.dtd_nsm_enhanced(t) * delta_t      # Fraction of stars that will explode as NSMs
        else:
            n_nsm = dtd.dtd_nsm(t) * delta_t      # Fraction of stars that will explode as NSMs
        #print('test', n_nsm)

    if empirical==False:
        # Load all sources of chemical yields
        nel, eps_sun, SN_yield, AGB_yield, M_SN, z_II, M_AGB, z_AGB = gce_yields.initialize_yields(
            Ia_source=params.Ia_source, II_source=params.II_source, 
            AGB_source=params.AGB_source, r_process_keyword=rprocess)
        atomic = SN_yield['atomic']

        # Linearly extrapolate supernova yields to min/max progenitor masses
        sn_min = SN_yield['II'][:,:,0] * params.M_SN_min/M_SN[0]             # Extrapolate yields to min progenitor mass
        sn_max = SN_yield['II'][:,:,-1] * params.M_SN_max/M_SN[-1]           # Extrapolate yields to max progenitor mass
        yield_ii = np.concatenate((sn_min[...,None], SN_yield['II'], sn_max[...,None]), axis=2)   # Concatenate yield tables
        M_SN = np.concatenate(([params.M_SN_min], M_SN, [params.M_SN_max]))     # Concatenate mass list

        # Linearly extrapolate AGB yields to min/max progenitor masses
        agb_min = AGB_yield['AGB'][:,:,0] * params.M_AGB_min/M_AGB[0]        # Extrapolate yields to min progenitor mass
        agb_max = AGB_yield['AGB'][:,:,-1] * params.M_AGB_max/M_AGB[-1]      # Extrapolate yields to max progenitor mass
        yield_agb = np.concatenate((agb_min[...,None], AGB_yield['AGB'], agb_max[...,None]), axis=2)   # Concatenate yield tables
        M_AGB = np.concatenate(([params.M_AGB_min], M_AGB, [params.M_AGB_max])) # Concatenate mass list 

        # If needed, linearly extrapolate SN yields to Z=0
        if ~np.isclose(z_II[0],0.):
            ii_z0 = yield_ii[:,0,:]+(0-z_II[0])*(yield_ii[:,1,:]-yield_ii[:,0,:])/(z_II[1]-z_II[0])
            yield_ii = np.concatenate((ii_z0[:,None,:], yield_ii), axis=1)   # Concatenate yield tables

            ia_z0 = SN_yield['Ia'][:,0]+(0-z_II[0])*(SN_yield['Ia'][:,1]-SN_yield['Ia'][:,0])/(z_II[1]-z_II[0])
            yield_ia = np.concatenate((ia_z0[:,None], SN_yield['Ia']), axis=1)   # Concatenate yield tables

            weight_z0 = SN_yield['weight_II'][:,0,:]+(0-z_II[0])*(SN_yield['weight_II'][:,1,:]-SN_yield['weight_II'][:,0,:])/(z_II[1]-z_II[0])
            weight_ii = np.concatenate((weight_z0[:,None,:], SN_yield['weight_II']), axis=1)   # Concatenate yield tables
            
            z_II = np.concatenate(([0],z_II))
        
        else:
            yield_ia = SN_yield['Ia']
            weight_ii = SN_yield['weight_II']

        # Linearly extrapolate AGB yields to Z = 0
        agb_z0 = yield_agb[:,0,:]+(0-z_AGB[0])*(yield_agb[:,1,:]-yield_agb[:,0,:])/(z_AGB[1]-z_AGB[0])
        yield_agb = np.concatenate((agb_z0[:,None,:], yield_agb), axis=1)   # Concatenate yield tables
        z_AGB = np.concatenate(([0],z_AGB))

        # Interpolate yield tables over mass
        f_ii_mass = interp1d(M_SN, yield_ii, axis=2, bounds_error=False, copy=False, assume_sorted=True)
        ii_yield_mass = f_ii_mass(m_himass) # Compute yields of masses of stars that will explode

        f_agb_mass = interp1d(M_AGB, yield_agb, axis=2, bounds_error=False, copy=False, assume_sorted=True)
        agb_yield_mass = f_agb_mass(m_intmass) # Compute yields of masses of stars that will produce AGB winds

        # Interpolate yield tables over metallicity
        f_ia_metallicity = interp1d(z_II, yield_ia, axis=1, bounds_error=False, copy=False, assume_sorted=True) 
        f_ii_metallicity = interp1d(z_II, ii_yield_mass, axis=1, bounds_error=False, copy=False, assume_sorted=True)
        f_agb_metallicity = interp1d(z_AGB, agb_yield_mass, axis=1, bounds_error=False, copy=False, assume_sorted=True) 

    # Get empirical yields
    else:
        nel, eps_sun, atomic, weight, f_ia_metallicity, f_ii_metallicity, f_agb_metallicity, f_nsm_metallicity = gce_yields.initialize_empirical(
            Ia_source=params.Ia_source, II_source=params.II_source, AGB_source=params.AGB_source, 
            r_process_keyword=rprocess, specialrprocess=specialrprocess, II_mass=m_himass, AGB_mass=m_intmass, fit=empiricalfit)
        #print(nel, atomic)

        #print('empirical', f_ii_metallicity(0)[np.where(atomic == 6)[0],:25])
        #return

    # Get indices for each tracked element. Will fail if element is not contained in SN_yield.
    snindex = {'h':np.where(atomic == 1)[0],
            'he':np.where(atomic == 2)[0],
            'c':np.where(atomic == 6)[0],
            'mg':np.where(atomic == 12)[0],
            'si':np.where(atomic == 14)[0],
            'ca':np.where(atomic == 20)[0],
            'ti':np.where(atomic == 22)[0],
            'fe':np.where(atomic == 26)[0],
            'ba':np.where(atomic == 56)[0],
            'mn':np.where(atomic == 25)[0],
            'ni':np.where(atomic == 28)[0]}

    # Create array to hold model outputs
    model = np.zeros(n, dtype=[('t','float64'),('f_in','float64'),('mgas','float64'),\
        ('Ia_rate','float64'),('II_rate','float64'),('AGB_rate','float64'),('NSM_rate','float64'),\
        ('de_dt','float64',(nel)),('dstar_dt','float64',(nel)),\
        ('abund','float64',(nel)),('eps','float64',(nel)),('mout','float64',(nel)),\
        ('z','float64'),('mdot','float64'),('mgal','float64'),('mstar','float64')])
    model['t'] = t

    # Read parameters from pars
    f_in_norm0 = pars[0]*1.e9    # Normalization of gas inflow rate (M_sun Gyr**-1)
    f_in_t0 = pars[1]       # Exponential decay time for gas inflow (Gyr)           
    f_out = pars[2]*1.e3    # Strength of gas outflow due to supernovae (M_sun SN**-1) 
    sfr_norm = pars[3]      # Normalization of SF law (10**6 M_sun Gyr**-1)
    sfr_exp = pars[4]       # Exponent of SF law
    model['mgas'][0] = pars[5]*1.e6    # Initial gas mass (M_sun)

    if empiricalfit:
        # Additional free parameters from yields
        if feh_denom:
            fe_ia = pars[6]         # Fe yield from IaSNe
            fe_idx = 6
        else:
            fe_ia = 0.8
            fe_idx = 5
        cexp_ii = pars[fe_idx+1]       # C exponent for CCSN yields
        mgnorm_ii = pars[fe_idx+2]     # Mg normalization for CCSN yields
        canorm_ii = pars[fe_idx+3]     # Ca normalization for CCSN yields
        cnorm_agb = pars[fe_idx+4]    # C normalization for AGB yields
        if len(pars) > 11:
            banorm_agb = pars[fe_idx+5]
            bamean_agb = pars[fe_idx+6]
        else:
            banorm_agb = 0.33
            bamean_agb = 0.3

    # Define parameters for pristine gas 
    pristine = np.zeros(nel)    # Pristine element fractions by mass (dimensionless)
    pristine[0] = 0.7514        # Hydrogen from BBN                                                                                      
    pristine[1] = 0.2486        # Helium from BBN
    pristine=pristine

    # Initialize gas inflow
    model['f_in'] = f_in_norm0 * model['t'] * np.exp(-model['t']/f_in_t0)    # Compute inflow rates (just a function of time)   
    if inflow=='const':
        model['f_in'] = np.ones_like(model['t']) * 0.03275 * 1e9
    elif inflow=='expdec':
        model['f_in'] = 0.074 * 1e9 * model['t'] * np.exp(-(model['t']-0.5)**2/(2*0.25**2))

    # Option to turn off inflow for reionization
    if reioniz:
        reioniz_idx = np.where(model['t'] > 0.6)
        model['f_in'][reioniz_idx] = 0.  # Reionization heating effectively stops gas inflow after z ~ 8.8

    # Initialize other model inputs                                                                             
    model['abund'][0,0] = model['mgas'][0]*pristine[0]    # Set initial gas mass of H
    model['abund'][0,1] = model['mgas'][0]*pristine[1]    # Set initial gas mass of He
    model['mgal'][0] = model['mgas'][0]   # Set the initial galaxy mass to the initial gas mass    

    # Prep arrays to hold yields
    M_Ia_arr = np.zeros((nel, n))    # Array of yields contributed by Type Ia SNe
    M_II_arr = np.zeros((nel, n))    # Array of yields contributed by Type II SNe
    M_AGB_arr = np.zeros((nel, n))   # Array of yields contributed by AGB stars

    # Step through time!
    timestep = 0 

    # While (the age of the universe at a given timestep is less than the age of the universe at z = 0)
    # AND [ (less than 10 Myr *or some other max time* has passed) 
    # OR (gas remains within the galaxy at a given time step AND the gas mass in iron at the previous timestep is subsolar) ]
    if delay:
        maxtime = 0.10
    else:
        maxtime = 0.01

    #nsmcheck = False
    while ((timestep < (n-1)) and ((timestep*delta_t <= maxtime) or 
    ( (model['mgas'][timestep] > 0.0) and (model['eps'][timestep-1,snindex['fe']] < 0.0) ) )):

        if model['mgas'][timestep] < 0.: 
            # If somehow the galaxy has negative gas mass (unphysical), set gas mass to zero instead
            model['mgas'][timestep] = 0.0

        # Eq. 3: Define the gas-phase absolute metal mass fraction (dimensionless)
        if model['mgas'][timestep-1] > 0.0:
            # If, at the previous timestep, the gas mass was nonzero, 
            # subtract contributions from H and He to the total gas mass
            model['z'][timestep] = (model['mgas'][timestep] - model['abund'][timestep-1,snindex['h']] - model['abund'][timestep-1,snindex['he']])/model['mgas'][timestep]
        else:
            # Otherwise, if the gas is depleted, the gas mass is zero
            model['z'][timestep] =  0.0

        #print(model['z'][timestep], model['mgas'][timestep], model['abund'][timestep-1,snindex['h']], model['abund'][timestep-1,snindex['he']])

        # Eq. 5: SFR (Msun Gyr**-1) set by generalization of K-S law
        if delay==False:
            model['mdot'][timestep] = sfr_norm * model['mgas'][timestep]**sfr_exp / 1.e6**(sfr_exp-1.0)
        else:
            if timestep > 50:
                model['mdot'][timestep] = sfr_norm * model['mgas'][timestep-50]**sfr_exp / 1.e6**(sfr_exp-1.0)
            else:
                model['mdot'][timestep] = 0.

        # Eq. 10: rate of Ia SNe that will explode IN THE FUTURE
        n_ia = model['mdot'][timestep] * n_wd       # Number of Type Ia SNe that will explode in the future
        model['Ia_rate'][timestep:] += n_ia[:(n-timestep)]     # Put Type Ia rate in future array

        # Eq. 11: Type Ia SNe yields IN THE FUTURE
        #f_Ia = SN_yield['Ia']                    # Mass ejected from each SN Ia (M_sun SN**-1) 
        #M_Ia = f_Ia * model['Ia_rate'][timestep]    # Eq. 11: Mass returned to the ISM by Ia SNe
        if model['z'][timestep] > 0.:
            # Put Type Ia yields in future array
            if empiricalfit:
                if mn is not None:
                    M_Ia_arr[:,timestep:] += n_ia[:(n-timestep)][None,:] * f_ia_metallicity(model['z'][timestep], fe_ia, mn_ia=mn)[:,None]
                elif ni is not None:
                    M_Ia_arr[:,timestep:] += n_ia[:(n-timestep)][None,:] * f_ia_metallicity(model['z'][timestep], fe_ia, ni_ia=ni)[:,None]
                else:
                    M_Ia_arr[:,timestep:] += n_ia[:(n-timestep)][None,:] * f_ia_metallicity(model['z'][timestep], fe_ia)[:,None]
            else:
                M_Ia_arr[:,timestep:] += n_ia[:(n-timestep)][None,:] * f_ia_metallicity(model['z'][timestep])[:,None]

        # Add in NSMs if needed
        if rprocess in ['rare_event_only', 'both']:
            n_nsmexplosion = model['mdot'][timestep] * n_nsm       # Number of NSMs that will explode in the future
            M_Ia_arr[:,timestep:] += n_nsmexplosion[:(n-timestep)][None,:] * f_nsm_metallicity()[:,None]
            model['NSM_rate'][timestep:] += n_nsmexplosion[:(n-timestep)]

        # Eq. 8: rate of Type II SNe that will explode IN THE FUTURE
        goodidxnew = goodidx
        n_ii = model['mdot'][timestep] * n_himass   # Number of stars formed now that will explode in the future
        if timestep + goodidx[-1] + 1 > n:
            n_ii = n_ii[:-(timestep + goodidx[-1] - n + 1)]
            goodidxnew = goodidx[:-(timestep + goodidx[-1] - n + 1)]
        model['II_rate'][timestep+goodidxnew] += n_ii  # Put Type II rate in future array

        # Eq. 7: Type II SNe yields IN THE FUTURE
        if model['z'][timestep] > 0.:
            # Put Type II yields in future array
            if empiricalfit:
                M_II_arr[:,timestep+goodidxnew] += n_ii * f_ii_metallicity(model['z'][timestep], cexp_ii, mgnorm_ii, canorm_ii)[:,:len(n_ii)]
            else:
                M_II_arr[:,timestep+goodidxnew] += n_ii * f_ii_metallicity(model['z'][timestep])[:,:len(n_ii)]

        # Rate of AGB stars that will explode IN THE FUTURE
        goodidxnew = goodidx_agb
        n_agb = model['mdot'][timestep] * n_intmass   # Number of stars formed now that will produce AGB winds in the future
        if timestep + goodidx_agb[-1] + 1 > n:
            n_agb = n_agb[:-(timestep + goodidx_agb[-1] - n + 1)]
            goodidxnew = goodidx_agb[:-(timestep + goodidx_agb[-1] - n + 1)]
        model['AGB_rate'][timestep+goodidxnew] += n_agb  # Put AGB rate in future array

        # Eq. 13: AGB yields IN THE FUTURE
        if model['z'][timestep] > 0.:
            # Put AGB yields in future array
            if empiricalfit:
                M_AGB_arr[:,timestep+goodidxnew] += n_agb * f_agb_metallicity(model['z'][timestep], cnorm_agb, banorm_agb, bamean_agb)[:,:len(n_agb)]
            else:
                M_AGB_arr[:,timestep+goodidxnew] += n_agb * f_agb_metallicity(model['z'][timestep])[:,:len(n_agb)]

        # Eq. 15: outflows IN CURRENT TIMESTEP (depends on gas mass fraction x_el)
        if model['mgas'][timestep] > 0.0: 
            # If there's currently gas within the galaxy, gas mass fraction depends on prev timestep
            x_el = model['abund'][timestep-1,:]/model['mgas'][timestep]
        else: 
            # Otherwise, if there is no gas, then the gas mass fraction is zero
            x_el = np.zeros(nel)

        if outflow=='none':
            outflowcoeff = f_out
        elif outflow=='inflow':
            outflowcoeff = f_out * model['f_in'][timestep]
        model['mout'][timestep,:] = outflowcoeff * x_el * (model['II_rate'][timestep] + model['Ia_rate'][timestep])
        if rampressure and model['eps'][timestep-1,snindex['fe']] > -1.5:
                model['mout'][timestep,:] += x_el * pars[13] * 1e6 #0.43 * (1+f_out) * sfr_norm

        # Now put all the parts of the model together!

        # Compute rate at which a given element is locked up in stars (M_sun Gyr**-1)
        # SFR - (gas returned from SNe and AGB stars)    
        model['dstar_dt'][timestep,:] = (x_el)*model['mdot'][timestep] - M_Ia_arr[:,timestep] - M_II_arr[:,timestep] - M_AGB_arr[:,timestep]

        # Compute change in gas mass (M_sun Gyr**-1) 
        # -(rate of locking stars up) - outflow + inflow                                                         
        model['de_dt'][timestep,:] = -model['dstar_dt'][timestep,:] - model['mout'][timestep,:] + model['f_in'][timestep]*pristine 

        # Update gas masses of individual elements (M_sun),
        # using trapezoidal rule to integrate from previous timestep
        int2 = delta_t * np.array([1., 1.]) / 2.  # Constants for trapezoidal integration                                                    
        if timestep > 0:
            model['abund'][timestep,:] = model['abund'][timestep-1,:] + np.sum(int2*model['de_dt'][timestep-1:timestep+1,:].T, axis=1)

        # Compute epsilon-notation gas-phase abundances (number of atoms in units of M_sun/amu = 1.20d57)
        for elem in range(nel):
            if model['abund'][timestep,elem] <= 0.0: 
                model['abund'][timestep,elem] = 0
                model['eps'][timestep,elem] = np.nan
            else:
                if empirical==False:
                    model['eps'][timestep,elem] = np.log10(model['abund'][timestep,elem]/np.interp(model['z'][timestep], z_II, weight_ii[elem,:,3]) )
                else:
                    model['eps'][timestep,elem] = np.log10(model['abund'][timestep,elem]/weight[elem])

        model['eps'][timestep] = 12.0 + model['eps'][timestep] - model['eps'][timestep,0] - eps_sun # Logarithmic number density relative to hydrogen, relative to sun
        #print('test', model['eps'][timestep,snindex['fe']])

        #if model['NSM_rate'][timestep] > 1 and nsmcheck==False:
        #    print('first NSM:', model['NSM_rate'][timestep], model['eps'][timestep, snindex['fe']], model['t'][timestep])
        #    nsmcheck=True

        # Calculate the stellar mass of the galaxy at a given timestep,
        # using trapezoidal rule to integrate from previous timestep
        if timestep > 0:
            model['mstar'][timestep] = model['mstar'][timestep-1] + np.sum(int2*np.sum(model['dstar_dt'][timestep-1:timestep+1], 1))
        else: model['mstar'][timestep] = 0.0

        #total galaxy mass (M_sun) at this timestep
        model['mgal'][timestep] = model['mgas'][timestep] + model['mstar'][timestep]  

        # Eq. 2: Compute total gas mass (M_sun), determined from individual element gas masses, for NEXT TIMESTEP
        model['mgas'][timestep+1] = model['abund'][timestep,snindex['h']] + model['abund'][timestep,snindex['he']] + \
            (model['abund'][timestep,snindex['mg']] + model['abund'][timestep,snindex['si']] + \
            model['abund'][timestep,snindex['ca']]+model['abund'][timestep,snindex['ti']])*10.0**(1.31) + \
            model['abund'][timestep,snindex['fe']]*10.0**(0.03) 

        #If somehow the galaxy has negative gas mass, it actually has zero gas mass   
        if model['mgas'][timestep+1] < 0.: model['mgas'][timestep+1] = 0.0

        # Increment timestep
        timestep += 1

    #print('why stop?', timestep, model['mgas'][timestep], model['eps'][timestep-1,snindex['fe']])
    #print('test Ba', model['abund'][:timestep-1, snindex['ba']])
    #print('test Ba', model['eps'][:timestep-1, snindex['ba']])

    if plot:
        gce_plot.makeplots(model[:timestep-1], atomic, title=title, plot=True, skip_end_dots=-10, 
        abunds=True, time=False, params=True, datasource='both', feh=feh_denom, mgenhance=mgenhance)

    if amr is not None:
        modeldata = np.hstack((model['eps'][:timestep-1,snindex['fe']], model['t'][:timestep-1, None]))
        np.save(amr, modeldata)

    if sfh is not None:
        modeldata = np.vstack((model['mdot'][:timestep-1], model['t'][:timestep-1]))
        np.save(sfh, modeldata)

    # Once model is done, define the model outputs that are useful for MCMC
    model = model[:timestep-1]

    if mgenhance:
        mg = model['eps'][:,snindex['mg']] - model['eps'][:,snindex['fe']] + 0.2
    else:
        mg = model['eps'][:,snindex['mg']] - model['eps'][:,snindex['fe']]

    elem_model = [model['eps'][:,snindex['fe']], #- model['eps'][:,snindex['h']], # [Fe/H]
        mg,	# [Mg/Fe]
        model['eps'][:,snindex['si']] - model['eps'][:,snindex['fe']],	# [Si/Fe]
        model['eps'][:,snindex['ca']] - model['eps'][:,snindex['fe']],	# [Ca/Fe]            
        model['eps'][:,snindex['c']] - model['eps'][:,snindex['fe']],     # [C/Fe] or [C/Mg]
        model['eps'][:,snindex['ba']] - model['eps'][:,snindex['fe']]]    # [Ba/Fe]
    elem_model = np.array(elem_model)[:,:,0]

    sfr = model['mdot']
    mstar_model = model['mstar'][-1]
    time = model['t']

    # Compute amount of leftover gas
    leftovergas = model['mgas'][-1]
    if (abs(model['mgas'][-2] - leftovergas) > 0.5*leftovergas) or (leftovergas < 0.): 
        leftovergas = 0.0

    # Compute AIC/BIC
    k = 13  # Number of free parameters
    if rampressure:
        k += 1
    if nomgas0:
        k += -1
    n = 474  # Number of stars
    negll = neglnlike(pars, model=[elem_model, sfr, mstar_model, time, leftovergas])
    aic = 2*k + 2*negll
    bic = np.log(n)*k + 2*negll

    #print('test', negll, aic)

    return model[:timestep-1], atomic, aic

    '''
    model2, atomic2 = gce_model(pars, n, delta_t, t, nel, eps_sun, SN_yield, AGB_yield, M_SN, z_II, M_AGB, z_AGB, snindex, pristine, n_wd, n_himass, f_ii_metallicity, n_intmass, f_agb_metallicity, agb_yield_mass, f_ia_metallicity)
    if plot:
        gce_plot.makeplots(model2, atomic2, title="Sculptor final", plot=True, skip_end_dots=-10, 
        abunds=True, time=True, params=True)

    return model2, atomic2
    '''