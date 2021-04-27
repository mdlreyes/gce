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
from scipy.interpolate import interp1d
import sys

# Import other files
import params
import dtd
import gce_yields as gce_yields
import gce_plot

def runmodel(pars, plot=False, title="", amr=None, empirical=False):
    """Galactic chemical evolution model.

    Takes in additional parameters from params.py, reads yields using gce_yields.py, 
    and computes delay-time distributions using dtd.py

    Args:
        pars (list): Parameters as described in paper = 
                    [A_in/1e6, tau_in, A_out/1000, A_star/1e6, alpha, M_gas_0]
        plot (bool): If True, run gce_plot function to make plots
        title (str): Title of output plots (if plot==True)
        amr (str): If not None, title of file to save age-metallicity relation
        empirical (bool): If True, use empirical parameterizations of yields 
                        rather than yield sets

    Returns:
        model (array): All outputs of model.
        atomic (array): Atomic numbers of all tracked elements.
    """  

    # Integration parameters
    delta_t = 0.001     # time step (Gyr)
    n = int(1.36/delta_t)     # number of timesteps in the model 
    t = np.arange(n)*delta_t    # time passed in model array -- age universe (Gyr)

    # Prepare arrays for SNe and AGB calculations
    n_wd = dtd.dtd_ia(t, params.ia_model) * delta_t      # Fraction of stars that will explode as Type Ia SNe in future

    m_himass, n_himass = dtd.dtd_ii(t, params.imf_model)       # Mass and fraction of stars that will explode in the future
    idx_bad = np.where((m_himass < params.M_SN_min) | (m_himass > params.M_SN_max)) # Limit to timesteps where stars between 10-100 M_sun will explode
    m_himass[idx_bad] = 0.
    n_himass[idx_bad] = 0.

    m_intmass, n_intmass = dtd.dtd_agb(t, params.imf_model)    # Mass and fraction of stars that become AGBs in the future
    idx_bad_agb = np.where((m_intmass < params.M_AGB_min) | (m_intmass > params.M_AGB_max)) # Limit to timesteps where stars between 0.865-10 M_sun will become AGB stars
    m_intmass[idx_bad_agb] = 0.
    n_intmass[idx_bad_agb] = 0.

    if empirical==False:
        # Load all sources of chemical yields
        nel, eps_sun, SN_yield, AGB_yield, M_SN, z_II, M_AGB, z_AGB = gce_yields.initialize_yields(
            Ia_source=params.Ia_source, II_source=params.II_source, 
            AGB_source=params.AGB_source, r_process_keyword=params.r_process_keyword)
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
        ii_yield_mass[:,:,idx_bad] = 0.

        f_agb_mass = interp1d(M_AGB, yield_agb, axis=2, bounds_error=False, copy=False, assume_sorted=True)
        agb_yield_mass = f_agb_mass(m_intmass) # Compute yields of masses of stars that will produce AGB winds
        agb_yield_mass[:,:,idx_bad_agb] = 0.

        # Interpolate yield tables over metallicity
        f_ia_metallicity = interp1d(z_II, yield_ia, axis=1, bounds_error=False, copy=False, assume_sorted=True) 
        f_ii_metallicity = interp1d(z_II, ii_yield_mass, axis=1, bounds_error=False, copy=False, assume_sorted=True)
        f_agb_metallicity = interp1d(z_AGB, agb_yield_mass, axis=1, bounds_error=False, copy=False, assume_sorted=True) 

        #print('yields', f_ii_metallicity(0)[np.where(atomic == 6)[0],:25])
        #return

    # Get empirical yields
    else:
        nel, eps_sun, atomic, weight, f_ia_metallicity, f_ii_metallicity, f_agb_metallicity = gce_yields.initialize_empirical(
            Ia_source=params.Ia_source, II_source=params.II_source, 
            AGB_source=params.AGB_source, r_process_keyword=params.r_process_keyword,
            II_mass=m_himass, AGB_mass=m_intmass)

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
            'mn':np.where(atomic == 25)[0]}

    # Create array to hold model outputs
    model = np.zeros(n, dtype=[('t','float64'),('f_in','float64'),('mgas','float64'),\
        ('Ia_rate','float64'),('II_rate','float64'),('AGB_rate','float64'),\
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
    print(model['mgas'][0])

    # Define parameters for pristine gas 
    pristine = np.zeros(nel)    # Pristine element fractions by mass (dimensionless)
    pristine[0] = 0.7514        # Hydrogen from BBN                                                                                      
    pristine[1] = 0.2486        # Helium from BBN
    pristine=pristine

    # Initialize model
    model['f_in'] = f_in_norm0 * model['t'] * np.exp(-model['t']/f_in_t0)    # Compute inflow rates (just a function of time)                                                                                
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
    # AND [ (less than 10 Myr has passed) 
    # OR (gas remains within the galaxy at a given time step AND the gas mass in iron at the previous timestep is subsolar) ]
    while ((timestep < (n-1)) and ((timestep*delta_t <= 0.010) or 
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
        model['mdot'][timestep] = sfr_norm * model['mgas'][timestep]**sfr_exp / 1.e6**(sfr_exp-1.0)

        # Eq. 10: rate of Ia SNe that will explode IN THE FUTURE
        n_ia = model['mdot'][timestep] * n_wd       # Number of Type Ia SNe that will explode in the future
        model['Ia_rate'][timestep:] += n_ia[:(n-timestep)]     # Put Type Ia rate in future array

        # Eq. 11: Type Ia SNe yields IN THE FUTURE
        #f_Ia = SN_yield['Ia']                    # Mass ejected from each SN Ia (M_sun SN**-1) 
        #M_Ia = f_Ia * model['Ia_rate'][timestep]    # Eq. 11: Mass returned to the ISM by Ia SNe
        if model['z'][timestep] > 0.:
            # Put Type Ia yields in future array
            M_Ia_arr[:,timestep:] += n_ia[:(n-timestep)][None,:] * f_ia_metallicity(model['z'][timestep])[:,None]

        # Eq. 8: rate of Type II SNe that will explode IN THE FUTURE
        n_ii = model['mdot'][timestep] * n_himass   # Number of stars formed now that will explode in the future
        model['II_rate'][timestep:] += n_ii[:(n-timestep)]  # Put Type II rate in future array

        # Eq. 7: Type II SNe yields IN THE FUTURE
        if model['z'][timestep] > 0.:
            # Put Type II yields in future array
            M_II_arr[:,timestep:] += n_ii[:(n-timestep)] * f_ii_metallicity(model['z'][timestep])[:,:(n-timestep)]

        # Rate of AGB stars that will explode IN THE FUTURE
        n_agb = model['mdot'][timestep] * n_intmass   # Number of stars formed now that will produce AGB winds in the future
        model['AGB_rate'][timestep:] += n_agb[:(n-timestep)]  # Put AGB rate in future array

        # Eq. 13: AGB yields IN THE FUTURE
        if model['z'][timestep] > 0.:
            # Put AGB yields in future array
            M_AGB_arr[:,timestep:] += n_agb[:(n-timestep)] * f_agb_metallicity(model['z'][timestep])[:,:(n-timestep)]

        # Eq. 15: outflows IN CURRENT TIMESTEP (depends on gas mass fraction x_el)
        if model['mgas'][timestep] > 0.0: 
            # If there's currently gas within the galaxy, gas mass fraction depends on prev timestep
            x_el = model['abund'][timestep-1,:]/model['mgas'][timestep]
        else: 
            # Otherwise, if there is no gas, then the gas mass fraction is zero
            x_el = np.zeros(nel)

        model['mout'][timestep,:] = f_out * x_el * (model['II_rate'][timestep] + model['Ia_rate'][timestep]) 

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

    print('why stop?', timestep, model['mgas'][timestep], model['eps'][timestep-1,snindex['fe']])
    #print('test Ba', model['abund'][:timestep-1, snindex['ba']])
    #print('test Ba', model['eps'][:timestep-1, snindex['ba']])

    if plot:
        gce_plot.makeplots(model[:timestep-1], atomic, title=title, plot=True, skip_end_dots=-10, 
        abunds=True, time=False, params=False, datasource='both')

    if amr is not None:
        modeldata = np.hstack((model['eps'][:timestep-1,snindex['fe']], model['t'][:timestep-1, None]))
        np.save(amr, modeldata)

    return model[:timestep-1], atomic

    '''
    model2, atomic2 = gce_model(pars, n, delta_t, t, nel, eps_sun, SN_yield, AGB_yield, M_SN, z_II, M_AGB, z_AGB, snindex, pristine, n_wd, n_himass, f_ii_metallicity, n_intmass, f_agb_metallicity, agb_yield_mass, f_ia_metallicity)
    if plot:
        gce_plot.makeplots(model2, atomic2, title="Sculptor final", plot=True, skip_end_dots=-10, 
        abunds=True, time=True, params=True)

    return model2, atomic2
    '''