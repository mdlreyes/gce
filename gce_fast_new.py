"""
This program is based on Gina Duggan's and Ivanna Escala's GCE model code. 
The original version of the code (described in n Kirby+2011b) was written by E.N.K in IDL.

Rather than computing integrals at each time step, this code tracks a "forward-looking" array.
"""

from time import time # For timing purposes

import numpy as np
import scipy
import sys
import params
import dtd
import gce_yields

def gce_model(pars):
    """Galactic chemical evolution model.

    Takes in additional parameters from gce_params.py, and reads yields using gce_yields.py

    Args:
        pars (list): Parameters as described in paper = [A_in/1e6, tau_in, A_out/1000, A_star/1e6, alpha, M_gas_0]

    Returns:
        model (array): All outputs of model.
        SN_yield['atomic'] (array): Atomic numbers of all tracked elements.
    """

    time1 = time()

    # Integration parameters
    n = 13600           # number of timesteps in the model 
    delta_t = 0.001     # time step (Gyr)
    t = np.arange(n)*delta_t    # time passed in model array -- age universe (Gyr)

    # Read parameters from pars
    f_in_norm0 = pars[0]    # Normalization of gas inflow rate (10**-6 M_sun Gyr**-1)
    f_in_t0 = pars[1]       # Exponential decay time for gas inflow (Gyr)           
    f_out = pars[2]*1.e3    # Strength of gas outflow due to supernovae (M_sun SN**-1) 
    sfr_norm = pars[3]      # Normalization of SF law (10**-6 M_sun Gyr**-1)
    sfr_exp = pars[4]       # Exponent of SF law
    mgas0 = 1.e6*pars[5]    # Initial gas mass (M_sun)

    # Create array to hold model outputs
    model = np.zeros(n, dtype=[('t','float64'),('f_in','float64'),('Ia_rate','float64'),\
    ('II_rate','float64'),('AGB_rate','float64'),('de_dt','float64',(nel)),('dstar_dt','float64',(nel)),\
    ('abund','float64',(nel)),('eps','float64',(nel)),('mout','float64',(nel)),\
    ('z','float64'),('mdot','float64'),('mgal','float64'),('mstar','float64'),\
    ('mgas','float64')])

    # Define parameters for pristine gas 
    pristine = np.zeros(nel)    # Pristine element fractions by mass (dimensionless)
    pristine[0] = 0.7514        # Hydrogen from BBN                                                                                      
    pristine[1] = 0.2486        # Helium from BBN
    pristine=pristine

    # Load all sources of chemical yields
    nel, eps_sun, SN_yield, AGB_yield, M_SN, _, z_II, M_AGB, z_AGB = gce_yields.initialize_yields_inclBa(AGB_source = AGB_source)

    # Linearly extrapolate supernova yields to min/max progenitor masses
    sn_min = SN_yield[elem]['II'][:,0] * M_SN_min/M_SN[0]               # Extrapolate to min progenitor mass
    sn_max = SN_yield[elem]['II'][:,-1] * M_SN_max/M_SN[-1]             # Extrapolate to max progenitor mass
    yield_ii = np.concatenate((sn_min.reshape(-1,1), SN_yield[elem]['II'], sn_max.reshape(-1,1)), axis=1)   # Concatenate yield tables
    M_SN = np.concatenate(([M_SN_min], [M_SN], [M_SN_max]))             # Concatenate mass list

    # Get indices for each tracked element. Will fail if element is not contained in SN_yield.
    snindex = {'h':np.where(SN_yield['atomic'] == 1)[0],
            'he':np.where(SN_yield['atomic'] == 2)[0],
            #'c':np.where(SN_yield['atomic'] == 6)[0],
            #'o':np.where(SN_yield['atomic'] == 8)[0],
            'mg':np.where(SN_yield['atomic'] == 12)[0],
            'si':np.where(SN_yield['atomic'] == 14)[0],
            'ca':np.where(SN_yield['atomic'] == 20)[0],
            'ti':np.where(SN_yield['atomic'] == 22)[0],
            'fe':np.where(SN_yield['atomic'] == 26)[0],
            'ba':np.where(SN_yield['atomic'] == 56)[0]}

    # Initialize model
    model['f_in'] = 1.e6 * f_in_norm0 * t * np.exp(-t/f_in_t0)    # Compute inflow rates (just a function of time)                                                                                
    model['abund'][0,0] = model['mgas'][0]*pristine[0]    # Set initial gas mass of H
    model['abund'][0,1] = model['mgas'][0]*pristine[1]    # Set initial gas mass of He
    model['mgal'][0] = model['mgas'][0]   # Set the initial galaxy mass to the initial gas mass   

    # Prepare arrays for Type II SNe calculations
    m_himass, n_himass = dtd_ii(t, params.imf_model)   # Mass and fraction of stars that will explode in the future
    idx_bad = np.where((m_himass < M_SN_min) or (m_himass > M_SN_max))          # Limit to timesteps where stars between 10-100 M_sun will explode
    m_himass[idx_bad] = 0.
    n_himass[idx_bad] = 0.
    for elem in range(nel):     
        ii_yield[:,elem] = interp_func(M_SN, yield_ii[:,m], m_himass)           # Compute yields of masses of stars that will explode

    # Step through time!
    timestep = 0

    # While (the age of the universe at a given timestep is less than the age of the universe at z = 0)
    # AND [ (less than 10 Myr has passed) 
    # OR (gas remains within the galaxy at a given time step AND the gas mass in iron at the previous timestep is subsolar) ]
    while ((timestep < (n - 1)) and ((timestep <= 10) or 
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
            z[timestep] =  0.0

        # Eq. 5: SFR (Msun Gyr**-1) set by generalization of K-S law                                                            
        model['mdot'][timestep] = sfr_norm * model['mgas'][timestep]**sfr_exp / 1.e6**(sfr_exp-1.0)

        # Eq. 10: rate of Ia SNe that will explode IN THE FUTURE
        n_ia = model['mdot'][timestep] * dtd.dtd_ia(t[timestep:] - t[timestep], params.ia_model)    # Number of Type Ia SNe that will explode in the future
        model['Ia_rate'][timestep:] = model['Ia_rate'][timestep:] + n_ia                            # Put Type Ia rate in future array

        # Eq. 11: Type Ia SNe yields IN CURRENT TIMESTEP
        f_Ia = SN_yield[:]['Ia']                    # Mass ejected from each SN Ia (M_sun SN**-1) [array size = (elem)]  
        M_Ia = f_Ia * model['Ia_rate'][timestep]    # Eq. 11: Mass returned to the ISM by Ia SNe  [array size = (elem)]

        # Eq. 8: rate of Type II SNe that will explode IN THE FUTURE
        n_ii = model['mdot'][timestep] * n_himass                           # Number of stars formed now that will explode in the future
        # TODO: Put Type II rate in future array

        # TODO: Eq. 7: Type II SNe yields IN THE FUTURE
        # Interpolate metallicity

        # TODO: Eq. 13: rate of AGB stars that will explode IN THE FUTURE

        # TODO: AGB yields IN THE FUTURE

        # TODO: Inflows

        # TODO: Outflows

        # Eq. 2: Compute gas mass (M_sun), determined from individual element gas masses
        model['mgas'][timestep+1] = model['abund'][timestep-1,snindex['h']] + model['abund'][timestep-1,snindex['he']] + \
            (model['abund'][timestep-1,snindex['mg']] + model['abund'][timestep-1,snindex['si']] + \
            model['abund'][timestep-1,snindex['ca']]+model['abund'][timestep-1,snindex['ti']])*10.0**(1.31) + \
            model['abund'][timestep-1,snindex['fe']]*10.0**(0.03) 

        # Increment timestep
        timestep += 1

    return