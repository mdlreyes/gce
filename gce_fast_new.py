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
from dtd_ia import dtd_ia

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

    #Get indices for each tracked element that has contributions
    #to its yield from SNe II. Will fail if element is not contained in SN_yield.
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

    # Prepare the arrays for IMF calculations 
    # TODO: make IMF an input variable?
    m_himass = np.logspace(np.log10(params.M_SN_min),np.log10(params.M_SN_max),200) # Mass range for massive stars/SNII (Msun)                                                                                       
    t_himass = 1.2 * m_himass**(-1.85) + 0.003                                      # Eq. 6: Total stellar lifetime for massive stars (Gyr)
    n_himass = 0.31 * m_himass[:-1]**(-2.7) * np.diff(m_himass)                     # From Kroupa IMF: Normalized number of massive stars that will explode

    m_lomass = np.logspace(np.log10(M_AGB_min), np.log10(M_AGB_max), 200)           # Mass range for low & intermediate mass stars/AGB stars (Msun)  
    t_lomass = np.zeros(len(m_lomass))                                              # Total stellar lifetime for more massive stars (Gyr)
    t_lomass[m_lomass >= 6.6] = 1.2*m_int2[m_lomass >= 6.6]**(-1.85) + 0.003        # Eq. 6
    t_lomass[m_lomass < 6.6] = 10**((0.334-np.sqrt(1.790-0.2232*(7.764-np.log10(m_int2[m_lomass < 6.6]))))/0.1116) # Eq. 12
    n_lomass = np.zeros(len(m_lomass))                                              # From Kroupa IMF: Normalized number of stars that will be AGB stars
    n_lomass[m_lomass >= 1] = 0.31 * m_lomass[m_lomass >= 1]**(-2.7)
    n_lomass[m_lomass < 1] = 0.31 * m_lomass[m_lomass >= 1]**(-2.2)
    n_lomass = n_lomass[:-1] * np.diff(m_lomass)                                  

    # Step through time!
    timestep = 0

    # While (the age of the universe at a given timestep is less than the age of the universe at z = 0)
    # AND [ (less than 10 Myr has passed) 
    # OR (gas remains within the galaxy at a given time step AND the gas mass in iron at the previous timestep is subsolar) ]
    while ((timestep < (n - 1)) and ((timestep <= 10) or 
    ( (model['mgas'][timestep] > 0.0) and (model['eps'][timestep-1,snindex['fe']] < 0.0) ) )):

        # Eq. 2: Compute gas mass (M_sun), determined from individual element gas masses
        model['mgas'][timestep] = model['abund'][timestep-1,snindex['h']] + model['abund'][timestep-1,snindex['he']] + \
            (model['abund'][timestep-1,snindex['mg']] + model['abund'][timestep-1,snindex['si']] + \
            model['abund'][timestep-1,snindex['ca']]+model['abund'][timestep-1,snindex['ti']])*10.0**(1.31) + \
            model['abund'][timestep-1,snindex['fe']]*10.0**(0.03) 
             
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

        # Eq. 10: Ia SNe
        n_ia = model['mdot'][timestep] * dtd_ia(t[timestep:] - t[timestep], ia_model)   # Number of Type Ia SNe that will explode in the future
        
        model['Ia_rate'][timestep:] = model['Ia_rate'][timestep:] + n_ia    # Put Type Ia rate in future array

        # TODO: Put IaSNe yields in future array

        # Eq. 7, 8: rate of Type II SNe
        n_ii = model['mdot'][timestep] * n_himass   # Number of stars formed now that will explode in the future
        t_ii = t_himass[:-1] + t[timestep]          # Times when they will explode (Gyr)
        idx_ii = np.searchsorted(t_ii, t)           # Indices to put yields in the future array

        model['II_rate'][idx_ii] = model['II_rate'][idx_ii] + n_ii  # Put Type II rate in future array

        # TODO: Put IISNe yields in future array

        # Eq. 13: rate of AGB stars
        n_agb = model['mdot'][timestep] * n_lomass  # Number of stars formed now that will explode in the future
        t_agb = t_lomass[:-1] + t[timestep]         # Times when they will produce winds (Gyr)
        idx_agb = np.searchsorted(t_agb, t)         # Indices to put yields in the future array

        model['AGB_rate'][idx_agb] = model['AGB_rate'][idx_agb] + n_agb  # Put AGB rate in future array

        # TODO: Put AGB yields in future array

        # Inflows

        # Outflows

    return