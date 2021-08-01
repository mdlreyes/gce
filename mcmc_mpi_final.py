"""
mcmc_mpi.py

Runs gce_fast on multiple processors.
A modified version that uses empirical yields as free parameters.
"""

# Import stuff to run model
from scipy.interpolate import interp1d
import params
import dtd
import gce_yields
from getdata import getdata

import numpy as np
import scipy.optimize as op
from scipy.integrate import trapz
import emcee
import os

# Packages for parallelization
from multiprocessing import Pool

# Test
from astropy.cosmology import FlatLambdaCDM  # needed to compute redshifts
cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)  # using Planck (2018) params
import astropy.units as u
from astropy.cosmology import z_at_value

# Run model
def gce_model(pars, n, t, inflow, nomgas0, reioniz, pristine, delay, delta_t, snindex, n_wd, goodidx, m_himass, n_himass, goodidx_agb, m_intmass, n_intmass, rampressure, outflow_z): #, n, delta_t, t, nel, eps_sun, SN_yield, AGB_yield, M_SN, z_II, M_AGB, z_AGB, snindex, pristine, n_wd, n_himass, f_ii_metallicity, n_intmass, f_agb_metallicity, agb_yield_mass):
    """Galactic chemical evolution model."""  

    # Get empirical yields
    nel, eps_sun, atomic, weight, f_ia_metallicity, f_ii_metallicity, f_agb_metallicity, _ = gce_yields.initialize_empirical(
        Ia_source=params.Ia_source, II_source=params.II_source, AGB_source=params.AGB_source, 
        r_process_keyword=params.r_process_keyword,
        II_mass=m_himass, AGB_mass=m_intmass, fit=True)

    # Create array to hold model outputs
    model = np.zeros(n, dtype=[('t','float64'),('f_in','float64'),('mgas','float64'),\
        ('Ia_rate','float64'),('II_rate','float64'),('AGB_rate','float64'),\
        ('de_dt','float64',(nel)),('dstar_dt','float64',(nel)),\
        ('abund','float64',(nel)),('eps','float64',(nel)),('mout','float64',(nel)),\
        ('z','float64'),('mdot','float64'),('mgal','float64'),('mstar','float64')])
    model['t'] = t

    # Read parameters from pars
    paramidx = 0
    f_in_norm0 = pars[0]*1.e9  # Normalization of gas inflow rate (10**-6 M_sun Gyr**-1)     
    if inflow=='const':
        paramidx += -1    
    else:
        f_in_t0 = pars[1]       # Exponential decay time for gas inflow (Gyr)  
    f_out = pars[paramidx+2]*1.e3    # Strength of gas outflow due to supernovae (M_sun SN**-1) 
    sfr_norm = pars[paramidx+3]      # Normalization of SF law (10**-6 M_sun Gyr**-1)
    sfr_exp = pars[paramidx+4]       # Exponent of SF law
    if nomgas0:
        model['mgas'][0] = 0.  # No initial gas mass
        paramidx += -1
    else:
        model['mgas'][0] = pars[paramidx+5]*1.e6  # Initial gas mass (M_sun)

    # Additional free parameters from yields
    fe_ia = pars[paramidx+6]         # Fe yield from IaSNe
    cexp_ii = pars[paramidx+7]       # C exponent for CCSN yields
    cnorm_agb = pars[paramidx+10]    # C normalization for AGB yields
    banorm_agb = pars[paramidx+11]   # Ba normalization for AGB yields
    bamean_agb = pars[paramidx+12]   # Ba mean for AGB yields
    mgnorm_ii = pars[paramidx+8]     # Mg normalization for CCSN yields
    canorm_ii = pars[paramidx+9]     # Ca normalization for CCSN yields
    if rampressure==False:
        paramidx += -1
    else:
        rampressureconst = pars[paramidx+13]

    # Initialize gas inflow
    if inflow is None:
        model['f_in'] = f_in_norm0 * model['t'] * np.exp(-model['t']/f_in_t0)    # Compute inflow rates (just a function of time)
    elif inflow=='const':
        model['f_in'] = f_in_norm0 * np.ones_like(model['t'])
    elif inflow=='expdec':
        model['f_in'] = f_in_norm0 * np.exp(-model['t']/f_in_t0)

    # Option to turn off inflow for reionization
    if reioniz:
        reioniz_idx = np.where(model['t'] > (0.6-0.5))
        model['f_in'][reioniz_idx] = 0.  # Reionization heating effectively stops gas inflow after z ~ 8.8

    # Initialize other model parameters
    model['abund'][0,0] = model['mgas'][0]*pristine[0]    # Set initial gas mass of H
    model['abund'][0,1] = model['mgas'][0]*pristine[1]    # Set initial gas mass of He
    model['mgal'][0] = model['mgas'][0]   # Set the initial galaxy mass to the initial gas mass    

    # Prep arrays to hold yields
    M_Ia_arr = np.zeros((nel, n))   # Array of yields contributed by Type Ia SNe
    M_II_arr = np.zeros((nel, n)) 	# Array of yields contributed by Type II SNe
    M_AGB_arr = np.zeros((nel, n))  # Array of yields contributed by AGB stars

    # Step through time!
    timestep = 0 

    # While (the age of the universe at a given timestep is less than the age of the universe at z = 0)
    # AND [ (less than 10 Myr has passed) 
    # OR (gas remains within the galaxy at a given time step AND the gas mass in iron at the previous timestep is subsolar) ]
    if delay:
        maxtime = 0.10
    else:
        maxtime = 0.01

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
        if model['z'][timestep] > 0.:
            # Put Type Ia yields in future array
            M_Ia_arr[:,timestep:] += n_ia[:(n-timestep)][None,:] * f_ia_metallicity(model['z'][timestep], fe_ia)[:,None]

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
            M_II_arr[:,timestep+goodidxnew] += n_ii * f_ii_metallicity(model['z'][timestep], cexp_ii, mgnorm_ii, canorm_ii)[:,:len(n_ii)]
            
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
            M_AGB_arr[:,timestep+goodidxnew] += n_agb * f_agb_metallicity(model['z'][timestep], cnorm_agb, banorm_agb, bamean_agb)[:,:len(n_agb)]

        # Eq. 15: outflows IN CURRENT TIMESTEP (depends on gas mass fraction x_el)
        if model['mgas'][timestep] > 0.0: 
            # If there's currently gas within the galaxy, gas mass fraction depends on prev timestep
            x_el = model['abund'][timestep-1,:]/model['mgas'][timestep]
        else: 
            # Otherwise, if there is no gas, then the gas mass fraction is zero
            x_el = np.zeros(nel)

        if outflow_z is None:
            model['mout'][timestep,:] = f_out * x_el * (model['II_rate'][timestep] + model['Ia_rate'][timestep]) 
        else:
            model['mout'][timestep,:] = f_out * np.exp(outflow_z[timestep]*pars[paramidx+14]) * x_el * (model['II_rate'][timestep] + model['Ia_rate'][timestep]) 
        if rampressure and model['eps'][timestep-1,snindex['fe']] > -1.5:
            model['mout'][timestep,:] += x_el * rampressureconst * 1e6 

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

    # Once model is done, define the model outputs that are useful for MCMC
    model = model[:timestep-1]

    elem_model = [model['eps'][:,snindex['fe']], #- model['eps'][:,snindex['h']], # [Fe/H]
        model['eps'][:,snindex['mg']] - model['eps'][:,snindex['fe']],	# [Mg/Fe]
        model['eps'][:,snindex['si']] - model['eps'][:,snindex['fe']],	# [Si/Fe]
        model['eps'][:,snindex['ca']] - model['eps'][:,snindex['fe']],	# [Ca/Fe]
        model['eps'][:,snindex['c']] - model['eps'][:,snindex['fe']],   # [C/Fe]
        model['eps'][:,snindex['ba']] - model['eps'][:,snindex['fe']]]  # [Ba/Fe]

    sfr = model['mdot']
    mstar_model = model['mstar'][-1]
    time = model['t']

    # Compute amount of leftover gas
    leftovergas = model['mgas'][-1]
    if (abs(model['mgas'][-2] - leftovergas) > 0.5*leftovergas) or (leftovergas < 0.): 
        leftovergas = 0.0

    return np.array(elem_model)[:,:,0], sfr, mstar_model, time, leftovergas

# Eq. 18: *Negative* log likelihood function
def neglnlike(parameters, elem_data, delem_data, n, t, inflow, nomgas0, reioniz, pristine, delay, delta_t, snindex, n_wd, goodidx, m_himass, n_himass, goodidx_agb, m_intmass, n_intmass, rampressure, outflow_z):

    # Get data from model
    elem_model, sfr, mstar_model, time, leftovergas = gce_model(parameters, n, t, inflow, nomgas0, reioniz, pristine, delay, delta_t, snindex, n_wd, goodidx, m_himass, n_himass, goodidx_agb, m_intmass, n_intmass, rampressure, outflow_z)

    # Get observed data
    nelems, nstars = elem_data.shape
    mstar_obs = 10**6.08
    dmstar_obs = 1.27e5
    mgas_obs = 3.2e3
    dmgas_obs = 1.e3

    # Check if model runs all the way
    goodidx = np.where((time > 0.007) & (np.all(np.isfinite(elem_model),axis=0)))
    if len(goodidx[0]) < 10:
        return 1e10
    sfr = sfr[goodidx]
    time = time[goodidx]
    elem_model = elem_model[:,goodidx]

    # Note that L is the *negative* log likelihood!
    L = 0.

    # Penalize models that don't have enough stars 
    if mstar_model < 1000.0: 
        L += 3e6

    # Compute probability of star forming at a given time
    prob = sfr/mstar_model
    prob /= trapz(sfr/mstar_model, x=time) # Normalize?

    # Loop over each star
    for star in range(nstars):

        # Loop over each element ratio
        product = 1.
        for elem in range(nelems):
            if ~np.isclose(elem_data[elem,star],-999.) and delem_data[elem,star] > 0.0 and delem_data[elem,star] < 0.4:
                product *= 1./(np.sqrt(2.*np.pi)*delem_data[elem,star]) * np.exp(-(elem_data[elem,star] - elem_model[elem,:])**2./(2.*delem_data[elem,star]**2.))[0,:]

        # Integrate as function of time
        integral = trapz(product*prob, x=time)

        if np.isfinite(integral) and integral > 0.:
            L -= np.log(integral)
        else:
            L += 5

    L += 0.1 * nstars * ((mstar_obs - mstar_model)**2./(2.*dmstar_obs**2.) + leftovergas**2./(2.*dmgas_obs**2.) 
            + np.log(2.*np.pi) + np.log(dmstar_obs) + np.log(dmgas_obs))

    #print(parameters, L)
    return L

# Define the priors
def lnprior(parameters, inflow, nomgas0, rampressure, outflow_z):

    # Default parameters
    paramidx = 0
    f_in_norm0 = parameters[0]
    if inflow=='const':
        f_in_t0 = 0.
        paramidx += -1
    else:
        f_in_t0 = parameters[1]
    f_out, sfr_norm, sfr_exp, mgas0 = parameters[paramidx+2:paramidx+6]
    cexp_ii = 1.
    cnorm_agb = 0.6
    fe_ia = 0.8
    banorm_agb=0.33
    bamean_agb=2.0
    ramconst = 0.
    outflowexp = 0.

    if nomgas0:
        mgas0 = 0.
        paramidx += -1

    fe_ia = parameters[paramidx+6]
    cexp_ii, mgnorm_ii, canorm_ii, cnorm_agb = parameters[paramidx+7:paramidx+11]
    banorm_agb, bamean_agb = parameters[paramidx+11:paramidx+13]
    if rampressure:
        ramconst = parameters[paramidx+13]
    else:
        paramidx += -1
    if outflow_z is not None:
        outflowexp = parameters[paramidx+14]

    # Define uniform priors, based on values in Table 2 of Kirby+11
    if not ((0. < f_in_norm0 < 5.) and (0. <= f_in_t0 < 5.) and (0. < f_out < 20.) and (0. < sfr_norm < 10.) and (0. < sfr_exp < 2.) and \
        (0. < fe_ia < 0.9) and (0. < cexp_ii < 2.) and (0. < mgnorm_ii < 2.) and (0. < canorm_ii < 0.5) and (0.4 < cnorm_agb < 10.) and \
        (0. < banorm_agb < 5.) and (0. <= ramconst < 5.) and (0. <= mgas0 < 1.) and (-2. <= outflowexp <= 2.)):
        return -np.inf
    # Add Gaussian prior on Ba norm
    mu = 2
    sigma = 0.25
    return np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(bamean_agb-mu)**2/sigma**2

# Define the full log-probability function
def lnprob(parameters, inflow, nomgas0, rampressure, outflow_z, elem_data, delem_data, n, t, reioniz, pristine, delay, delta_t, snindex, n_wd, goodidx, m_himass, n_himass, goodidx_agb, m_intmass, n_intmass):
    lp = lnprior(parameters, inflow, nomgas0, rampressure, outflow_z)
    ll = neglnlike(parameters, elem_data, delem_data, n, t, inflow, nomgas0, reioniz, pristine, delay, delta_t, snindex, n_wd, goodidx, m_himass, n_himass, goodidx_agb, m_intmass, n_intmass, rampressure, outflow_z)
    if np.isfinite(lp) and np.isfinite(ll):
        return lp - ll
    else:
        return -np.inf

def runmcmc(outputname, nsteps=50, nwalkers=32, delay=False, reioniz=False, rampressure=False, nomgas0=False, inflow=None, outflow=None, imf='kroupa93', ia_dtd='maoz10'):

    try:

        # Put in initial guesses for parameters 
        params_init = [1.07, 0.16, 4.01, 0.89, 0.82, 0.59, 0.8, 1., 1., 0., 0.6, 0.33, 2.0, 0., 0.] # initial values
        if outflow is None:
            del params_init[14]
        if rampressure==False:
            del params_init[13]
        if nomgas0:
            del params_init[5]
        if inflow=='const':
            del params_init[1]

        # Model prep!

        # Integration parameters
        delta_t = 0.001     # time step (Gyr)
        n = int(1.36/delta_t)     # number of timesteps in the model 
        t = np.arange(n)*delta_t    # time passed in model array -- age universe (Gyr)

        # Convert times to a redshift array
        if outflow is not None:
            ages = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.5, 1.8, 2, 3])*u.Gyr
            redshifts = [z_at_value(cosmo.age, age) for age in ages]
            outflow_z = np.interp((t + 0.5), ages.value, redshifts)
        else:
            outflow_z = None

        # Prepare arrays for SNe and AGB calculations
        n_wd = dtd.dtd_ia(t, ia_dtd) * delta_t      # Fraction of stars that will explode as Type Ia SNe in future

        m_himass, n_himass = dtd.dtd_ii(t, imf)       # Mass and fraction of stars that will explode in the future
        goodidx = np.where((m_himass > params.M_SN_min) & (m_himass < params.M_SN_max))[0]  # Limit to timesteps where stars will explode as CCSN
        m_himass = m_himass[goodidx]    
        n_himass = n_himass[goodidx]

        m_intmass, n_intmass = dtd.dtd_agb(t, imf)    # Mass and fraction of stars that become AGBs in the future
        goodidx_agb = np.where((m_intmass > params.M_AGB_min) & (m_intmass < params.M_AGB_max))[0] # Limit to timesteps where stars between 0.865-10 M_sun will become AGB stars
        m_intmass = m_intmass[goodidx_agb]
        n_intmass = n_intmass[goodidx_agb]

        # Get empirical yields
        nel, _, atomic, _, _, _, _, _ = gce_yields.initialize_empirical(
            Ia_source=params.Ia_source, II_source=params.II_source, AGB_source=params.AGB_source, 
            r_process_keyword=params.r_process_keyword,
            II_mass=m_himass, AGB_mass=m_intmass, fit=True)

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
                'eu':np.where(atomic == 63)[0]}

        # Define parameters for pristine gas 
        pristine = np.zeros(nel)    # Pristine element fractions by mass (dimensionless)
        pristine[0] = 0.7514        # Hydrogen from BBN                                                                                      
        pristine[1] = 0.2486        # Helium from BBN
        pristine=pristine

        # Define observed data
        elem_dart, delem_dart, _ = getdata(galaxy='Scl', source='dart', c=True, ba=True, removerprocess='statistical', feh_denom=True) #, eu=baeu)
        elem_deimos, delem_deimos, _ = getdata(galaxy='Scl', source='deimos', c=True, ba=True, removerprocess='statistical', feh_denom=True) #, eu=baeu)

        # Don't use [Fe/H] from DART?
        elem_dart[0,:] = -999.

        # Don't use [Ba/Fe] from DEIMOS?
        elem_deimos[-1,:] = -999.

        # Combine datasets
        elem_data = np.hstack((elem_dart, elem_deimos))
        delem_data = np.hstack((delem_dart, delem_deimos))

        # Sample the log-probability function using emcee - first, initialize the walkers
        ndim = len(params_init)
        dpar = [0.052456082, 0.0099561587, 0.15238868, 0.037691148, 0.038053383, 0.26619513, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01] / np.sqrt(14.)
        if outflow is None:
            dpar = np.delete(dpar,14)
        if rampressure==False:
            dpar = np.delete(dpar,13)
        if nomgas0:
            dpar = np.delete(dpar,5)
        if inflow=='const':
            dpar = np.delete(dpar,1)

        pos = []
        for i in range(nwalkers):
            a = params_init + dpar*np.random.randn(ndim)
            pos.append(a)

        print('Starting sampler')

        # Run parallel MCMC
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, args=[inflow, nomgas0, rampressure, outflow_z, elem_data, delem_data, n, t, reioniz, pristine, delay, delta_t, snindex, n_wd, goodidx, m_himass, n_himass, goodidx_agb, m_intmass, n_intmass])
            sampler.run_mcmc(pos, nsteps, progress=True)

            # Save the chain so we don't have to run this again
            np.save(outputname, sampler.chain, allow_pickle=True, fix_imports=True)
    
    except:
        print('Oops! Model '+outputname+' failed')

    return

if __name__=="__main__":
    #runmcmc('test', nsteps=50, nwalkers=32, delay=False, reioniz=False, rampressure=False, nomgas0=True, inflow='expdec', outflow=None)
    #runmcmc('inflowconst', nsteps=20000, nwalkers=32, delay=False, reioniz=False, rampressure=False, nomgas0=True, inflow='const', outflow=None)
    #runmcmc('inflowexp', nsteps=15625, nwalkers=32, delay=False, reioniz=False, rampressure=False, nomgas0=True, inflow='expdec', outflow=None)
    #runmcmc('outflow', nsteps=15625, nwalkers=32, delay=False, reioniz=False, rampressure=False, nomgas0=True, inflow=None, outflow='test')
    #runmcmc('sfdelay', nsteps=15625, nwalkers=32, delay=True, reioniz=False, rampressure=False, nomgas0=True, inflow=None, outflow=None)
    #runmcmc('rampressure', nsteps=15625, nwalkers=32, delay=False, reioniz=False, rampressure=True, nomgas0=True, inflow=None, outflow=None)
    #runmcmc('imf_chabrier', nsteps=15625, nwalkers=32, delay=False, reioniz=False, rampressure=False, nomgas0=True, inflow=None, outflow=None, imf='chabrier03')
    #runmcmc('imf_salpeter', nsteps=15625, nwalkers=32, delay=False, reioniz=False, rampressure=False, nomgas0=True, inflow=None, outflow=None, imf='salpeter55')
    #runmcmc('dtd_tmin200', nsteps=15625, nwalkers=32, delay=False, reioniz=False, rampressure=False, nomgas0=True, inflow=None, outflow=None, ia_dtd='medhidelay')
    runmcmc('dtd_tmin50', nsteps=15625, nwalkers=32, delay=False, reioniz=False, rampressure=False, nomgas0=True, inflow=None, outflow=None, ia_dtd='lowmindelay')
    #runmcmc('dtd_index05', nsteps=15625, nwalkers=32, delay=False, reioniz=False, rampressure=False, nomgas0=True, inflow=None, outflow=None, ia_dtd='index05')
    runmcmc('dtd_index15', nsteps=15625, nwalkers=32, delay=False, reioniz=False, rampressure=False, nomgas0=True, inflow=None, outflow=None, ia_dtd='index15')
    #runmcmc('dtd_cutoff', nsteps=15625, nwalkers=32, delay=False, reioniz=False, rampressure=False, nomgas0=True, inflow=None, outflow=None, ia_dtd='cutoff')
    #runmcmc('reioniz', nsteps=15625, nwalkers=32, delay=False, reioniz=True, rampressure=False, nomgas0=True, inflow=None, outflow=None)