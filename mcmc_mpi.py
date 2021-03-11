"""
mcmc_mpi.py

Runs gce_fast on multiple processors
"""

# Import stuff to run model
from scipy.interpolate import interp1d
import params
import dtd
import gce_yields as gce_yields
import gce_plot
from getdata import getdata

import sys
import numpy as np
import scipy.optimize as op
from scipy.integrate import trapz
import emcee

# Packages for parallelization
from multiprocessing import Pool

# Variables for MCMC run
nsteps = 1000
nwalkers = 20
parallel = True

# Put in initial guesses for parameters 
#params_init = [0.91144016, 0.19617321, 4.42241379, 4.45999299, 1.97494677, 0.17709949] # from Powell optimization
#params_init = [2.6988, 0.27, 5.37, 4.46, 0.85, 0.] # from dsph_gce.dat
params_init = [0.70157967, 0.26730922, 5.3575732, 0.47251228, 0.82681450, 0.49710685] # (based on results from Kirby+11)

# Model prep!

# Integration parameters
delta_t = 0.001     # time step (Gyr)
n = int(1.36/delta_t)     # number of timesteps in the model 
t = np.arange(n)*delta_t    # time passed in model array -- age universe (Gyr)

# Load all sources of chemical yields
nel, eps_sun, SN_yield, AGB_yield, M_SN, z_II, M_AGB, z_AGB = gce_yields.initialize_yields(
    Ia_source=params.Ia_source, II_source=params.II_source, 
    AGB_source=params.AGB_source, r_process_keyword=params.r_process_keyword)
    
# Get indices for each tracked element. Will fail if element is not contained in SN_yield.
snindex = {'h':np.where(SN_yield['atomic'] == 1)[0],
        'he':np.where(SN_yield['atomic'] == 2)[0],
        'c':np.where(SN_yield['atomic'] == 6)[0],
        #'o':np.where(SN_yield['atomic'] == 8)[0],
        'mg':np.where(SN_yield['atomic'] == 12)[0],
        'si':np.where(SN_yield['atomic'] == 14)[0],
        'ca':np.where(SN_yield['atomic'] == 20)[0],
        'ti':np.where(SN_yield['atomic'] == 22)[0],
        'fe':np.where(SN_yield['atomic'] == 26)[0],
        'ba':np.where(SN_yield['atomic'] == 56)[0],
        'mn':np.where(SN_yield['atomic'] == 25)[0]}

# Define parameters for pristine gas 
pristine = np.zeros(nel)    # Pristine element fractions by mass (dimensionless)
pristine[0] = 0.7514        # Hydrogen from BBN                                                                                      
pristine[1] = 0.2486        # Helium from BBN
pristine=pristine

# Linearly extrapolate supernova yields to min/max progenitor masses
sn_min = SN_yield[:]['II'][:,:,0] * params.M_SN_min/M_SN[0]             # Extrapolate yields to min progenitor mass
sn_max = SN_yield[:]['II'][:,:,-1] * params.M_SN_max/M_SN[-1]           # Extrapolate yields to max progenitor mass
yield_ii = np.concatenate((sn_min[...,None], SN_yield[:]['II'], sn_max[...,None]), axis=2)   # Concatenate yield tables
M_SN = np.concatenate(([params.M_SN_min], M_SN, [params.M_SN_max]))     # Concatenate mass list

# Linearly extrapolate AGB yields to min/max progenitor masses
agb_min = AGB_yield[:]['AGB'][:,:,0] * params.M_AGB_min/M_AGB[0]        # Extrapolate yields to min progenitor mass
agb_max = AGB_yield[:]['AGB'][:,:,-1] * params.M_AGB_max/M_AGB[-1]      # Extrapolate yields to max progenitor mass
yield_agb = np.concatenate((agb_min[...,None], AGB_yield[:]['AGB'], agb_max[...,None]), axis=2)   # Concatenate yield tables
M_AGB = np.concatenate(([params.M_AGB_min], M_AGB, [params.M_AGB_max])) # Concatenate mass list 

# Linearly extrapolate AGB yields to Z = 0
agb_z0 = yield_agb[:,0,:]+(0-z_AGB[0])*(yield_agb[:,1,:]-yield_agb[:,0,:])/(z_AGB[1]-z_AGB[0])
yield_agb = np.concatenate((agb_z0[:,None,:], yield_agb), axis=1)   # Concatenate yield tables
z_AGB = np.concatenate(([0],z_AGB))

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

# Interpolate yield tables over mass
f_ii_mass = interp1d(M_SN, yield_ii, axis=2, bounds_error=False, copy=False, assume_sorted=True)
ii_yield_mass = f_ii_mass(m_himass)  # Compute yields of masses of stars that will explode
ii_yield_mass[:,:,idx_bad] = 0.

f_agb_mass = interp1d(M_AGB, yield_agb, axis=2, bounds_error=False, copy=False, assume_sorted=True)
agb_yield_mass = f_agb_mass(m_intmass)  # Compute yields of masses of stars that will produce AGB winds
agb_yield_mass[:,:,idx_bad_agb] = 0.

# Interpolate yield tables over metallicity
f_ia_metallicity = interp1d(z_II, SN_yield['Ia'], axis=1, bounds_error=False, copy=False, assume_sorted=True) 
f_ii_metallicity = interp1d(z_II, ii_yield_mass, axis=1, bounds_error=False, copy=False, assume_sorted=True) 
f_agb_metallicity = interp1d(z_AGB, agb_yield_mass, axis=1, bounds_error=False, copy=False, assume_sorted=True) 

# Run model
def gce_model(pars): #, n, delta_t, t, nel, eps_sun, SN_yield, AGB_yield, M_SN, z_II, M_AGB, z_AGB, snindex, pristine, n_wd, n_himass, f_ii_metallicity, n_intmass, f_agb_metallicity, agb_yield_mass):
    """Galactic chemical evolution model."""  

    # Create array to hold model outputs
    model = np.zeros(n, dtype=[('t','float64'),('f_in','float64'),('mgas','float64'),\
        ('Ia_rate','float64'),('II_rate','float64'),('AGB_rate','float64'),\
        ('de_dt','float64',(nel)),('dstar_dt','float64',(nel)),\
        ('abund','float64',(nel)),('eps','float64',(nel)),('mout','float64',(nel)),\
        ('z','float64'),('mdot','float64'),('mgal','float64'),('mstar','float64')])
    model['t'] = t

    # Read parameters from pars
    f_in_norm0 = pars[0]*1.e9  # Normalization of gas inflow rate (10**-6 M_sun Gyr**-1)
    f_in_t0 = pars[1]       # Exponential decay time for gas inflow (Gyr)           
    f_out = pars[2]*1.e3    # Strength of gas outflow due to supernovae (M_sun SN**-1) 
    sfr_norm = pars[3]      # Normalization of SF law (10**-6 M_sun Gyr**-1)
    sfr_exp = pars[4]       # Exponent of SF law
    model['mgas'][0] = pars[5]*1.e6  # Initial gas mass (M_sun)

    # Initialize model
    model['f_in'] = f_in_norm0 * model['t'] * np.exp(-model['t']/f_in_t0)    # Compute inflow rates (just a function of time)                                                                                
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

        # Eq. 5: SFR (Msun Gyr**-1) set by generalization of K-S law                                                            
        model['mdot'][timestep] = sfr_norm * model['mgas'][timestep]**sfr_exp / 1.e6**(sfr_exp-1.0)

        # Eq. 10: rate of Ia SNe that will explode IN THE FUTURE
        n_ia = model['mdot'][timestep] * n_wd       # Number of Type Ia SNe that will explode in the future
        model['Ia_rate'][timestep:] += n_ia[:(n-timestep)]     # Put Type Ia rate in future array

        # Eq. 11: Type Ia SNe yields IN THE FUTURE
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
                model['eps'][timestep,elem] = np.log10( model['abund'][timestep,elem]/np.interp(model['z'][timestep], z_II, SN_yield[elem]['weight_II'][:,3]) )

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

    elem_model = [ model['eps'][:,snindex['fe']], #- model['eps'][:,snindex['h']],		# [Fe/H]
            model['eps'][:,snindex['mg']] - model['eps'][:,snindex['fe']] + 0.2,		# [Mg/Fe]
            model['eps'][:,snindex['si']] - model['eps'][:,snindex['fe']],		# [Si/Fe]
            model['eps'][:,snindex['ca']] - model['eps'][:,snindex['fe']],		# [Ca/Fe]
            model['eps'][:,snindex['c']] - model['eps'][:,snindex['fe']] 	# [C/Fe]
            #model['eps'][:,snindex['ba']] - model['eps'][:,snindex['fe']],	# [Ba/Fe]
            #model['eps'][:,snindex['mn']] - model['eps'][:,snindex['fe']]	# [Mn/Fe]
        ]
    sfr = model['mdot']
    mstar_model = model['mstar'][-1]
    time = model['t']

    # Compute amount of leftover gas
    leftovergas = model['mgas'][-1]
    if (abs(model['mgas'][-2] - leftovergas) > 0.5*leftovergas) or (leftovergas < 0.): 
        leftovergas = 0.0

    return np.array(elem_model)[:,:,0], sfr, mstar_model, time, leftovergas

# Define observed data
elem_data, delem_data = getdata(galaxy='Scl', c=True) #, ba=True, mn=True)
nelems, nstars = elem_data.shape
print('Numbers:', nelems, nstars)
mstar_obs = 10**6.08
dmstar_obs = 1.27e5
mgas_obs = 3.2e3
dmgas_obs = 1.e3

# Eq. 18: *Negative* log likelihood function
def neglnlike(parameters):

    # Don't even bother to compute likelihood if any of the parameters are negative
    if np.any(np.asarray(parameters) < 0.):
        return 1e10

    # Get data from model
    elem_model, sfr, mstar_model, time, leftovergas = gce_model(parameters)

    # Check if model runs all the way
    #if time[-1] < 0.9:
    #    return 1e10
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
def lnprior(parameters):

    f_in_norm0, f_in_t0, f_out, sfr_norm, sfr_exp, mgas0 = parameters

    # Define uniform priors, based on values in Table 2 of Kirby+11
    if (0. < f_in_norm0 < 5.) and (0. < f_in_t0 < 1.) and (0. < f_out < 20.) and (0. < sfr_norm < 10.) and (0. < sfr_exp < 2.) and (0. < mgas0 < 20.):
        return 0.0
    return -np.inf

# Define the full log-probability function
def lnprob(parameters):
    lp = lnprior(parameters)
    ll = neglnlike(parameters)
    if np.isfinite(lp) and np.isfinite(ll):
        return lp - ll
    else:
        return -np.inf

# Test likelihood function
print('actual values:', neglnlike([0.70157967, 0.26730922, 5.3575732, 0.47251228, 0.82681450, 0.49710685]))
print('initial values:', neglnlike([2.6988, 0.27, 5.37, 4.46, 0.85, 0.]))

# Original values
#print('values after powell:', neglnlike([0.66951658, 0.21648284, 3.96625663, 0.17007564, 1.81059811, 1.44096689])) 
#print('values after mcmc:', neglnlike([0.79, 0.20, 4.11, 0.25, 1.65, 1.72]))

# With C
print('values after powell:', neglnlike([0.91144016, 0.19617321, 4.42241379, 4.45999299, 1.97494677, 0.17709949]))
print('values after mcmc:', neglnlike([1.01, 0.18, 4.30, 1.28, 0.74, 0.11]))
print('values after mcmc (starting from kirby+11, Maoz+10 DTD):', neglnlike([1.01, 0.17, 4.31, 1.23, 0.76, 0.21]))
print('values after mcmc (starting from Kirby+11, Mannucci+06 DTD):', neglnlike([4.86509872, 0.05459378, 3.13738242, 4.87828528, 0.4670316, 0.17314514]))

'''
# Start by doing basic max-likelihood fit to get initial values
result = op.minimize(neglnlike, [2.6988, 0.27, 5.37, 4.46, 0.85, 0.], method='powell', options={'ftol':1e-6, 'maxiter':100000, 'direc':np.diag([-0.05,0.05,1.0,0.01,0.01,0.01])}) 
print(result)
params_init = result["x"]
'''
'''
# Sample the log-probability function using emcee - first, initialize the walkers
ndim = len(params_init)
dpar = [0.052456082, 0.0099561587, 0.15238868, 0.037691148, 0.038053383, 0.26619513] / np.sqrt(6.)
pos = []
for i in range(nwalkers):
    a = params_init + dpar*np.random.randn(ndim)
    a[a < 0.] = 0.
    pos.append(a)

print('Starting sampler')

if parallel:
    # Run parallel MCMC
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)

        # Save the chain so we don't have to run this again
        np.save('chain', sampler.chain, allow_pickle=True, fix_imports=True)

else:
    # Run serial MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    sampler.run_mcmc(pos, nsteps, progress=True)

    # Save the chain so we don't have to run this again
    np.save('chain', sampler.chain, allow_pickle=True, fix_imports=True)
'''