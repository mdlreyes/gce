"""
run_gce_mpi.py

Runs gce_fast on multiple processors
"""

#Backend for python3 on stravinsky
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Import files to run model
import params
import dtd
import gce_yields
import gce_plot

import sys
import numpy as np
import scipy.optimize as op
import emcee

# Packages for parallelization
from multiprocessing import Pool

def mcmc(params, nsteps):

	# Model prep!

	# Integration parameters
    n = 1360           # number of timesteps in the model 
    delta_t = 0.001     # time step (Gyr)
    t = np.arange(n)*delta_t    # time passed in model array -- age universe (Gyr)

    # Load all sources of chemical yields
    nel, eps_sun, SN_yield, AGB_yield, M_SN, _, z_II, M_AGB, z_AGB = gce_yields.initialize_yields_inclBa(AGB_source = params.AGB_source)

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

    # TODO: Linearly extrapolate AGB yields to Z = 0

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
    ii_yield_mass = f_ii_mass(m_himass) # Compute yields of masses of stars that will explode
    ii_yield_mass[:,:,idx_bad] = 0.

    f_agb_mass = interp1d(M_AGB, yield_agb, axis=2, bounds_error=False, copy=False, assume_sorted=True)
    agb_yield_mass = f_agb_mass(m_intmass) # Compute yields of masses of stars that will produce AGB winds
    agb_yield_mass[:,:,idx_bad_agb] = 0.

    # Interpolate yield tables over metallicity
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
		f_in_norm0 = pars[0]    # Normalization of gas inflow rate (10**-6 M_sun Gyr**-1)
		f_in_t0 = pars[1]       # Exponential decay time for gas inflow (Gyr)           
		f_out = pars[2]*1.e3    # Strength of gas outflow due to supernovae (M_sun SN**-1) 
		sfr_norm = pars[3]      # Normalization of SF law (10**-6 M_sun Gyr**-1)
		sfr_exp = pars[4]       # Exponent of SF law
		model['mgas'][0] = 1.e6*pars[5]    # Initial gas mass (M_sun)

		# Initialize model
		model['f_in'] = 1.e6 * f_in_norm0 * model['t'] * np.exp(-model['t']/f_in_t0)    # Compute inflow rates (just a function of time)                                                                                
		model['abund'][0,0] = model['mgas'][0]*pristine[0]    # Set initial gas mass of H
		model['abund'][0,1] = model['mgas'][0]*pristine[1]    # Set initial gas mass of He
		model['mgal'][0] = model['mgas'][0]   # Set the initial galaxy mass to the initial gas mass    

		# Prep arrays to hold yields
		M_II_arr = np.zeros((nel, n)) 	# Array of yields contributed by Type II SNe
		M_AGB_arr = np.zeros((nel, n))  # Array of yields contributed by AGB stars

		# Step through time!
		timestep = 0 

		# While (the age of the universe at a given timestep is less than the age of the universe at z = 0)
		# AND [ (less than 10 Myr has passed) 
		# OR (gas remains within the galaxy at a given time step AND the gas mass in iron at the previous timestep is subsolar) ]
		while ((timestep < (n-1)) and ((timestep <= 10) or 
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

			# Eq. 11: Type Ia SNe yields IN CURRENT TIMESTEP
			f_Ia = SN_yield[:]['Ia']                    # Mass ejected from each SN Ia (M_sun SN**-1) 
			M_Ia = f_Ia * model['Ia_rate'][timestep]    # Eq. 11: Mass returned to the ISM by Ia SNe

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
				if model['z'][timestep] < min(z_AGB):
					M_AGB_arr[:,timestep:] += n_agb[:(n-timestep)] * agb_yield_mass[:,0,:(n-timestep)]
				else:
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
			model['dstar_dt'][timestep,:] = (x_el)*model['mdot'][timestep] - M_II_arr[:,timestep] - M_AGB_arr[:,timestep] - M_Ia

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
					model['eps'][timestep,elem] = np.log10(model['abund'][timestep,elem]/interp_func(z_II,SN_yield[elem]['weight_II'][:,3], model['z'][timestep]))

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
		model = model[:timestep]

		elem_model = [ #model['eps'][:,snindex['c']] - model['eps'][:,snindex['fe']], 	# [C/Fe]
				#model['eps'][:,snindex['ba']] - model['eps'][:,snindex['fe']],	# [Ba/Fe]
				model['eps'][:,snindex['fe']] - model['eps'][:,snindex['h']],		# [Fe/H]
				model['eps'][:,snindex['mg']] - model['eps'][:,snindex['fe']],		# [Mg/Fe]
				model['eps'][:,snindex['si']] - model['eps'][:,snindex['fe']],		# [Si/Fe]
				model['eps'][:,snindex['ca']] - model['eps'][:,snindex['fe']]		# [Ca/Fe]
			]
		sfr = model['mdot']
		mstar_model = model['mstar'][-1]

		return elem_model, sfr, mstar_model

	# TODO: Define observed data
	elem_data = 
	delem_data = 
	mstar_obs = 12e5.
	dmstar_obs = 5e5.
	mgas_obs = 
	dmgas_obs = 1e3.

	# Eq. 18: Log likelihood function
	def lnlike(params):

		L = 0.

		# Get data from model
		elem_model, sfr, mstar_model = gce_model(params)

		# Loop over each star
		for star in nstars:

			# Integrate over time
			integral = 0.
			for t in range(len(sfr)):

				# Multiply by probability of star forming
				product = sfr[t]/mstar_model
		
				# Loop over each element ratio
				for elem in nel:
					product *= 1./(np.sqrt(2.*np.pi)*delem_data[star, elem]) * np.exp(-(elem_data[star,elem] - elem_model[elem][t])**2./(2.*delem_data[star, elem]**2.))

				integral += product

		L += -np.log(integral)

		L += 0.1 * nstars * ((mstar_obs - mstar_model)**2./(2.*dmstar_obs**2.) + mgas_obs**2./(2.*dmgas_obs**2.) 
		     + np.log(2.*np.pi) + np.log(dmstar_obs) + np.log(dmgas_obs))

		return L

	# Define the priors
	def lnprior(params):

		f_in_norm0, f_in_t0, f_out, sfr_norm, sfr_exp, mgas0 = params

		# Define uniform priors, based on values in Table 2 of Kirby+11
		if (0. < f_in_norm0 < 5.) and (0. < f_in_t0 < 1.) and (0. < f_out < 20.) and (0. < sfr_norm < 10.) and (0. < sfr_exp < 2.) and (0. < mgas0 < 20.):
			return 0.0
		return -np.inf

	# Define the full log-probability function
	def lnprob(params):
		lp = lnprior(params)
		ll = lnlike(params)
		if np.isfinite(lp) and np.isfinite(ll):
			return lp + ll
		else:
			return lp + ll
		
	# Start by doing basic max-likelihood fit to get initial values
	nll = lambda *args: -lnlike(*args)

	# TODO: put in initial guesses for parameters (or use results from Kirby+11)
	result = op.minimize(nll, [0., 0.], args=())
	params_init = result["x"]

	# Sample the log-probability function using emcee - first, initialize the walkers
	ndim = len(params)
	nwalkers = 100
	pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

	# Run MCMC

	with Pool() as pool:
		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
		sampler.run_mcmc(pos, nsteps, progress=True)
		multi_time = end - start

	'''
	# Plot walkers
	fig, ax = plt.subplots(2,1,figsize=(8,8), sharex=True)
	ax = ax.ravel()

	names = [r"$\theta$", r"$b_{\bot}$"]
	for i in range(ndim):
		for j in range(nwalkers):
			chain = sampler.chain[j,:,i]
			ax[i].plot(range(len(chain)), chain, 'k-', alpha=0.5)
			ax[i].set_ylabel(names[i], fontsize=16)
			for label in (ax[i].get_xticklabels() + ax[i].get_yticklabels()):
				label.set_fontsize(14)

	ax[0].set_title(title, fontsize=18)
	ax[1].set_xlabel('Steps', fontsize=16)

	plt.savefig(outfile+'_walkers.png', bbox_inches='tight')
	plt.close()

	# Make corner plots
	samples = sampler.chain[:,100:, :].reshape((-1, ndim))
	cornerfig = corner.corner(samples, labels=[r"$\theta$", r"$b_{\bot}$"],
								quantiles=[0.16, 0.5, 0.84],
								show_titles=True, title_kwargs={"fontsize": 12})
	cornerfig.savefig(outfile+'_cornerfig.png')
	plt.close(cornerfig)

	# Compute 16th, 50th, and 84th percentiles of parameters
	theta = np.array([np.percentile(samples[:,0],16), np.percentile(samples[:,0],50), np.percentile(samples[:,0],84)])
	bperp = np.array([np.percentile(samples[:,1],16), np.percentile(samples[:,1],50), np.percentile(samples[:,1],84)])

	print(theta, bperp)
	'''
	
	return

def main():

	pass

if __name__ == "__main__":
	main()
