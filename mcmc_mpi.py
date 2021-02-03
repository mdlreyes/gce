"""
run_gce_mpi.py

Runs gce_fast on multiple processors
"""

#Backend for python3 on stravinsky
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sys
import numpy as np
import scipy.optimize as op
import emcee
import gce_fast as gce

# Packages for parallelization
from multiprocessing import Pool

def mcmc(params, nsteps):

	# TODO: restructure gce_fast and code here so that we don't have to load yields, etc. every time

	# Run model
	model, atomic = gce.gce_model(params)

	# Get indices for each tracked element
    elem_idx = {'h':np.where(atomic == 1)[0],
				'he':np.where(atomic == 2)[0],
				#'c':np.where(atomic == 6)[0],
				#'o':np.where(atomic == 8)[0],
				'mg':np.where(atomic == 12)[0],
				'si':np.where(atomic == 14)[0],
				'ca':np.where(atomic == 20)[0],
				'ti':np.where(atomic == 22)[0],
				'fe':np.where(atomic == 26)[0],
				'ba':np.where(atomic == 56)[0]}
	
	# Define model data
	elem_model = [ #model['eps'][:,elem_idx['c']] - model['eps'][:,elem_idx['fe']], 	# [C/Fe]
				   #model['eps'][:,elem_idx['ba']] - model['eps'][:,elem_idx['fe']],	# [Ba/Fe]
				   model['eps'][:,elem_idx['fe']] - model['eps'][:,elem_idx['h']],		# [Fe/H]
				   model['eps'][:,elem_idx['mg']] - model['eps'][:,elem_idx['fe']],		# [Mg/Fe]
				   model['eps'][:,elem_idx['si']] - model['eps'][:,elem_idx['fe']],		# [Si/Fe]
				   model['eps'][:,elem_idx['ca']] - model['eps'][:,elem_idx['fe']]		# [Ca/Fe]
				]
	sfr = model['mdot']
	mstar_model = model['mstar'][-1]

	# TODO: Define observed data
	elem_data = 
	delem_data = 
	mstar_obs = 
	dmstar_obs = 
	mgas_obs = 
	dmgas_obs = 1e3.

	# Eq. 18: Log likelihood function
	def lnlike(params):

		L = 0.

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

	# TODO: define the priors
	def lnprior(params):
		theta, bperp = params
		if -np.pi/2. < theta < np.pi/2. and -10.0 < bperp < 10.0:
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

	# TODO: put in parameters
	result = op.minimize(nll, [0., 0.], args=())
	theta_init, b_init = result["x"]

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
