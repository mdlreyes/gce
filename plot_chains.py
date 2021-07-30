"""
plot_chains.py

Make plots of MCMC chains
"""

#Backend for python3 on stravinsky
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Do some formatting stuff with matplotlib
from matplotlib import rc
rc('font', family='serif')
rc('text',usetex=True)

# Import other packages
import numpy as np
import scipy.stats as stats
import corner

def plotmcmc(file='chain.npy', outfile='plots', burnin=100, empiricalfit=False, c=True, fe=True, rampressure=True, nomgas0=True, inflow=None, outflow=None):

    # Load file
    chainfile = np.load(file)
    nwalkers, nsteps, ndim = np.shape(chainfile)

    # Plot walkers
    fig, ax = plt.subplots(ndim,1,figsize=(12,12), sharex=True)
    ax = ax.ravel()

    names = [r"$A_{\mathrm{in}}$", r"$\tau_{\mathrm{in}}$", r"$A_{\mathrm{out}}$", r"$A_{\star}$", r"$\alpha$", r"$M_{\mathrm{gas},0}$"]
    if empiricalfit:
        names += [r"$\mathrm{Fe}_{\mathrm{Ia}}$", r"$\mathrm{expC}_{\mathrm{II}}$", r"$\mathrm{normMg}_{\mathrm{II}}$", r"$\mathrm{normCa}_{\mathrm{II}}$", r"$\mathrm{normC}_{\mathrm{AGB}}$", r"$\mathrm{normBa}_{\mathrm{AGB}}$", r"$\mathrm{meanBa}_{\mathrm{AGB}}$", r"$\eta$", r"$k_{\mathrm{outflow}}$"]
        if outflow is None:
            del names[14]
        if rampressure==False:
            del names[13]
        if c==False:
            del names[10]
            del names[7]
        if fe==False:
            del names[6]
    if nomgas0:
        del names[5]
    if inflow=='const':
        del names[1]

    for i in range(ndim):
        for j in range(nwalkers):
            chain = chainfile[j,:,i]
            ax[i].plot(range(len(chain)), chain, 'k-', alpha=0.5)
            ax[i].set_ylabel(names[i], fontsize=16)
            for label in (ax[i].get_xticklabels() + ax[i].get_yticklabels()):
                label.set_fontsize(14)

    ax[0].set_title('Chain', fontsize=18)
    ax[ndim-1].set_xlabel('Steps', fontsize=16)

    plt.savefig(outfile+'/walkers.pdf', bbox_inches='tight')
    plt.show()

    # Make corner plots
    samples = chainfile[:,burnin:, :].reshape((-1, ndim))
    cornerfig = corner.corner(samples, labels=names,
                                quantiles=[0.16, 0.5, 0.84], rasterized=True,
                                show_titles=True, title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16})
    cornerfig.savefig(outfile+'/cornerfig.pdf', bbox_inches='tight', dpi=400)
    plt.show()

    # Compute 16th, 50th, and 84th percentiles of parameters
    for i in range(ndim):
        median = np.percentile(samples[:,i],50)
        percentile = np.array([np.percentile(samples[:,i],50), np.percentile(samples[:,i],84)-median, median-np.percentile(samples[:,i],16)])
        print(names[i]+' range:', "${:.2f}^+{:.2f}_-{:.2f}$".format(*percentile))

    # Print the output parameters
    outputparams = np.percentile(samples,50, axis=0)
    if inflow=='const':
        outputparams = np.insert(outputparams, 1, 0.)
    if nomgas0:
        outputparams = np.insert(outputparams, 5, 0.)
    if fe==False:
        outputparams = np.insert(outputparams, 6, 0.8)
    if c==False:
        outputparams = np.insert(outputparams, 7, 1.)
        outputparams = np.append(outputparams, 0.6)
    if rampressure==False:
        outputparams = np.insert(outputparams, 13, 0.)
    if outflow is None:
        outputparams = np.append(outputparams, 0.)
    print(*outputparams, sep=',')

    # Check MAP of Mgas0 parameter
    if not nomgas0:
        print('Mgas0:', stats.mode(samples[:,5]))

    return

if __name__ == "__main__":

    #plotmcmc(file='output/final/inflowconst.npy', burnin=10000, empiricalfit=True, c=True, fe=True, rampressure=False, inflow='const', nomgas0=True, outflow=None)
    #plotmcmc(file='output/final/inflowexp.npy', burnin=5000, empiricalfit=True, c=True, fe=True, rampressure=False, inflow='expdec', nomgas0=True, outflow=None)
    #plotmcmc(file='output/final/outflow.npy', burnin=5000, empiricalfit=True, c=True, fe=True, rampressure=False, inflow=None, nomgas0=True, outflow='test')
    #plotmcmc(file='output/final/sfdelay.npy', burnin=5000, empiricalfit=True, c=True, fe=True, rampressure=False, inflow=None, nomgas0=True, outflow=None)
    plotmcmc(file='output/final/reioniz.npy', burnin=5000, empiricalfit=True, c=True, fe=True, rampressure=False, inflow=None, nomgas0=True, outflow=None)
    #plotmcmc(file='output/final/rampressure.npy', burnin=5000, empiricalfit=True, c=True, fe=True, rampressure=True, inflow=None, nomgas0=True, outflow=None)
    #plotmcmc(file='output/final/imf_chabrier.npy', burnin=5000, empiricalfit=True, c=True, fe=True, rampressure=False, inflow=None, nomgas0=True, outflow=None)
    #plotmcmc(file='output/final/imf_salpeter.npy', burnin=5000, empiricalfit=True, c=True, fe=True, rampressure=False, inflow=None, nomgas0=True, outflow=None)
    #plotmcmc(file='output/final/dtd_tmin200.npy', burnin=5000, empiricalfit=True, c=True, fe=True, rampressure=False, inflow=None, nomgas0=True, outflow=None)