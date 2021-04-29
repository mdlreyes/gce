"""
plot_chains.py

Make plots of MCMC chains
"""

#Backend for python3 on stravinsky
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Import other packages
import numpy as np
import corner

def plotmcmc(file='chain.npy', outfile='plots', burnin=100, empiricalfit=False):

    # Load file
    chainfile = np.load(file)
    nwalkers, nsteps, ndim = np.shape(chainfile)

    # Plot walkers
    fig, ax = plt.subplots(ndim,1,figsize=(12,12), sharex=True)
    ax = ax.ravel()

    names = [r"$A_{\mathrm{in}}$", r"$\tau_{\mathrm{in}}$", r"$A_{\mathrm{out}}$", r"$A_{\star}$", r"$\alpha$", r"$M_{\mathrm{gas},0}$"]
    if empiricalfit:
        names += [r"$\mathrm{Fe}_{\mathrm{Ia}}$", r"$\mathrm{expC}_{\mathrm{II}}$", r"$\mathrm{normMg}_{\mathrm{II}}$", r"$\mathrm{normCa}_{\mathrm{II}}$", r"$\mathrm{normC}_{\mathrm{AGB}}$"]
    for i in range(ndim):
        for j in range(nwalkers):
            chain = chainfile[j,:,i]
            ax[i].plot(range(len(chain)), chain, 'k-', alpha=0.5)
            ax[i].set_ylabel(names[i], fontsize=16)
            for label in (ax[i].get_xticklabels() + ax[i].get_yticklabels()):
                label.set_fontsize(14)

    ax[0].set_title('Chain', fontsize=18)
    ax[ndim-1].set_xlabel('Steps', fontsize=16)

    plt.savefig(outfile+'/walkers.png', bbox_inches='tight')
    plt.show()

    # Make corner plots
    samples = chainfile[:,burnin:, :].reshape((-1, ndim))
    cornerfig = corner.corner(samples, labels=names,
                                quantiles=[0.16, 0.5, 0.84],
                                show_titles=True, title_kwargs={"fontsize": 12})
    cornerfig.savefig(outfile+'/cornerfig.png')
    plt.show()

    # Compute 16th, 50th, and 84th percentiles of parameters
    for i in range(ndim):
        percentile = np.array([np.percentile(samples[:,i],16), np.percentile(samples[:,i],50), np.percentile(samples[:,i],84)])
        print(names[i]+' range:', percentile)

    print(np.percentile(samples,50, axis=0))

    return

if __name__ == "__main__":

    plotmcmc(file='chain.npy', burnin=800, empiricalfit=True)