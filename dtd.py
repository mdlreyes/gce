import numpy as np
import scipy.integrate
import sys

# Backend for matplotlib on mahler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def dtd_ia(t,ia_model): #(Gyr)

    if ia_model == 'maoz10':
    #Note that this DTD is for cluster environments, where the SN Ia rate is based
    #on data from galaxy clusters from z = 0 to z = 1.45 (Maoz et al. 2010)
    #The given exponent is based on the minimum iron constraint (-1.1 \pm 0.2)
    # Actually, check Freundlich & Maoz (2021) for DTD...
        t_ia = 1e-1 #Gyr
        rate = (1e-3)*t**(-1.1)
        w = np.where(t <= t_ia)[0]
        if len(w) > 0: rate[w] = 0.0
        return rate #SNe Gyr**-1 (M_sun)**-1

    elif ia_model == 'maoz17':
        #Note that this DTD is for field galaxies (Maoz & Graur 2017), 
        #where the SN Ia rate is lower than the rate in cluster galaxies
        t_ia = 1e-1 #Gyr
        rate = (0.2e-3)*t**(-1.1)
        w = np.where(t <= t_ia)[0]
        if len(w) > 0: rate[w] = 0.0
        return rate #SNe Gyr**-1 (M_sun)**-1

    elif ia_model == 'highmindelay':
        #Higher minimum delay time
        t_ia = 5e-1 #Gyr
        rate = (1e-3)*t**(-1.1)
        w = np.where(t <= t_ia)[0]
        if len(w) > 0: rate[w] = 0.0
        return rate #SNe Gyr**-1 (M_sun)**-1

    elif ia_model == 'medhidelay':
        #Only slightly higher minimum delay time
        t_ia = 2e-1 #Gyr
        rate = (1e-3)*t**(-1.1)
        w = np.where(t <= t_ia)[0]
        if len(w) > 0: rate[w] = 0.0
        return rate #SNe Gyr**-1 (M_sun)**-1

    elif ia_model == 'lowmindelay':
        #Lower minimum delay time
        t_ia = 5e-2 #Gyr
        rate = (1e-3)*t**(-1.1)
        w = np.where(t <= t_ia)[0]
        if len(w) > 0: rate[w] = 0.0
        return rate #SNe Gyr**-1 (M_sun)**-1

    elif ia_model == 'index05':
        #Power law index = -0.5
        t_ia = 1e-1 #Gyr
        rate = (1e-3)*t**(-0.5)
        w = np.where(t <= t_ia)[0]
        if len(w) > 0: rate[w] = 0.0
        return rate #SNe Gyr**-1 (M_sun)**-1

    elif ia_model == 'index15':
        #Power law index = -1.5
        t_ia = 1e-1 #Gyr
        rate = (1e-3)*t**(-1.5)
        w = np.where(t <= t_ia)[0]
        if len(w) > 0: rate[w] = 0.0
        return rate #SNe Gyr**-1 (M_sun)**-1

    elif ia_model == 'cutoff':
        #Mimicking a single degenerate Ia DTD, with a cut-off at 1 Gyr where power law index goes from -1 to -2
        t_ia = 1e-1 #Gyr
        rate = (1e-3)*t**(-1)
        w = np.where(t <= t_ia)[0]
        if len(w) > 0: rate[w] = 0.0
        w_cutoff = np.where(t > 1)[0] # Gyr
        if len(w_cutoff) > 0: rate[w_cutoff] = (1e-3)*t[w_cutoff]**(-2)
        return rate #SNe Gyr**-1 (M_sun)**-1
        
    elif ia_model == 'mannucci06':
        t_ia = 3.752e-2 #Gyr
        delayed = 5.3e-5 #SNe Gyr^(-1) Msun^(-1)
        prompt = 1.6e-2*np.exp(-((1.e3*t - 50)/10.)**2.) #SNe Gyr^(-1) Msun^(-1)
        rate = prompt + delayed
        w = np.where(t <= t_ia)[0]
        if len(w) > 0: rate[w] = 0.0
        return rate #SNe Gyr**-1 Msun**(-1)
    
    else: 
        sys.stderr.write('Type Ia SNe DTD '+ia_model+' not recognized')
        sys.exit()

def dtd_ii(t,imf_model):

    # Eq. 6 (inverted): Massive stars that will explode in the future (M_sun)
    m_himass = ((t - 0.003)/1.2)**(-1/1.85)

    # Fraction of massive stars that will explode (as a function of time in Gyr)
    if imf_model == 'kroupa93':
        # Integral of Kroupa IMF of M(t)
        n_himass = 0.31 * (m_himass)**(-2.7) * -np.concatenate((np.diff(m_himass),[0]))

    m_himass[~np.isfinite(m_himass)] = 0.
    n_himass[~np.isfinite(m_himass)] = 0.

    return m_himass, n_himass

def dtd_agb(t,imf_model):

    # Eq. 6 (inverted): Massive stars (> 6.6 M_sun) that will explode in the future (M_sun)
    m_himass = ((t - 0.003)/1.2)**(-1/1.85)

    # Eq. 12 (inverted): Less-massive stars (< 6.6 M_sun) that will produce AGB winds in the future (M_sun)
    m_lomass = 10. ** (7.764 - ((1.790 - (0.334 - 0.1116 * np.log10(t)) ** 2.) / 0.2232) )

    # Fraction of massive stars that will produce AGB winds (as a function of time in Gyr)
    if imf_model == 'kroupa93':
        # Integral of Kroupa IMF of M(t) for M > 6.6 M_sun (imf_himass, m_himass)
        n_himass = 0.31 * (m_himass)**(-2.7) * -np.concatenate((np.diff(m_himass),[0]))
        idx_himass = np.where((m_himass >= 6.6) & (m_himass < 10))

        # Integral of Kroupa IMF of M(t) for 1 < M < 6.6 M_sun (imf_himass, m_lomass)
        n_intmass = 0.31 * (m_lomass)**(-2.7) * -np.concatenate((np.diff(m_lomass),[0]))
        idx_intmass = np.where((m_lomass >= 1) & (m_lomass < 6.61))

        # Integral of Kroupa IMF of M(t) for 0.5 < M < 1 M_sun (imf_lomass, m_lomass)
        n_lomass = 0.31 * (m_lomass)**(-2.2) * -np.concatenate((np.diff(m_lomass),[0]))
        idx_lomass = np.where((m_lomass >= 0.865) & (m_lomass < 1))

        # Combine arrays to get total fraction of future AGB stars
        n_agb = np.zeros(len(t))
        n_agb[idx_himass] = n_himass[idx_himass]
        n_agb[idx_intmass] = n_intmass[idx_intmass]
        n_agb[idx_lomass] = n_lomass[idx_lomass]

        m_agb = np.zeros(len(t))
        m_agb[idx_himass] = m_himass[idx_himass]
        m_agb[idx_intmass] = m_lomass[idx_intmass]
        m_agb[idx_lomass] = m_lomass[idx_lomass]

    m_agb[~np.isfinite(m_agb)] = 0.
    n_agb[~np.isfinite(m_agb)] = 0.

    return m_agb, n_agb

def dtd_nsm(t):
    """DTD for NSMs"""

    t_ia = 1e-2  # (Gyr) from Fig 7 of Cote+17
    rate = (0.0059*1.6e-2)*t**(-1.5)  # normalization from Simonetti+19
    w = np.where(t <= t_ia)[0]
    if len(w) > 0: rate[w] = 0.0
    return rate  # NSM Gyr**-1 (M_sun)**-1

def plot_dtd(model):
    """Plot delay-time distributions."""

    t = np.arange(0.001,13.6,0.001)

    dtd_ia_maoz10 = dtd_ia(t, ia_model='maoz10')
    dtd_ia_mannucci06 = dtd_ia(t, ia_model='mannucci06')
    dtd_ia_model = dtd_ia(t, ia_model=model)
    dtd_nsm_model = dtd_nsm(t)

    plt.loglog(t*1e9, dtd_ia_maoz10, 'k-', label='Type Ia (Maoz et al. 2010)')
    plt.plot(t*1e9, dtd_ia_mannucci06, 'k--', label='Type Ia (Mannucci et al. 2006)')
    plt.plot(t*1e9, dtd_ia_model, 'k:', label='Type Ia ('+model+')')
    plt.plot(t*1e9, dtd_nsm_model, 'r', label='NSM')
    #plt.plot(t, dtd_agb(t, imf_model)[0], 'b-', label='AGB')
    #plt.plot(t, dtd_ii(t, imf_model)[0], 'r-', label='Type Ia')
    plt.legend()
    plt.xlabel('Delay time (yr)')
    plt.ylabel('')
    #plt.xlim(0,1e8)
    plt.show()

    return

if __name__ == "__main__":
    plot_dtd('cutoff')
