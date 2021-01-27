import numpy as np
import sys

def dtd_ia(t,ia_model): #(Gyr)

    if ia_model == 'maoz10':
    #Note that this DTD is for cluster environments, where the SN Ia rate is based
    #on data from galaxy clusters from z = 0 to z = 1.45 (Maoz et al. 2010)
    #The given exponent is based on the minimum iron constraint (-1.1 \pm 0.2)
        t_ia = 1e-1 #Gyr
        rate = (1e-3)*t**(-1.1)
        w = np.where(t <= t_ia)[0]
        if len(w) > 0: rate[w] = 0.0
        return rate #SNe Gyr**-1 (M_sun)**-1
        
    elif ia_model == 'mannucci06':
        t_ia = 3.752e-2 #Gyr
        prompt = 5.3e-5 #SNe Gyr^(-1) Msun^(-1)
        delayed = 1.6e-2*np.exp(-((1.e3*t - 50)/10.)**2.) #SNe Gyr^(-1) Msun^(-1)
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
    if imf_model == 'kroupa':
        # Integral of Kroupa IMF of M(t)
        n_himass = 0.0966 * (t - 0.003)**2.459

    return m_himass, n_himass

def dtd_agb(t,imf_model):

    # Eq. 6 (inverted): Massive stars that will explode in the future (M_sun)
    m_himass = ((t - 0.003)/1.2)**(-1/1.85)

    # Fraction of massive stars that will explode (as a function of time in Gyr)
    if imf_model == 'kroupa':
        # Integral of Kroupa IMF of M(t)
        n_himass = 0.0966 * (t - 0.003)**2.459

    return m_himass, n_himass
