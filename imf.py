import numpy as np
import sys

# Backend for matplotlib on mahler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Note that these IMFs are normalized by *mass*

def imf(mass_array, imf_model):

    if imf_model == 'kroupa93':
    
        if len(mass_array)>0:
            dN_dM = np.zeros(len(mass_array)) #dN/dm
            wmed = np.where((mass_array >= 0.5) & (mass_array < 1))[0]
            whi = np.where(mass_array >= 1.0)[0]
            wlo = np.where(mass_array < 0.5)[0]
            coeff = 0.309866 
            #Kroupa IMF coefficient for high-mass stars (M > 1.0 M_sun)
            if len(whi)>0: dN_dM[whi] = coeff*mass_array[whi]**-2.7 
            #Kroupa IMF coefficient for low-mass stars (0.5 < M/M_sun < 1.0)
            if len(wmed)>0: dN_dM[wmed] = coeff*mass_array[wmed]**-2.2 
            if len(wlo)>0: dN_dM[wlo] = 0.619732*mass_array[wlo]**-1.2
        else: dN_dM = []
        
    elif imf_model == 'kroupa01':
        #Define the properties characteristic of a Kroupa 2001 IMF
        #Note: option in Kroupa 2001 IMF for m/Msun > 1 s.t. alpha = 2.7 \pm 0.3
        #For information concerning the IMF constants, see Table 1 of Kroupa 2002
    
        alpha_lo, alpha_med, alpha_hi = -0.3, -1.3, -2.3 #(\pm 0.7, 0.5, 0.3)
        m1, m2, m3 = 0.08, 0.50, 1.
        #k = 0.877 #(\pm 0.045 stars pc^(-3) Msun^(-1))
        k = 0.877/0.0746137 # Edited to be mass-normalized over the range 0.08-100 Msun

        if len(mass_array) > 0:
    
            dN_dM = np.zeros(len(mass_array)) #dN/dm
        
            wlo = np.where(mass_array < m1)[0]
            wmed = np.where((mass_array >= m1)&(mass_array < m2))[0]
            whi = np.where(mass_array >= m2)[0]
        
            if len(wlo) > 0:
                dN_dM[wlo] = k*(mass_array[wlo]/m1)**alpha_lo
        
            if len(wmed) > 0:
                dN_dM[wmed] = k*(mass_array[wmed]/m1)**alpha_med
        
            if len(whi) > 0:
                dN_dM[whi] = k*((m2/m1)**alpha_med) * (mass_array[whi]/m2)**alpha_hi
            
        else: dN_dM = []
        
    elif imf_model == 'chabrier03':
    
        if len(mass_array)>0:
            dN_dM = np.zeros(len(mass_array)) #dN/dm
            whi = np.where(mass_array >= 1.0)[0]
            wlo = np.where(mass_array < 1.0)[0]
            if len(whi)>0: 
                dN_dM[whi] = 0.232012599*mass_array[whi]**-2.3
            if len(wlo)>0: 
                dN_dM[wlo] = 1.8902/(mass_array[wlo]*np.log(10))*np.exp(-(np.log10(mass_array[wlo])-np.log10(0.08))**2/(2*0.69**2))
        else: dN_dM = []
    
    elif imf_model == 'salpeter55':
    
        if len(mass_array)>0:
            dN_dM = 0.156713*mass_array**-2.35
        else: dN_dM = []
        
    else:
        sys.stderr.write('IMF model '+imf_model+' not recognized')
        sys.exit()

    #print(imf_model, np.trapz(dN_dM,x=mass_array))
    return dN_dM

def plot_imf():
    """Plot delay-time distributions."""

    # Stellar masses
    m = np.arange(0.08,100,0.001)

    plt.loglog(m[:-1], imf(m,'kroupa93')[:-1]*np.diff(m), 'b-', label='Kroupa et al. (1993)')
    plt.plot(m[:-1], imf(m, 'kroupa01')[:-1]*np.diff(m), 'k-', label='Kroupa et al. (2001)')
    plt.plot(m[:-1], imf(m, 'chabrier03')[:-1]*np.diff(m), 'k--', label='Chabrier et al. (2003)')
    plt.plot(m[:-1], imf(m, 'salpeter55')[:-1]*np.diff(m), 'k:', label='Salpeter (1955)')
    plt.axvspan(0.861, 47, color='r', alpha=0.5)
    plt.legend()
    plt.xlabel(r'M $(M_{\odot})$')
    plt.ylabel('dN/dM')
    plt.savefig('plots/imf.png', bbox_inches="tight")
    plt.show()

    return

if __name__ == "__main__":
    plot_imf()