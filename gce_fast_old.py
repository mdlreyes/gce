"""
This program is a python version wrriten by Ivanna Escala of the GCE model code 
from Kirby et al. 2011b. The original version of the code was written by E.N.K in IDL.
The purpose of the I.E. modifications are for compatibility 
with the FIRE-2 dwarf galaxy simulations (Hopkins et al. 2017).
"""
import numpy as np
import scipy
import scipy.integrate
import sys
#Load other files
sys.path.insert(0, '/raid/gduggan/gce')
import params as gce_params
import gce_yields
import imf
sys.path.insert(0, '/raid/gduggan/gce/dtd')
import dtd as dtd_ia

"""
Mass ejected by supernova (courtesy Hai Fu):

Thornton et al (1998) calculated that only 10% of the initial SN
energy (1E51 erg) is released in the form of kinetic energy (8.5E49
erg). From virial theorem, M = 5 r sigma**2/G, so the escape velocity,
v_e = sqrt(2 G M / r) = sqrt(10 sigma**2). So the mass ejected equals
M_eject = E/v_e**2 = ((8.5e49 erg) / ((10 km (s**(-1)))**2)) / 10 = 4274 M_sun
"""

def idl_tabulate(x, f, p=5) :
    def newton_cotes(x, f) :
        if x.shape[0] < 2 :
            return 0.
        rn = (x.shape[0] - 1) * (x - x[0]) / (x[-1] - x[0])
        weights = scipy.integrate.newton_cotes(rn)[0]
        return (x[-1] - x[0]) / (x.shape[0] - 1) * np.dot(weights, f)
    ret = 0
    for idx in xrange(0, x.shape[0], p - 1):
        ret += newton_cotes(x[idx:idx + p], f[idx:idx + p])
    return ret
    
def interp_func(x,y,xinterp):
    index = np.argsort(x)
    if len(xinterp.shape) !=0:
        indexinterp = np.argsort(xinterp)
        yinterp = np.interp(xinterp[indexinterp],x[index],y[index])[indexinterp]
    else: yinterp = np.interp(xinterp,x[index],y[index])           
    return yinterp


def gce_model(pars, imf_model = '', r_process_keyword = '', AGB_source = '',
                 AGB_yield_mult = np.nan, SNII_yield_mult = np.nan, NSM_yield_mult = np.nan):  #SFR fixed to gas mass

    # set up type of model (load keywords)
    # initialize structured array
    # set constants for newton cotes integration (if needed)
    # set 't' array
    # set 'f_in' array
    # set t=0 parameters
    # for name = 'sfr-law' and npars=6: 
    #pars = [f_in_norm0, f_in_t0, f_out/1e3, sfr_norm, sfr_exp, model[0]['mgas']/1e6]
    #using names used in the paper:  
    #pars = [A_in/1e6, tau_in, A_out/1e3, A_star/1e6, alpha, mgas_0]
    ################## set all input parameters of the model ##################
    #Load parameters set in gce_params.py needed for yield table and M_SN_ej
    #if value isn't passed to the function, use the value saved in gce_params.py
    if len(r_process_keyword) == 0:             
        r_process_keyword = gce_params.r_process_keyword
    if len(AGB_source) == 0:
        AGB_source = gce_params.AGB_source
    M_AGB_min = gce_params.M_AGB_min
    M_AGB_max = gce_params.M_AGB_max
    M_SN_min = gce_params.M_SN_min
    M_SN_max = gce_params.M_SN_max
    M_HN_min = gce_params.M_HN_min
    name = gce_params.name
    ia_model = gce_params.ia_model
    if len(imf_model) == 0:
        imf_model = gce_params.imf_model
    n = int(gce_params.n)                        #Number of timesteps in the model. 
    delta_t = gce_params.delta_t            #time step (Gyr)
    epsilon_HN = gce_params.epsilon_HN
    ##### multiplier of barium ejecta of each source
    if np.isnan(AGB_yield_mult) == True:
        AGB_yield_mult = gce_params.AGB_yield_mult
    if np.isnan(SNII_yield_mult) == True:
        SNII_yield_mult = gce_params.SNII_yield_mult
    if np.isnan(NSM_yield_mult) == True:
        NSM_yield_mult = gce_params.NSM_yield_mult
    #Load parameters saved in 'pars'
    npars = len(pars) 
    if npars == 4: ipar = 2                 #no gas infall 
    else:                                   #with gas infall
        ipar = 0       
        #f_in_norm: Normalization of gas inflow rate (10**6 M_sun Gyr**-1)
        f_in_norm0 = pars[0]
        #f_in_t0: Exponential decay time for gas inflow (Gyr)
        f_in_t0 = pars[1]
    #f_out: Strength of gas outflow due to supernovae (M_sun SN**-1)           
    f_out = pars[2-ipar]*1.e3    
    sfr_norm = pars[3-ipar]
    sfr_exp = pars[4-ipar]
    #Load all sources of chemical yields
    nel, eps_sun, SN_yield, AGB_yield, M_SN, M_HN, z_II, M_AGB, z_AGB = gce_yields.initialize_yields_inclBa(
        r_process_keyword = r_process_keyword, AGB_source = AGB_source)

    #redefine AGB masses to cover full AGB mass range   
    M_AGB_ej = np.unique(np.concatenate(([M_AGB_min],M_AGB,[M_AGB_max]))) #SNeIa, AGB, SNe II
    #Make sure that the hypernovae minimum mass is a grid point for both the supernovae II and hypernovae.
    #If it isn't, return an error and set the hypernovae minimum mass to the closest mass in the yield grids. 
    #(If desired, you could extrapolate the yield for the hypernovae minimum mass and update SN_yield, M_SN, and M_HN.)
    if (M_HN_min not in M_SN) or (M_HN_min not in M_HN):
        print("ERROR: Select a minimum hypernovae mass (M_HN_min) in gce_params.py that exists in the yield tables.")
        mass_options = np.intersect1d(M_SN, M_HN)
        print("You selected %g. Acceptable masses are:"%M_HN_min,mass_options) 
        M_HN_min = mass_options[np.argmin(np.abs(mass_options-M_HN_min))]
        print("Setting M_HN_min to %g M_sun"%M_HN_min)
    #redefine SN masses to cover full SN mass range and to turn on hypernovae
    #subtract 0.01 from hypernovae minimum mass to define an endpoint for SN and a start point for HN (M_sun)                                                                                       
    #Note that the following are the masses for which we have supernovae models
    #M_SN = [13, 15, 18, 20, 25, 30, 40]
    #M_SN_ej = [10, 13, 15, 18, 19.99, 20, 25, 30, 40, 100]                                                                                          
    M_SN_ej = np.concatenate(([M_SN_min, M_SN_max, M_HN_min, M_HN_min-0.01], M_SN))
    M_SN_ej = np.sort(np.unique(M_SN_ej))
    #set SN masses to cover full SN mass range and to turn on hypernovae
    M_SN_ej = M_SN_ej
    index_M_HN_min_M_SN = np.where(M_SN == M_HN_min)[0][0] #index in M_SN where hypernovae start to be considered
    ###TODO: Interpolate if the minimum hypernova mass (M_HN_min) doesn't happen to be included in the supernova grid (M_SN)
    index_M_HN_min_M_SN_ej = np.where(M_SN_ej == M_HN_min)[0][0] #index in M_SN_ej where hypernovae start to be considered
    #pristine element fractions by mass (dimensionless)
    pristine = np.zeros(nel)
    pristine[0] = 0.7514 #Hydrogen from BBN                                                                                      
    pristine[1] = 0.2486 #Helium from BBN
    pristine=pristine
    #constants for Newton-Cotes integration
    int2 = delta_t * np.array([1., 1.]) / 2. #trapezoid rule                                                                     
    int3 = delta_t * np.array([1., 4., 1.]) / 3. #simpsons rule
    int4 = delta_t * np.array([1., 3., 3., 1.]) * 3. / 8. #simpsons 3/8 rule
    int5 = delta_t * np.array([7., 32., 12., 32., 7.]) * 2. / 45. #boole's rule
    ################## define dummy integration arrays ##################
    #(M_sun), 200 steps btw minimum and maximum mass for massive stars/SNII                                                                              
    m_int1 = np.logspace(np.log10(M_SN_min),np.log10(M_SN_max),200)         
    #total stellar lifetime for massive stars in (Gyr), Eq. 6                                                                            
    t_int1 = 1.2*m_int1**(-1.85) + 0.003                                                                         
    #(M_sun), 200 steps between minimum and maximum mass for low and intermediate mass stars/AGB stars
    m_int2 = np.logspace(np.log10(M_AGB_min), np.log10(M_AGB_max), 200) #Up to 8 in Evan's GCE code
    wlo = m_int2 < 6.6   #Define threshold for "low" mass stars at 6.6 Msun
    whi = m_int2 >= 6.6  #Define threshold for intermediate mass stars
    #total stellar lifetime for low- and intermediate-mass stars (Gyr)
    t_int2 = np.zeros(len(m_int2))
    t_int2[whi] = 1.2*m_int2[whi]**(-1.85) + 0.003  #same as for the massive stars, Eq. 6
    #Define the lower mass limits for contributions from certain types of ejecta                                                            
    # Eq. 12 for "low" mass stars
    t_int2[wlo] = 10**((0.334-np.sqrt(1.790-0.2232*(7.764-np.log10(m_int2[wlo]))))/0.1116)
    #Define mass-dependent lifetime for AGB stars, Eq. 12                          
    t_agb = 10**((0.334-np.sqrt(1.790-0.2232*(7.764-np.log10(M_AGB))))/0.1116)                                    
    #set atomic indices and check that they are found properly
    #Set the index for each tracked element that has contributions
    #to its yield from SNe II. Will fail if element is not contained in SN_yield.
    h_sn_index = np.where(SN_yield['atomic'] == 1)[0]
    he_sn_index = np.where(SN_yield['atomic'] == 2)[0]
    #c_sn_index = np.where(SN_yield['atomic'] == 6)[0]
    #o_sn_index = np.where(SN_yield['atomic'] == 8)[0]
    mg_sn_index = np.where(SN_yield['atomic'] == 12)[0]
    si_sn_index = np.where(SN_yield['atomic'] == 14)[0]
    ca_sn_index = np.where(SN_yield['atomic'] == 20)[0]
    ti_sn_index = np.where(SN_yield['atomic'] == 22)[0]
    fe_sn_index = np.where(SN_yield['atomic'] == 26)[0]
    ba_sn_index = np.where(SN_yield['atomic'] == 56)[0]
    ##Similarly define indices for tracked elements with contribution from AGB winds
    #h_agb_index = np.where(AGB_yield['atomic'] == 1)[0]
    #he_agb_index = np.where(AGB_yield['atomic'] == 2)[0]
    #c_agb_index = np.where(AGB_yield['atomic'] == 6)[0]
    #o_agb_index = np.where(AGB_yield['atomic'] == 8)[0]
    #mg_agb_index = np.where(AGB_yield['atomic'] == 12)[0]
    #si_agb_index = np.where(AGB_yield['atomic'] == 14)[0]
    #fe_agb_index = np.where(AGB_yield['atomic'] == 26)[0]
    ba_agb_index = np.where(AGB_yield['atomic'] == 56)[0]   
    #Multiply barium yields by the given factor
    SN_yield[ba_sn_index]['II'] = SN_yield[ba_sn_index]['II']*SNII_yield_mult
    AGB_yield[ba_agb_index]['AGB'] = AGB_yield[ba_agb_index]['AGB']*AGB_yield_mult
    SN_yield[ba_sn_index]['Ia'] = SN_yield[ba_sn_index]['Ia']*NSM_yield_mult
    ################## set temporary model variables ##################
    #Define the arrays for the ejected gas mass as a function of SN/HN/AGB
    #progenitor mass
    x_sn = np.zeros(len(M_SN))
    x_hn = np.zeros(len(M_HN))
    x_agb = np.zeros(len(M_AGB)+2)
    #Define arrays for the mass of each tracked element contributed by each source 
    M_II_arr = np.zeros(nel)
    M_agb_arr = np.zeros(nel)
    M_Ia_arr = np.zeros(nel)
    ################## initialize model itself ##################                                                                                  
    t = np.arange(n)*delta_t #time passed in model array -- age universe (Gyr)
    #gas inflow rate (M_sun  Gyr**-1), Eq. 14, Weschler et al. 2002  
    f_in = 1.e6*f_in_norm0*t*np.exp(-1.0*t/f_in_t0)   
    Ia_rate = np.zeros(n)
    II_rate = np.zeros(n)
    AGB_rate = np.zeros(n)
    de_dt = np.zeros((n,nel))
    dstar_dt = np.zeros((n,nel))
    abund = np.zeros((n,nel))
    eps = np.zeros((n,nel))
    mout = np.zeros((n,nel))
    z = np.zeros(n)
    mdot = np.zeros(n)
    mgal = np.zeros(n)
    mstar = np.zeros(n)
    mgas = np.zeros(n)    
    ################## set t=0 values ##################                                                                                  
    #initial star formation rate (M_sun Gyr**-1)
    #mdot[0] = 0.0                                                                     
    #initial gas mass (M_sun) for no gas infall
    if npars == 4: mgas[0] = 1.e6*pars[3] 
    #initial gas mass (M_sun) for gas infall                                                                 
    if npars == 6: mgas[0] = 1.e6*pars[5]
    #Otherwise assume no initial gas mass (M_sun)                                                                  
    else: mgas[0] = 0.                                                                       
    #Set the abundances for H and He, where abundance is the gas mass in the element                                                                                
    abund[0,0] = mgas[0]*pristine[0] 
    abund[0,1] = mgas[0]*pristine[1]
    #Set the initial galaxy mass to the initial gas mass
    mgal[0] = mgas[0]

    ########## NOW STEP THROUGH TIME ##########
    
    j = 0
    #While the age of the universe at a given timestep is less than 
    #the age of the universe at z = 0, AND more than 10 Myr has not passed
    #OR gas remains within the galaxy at a given time step 
    #AND the gas mass in iron at the previous timestep is subsolar
    while ((j < (n - 1)) and ((j <= 10) or ( (mgas[j] > 0.0) 
    and (eps[j-1,fe_sn_index] < 0.0) ) )):
    
        #abund=[H, He, C, O, Mg, Si, Ca, Ti, Fe, Ba]
        
        #If we are at the initial timestep in the model (t = 0)
        if j == 0:
            #gas mass (M_sun), determined from individual element gas masses Eq. 2
            mgas[j] = abund[j-1,h_sn_index] + abund[j-1,he_sn_index] + \
                (abund[j-1,mg_sn_index] + abund[j-1,si_sn_index] + \
                 abund[j-1,ca_sn_index] + abund[j-1,ti_sn_index])*10.0**(1.31) + \
                 abund[j-1,fe_sn_index]*10.0**(0.03)
        
        #SFR for power law model (M_sun Gyr**-1), Eq. 5                                                                    
        mdot[j] = sfr_norm*mgas[j]**sfr_exp / 1.e6**(sfr_exp-1.0)

        #Define the gas phase absolute metal mass fraction (dimensionless), Eq. 3
        #If, at the previous timestep, the gas mass was nonzero
        if mgas[j-1] > 0.0:
        
            #Subtract off contributions from H and He to the total gas mass
            z[j] = (mgas[j] - abund[j-1,h_sn_index] - \
            abund[j-1,he_sn_index])/mgas[j]
            
        #Otherwise, if the gas is depleted, the gas mass is zero
        else:
            z[j] =  0.0
            
        #SN Ia delay time distribution (SNe Gyr**-1 (M_sun)**-1)
        #returns Phi_Ia, or the SNe Ia rate, for each t (t_delay in paper), Eq 9  
        dtd = dtd_ia.dtd_ia(t[0:j+1], ia_model)[::-1]
        
        #If some time has passed in the universe, such that SNe Ia might go off                                                                    
        if j > 1:
            
            #Integrate to determine the instantaneous SN Ia rate (SN Gyr**-1), Eq. 10 
            Ia_rate[j] = scipy.integrate.simps(mdot[0:j+1]*dtd,dx=delta_t)
            
        #Otherwise, not enough time passed for SN Ia
        else: 
            Ia_rate[j] = 0.0             

        #If the lifetime of massive stars at a given timestep is less than the current
        #age of the universe in the model, then determine which stars will explode
        wexplode = np.where(t_int1 < t[j])[0]                                                      
        #If there are massive stars to explode
        if len(wexplode) > 0:
        
            #SFR interpolated onto t_int1 grid (M_sun Gyr**-1)
            mdot_int1 = interp_func(t[j]-t[0:j+1], mdot[0:j+1],
            t_int1[wexplode])
            #instantaneous SN II rate (SN Gyr**-1), Eq. 8                  
            R_II = scipy.integrate.simps(mdot_int1*imf.imf(m_int1[wexplode], imf_model),
            m_int1[wexplode])
            #imf_dat = imf.imf(m_int1[wexplode], imf_model)
            #R_II = idl_tabulate(m_int1[wexplode], mdot_int1*imf_dat)            
        #Otherwise the SNe II rate is zero               
        else: 
            R_II = 0.0
        II_rate[j] = R_II
        #where the time array includes AGB stars for M_init <= 8 M_sun
        wagb = np.where(t_int2 < t[j])[0]                                                               
        if len(wagb) > 0:
            #SFR interpolated onto t_int2 grid (M_sun Gyr**-1)
            mdot_int2 = interp_func(t[j]-t[0:j+1], mdot[0:j+1],
            t_int2[wagb])
            #instantaneous AGB rate (AGB Gyr**-1)                 
            R_AGB = scipy.integrate.simps(mdot_int2*imf.imf(m_int2[wagb], imf_model),
            m_int2[wagb])
        #Otherwise the AGB rate is zero               
        else: R_AGB = 0.0
        AGB_rate[j] = R_AGB
        ########## NOW STEP THROUGH ELEMENTS ##########
        for i in list(range(nel)):
            ##### SN II #####
            if len(wexplode) > 0:
                #gas phase metal fraction at time the supernova progenitor was born
                z_sn = interp_func(t[j]-t[0:j+1], z[0:j+1],
                1.2*M_SN**(-1.85)+0.003)
                #ejected mass as a function of SN mass (M_sun)     
                for m in list(range(len(M_SN))): 
                    x_sn[m] = interp_func(z_II, SN_yield[i]['II'][:,m],
                    min(max(z_sn[m], min(z_II)), max(z_II)))
                #gas phase metal fraction at time the hypernova progenitor was born            
                z_hn = interp_func(t[j]-t[0:j+1], z[0:j+1],
                1.2*M_HN**(-1.85)+0.003)
                #ejected mass as a function of HN mass (M_sun)                 
                for m in list(range(len(M_HN))): 
                    x_hn[m] =  interp_func(z_II, SN_yield[i]['HN'][:,m],
                    min(max(z_hn[m],min(z_II)),max(z_II)))
                #create an array of the ejected mass for SN and HN (M_sun) to match M_SN_ej. 
                #Extrapolate what the ejecta would be at the minimum and maximum SN mass
                #Duplicate the ejecta at the minimum hypernova mass, because we added M_HN_min-0.01 in M_SN_ej.
                x_ej = np.concatenate(([x_sn[0]*M_SN_ej[0]/M_SN_ej[1]],
                            x_sn[:index_M_HN_min_M_SN],
                            [x_sn[index_M_HN_min_M_SN]],
                            x_sn[index_M_HN_min_M_SN:],
                            [x_sn[-1]*M_SN_ej[-1]/M_SN_ej[-2]]))
                if epsilon_HN > 0:                                              #if we are including hypernovae
                    #decreased SN ejecta by HN fraction (M_sun)      
                    x_ej[index_M_HN_min_M_SN_ej:] = x_ej[index_M_HN_min_M_SN_ej:]*(1.0 - epsilon_HN)
                    #augment HN ejecta by HN fraction (M_sun)       
                    x_ej[index_M_HN_min_M_SN_ej:] = x_ej[index_M_HN_min_M_SN_ej:] + \
                        epsilon_HN*(np.concatenate((x_hn, 
                                                [x_hn[index_M_HN_min_M_SN]*
                                                 M_SN_ej[-1]/M_HN[index_M_HN_min_M_SN]])))
                #interpolate ejected mass onto dummy integration mass variable (M_sun)                                        
                x_ej = interp_func(M_SN_ej,x_ej, m_int1[wexplode])
                #If assuming a certain SFR law and a certain abundance for metallicity
                #dependent models                                                
                if ((name == 'sfr-law_perets1') and (z[i] < 0.019*10.0**(-2.0))): 
                    x_ej = 0.996*x_ej + 0.004*perets[i].dotIa_1*m_int1[wexplode]
                if ((name == 'sfr-law_perets2') and (z[i] < 0.019*10.0**(-2.0))): 
                    x_ej = 0.996*x_ej + 0.004*perets[i].dotIa_2*m_int1[wexplode]
                #mass ejected from SN II at this time step (M_sun Gyr**-1), Eq. 7
                M_II_arr[i] = scipy.integrate.simps(x_ej*mdot_int1*imf.imf(m_int1[wexplode],imf_model),
                m_int1[wexplode]) 
            #Otherwise, if there are no contributions from SNe II at this timestep
            else: 
                M_II_arr[i] = 0.0
            ##### AGB #####
            #If there are contributions from AGB stars at this timestep
            if len(wagb) > 0:
                #metallicity at the time low- and intermediate-mass stars in AGB phase 
                #were born                                                                              
                z_agb = interp_func(t[j]-t[0:j+1], z[0:j+1],t_agb)
                #Based on the element currently considered, determine if
                #there are contributions to the stars from AGB yields   
                #GD: This used to be fixed specifically to elements tracket in Karakas 2010.                               
                for m in list(range(AGB_yield[i]['AGB'].shape[1])):
                    #ejected mass as a function of LIMS mass (M_sun)
                    x_agb[m+1] = interp_func(z_AGB, AGB_yield[i]['AGB'][:,m],
                    min(max(z_agb[m],min(z_AGB)),max(z_AGB)))
                #Linearly extrapolate the model yields to the maximum and minimum AGB mass considered
                x_agb[0] = x_agb[1]*M_AGB_ej[0]/M_AGB_ej[1]
                x_agb[len(x_agb)-1] = x_agb[len(x_agb)-2]*M_AGB_ej[len(M_AGB_ej)-1]/M_AGB_ej[len(M_AGB_ej)-2]
                #interpolate ejected mass onto dummy integration mass variable (M_sun)
                x_agb_ej = interp_func(M_AGB_ej, x_agb, m_int2[wagb])
                #mass ejected from AGB stars at this time step (M_sun Gyr**-1), Eq. 13
                M_agb_arr[i] = scipy.integrate.simps(x_agb_ej*mdot_int2*imf.imf(m_int2[wagb],imf_model),
                m_int2[wagb])                       
            #Otherwise, there is no contribution from AGB stars  
            else: 
                M_agb_arr[i] = 0.0
            ##### SN Ia #####
            f_Ia = SN_yield[i]['Ia'] #mass ejected from each SN Ia (M_sun SN**-1)                                                                      
            M_Ia_arr[i] = f_Ia*Ia_rate[j] #mass returned to the ISM, Eq. 11                                                                

        ########## NOW STEP THROUGH ELEMENTS AGAIN ##########
        for i in list(range(nel)):
        
            #If, at a given timestep, there is gas within the galaxy                                                              
            if mgas[j] > 0.0: 
                #Then the gas mass fraction for a given element is calculated based on the
                #previous timestep
                x_el = abund[j-1,i]/mgas[j]
            #Otherwise, if there is no gas, then the gas mass fraction is zero
            else: x_el = 0.0                               
            #mass loading formulation of supernova ejecta 
            #(SN ejecta is not instantaneously mixed)
            ##### CONSIDER OTHER GCE MODELS #####
            if name == 'massloading':                                                                 
                chi = pars[2]  #fraction of SN ejected mass that escapes the galaxy 
                #amount of ISM entrained in SN wind relative to escaping SN ejecta
                if npars == 6: eta = pars[5] 
                else: eta = pars[4]
                #SN ejecta that escapes the galaxy (M_sun)                                                                   
                sn_ejecta = chi*(M_II_arr[i]+M_Ia_arr[i])
                #total mass in Type II SN ejecta
                M_II_tot = np.sum(M_II_arr[0:1]) + np.sum(M_II_arr[3:6])*10.0**(1.31) + \
                M_II_arr[7]*10.0**(0.03)
                #total mass in Type Ia SN ejecta 
                M_Ia_tot = np.sum(M_Ia_arr[0:1]) + np.sum(M_Ia_arr[3:6])*10.0**(1.31) + \
                M_Ia_arr[7]*10.0**(0.03)
                #total mass in Type II SN ejecta  
                M_SN_tot = M_II_tot + M_Ia_tot   
                #ISM entrained in SN wind (M_sun) 
                entrained_ism = eta*x_el*M_SN_tot
                #gas outflow, equal to SN ejecta and entrained ISM (M_sun)                                                              
                mout[j,i] = sn_ejecta + entrained_ism   
            #Define the metal enhancement of the SN winds (dimensionless)
            else:                                                         
                if name == 'Zwind':
                    if len(pars) == 7: z_enhance = pars[6]
                    else: z_enhance = pars[5]
                elif name == 'sfr-law_Zwind': 
                    z_enhance = 0.01
                else: 
                    z_enhance = 0.0   
                #Now determine the gas outflow rate for the given element
                #Include some metallicity dependence based on which element is
                #being considered
                
                if i <= 1:                                                      # for hydrogen and helium
                    #gas outflow, proportional to Types II and Ia SNe rate (M_sun Gyr**-1)
                    #Eq. 15 
                    mout[j,i] = f_out*(1.0-z_enhance)*x_el*(II_rate[j] + \
                    Ia_rate[j])                   
                elif i > 1 and z[j] > 0: 
                    mout[j,i] = f_out*(z_enhance*((z[j])**(-1.0)-1.0)+1.0)*x_el*(II_rate[j] + \
                    Ia_rate[j])
                else: 
                    mout[j,i] = 0.0
            #Now determine the star formation rate for a given element
            #(The rate at which the element is locked up in stars)
            #At the given time step gas locked into stars (M_sun Gyr**-1) minus
            #gas returned from Type II SNe, AGB stars, and Type Ia SNe (M_sun Gyr**-1)                                                                                                    
            dstar_dt[j,i] = (x_el)*mdot[j] - M_II_arr[i] - \
            M_agb_arr[i] - M_Ia_arr[i] 
            #change in gas phase abundance, owing to star formation (M_sun Gyr**-1) 
            #minus gas outflow from SN winds (M_sun Gyr**-1) 
            #plus PRISTINE gas infall, constant rate (M_sun Gyr**-1)                                                           
            de_dt[j,i] = -1.0*dstar_dt[j,i] - mout[j,i] + \
            f_in[j]*pristine[i] 
            #Calculate the gas phase mass fraction (M_sun) according to timestep
            #If it is not the first timestep  
            #I.E. This was very wrong beforehand                                                         
            if j > 0:
                if j < 4:
                    if j-1 == 0: abund[j,i] = abund[j-1,i] + np.sum(int2*de_dt[j-1:j+1,i])
                    elif j-1 == 1: abund[j,i] = abund[j-2,i] + np.sum(int3*de_dt[j-2:j+1,i])
                    elif j-1 == 2: abund[j,i] = abund[j-3,i] + np.sum(int4*de_dt[j-3:j+1,i])
                else: abund[j,i] = abund[j-4,i] + np.sum(int5*de_dt[j-4:j+1,i])
                    
            if abund[j,i] <= 0.0: 
                abund[j,i] = 0
                eps[j,i] = np.nan
            else:
                #gas phase abundance (number of atoms in units of M_sun/amu = 1.20d57)                
                eps[j,i] = np.log10(abund[j,i]/interp_func(z_II,
                SN_yield[i]['weight_II'][:,3], z[j]))
        ##### NOW BACK INTO LARGER TIMESTEP FOR LOOP #####
        #gas phase abundance (logarithmic number density relative to hydrogren relative to sun)
        eps[j] = 12.0 + eps[j] - eps[j,0] - eps_sun
        #Calculate the stellar mass of the galaxy at a given timestep
        #If more than 1 Myr has passed
        if j > 0:
            if j < 4:
                if j-1 == 0: mstar[j] = mstar[j-1] + np.sum(int2*np.sum(dstar_dt[j-1:j+1], 1))
                elif j-1 == 1: mstar[j] = mstar[j-2] + np.sum(int3*np.sum(dstar_dt[j-2:j+1], 1))
                elif j-1 == 2: mstar[j] = mstar[j-3] + np.sum(int4*np.sum(dstar_dt[j-3:j+1], 1))
            else: mstar[j] = mstar[j-4] + np.sum(int5*np.sum(dstar_dt[j-4:j+1], 1))
        #If it is the first timestep, there is no stellar mass
        else: mstar[j] = 0.0
        #total galaxy mass (M_sun) at this timestep
        mgal[j] = mgas[j] + mstar[j]                                                            
        #Increment in time
        j = j + 1
        #Update the gas mass for the following timestep
        if j < n:
            mgas[j] = abund[j-1,h_sn_index] + abund[j-1,he_sn_index] + \
            (abund[j-1,mg_sn_index] + abund[j-1,si_sn_index] + \
            abund[j-1,ca_sn_index]+abund[j-1,ti_sn_index])*10.0**(1.31) + \
            abund[j-1,fe_sn_index]*10.0**(0.03) 
            #If somehow the galaxy has negative gas mass, it actually has zero gas mass   
            if mgas[j] < 0.: mgas[j] = 0.0
    ##### OUTSIDE OF ALL FOR LOOPS #####
    #Define all of the data to be saved for the model for each timestep
    model = np.zeros(n, dtype=[('t','float64'),('f_in','float64'),('Ia_rate','float64'),\
    ('II_rate','float64'),('AGB_rate','float64'),('de_dt','float64',(nel)),('dstar_dt','float64',(nel)),\
    ('abund','float64',(nel)),('eps','float64',(nel)),('mout','float64',(nel)),\
    ('z','float64'),('mdot','float64'),('mgal','float64'),('mstar','float64'),\
    ('mgas','float64')])
    #Save arrays in model
    model['t']=t
    model['f_in'] = f_in
    model['Ia_rate'] = Ia_rate
    model['II_rate'] = II_rate
    model['AGB_rate'] = AGB_rate
    model['de_dt'] = de_dt
    model['dstar_dt'] = dstar_dt
    model['abund'] = abund
    model['eps'] = eps
    model['mout'] = mout
    model['z'] = z
    model['mdot'] = mdot 
    model['mgal'] = mgal 
    model['mstar'] = mstar 
    model['mgas'] = mgas
    #Now return only the model data up to the point at which the galaxy ran out of gas
    model = model[0:j]
    return model, SN_yield['atomic']