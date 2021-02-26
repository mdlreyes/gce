"""
This program sets the parameters to select a certain GCE model
(closed box, leaky box, accretion, etc.) and combination of
Type Ia SNe DTD and IMF
"""
  
name = 'sfr-law'   #closed box model
#name: 'sfr-law', ...
ia_model = 'maoz10'
#ia_model = 'mannucci06', 'maoz10'
imf_model = 'kroupa93'
#imf_model: 'kroupa01', 'kroupa93', 'chabrier03', 'salpeter55'
r_process_keyword='none'
# r_process_keyword: 'none', 'typical_SN_only', 'rare_event_only', 'both'
AGB_source = 'cri15'#'kar16'
# AGB_source: 'cri15', 'kar16'

##### multiplier of barium ejecta of each source
AGB_yield_mult = 1
SNII_yield_mult = 1
NSM_yield_mult = 1

#For delta_t = 0.001, the model runs for 13.6 Gyr    
#max number of time steps
n = 13600.
#duration of a time step in Gyrs
delta_t = 0.001
#Minimum and maximum AGB masses
M_AGB_min = 0.865   #lower mass limit for AGB stars (M_sun)
M_AGB_max = 10.  #upper mass limit for AGB stars (M_sun)
#Minimum and maximum SN II masses
M_SN_min = 10.   #lower mass limit for Type II SN to explode (M_sun)
M_SN_max = 100.  #upper mass limit for Type II SN to explode (M_sun)
#hypernovae fraction for stars with M >= 20 M_sun (dimensionless)
epsilon_HN = 0.3
#Minimum mass of a hypernovae, i.e., turn-on mass for hypernovae
M_HN_min = 20.