"""
This program sets the parameters to select a certain GCE model
(closed box, leaky box, accretion, etc.) and combination of
Type Ia SNe DTD and IMF
"""
  
# Closed box model
name = 'sfr-law'

# Ia delay-time distribution: 'mannucci06', 'maoz10'
ia_model = 'maoz10'

# Initial mass function: 'kroupa01', 'kroupa93', 'chabrier03', 'salpeter55'
imf_model = 'kroupa93'

# AGB yield source: 'cri15', 'kar'
AGB_source = 'cri15'

# SNIa yield source: 'iwa99', 'leu20'
Ia_source = 'leu20'

# SNII yield source: 'nom06', 'nom13', 'lim18'
II_source = 'nom13'

# r-process options: 'none', 'typical_SN_only', 'rare_event_only', 'both'
r_process_keyword='rare_event_only'

# Minimum and maximum AGB masses
M_AGB_min = 0.865   # lower mass limit for AGB stars (M_sun)
M_AGB_max = 10.  # upper mass limit for AGB stars (M_sun)

# Minimum and maximum SN II masses
M_SN_min = 10.   # lower mass limit for Type II SN to explode (M_sun)
M_SN_max = 100.  # upper mass limit for Type II SN to explode (M_sun)
