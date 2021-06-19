"""
This program sets the parameters to select a certain GCE model
(closed box, leaky box, accretion, etc.) and combination of
Type Ia SNe DTD and IMF
"""
  
# Closed box model
name = 'sfr-law'

# Ia delay-time distribution: 'maoz10', 'maoz17', 'highmindelay', 'lowmindelay'
ia_model = 'lowmindelay'

# Initial mass function: 'kroupa93' ('kroupa01', 'chabrier03', 'salpeter55' - not yet implemented)
imf_model = 'kroupa93'

# AGB yield source: 'cri15' ('kar' - note that Ba abundances are probably wrong!)
AGB_source = 'cri15'

# SNIa yield source: 
# 'iwa99' (old), 'leu20' (subMCh with He shell), 'shen18' (subMCh with bare WD), 'leu18_ddt' (MCh DDT), 'leu18_def' (MCh pure def)
Ia_source = 'leu20'

# SNII yield source: 'nom06', 'nom13', 'lim18'
II_source = 'nom13'

# r-process options: 'none', 'typical_SN_only', 'rare_event_only', 'both'
r_process_keyword='none' #'rare_event_only'

# Minimum and maximum AGB masses
M_AGB_min = 0.865   # lower mass limit for AGB stars (M_sun)
M_AGB_max = 10.  # upper mass limit for AGB stars (M_sun)

# Minimum and maximum SN II masses
M_SN_min = 10.   # lower mass limit for Type II SN to explode (M_sun)
M_SN_max = 100.  # upper mass limit for Type II SN to explode (M_sun)
