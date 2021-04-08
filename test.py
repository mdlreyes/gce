#Reduced version of "yield_analysis_lc18.py"
#Note that the parts havent been edited to run but 
#to show the main parts of the pri
​
##########FUNCTIONS##########
def get_yields_lc18(isotope_X_lc18, yields_table_lc18, velocity_val_lc18, metallicity_val_lc18, model_massess_lc18):
    isotopeX_yield_mass_for_modeled_masses_lc18 = np.zeros(len(model_massess_lc18))
    for i in range(len(model_massess_lc18)):
        abundance_ind_lc18 = np.where((yields_table_lc18['Isotope'] == isotope_X_lc18) & (yields_table_lc18['Vel'] == velocity_val_lc18) & (yields_table_lc18['[Fe/H]'] == metallicity_val_lc18))
        mass_model_lc18 = model_massess_lc18[i]+'M' 
        isotopeX_yield_lc18 = yields_table_lc18[abundance_ind_lc18][mass_model_lc18]
        isotopeX_yield_mass_for_modeled_masses_lc18[i] = isotopeX_yield_lc18
    return isotopeX_yield_mass_for_modeled_masses_lc18
​
​
##########DEFINING PATHS##########
#Limongi  & Chieffi 2018
yield_file_lc18_sn = '/Users/evanhazenunez/cit_research/data/ccsn_yields/lc18_sn.txt'
yield_file_lc18_winds = '/Users/evanhazenunez/cit_research/data/ccsn_yields/lc18_winds.txt'
yield_table_lc18_sn = Table.read(yield_file_lc18_sn, format="ascii.cds")
yield_table_lc18_winds = Table.read(yield_file_lc18_winds, format="ascii.cds")
​
##########CONSTANTS##########
#####Limongi and Chieffi 2018
metallicity_lc18 = -3 #[Fe/H] Also 0,-1,-2 are available
vel_lc18 = np.array([0, 150, 300])
​
​
​
##########MAIN PROGRAM##########
#List of radioactive isotopes that decay to stable isotopes of each of our element of interest
#Doing this by hand for now
Fe_rad_iso = np.array(['Mn56', 'Co56', 'Ni56', 'Mn57', 'Co57', 'Ni57', 'Cu57', 'Co58', 'Cu58' ])
C_rad_iso = np.array(['N13'])
N_rad_iso = np.array(['C14', 'O15'])
O_rad_iso = np.array(['N16', 'F17', 'F18'])
Al_rad_iso = np.array(['Mg27', 'Si27'])
Si_rad_iso = np.array(['Al28', 'P29', 'P30'])
S_rad_iso = np.array(['Si32', 'P32', 'P33', 'Cl33', 'P34', 'Cl34', 'Cl36'])
​
​
#List of stable isotopes to be extracted from the yield table
Fe_st_iso = np.array(['Fe54', 'Fe56', 'Fe57', 'Fe58'])
C_st_iso = np.array(['C12', 'C13'])
N_st_iso = np.array(['N14', 'N15'])
O_st_iso = np.array(['O16', 'O17', 'O18'])
Al_st_iso = np.array(['Al27'])
Si_st_iso = np.array(['Si28', 'Si29', 'Si30'])
S_st_iso = np.array(['S32', 'S33', 'S34', 'S36'])
​
#Full list of isotopes for our elements of interest (basically accounting for radioactive decay)
Fe_isotopes_lc18 = np.append(Fe_st_iso, Fe_rad_iso)
C_isotopes_lc18 = np.append(C_st_iso, C_rad_iso)
N_isotopes_lc18 = np.append(N_st_iso, N_rad_iso)
O_isotopes_lc18 = np.append(O_st_iso, O_rad_iso)
Al_isotopes_lc18 = np.append(Al_st_iso, Al_rad_iso)
Si_isotopes_lc18 = np.append(Si_st_iso, Si_rad_iso)
S_isotopes_lc18 = np.append(S_st_iso, S_rad_iso)
​
#Declaring the empty arrays to store [X/Fe] for 
Fe_lc18_sn, Fe_lc18_snwinds = np.zeros(len(vel_lc18)), np.zeros(len(vel_lc18))
CFe_lc18_sn, CFe_lc18_snwinds = np.zeros(len(vel_lc18)), np.zeros(len(vel_lc18))
NFe_lc18_sn, NFe_lc18_snwinds = np.zeros(len(vel_lc18)), np.zeros(len(vel_lc18))
OFe_lc18_sn, OFe_lc18_snwinds = np.zeros(len(vel_lc18)), np.zeros(len(vel_lc18))
AlFe_lc18_sn, AlFe_lc18_snwinds = np.zeros(len(vel_lc18)), np.zeros(len(vel_lc18))
SiFe_lc18_sn, SiFe_lc18_snwinds = np.zeros(len(vel_lc18)), np.zeros(len(vel_lc18))
SFe_lc18_sn, SFe_lc18_snwinds = np.zeros(len(vel_lc18)), np.zeros(len(vel_lc18))
​
#Calculating XFe for all values of Velocity
for i, vel in enumerate(vel_lc18):
    Fe_lc18_sn[i], Fe_lc18_snwinds[i] = Fe_main_lc18(Fe_isotopes_lc18, yield_table_lc18_sn, yield_table_lc18_winds, vel, metallicity_lc18)
    CFe_lc18_sn[i], CFe_lc18_snwinds[i] = XFe_main_lc18(C_isotopes_lc18, 'C', yield_table_lc18_sn, yield_table_lc18_winds, vel, metallicity_lc18, Fe_lc18_sn[i], Fe_lc18_snwinds[i], solar_abundances)
    NFe_lc18_sn[i], NFe_lc18_snwinds[i] = XFe_main_lc18(N_isotopes_lc18, 'N', yield_table_lc18_sn, yield_table_lc18_winds, vel, metallicity_lc18, Fe_lc18_sn[i], Fe_lc18_snwinds[i], solar_abundances)
    OFe_lc18_sn[i], OFe_lc18_snwinds[i] = XFe_main_lc18(O_isotopes_lc18, 'O', yield_table_lc18_sn, yield_table_lc18_winds, vel, metallicity_lc18, Fe_lc18_sn[i], Fe_lc18_snwinds[i], solar_abundances)
    AlFe_lc18_sn[i], AlFe_lc18_snwinds[i] = XFe_main_lc18(Al_isotopes_lc18, 'Al', yield_table_lc18_sn, yield_table_lc18_winds, vel, metallicity_lc18, Fe_lc18_sn[i], Fe_lc18_snwinds[i], solar_abundances)
    SiFe_lc18_sn[i], SiFe_lc18_snwinds[i] = XFe_main_lc18(Si_isotopes_lc18, 'Si', yield_table_lc18_sn, yield_table_lc18_winds, vel, metallicity_lc18, Fe_lc18_sn[i], Fe_lc18_snwinds[i], solar_abundances)
    SFe_lc18_sn[i], SFe_lc18_snwinds[i] = XFe_main_lc18(S_isotopes_lc18, 'S', yield_table_lc18_sn, yield_table_lc18_winds, vel, metallicity_lc18, Fe_lc18_sn[i], Fe_lc18_snwinds[i], solar_abundances)
​
#Storing the vaiables in as pickles
XFe_lc18 = {'CFe_lc18_sn':CFe_lc18_sn,
            'CFe_lc18_snwinds:':CFe_lc18_snwinds,
            'NFe_lc18_sn':NFe_lc18_sn,
            'NFe_lc18_snwinds':NFe_lc18_snwinds,
            'OFe_lc18_sn':OFe_lc18_sn,
            'OFe_lc18_snwinds':OFe_lc18_snwinds,
            'AlFe_lc18_sn':AlFe_lc18_sn,
            'AlFe_lc18_snwinds':AlFe_lc18_snwinds,
            'SiFe_lc18_sn':SiFe_lc18_sn,
            'SiFe_lc18_snwinds':SiFe_lc18_snwinds,
            'SFe_lc18_sn':SFe_lc18_sn,
            'SFe_lc18_snwinds':SFe_lc18_snwinds}
pickle.dump(XFe_lc18, open('XFe_lc18_v'+version_number+'_'+m_lo+'_'+m_hi+'.p','wb'))
​