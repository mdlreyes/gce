"""
gce_yields.py

This program is based on Gina Duggan's code to read in yields for a GCE model.
"""

# Import useful packages
import numpy as np
from astropy.io import ascii
import re
              
def initialize_yields(yield_path='yields/', r_process_keyword='none', 
    AGB_source='cri15', verbose=False):

    """Reads in yield tables.

    Args:
        yield_path (str): Path to folder with yields.
        r_process_keyword (str): How to handle r-process elements: 'none', 'typical_SN_only', 'rare_event_only', 'both'
        AGB_source (str): Source of AGB yields: 'cri15', 'kar16'
        verbose (bool): Keyword to print additional output.

    Returns:
        nel (int): Number of elements.
        eps_sun (array): Solar abundances in epsilon notation. 
        SN_yield (array): 
        AGB_yield (array): 
        M_SN (array): Supernova masses.
        z_II (array): Supernova metallicities. 
        M_AGB (array): AGB masses.
        z_AGB (array): AGB metallicities.
    """

    # Atomic data
    eps_sun = np.array([12.00, 10.99, 3.31, 1.42, 2.88, 8.56, 8.05, 8.93, 4.56,
                        8.09, 6.33, 7.58, 6.47, 7.55, 5.45, 7.21, 5.5, 6.56,
                        5.12, 6.36, 3.10, 4.99, 4.00, 5.67, 5.39, 7.52, 4.92,
                        6.25, 4.21, 4.60, 2.88, 3.41, 2.90, 2.13, 1.22, 0.51])
    atomic_weight = np.array([1.00794, 4.00602, 6.941, 9.012182, 10.811, 12.0107,
                            14.0067, 15.9994, 18.9984032, 20.1797, 22.98976928,
                            24.3050, 26.9815386, 28.0355, 30.973762, 32.065,
                            35.453, 39.948, 39.0983, 40.078, 44.955912, 47.957,
                            50.9415, 51.9951, 54.9438045, 55.845, 58.933195,
                            58.6934, 63.546, 65.38, 69.723, 72.64, 87.62,
                            137.522, 138.9055, 151.964])
    atomic_num = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
                        20,21,22,23,24,25,26,27,28,29,30,31,32,38,56,57,63])
    atomic_names = np.array(['H','He','Li','Be','B','C','N','O','F','Ne','Na',
                        'Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti',
                        'V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','Sr',
                        'Ba','La','Eu'])

    # Extract info for the elements we want
    elem_atomic = [1, 2, 6, 8, 12, 14, 20, 22, 26, 56] # 25, 
    nel = len(elem_atomic)
    elem_idx = np.where(np.isin(atomic_num, elem_atomic))[0]

    eps_sun = eps_sun[elem_idx]
    atomic_num = elem_atomic
    atomic_weight = atomic_weight[elem_idx]
    atomic_names = atomic_names[elem_idx]

    # Prep array to hold SN yields
    M_SN = np.array([13.0, 15.0, 18.0, 20.0, 25.0, 30.0, 40.0])
    z_II = np.array([0.0, 0.001, 0.004, 0.02])
    SN_yield = np.zeros(nel,dtype=[('atomic','float64'),('II','float64',(len(z_II),len(M_SN))),
                                        ('weight_II','float64',(len(z_II),len(M_SN))),
                                        ('Ia','float64'),('weight_Ia','float64')])
    SN_yield['atomic'] = atomic_num

    # Read in IISN yields from Nomoto+06
    ii_table = ascii.read(yield_path+'nom06/tab1.txt')
    elem_name = np.array([ii_table['Element'][j].strip('0123456789-*^') for j in range(len(ii_table['Element']))])  # Element names
    mass_num = np.array([re.sub("\D","",ii_table['Element'][k]) for k in range(len(ii_table['Element']))])  # Mass numbers
    mass_array = ii_table['P13', 'P15', 'P18', 'P20', 'P25', 'P30', 'P40']

    for elem_idx, elem in enumerate(elem_name):
        # Loop over each line in yield table
        if elem == 'p':
            elem = 'H'
            isotope_mass = 1
        elif elem == 'd':
            elem = 'H'
            isotope_mass = 2
        elif (elem not in atomic_names): 
            continue
        else:
            isotope_mass = int(mass_num[elem_idx])

        # Get indices for storing yields in SN_yield table
        wz = np.where(z_II == ii_table['Z'][elem_idx])[0][0]
        wa = np.where(atomic_names == elem)[0][0]
        
        # Add yields to SN_yield table
        for mass_idx, mass in enumerate(M_SN):
            SN_yield[wa]['weight_II'][wz,mass_idx] += isotope_mass * mass_array['P'+str(int(mass))][elem_idx]     # Mass weighted by isotopic weight
            SN_yield[wa]['II'][wz,mass_idx] += mass_array['P'+str(int(mass))][elem_idx]

    # Read in IaSN He yields from Nomoto+06
    ia_he_table = ascii.read(yield_path+'nom06/tab3.txt')
    elem_name = np.array([ia_he_table['Element'][j].strip('0123456789-*^') for j in range(len(ia_he_table['Element']))])  
    mass_num = np.array([re.sub("\D","",ia_he_table['Element'][k]) for k in range(len(ia_he_table['Element']))])  

    for elem_idx, elem in enumerate(elem_name):
        # Loop over each line in yield table
        if elem == 'He':    
            isotope_mass = int(mass_num[elem_idx])

            # Get index for storing yields in SN_yield table
            wa = np.where(atomic_names == elem)[0][0]

            # Add yields to SN_yield table
            SN_yield[wa]['weight_Ia'] += isotope_mass * ia_he_table['SNIa'][elem_idx]  # Mass weighted by isotopic weight
            SN_yield[wa]['Ia'] += ia_he_table['SNIa'][elem_idx]                      
    
    # Read in other IaSN yields from Iwa+99
    ia_file = ascii.read(yield_path+'iwa99/tab3.txt')
    elem_name = np.array([ia_file['isotope'][j].strip('0123456789-*^') for j in range(len(ia_file['isotope']))])    
    mass_num = np.array([re.sub("\D","",ia_file['isotope'][k]) for k in range(len(ia_file['isotope']))])

    for elem_idx, elem in enumerate(elem_name): 
        # Loop over each line in yield table   
        if (elem not in atomic_names): 
            continue
        else:
            isotope_mass = int(mass_num[elem_idx])

        # Get index for storing yields in SN_yield table
        wa = np.where(atomic_names == elem)[0][0]

        # Add yields to SN_yield table
        SN_yield[wa]['weight_Ia'] += isotope_mass * ia_file['W7'][elem_idx]  # Mass weighted by isotopic weight
        SN_yield[wa]['Ia'] += ia_file['W7'][elem_idx]                           
    
    # Divide weighted mass by total yield mass to get average isotope mass
    SN_yield['weight_II'][SN_yield['II']>0] /= SN_yield['II'][SN_yield['II']>0]
    SN_yield['weight_Ia'][SN_yield['Ia']>0] /= SN_yield['Ia'][SN_yield['Ia']>0]

    # Read in AGB yield files

    # Add in barium abundances from SNe/rare r-process events

    return nel, eps_sun

if __name__ == "__main__":

    initialize_yields()