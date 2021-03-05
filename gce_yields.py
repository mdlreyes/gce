"""
gce_yields.py

This program is based on Gina Duggan's code to read in yields for a GCE model.
"""

# Import useful packages
import numpy as np
from astropy.io import ascii
import re

def load_II(II_source, yield_path, SN_yield, atomic_names, z_II, M_SN):
    """Reads in II yield files."""

    if II_source == 'nom06':
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

    return SN_yield

def load_Ia(Ia_source, yield_path, SN_yield, atomic_names):
    """Reads in Ia yield files."""

    if Ia_source == 'iwa99':
        # Read in He yields from Nomoto+06
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

    return SN_yield

def load_AGB(AGB_source, yield_path, atomic_num, atomic_names):
    """Reads in AGB yield files."""

    if AGB_source == 'cri15':
        # Masses and metallicities from FRUITY files
        M_cri = np.array([1.3,1.5,2.0,2.5,3.0,4.0,5.0,6.0])
        z_cri = np.array([0.001,0.002,0.003,0.006,0.008,0.01,0.014,0.02])

        # Define the output array of yields
        cri15 = np.zeros(len(atomic_num),dtype=[('atomic','float64'),
                        ('AGB','float64',(len(z_cri),len(M_cri))),
                        ('weight','float64',(len(z_cri),len(M_cri)))])
        cri15['atomic'] = atomic_num
        
        # Convert atomic names to lowercase to match FRUITY files
        sub_string = [re.sub('[A-Z]+', lambda m: m.group(0).lower(), name) for name in atomic_names]

        # Loop over each element file
        cri15files = [yield_path + 'FRUITY/'+ sub_string[i]+'.txt' for i in range(len(sub_string))]
        for elem_idx, filename in enumerate(cri15files): 

            # Open file
            data = ascii.read(filename)

            # Get isotopes (for everything except hydrogen)
            isotope_names = data.colnames[3:]
            if elem_idx == 0:
                isotope = [1]
            else:
                isotope = [int(re.findall(r'\d+', name)[0]) for name in isotope_names]
                if atomic_num[elem_idx] == 56: isotope = np.array(isotope) + 100

            # Compute yields and weights
            if len(isotope) == 1:    
                agb_yield = data[isotope_names[0]]
                weight = np.zeros(len(agb_yield)) + isotope
            else:    
                iso_array = np.array(data[isotope_names])
                iso_array = iso_array.view(np.float).reshape(iso_array.shape + (-1,))  # Convert structured array to ndarray
                agb_yield = np.sum(iso_array,axis=1)                                      
                weight = np.dot(iso_array,isotope)/agb_yield  # Average isotopic weight
            
            # Add to yield table
            for idx in range(len(agb_yield)):
                m_idx = np.where(M_cri == data['Mass'][idx])[0]
                z_idx = np.where(z_cri == data['Metallicity'][idx])[0]
                if len(m_idx) == 1 and len(z_idx) == 1:
                    cri15[elem_idx]['AGB'][z_idx[0], m_idx[0]] = agb_yield[idx]
                    cri15[elem_idx]['weight'][z_idx[0], m_idx[0]] = weight[idx]

        return M_cri, z_cri, cri15

    # TODO: Add Karakas+16 yields as an option!
    if AGB_source == 'kar16':
        pass
              
def initialize_yields(yield_path='yields/', r_process_keyword='none', AGB_source='cri15', Ia_source='iwa99', II_source='nom06'):
    """Reads in yield tables.

    Args:
        yield_path (str): Path to folder with yields.
        r_process_keyword (str): How to handle r-process elements: 'none', 'typical_SN_only', 'rare_event_only', 'both'
        AGB_source (str): Source of AGB yields: 'cri15', 'kar16'
        Ia_source (str): Source of Ia yields: 'iwa99', 'leu20'

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
    elem_atomic = [1, 2, 6, 12, 14, 20, 22, 25, 26, 56] #8, 
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

    # Read in SN yield files
    SN_yield = load_II(II_source, yield_path, SN_yield, atomic_names, z_II, M_SN)
    SN_yield = load_Ia(Ia_source, yield_path, SN_yield, atomic_names)                           
    
    # Divide weighted mass by total yield mass to get average isotope mass
    SN_yield['weight_II'][SN_yield['II']>0] /= SN_yield['II'][SN_yield['II']>0]
    SN_yield['weight_Ia'][SN_yield['Ia']>0] /= SN_yield['Ia'][SN_yield['Ia']>0]

    # Read in AGB yield files
    M_AGB, z_AGB, AGB_yield = load_AGB(AGB_source, yield_path, atomic_num, atomic_names)

    # Add in barium abundances from SNe/rare r-process events
    ba_idx = np.where((np.asarray(elem_atomic) == 56))[0][0]

    if r_process_keyword in ['typical_SN_only','both']:

        # Barium yield for weak r-process event as a function of mass (Li+2014)
        li14_weakr = [1.38e-8, 2.83e-8, 5.38e-8, 6.84e-8, 9.42e-8, 0, 0]

        # Linearly interpolate to high-mass end
        li14_weakr[-2] = li14_weakr[-3]*M_SN[-2]/M_SN[-3]
        li14_weakr[-1] = li14_weakr[-2]*M_SN[-1]/M_SN[-2]

        # Add to SNII yield arrays
        for i in range(len(z_II)):
            SN_yield['II'][ba_idx][i,:] = li14_weakr  # Assume no Z-dependence
        SN_yield['weight_II'][ba_idx] = np.mean(AGB_yield['weight'][9])
            
    if r_process_keyword in ['rare_event_only','both']:   

        # Note: this is the average barium yield for a main r-process event (Li+2014)
        ba_yield = 2e-6

        # Since NSM have a similar DTD as SN Ia, put yields in SNIa array as a proxy. 
        # (Scale barium yield up and reduce rate of events proportionally to compare to models.)
        SN_yield['Ia'][ba_idx] = ba_yield
        SN_yield['weight_Ia'][ba_idx] = np.mean(AGB_yield['weight'][9])

    return nel, eps_sun, SN_yield, AGB_yield, M_SN, z_II, M_AGB, z_AGB

if __name__ == "__main__":

    initialize_yields()