"""
gce_yields.py

This program is based on Gina Duggan's code to read in yields for a GCE model.
"""

# Import useful packages
import numpy as np
from astropy.io import ascii
import re

# Set up parameters
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
elem_atomic = [1, 2, 6, 12, 14, 20, 22, 25, 26, 56, 63] #8, 
nel = len(elem_atomic)
elem_idx = np.where(np.isin(atomic_num, elem_atomic))[0]

eps_sun = eps_sun[elem_idx]
atomic_num = elem_atomic
atomic_weight = atomic_weight[elem_idx]
atomic_names = atomic_names[elem_idx]

def load_II(II_source, yield_path, nel, atomic_names, atomic_num):
    """Reads in II yield files."""

    if II_source == 'nom06':
        # Prep array to hold SN yields
        M_SN = np.array([13.0, 15.0, 18.0, 20.0, 25.0, 30.0, 40.0])
        z_II = np.array([0.0, 0.001, 0.004, 0.02])
        SN_yield = np.zeros(nel,dtype=[('atomic','float64'),('II','float64',(len(z_II),len(M_SN))),
                                            ('weight_II','float64',(len(z_II),len(M_SN))),
                                            ('Ia','float64',(len(z_II))),('weight_Ia','float64',(len(z_II)))])
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

    elif II_source == 'nom13':

        # Prep array to hold SN yields
        M_SN = np.array([13., 15., 18., 20., 25., 30., 40.])
        z_II = np.array([0., 0.001, 0.004, 0.008, 0.02, 0.05])
        SN_yield = np.zeros(nel,dtype=[('atomic','float64'),('II','float64',(len(z_II),len(M_SN))),
                                            ('weight_II','float64',(len(z_II),len(M_SN))),
                                            ('Ia','float64',(len(z_II))),('weight_Ia','float64',(len(z_II)))])
        SN_yield['atomic'] = atomic_num

        # Read in IISN yields from Nomoto+13
        with open(yield_path+'nom13/yieldstable.txt','r') as nomfile:  
            use = True
            for line in nomfile:
                ln_list = line.split()

                # Get metallicity
                if line.startswith("Z="):
                    z_idx = np.where(z_II==float(ln_list[1]))[0]

                # Read in all masses
                elif ln_list[0] == 'M':
                    if float(ln_list[1])==20:
                        use = False
                    else:
                        use = True
                        massdict = {}
                        for colidx, mass in enumerate(ln_list[1:]):
                            if float(mass) in M_SN:
                                massdict[colidx+2] = np.where((M_SN==float(mass)))[0][0]

                elif use and ln_list[0] in atomic_names:
                    atom_idx = np.where(atomic_names == ln_list[0])
                    for mass_idx in massdict.keys():
                        col_idx = massdict[mass_idx]
                        SN_yield['weight_II'][atom_idx, z_idx,col_idx] += float(ln_list[1]) * float(ln_list[mass_idx])     # Mass weighted by isotopic weight
                        SN_yield['II'][atom_idx, z_idx,col_idx] += float(ln_list[mass_idx])

    elif II_source == 'lim18': 

        # Prep array to hold SN yields
        M_SN = np.array([13., 15., 20., 25., 30., 40.]) #, 60., 80.])
        feh_II = np.array([-3., -2., -1., 0.])
        z_II = 10.**feh_II * 0.02 # Note that Z_sun = 0.02
        SN_yield = np.zeros(nel,dtype=[('atomic','float64'),('II','float64',(len(z_II),len(M_SN))),
                                            ('weight_II','float64',(len(z_II),len(M_SN))),
                                            ('Ia','float64',(len(z_II))),('weight_Ia','float64',(len(z_II)))])
        SN_yield['atomic'] = atomic_num

        # For each elem, get list of all isotopes that need to be included
        isolist = {'H':['H1','H2'],
                    'He':['H3','He3','He4'],
                    'C':['C12','C13','N13'],
                    'Mg':['Na24','Mg24','Mg25','Mg26','Al25','Al26'],
                    'Si':['Al28','Si28','Si29','Si30','P29','P30'],
                    'Ca':['K42','Ca40','Ca42','Ca43','Ca44','Ca46','Ca48','Sc42','Sc43','Sc44','Ti44'],
                    'Ti':['Ca47','Ca49','Sc46','Sc47','Sc48','Sc49','Ti46','Ti47','Ti48','Ti49','Ti50','V46','V47','V48','V49','Cr48','Cr49','Mn50'],
                    'Mn':['Mn55','Fe55','Co55','Cr55'],
                    'Fe':['Mn56','Mn57', 'Fe54','Fe56','Fe57','Fe58','Co56','Co57','Co58','Ni56','Ni57','Cu57','Cu58'],
                    'Ba':['Xe135','Cs134','Cs135','Cs136','Cs137','Cs138','Ba134','Ba135','Ba136','Ba137','Ba138'],
                    'Eu':[]}

        # Read in IISN yields from Limongi & Chieffi (2018)
        ii_table = ascii.read(yield_path+'lim18/tab8.txt')

        # Get element names and mass numbers
        elem_name = []
        mass_num = []
        for elem in ii_table['Isotope']:  # Element names
            match = re.match(r"([A-Za-z]+)([0-9]+)", elem, re.I)
            if match:
                items = match.groups()
                elem_name.append(items[0])
                mass_num.append(items[1])

        elem_name = np.asarray(elem_name)
        mass_num = np.asarray(mass_num)

        mass_array = ii_table['13M', '15M', '20M', '25M', '30M', '40M'] #, '60M', '80M']

        # Rotational velocity weights (from Prantzos+18, Fig 4)
        # rotvelweights[metallicity, rotvel]
        rotvel = np.array([0,150,300])
        rotvelweights = np.array([[0.05, 0.72, 0.23],
                                [0.50, 0.48, 0.02],
                                [0.63, 0.36, 0.01],
                                [0.67, 0.32, 0.01]])

        #print('rotvelweights', rotvelweights[0][1])

        # Loop over each element needed in final table
        for elem_idx, elem in enumerate(atomic_names):
            if (elem not in elem_name): 
                continue
            
            # Loop over each metallicity
            for z_idx, feh in enumerate(feh_II):
                
                # Loop over each rotational velocity
                for vel_idx, vel in enumerate(rotvel):

                    weight = rotvelweights[z_idx, vel_idx]

                    # Loop over all isotopes
                    for i in range(len(isolist[elem])):

                        # Find lines in table where everything matches up
                        idx = np.where((ii_table['Isotope']==isolist[elem][i]) & (ii_table['[Fe/H]']==int(feh)) & (ii_table['Vel']==int(vel)))[0]
                        #print(idx, elem, isolist[elem][i], feh, vel, weight)

                        isotope_mass = float(mass_num[idx])
                        test = np.array(mass_array[idx])
                        test = test.view((float, len(test.dtype.names))).reshape(6,)

                        # Add yields to SN_yield table
                        SN_yield['weight_II'][elem_idx, z_idx, :] += isotope_mass * weight * test     # Mass weighted by isotopic weight
                        SN_yield['II'][elem_idx, z_idx, :] += weight * test

    return SN_yield, M_SN, z_II

def load_Ia(Ia_source, yield_path, SN_yield, atomic_names, z_II):
    """Reads in Ia yield files."""

    if Ia_source == 'iwa99':
        # Iwa+99 doesn't have He, so read in He yields from Nomoto+06
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

    elif Ia_source == 'leu20':
        # Read in Ia yields from Leung & Nomoto 2020
        ia_table = ascii.read(yield_path+'leu20/tab6.txt', format='basic')
        elem_name = np.array([ia_table['Isotope'][j].strip('0123456789-*^') for j in range(len(ia_table['Isotope']))])  
        mass_num = np.array([re.sub("\D","",ia_table['Isotope'][k]) for k in range(len(ia_table['Isotope']))])  

        # Get metallicity and yield arrays
        z_arr = np.asarray(ia_table.colnames[1:], dtype='float')
        yields = np.array(ia_table[ia_table.colnames[1:]])
        yields = yields.view(np.float64).reshape(yields.shape + (-1,)) # Convert structured array to ndarray
        
        for elem_idx, elem in enumerate(elem_name): 
            # Loop over each line in yield table   
            if (elem not in atomic_names): 
                continue
            else:
                isotope_mass = int(mass_num[elem_idx])

            # Get index for storing yields in SN_yield table
            wa = np.where(atomic_names == elem)[0][0]

            # Interpolate as a function of metallicity (to match IISNe metallicities)
            elem_yields = np.interp(z_II, z_arr, yields[elem_idx])

            # Add yields to SN_yield table
            SN_yield[wa]['weight_Ia'] += isotope_mass * elem_yields  # Mass weighted by isotopic weight
            SN_yield[wa]['Ia'] += elem_yields

    return SN_yield

def load_AGB(AGB_source, yield_path, atomic_num, atomic_names, atomic_weight):
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
                if atomic_num[elem_idx] == 63: isotope = np.array(isotope) + 100

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

    if AGB_source == 'kar':
        # Masses and metallicities from Karakas files
        M_kar = np.array([1.00, 1.25, 1.50, 1.90, 2.00, 2.10, 2.25, 2.50, 
                2.75, 3.00, 3.25, 3.50, 4.00, 4.50, 5.00, 5.50, 6.00, 7.00])
        z_kar = np.array([0.0028, 0.007, 0.014, 0.03])

        # Define the output array of yields
        kar_temp = np.zeros((len(atomic_num),len(z_kar), len(M_kar)))
        kar = np.zeros(len(atomic_num),dtype=[('atomic','float64'),
                        ('AGB','float64',(len(z_kar),len(M_kar)-2)),
                        ('weight','float64',(len(z_kar),len(M_kar)-2))])
        kar['atomic'] = atomic_num
        for elem in range(len(kar['atomic'])):
            kar[elem]['weight'] = atomic_weight[elem]

        # Get Karakas files
        sub_string = ['007', '014', '03']
        karfiles = [yield_path+'kar16/yield_z'+ sub_string[i] +'.dat' for i in range(len(sub_string))]
        karfiles.insert(0,yield_path+'kar18/yields_z0028.dat')

        # Loop over each file
        for filename in karfiles:
            with open(filename,'r') as karfile:  
                for line in karfile:
                    ln_list = line.split()

                    if line.startswith("#"):
                        if ln_list[1] == "Initial":

                            # Bool to decide whether to use value
                            use = True

                            # Read in model parameters
                            m_in = float(ln_list[4].strip(','))
                            z_in = float(ln_list[7].strip(','))
                            m_mix = float(ln_list[13].strip(','))
                            
                            # Get indices to store yields
                            wm = np.where(M_kar == m_in)[0]
                            wz = np.where(z_kar == z_in)[0]

                            # Deal with overshoot conditions
                            if use and "N_ov" in ln_list:
                                N_ov = float(ln_list[-1])
                                if (z_in < 0.02 and N_ov != 0.0):
                                    use = True
                                elif (z_in > 0.02 and N_ov < 1.0):
                                    use = True
                                else: use = False
                            else:
                                N_ov = 0.0

                            # Deal with M_mix conditions
                            if use: 
                                if z_in > 0.0028:
                                    if (m_in >= 5.0 and ~np.isclose(m_mix, 0.0)) or \
                                        (m_in < 5.0 and m_in > 4.0 and ~np.isclose(m_mix, 1.e-4)) or \
                                        (m_in <= 4.0 and m_in > 3.0 and ~np.isclose(m_mix, 1.e-3)) or \
                                        (z_in < 0.03 and m_in <= 3.0 and m_in > 1.25 and ~np.isclose(m_mix, 2.e-3)):
                                        use = False
                                # Note that z=0.0028 (from Kar+18) has slightly different conditions
                                else:
                                    if (m_in > 4.0 and ~np.isclose(m_mix, 0.0)) or \
                                        (m_in < 4.5 and m_in > 3.75 and ~np.isclose(m_mix, 1.e-4)) or \
                                        (m_in < 4.0 and m_in >= 3.0 and ~np.isclose(m_mix, 1.e-3)) or \
                                        (m_in < 3.0 and m_in > 1.0 and ~np.isclose(m_mix, 2.e-3)):
                                            use = False

                                    # Add additional conditions for mass-loss prescriptions
                                    if use and ((np.isclose(m_in, 3.75) and ln_list[-1] == 'B95') or
                                        (np.isclose(m_in, 5.0) or np.isclose(m_in, 7.0)) and ln_list[-1] == 'VW93'):
                                        use = False

                    # Skip lines at the end of each table
                    elif line.startswith("   "):
                        pass

                    # Store yields
                    elif use & (len(wm) == 1) & (len(wz) == 1):
                        elname, atom, masslost_in = ln_list[0], int(ln_list[1]), float(ln_list[-1])

                        # Make sure the isotope, mass, and metallicity all exist in the output array
                        if (atom in atomic_num):

                            # Store yield in output array
                            wa = np.where(np.asarray(atomic_num) == atom)[0][0]
                            kar_temp[wa,wz[0],wm[0]] += masslost_in  

        # Average non-zero yields for 1.9, 2, 2.1 -> 2.0
        for metal in range(len(z_kar)):
            if kar_temp[0,metal,3] != 0:
                kar_temp[:,metal,4] = np.average([kar_temp[:,metal,3],kar_temp[:,metal,5]], axis=0)
        kar_temp = np.delete(kar_temp, [3,5], 2)
        M_kar = np.delete(M_kar, [3,5])

        # Put new yields in final array
        kar['AGB'] = kar_temp

        return M_kar, z_kar, kar
              
def initialize_yields(yield_path='yields/', r_process_keyword='none', AGB_source='cri15', Ia_source='iwa99', II_source='nom06'):
    """Reads in yield tables.

    Args:
        yield_path (str): Path to folder with yields.
        r_process_keyword (str): How to handle r-process elements: 'none', 'typical_SN_only', 'rare_event_only', 'both'
        AGB_source (str): Source of AGB yields: 'cri15', 'kar'
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

    # Read in SN yield files
    SN_yield, M_SN, z_II = load_II(II_source, yield_path, nel, atomic_names, atomic_num)
    SN_yield = load_Ia(Ia_source, yield_path, SN_yield, atomic_names, z_II)                           
    
    # Divide weighted mass by total yield mass to get average isotope mass
    SN_yield['weight_II'][SN_yield['II']>0] /= SN_yield['II'][SN_yield['II']>0]
    SN_yield['weight_Ia'][SN_yield['Ia']>0] /= SN_yield['Ia'][SN_yield['Ia']>0]

    # Read in AGB yield files
    M_AGB, z_AGB, AGB_yield = load_AGB(AGB_source, yield_path, atomic_num, atomic_names, atomic_weight)

    # Add in Ba, Eu abundances from SNe/rare r-process events
    ba_idx = np.where((np.asarray(elem_atomic) == 56))[0][0]
    eu_idx = np.where((np.asarray(elem_atomic) == 63))[0][0]

    SN_yield['weight_II'][ba_idx] = np.mean(AGB_yield['weight'][9])
    SN_yield['weight_II'][eu_idx] = np.mean(AGB_yield['weight'][10])

    if r_process_keyword in ['typical_SN_only','both']:

        # Note: no Eu abundances from Li+2014?

        # Barium yield for weak r-process event as a function of mass (Li+2014)
        # These correspond to masses M_SN = [13.0, 15.0, 18.0, 20.0, 25.0, 30.0, 40.0]
        ba_li14_weakr = 10. * np.asarray([1.38e-8, 2.83e-8, 5.38e-8, 6.84e-8, 9.42e-8, 0, 0])

        # Linearly interpolate to high-mass end
        ba_li14_weakr[-2] = ba_li14_weakr[-3]*M_SN[-2]/M_SN[-3]
        ba_li14_weakr[-1] = ba_li14_weakr[-2]*M_SN[-1]/M_SN[-2]

        # Interpolate to match SN mass array
        M_li14 = np.array([13.0, 15.0, 18.0, 20.0, 25.0, 30.0, 40.0])
        ba_li14_weakr = np.interp(M_SN, M_li14, ba_li14_weakr)

        # Add to SNII yield arrays
        for i in range(len(z_II)):
            SN_yield['II'][ba_idx][i,:] = ba_li14_weakr  # Assume no Z-dependence
            
    if r_process_keyword in ['rare_event_only','both']:   

        # Note: these are average Ba, Eu yields for a "main" r-process event (Li+2014)
        ba_yield = 2.3e-6
        eu_yield = 2.27e-7

        # Since NSM have a similar DTD as SN Ia, put yields in SNIa array as a proxy. 
        # (Scale Ba and Eu yields up and reduce rate of events proportionally to compare to models.)
        SN_yield['Ia'][ba_idx] = ba_yield
        SN_yield['Ia'][eu_idx] = eu_yield
        SN_yield['weight_Ia'][ba_idx] = np.mean(AGB_yield['weight'][9])
        SN_yield['weight_Ia'][eu_idx] = np.mean(AGB_yield['weight'][10])

    return nel, eps_sun, SN_yield, AGB_yield, M_SN, z_II, M_AGB, z_AGB

if __name__ == "__main__":

    nel, eps_sun, SN_yield, AGB_yield, M_SN, z_II, M_AGB, z_AGB = initialize_yields(II_source='lim18', r_process_keyword='none')
    #print(SN_yield['weight_II'][0])
    #print(SN_yield['II'][0,:,:]) # elem, Z, M
    #print(z_II, M_SN)
    #print(np.isclose(z_II,0))
