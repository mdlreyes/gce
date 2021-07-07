"""
gce_yields.py

This program is based on Gina Duggan's code to read in model
nucleosynthetic yields for a GCE model.
"""

# Import useful packages
import numpy as np
from scipy.interpolate import interp1d
from astropy.io import ascii
import re
import pandas as pd
import os
from scipy.stats import lognorm, norm
import imf

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
elem_atomic = [1, 2, 6, 12, 14, 20, 22, 25, 26, 28, 56, 63] #8, 
nel = len(elem_atomic)
elem_idx = np.where(np.isin(atomic_num, elem_atomic))[0]

eps_sun = eps_sun[elem_idx]
atomic_num = elem_atomic
atomic_weight = atomic_weight[elem_idx]
atomic_names = atomic_names[elem_idx]

# Read in the data from Lugaro+12
def readkaryields():
    path = '/Users/miadelosreyes/Documents/Research/MnDwarfs_DTD/code/gce/yields/AGBdata/z0001models/elemental_yields/'
    outfile = '/Users/miadelosreyes/Documents/Research/MnDwarfs_DTD/code/gce/yields/lug12/yields_z0001.dat'
    
    for filename in os.listdir(path):
        if filename.startswith("yield"): 
            # open both files
            with open(path+filename,'r') as firstfile, open(outfile,'a') as secondfile:
                
                # read content from first file
                for line in firstfile:
                        
                        # write content to second file
                        secondfile.write(line)
            continue
    
    return

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
                    'Fe':['Mn56','Mn57','Fe54','Fe56','Fe57','Fe58','Co56','Co57','Co58','Ni56','Ni57','Cu57','Cu58'],
                    'Ni':['Fe60','Fe61','Co60','Co61','Ni58','Ni60','Ni61','Ni62','Ni64','Cu60','Cu61','Cu62','Cu64','Zn60','Zn61','Zn62','Ga62','Ga64','Ge64'],
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

    elif Ia_source == 'leu18_ddt':
        # Read in Ia yields from Leung & Nomoto 2018
        ia_table = ascii.read(yield_path+'leu18/tab4.txt', delimiter='\t')
        elem_name = np.array([ia_table['Isotopes'][j].strip('0123456789-*^') for j in range(len(ia_table['Isotopes']))])  
        mass_num = np.array([re.sub("\D","",ia_table['Isotopes'][k]) for k in range(len(ia_table['Isotopes']))])  

        # Get metallicity and yield arrays
        z_arr = np.asarray([col[4:] for col in ia_table.colnames[1:7]], dtype='float')
        yields = np.array(ia_table[ia_table.colnames[1:7]])
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
    
    elif Ia_source == 'leu18_def':
        # Read in Ia yields from Leung & Nomoto 2018
        ia_table = ascii.read(yield_path+'leu18/tab9.txt', delimiter='\t')
        elem_name = np.array([ia_table['Isotopes'][j].strip('0123456789-*^') for j in range(len(ia_table['Isotopes']))])  
        mass_num = np.array([re.sub("\D","",ia_table['Isotopes'][k]) for k in range(len(ia_table['Isotopes']))])  

        # Get yield array
        yields = np.array(ia_table['300-1-c3-1P'])

        for elem_idx, elem in enumerate(elem_name): 
            # Loop over each line in yield table   
            if (elem not in atomic_names): 
                continue
            else:
                isotope_mass = int(mass_num[elem_idx])

            # Get index for storing yields in SN_yield table
            wa = np.where(atomic_names == elem)[0][0]

            # Interpolate as a function of metallicity (to match IISNe metallicities)
            elem_yields = yields[elem_idx] * np.ones_like(z_II)

            # Add yields to SN_yield table
            SN_yield[wa]['weight_Ia'] += isotope_mass * elem_yields  # Mass weighted by isotopic weight
            SN_yield[wa]['Ia'] += elem_yields

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

    elif Ia_source == 'shen18':
        # Read in Ia yields from Shen et al. (2018)
        z_arr = np.array([0.000,0.005,0.010,0.020])

        # Create empty arrays to hold yields
        yields = np.zeros((len(atomic_names),len(z_arr))) # shape (elems, Z)
        weightedyields = np.zeros((len(atomic_names),len(z_arr)))

        for z_idx in range(len(z_arr)):
            data = pd.read_csv(yield_path+'shen18/shen18.txt',delimiter='\s+',skiprows=8*z_idx+3,nrows=5)
            idx_mass = np.where(data['mass']==1.10)[0]
            for col in data.columns:
                # Get element name
                if col == 'mass':
                    continue
                elem_name = col.strip('0123456789-*^')
                mass_num = int(re.sub("\D","",col))
                if (elem_name not in atomic_names):
                    continue

                # Get index for storing yields in SN_yield table
                wa = np.where(atomic_names == elem_name)[0][0]

                # Put into temp yield tables
                yields[wa, z_idx] += data[col][idx_mass]
                weightedyields[wa, z_idx] += data[col][idx_mass] * mass_num # Mass weighted by isotopic weight

        # If needed, extrapolate over metallicity to match max(z_II)
        if max(z_arr) < max(z_II):
            weightedia_zmax = weightedyields[:,-1]+(max(z_II)-z_arr[-1])*(weightedyields[:,-1]-weightedyields[:,-2])/(z_arr[-1]-z_arr[-2])
            weightedyields = np.concatenate((weightedyields, weightedia_zmax[:,None]), axis=1)   # Concatenate yield tables

            ia_zmax = yields[:,-1]+(max(z_II)-z_arr[-1])*(yields[:,-1]-yields[:,-2])/(z_arr[-1]-z_arr[-2])
            yields = np.concatenate((yields, ia_zmax[:,None]), axis=1)   # Concatenate yield tables

            z_arr = np.concatenate((z_arr,[max(z_II)]))

        # Interpolate to match z_II metallicities
        SN_yield['weight_Ia'] = interp1d(z_arr,weightedyields,axis=1)(z_II)  # Mass weighted by isotopic weight
        SN_yield['Ia'] = interp1d(z_arr,yields,axis=1)(z_II)

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
        z_kar = np.array([0.0001, 0.0028, 0.007, 0.014, 0.03])

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
        karfiles.insert(0,yield_path+'lug12/yields_z0001.dat')

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
        M_li14 = np.array([13.0, 15.0, 18.0, 20.0, 25.0, 30.0, 40.0])
        ba_li14_weakr = 10. * np.asarray([1.38e-8, 2.83e-8, 5.38e-8, 6.84e-8, 9.42e-8, 0, 0])

        # Eu yield for weak r-process event as a function of mass (Cescutti+2006)
        M_ces06 = np.array([12.0, 15.0, 30.0, 40.0])
        eu_ces06_weakr = np.array([4.5e-8, 3.0e-8, 5.0e-10, 0.])

        # Linearly interpolate to high-mass end
        ba_li14_weakr[-2] = ba_li14_weakr[-3]*M_li14[-2]/M_li14[-3]
        ba_li14_weakr[-1] = ba_li14_weakr[-2]*M_li14[-1]/M_li14[-2]
        eu_ces06_weakr[-1] = eu_ces06_weakr[-2]*M_ces06[-1]/M_ces06[-2]

        # Interpolate to match SN mass array
        ba_li14_weakr = np.interp(M_SN, M_li14, ba_li14_weakr)
        eu_ces06_weakr = np.interp(M_SN, M_ces06, eu_ces06_weakr)

        # Add to SNII yield arrays, assuming no Z dependence
        for i in range(len(z_II)):
            SN_yield['II'][ba_idx][i,:] += ba_li14_weakr
            SN_yield['II'][eu_idx][i,:] += eu_ces06_weakr 
            
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

def initialize_empirical(yield_path='yields/', r_process_keyword='none', imfweight='kroupa93',
        AGB_source='cri15', Ia_source='leu20', II_source='nom06', 
        II_mass=None, AGB_mass=None, fit=False): 
    """Reads in yield tables.

    Args:
        yield_path (str): Path to folder with yields.
        r_process_keyword (str): How to handle r-process elements: 'none', 'typical_SN_only', 'rare_event_only', 'both'
        AGB_source (str): Source of AGB yields: 'cri15', 'kar'
        Ia_source (str): Source of Ia yields: 'leu20', 'leu18_ddt', 'leu18_def', 'shen18'
        II_source (str): Source of II yields: 'nom13', 'lim18'
        II_mass, AGB_mass (float array): Masses for which to compute yields
        fit (bool): If True, try fitting yield parameters

    Returns:
        nel (int): Number of elements.
        eps_sun (array): Solar abundances in epsilon notation. 
        atomic_num (array): List of atomic numbers (equivalent to SN_yields['atomic']).
        atomic_weight (array): List of atomic weights (equivalent to SN_yields['weight_II']).
        f_ia_metallicity (func): Function that takes Z as input, outputs IaSN yields
        f_ii_metallicity (func): Function that takes Z as input, outputs CCSN yields
        f_agb_metallicity (func): Function that takes Z as input, outputs AGB yields
    """

    # Constant for normal distribution
    normpdfc = np.sqrt(2*np.pi)

    # Use set yield parameters
    if fit==False:
        # Ia yield function
        def f_ia_metallicity(metallicity):
            # Default Ia yields
            yields = np.array([0., 0., 1.e-3, 1.e-2, 0.15, 2.e-2, 1.e-3, 1., 0.8, 1.5e-2, 0., 0.])
                
            # Put in Mn yields
            mnyields = {'leu18_ddt':7.25e-3, 'leu18_def':8.21e-3, 'leu20':1.79e-3, 'shen18':0.14e-3}
            yields[7] = mnyields[Ia_source]

            # Put in Ni yields
            niyields = {'leu18_ddt':5.65e-2, 'leu18_def':5.89e-2, 'leu20':1.53e-2, 'shen18':2.19e-2}
            yields[9] = niyields[Ia_source]

            # Pure deflagration yields
            if Ia_source=='leu18_def':
                yields[2] = 0.36   # C
                yields[5] = 0.15e-2  # Ca
                yields[8] = 0.45   # Fe

            return yields

        # CCSN yield function
        def f_ii_metallicity(metallicity):
            yields = np.zeros((nel,len(II_mass)))
            dN_dM = imf.imf(II_mass, imfweight)

            # Common yields
            yields[0,:] = 1e-3 * (255 * II_mass**(-1.88) - 0.1)  # H
            yields[1,:] = 1e-3 * (45 * II_mass**(-1.35) - 0.2)  # He
            yields[6,:] = 1e-8 * (1000 * II_mass**(-2.3))  # Ti
            yields[7,:] = 1e-7 * (30 * II_mass**(-1.32) - 0.25)  # Mn
            yields[8,:] = 1e-5 * (2722 * II_mass**(-2.77))  # Fe

            if II_source=='nom13':
                yields[2,:] = 1e-5 * (100 * II_mass**(-1.35))  # C
                yields[3,:] = 1e-5 * (261 * II_mass**(-1.8) + 0.33)  # Mg
                yields[4,:] = 1e-5 * (2260 * II_mass**(-2.83) + 0.8)  # Si
                yields[5,:] = 1e-6 * (15.4 * II_mass**(-1) + 0.06)  # Ca
                yields[9,:] = 1e-6 * (8000 * II_mass**(-3.6))  # Ni
            elif II_source=='lim18':
                yields[2,:] = 1e-5 * (100 * II_mass**(-1.35))  # C
                yields[2,np.where(II_mass > 30)] = 0.
                yields[3,:] = 1e-5 * 13*np.exp(-((II_mass-19)/6.24)**2/2)/(6.24*normpdfc) #norm.pdf(II_mass, loc=19, scale=6.24)  # Mg
                yields[4,:] = 1e-5 * (28 * II_mass**(-0.34) - 8.38)  # Si
                yields[5,:] = 1e-6 * 40*np.exp(-((II_mass-(17.5-3000*metallicity))/3)**2/2)/(3*normpdfc) #norm.pdf(II_mass, loc=(17.5-3000*metallicity), scale=3)  # Ca
                yields[9,:] = 1e-6 * (8000 * II_mass**(-3.2))  # Ni

            yields /= dN_dM
            return yields

        def f_agb_metallicity(metallicity):
            yields = np.zeros((nel,len(AGB_mass)))
            dN_dM = imf.imf(AGB_mass, imfweight)

            # Common yields
            yields[0] = 1e-1 * (1.1 * AGB_mass**(-0.9) - 0.15)  # H
            yields[1] = 1e-2 * (4 * AGB_mass**(-1.07) - 0.22)  # He
            yields[3] = 1e-5 * ((400*metallicity + 1.1) * AGB_mass**(0.08 - 340*metallicity) + (360*metallicity - 1.27))  # Mg
            yields[4] = 1e-5 * ((800*metallicity) * AGB_mass**(-0.9) - (0.03 + 80*metallicity))  # Si
            yields[5] = 1e-6 * ((-0.1 + 800*metallicity) * AGB_mass**(-0.96) - (80*metallicity))  # Ca
            yields[6] = 1e-8 * ((3400*metallicity) * AGB_mass**(-0.88) - (480*metallicity))  # Ti
            yields[7] = 1e-7 * ((1500*metallicity) * AGB_mass**(-0.95) - (160*metallicity))  # Mn
            yields[8] = 1e-5 * ((1500*metallicity) * AGB_mass**(-0.95) - (160*metallicity))  # Fe
            yields[9] = 1.e-6* ((840*metallicity) * AGB_mass**(-0.92) - (80*metallicity + 0.04)) # Ni

            if AGB_source=='cri15':
                yields[2] = 1e-3 * 0.89*np.exp(-((AGB_mass-1.9)/0.58)**2/2)/(0.58*normpdfc) #norm.pdf(AGB_mass, loc=1.9, scale=0.58)  # C
                yields[10] = 1e-8 * (400*metallicity - 0.1)*np.exp(-((AGB_mass-2)/0.5)**2/2)/(0.5*normpdfc) #norm.pdf(AGB_mass, loc=2, scale=0.5)  # Ba
                yields[11] = 1e-11 * (2000*metallicity - 0.6)*np.exp(-((AGB_mass-2)/0.65)**2/2)/(0.65*normpdfc) #norm.pdf(AGB_mass, loc=2, scale=0.65)  # Eu
            elif AGB_source=='kar':
                yields[2] = 1e-3 * (1.68-220*metallicity)*np.exp(-((AGB_mass-2)/0.6)**2/2)/(0.6*normpdfc) #norm.pdf(AGB_mass, loc=2, scale=0.6)  # C
                yields[3] += 1e-5 * (0.78-300*metallicity)*np.exp(-((AGB_mass-2.3)/0.14)**2/2)/(0.14*normpdfc) #norm.pdf(AGB_mass, loc=2.3, scale=0.14)  # Mg
                yields[10] = 1e-8 * (1000*metallicity + 0.2)*np.exp(-((AGB_mass-2.3)/(0.75-100*metallicity))**2/2)/((0.75-100*metallicity)*normpdfc) #norm.pdf(AGB_mass, loc=2.3, scale=(0.75-100*metallicity))  # Ba
                yields[11] = 1e-11 * (3400*metallicity + 0.4)*np.exp(-((AGB_mass-2.2)/0.65)**2/2)/(0.65*normpdfc) #norm.pdf(AGB_mass, loc=2.2, scale=0.65)  # Eu

            yields /= dN_dM
            return yields

    # Return functions with free parameters
    else:

        # NSM yield function
        if r_process_keyword in ['rare_event_only','both']:   
            def f_nsm_metallicity():
                yields = np.zeros(12)
                # Note: these are average Ba, Eu yields for a "main" r-process event (Li+2014)
                yields[10] = 2.3e-6 # Ba
                yields[11] = 2.27e-7 # Eu
                return yields
        else:
            f_nsm_metallicity = None

        # Ia yield function
        def f_ia_metallicity(metallicity, fe_ia, mn_ia=2e-3, ni_ia=1.5e-2):
            yields = np.array([0., 0., 1.e-3, 1.e-2, 0.15, 2.e-2, 1.e-3, 2.e-3, 0.8, 1.5e-2, 0., 0.])
            yields[7] = mn_ia  # Mn
            yields[8] = fe_ia  # Fe
            yields[9] = ni_ia  # Ni

            return yields

        # CCSN yield function
        def f_ii_metallicity(metallicity, cexp_ii, mgnorm_ii, canorm_ii):
            yields = np.zeros((nel,len(II_mass)))
            dN_dM = imf.imf(II_mass, imfweight)

            # Common yields
            yields[0,:] = 1e-3 * (255 * II_mass**(-1.88) - 0.1)  # H
            yields[1,:] = 1e-3 * (45 * II_mass**(-1.35) - 0.2)  # He
            yields[4,:] = 1e-5 * (2260 * II_mass**(-2.83) + 0.8)  # Si
            yields[5,:] = 1e-6 * (15.4* II_mass**(-1) + 0.06)  # Ca
            yields[6,:] = 1e-8 * (1000 * II_mass**(-2.3))  # Ti
            yields[7,:] = 1e-7 * (30 * II_mass**(-1.32) - 0.25)  # Mn
            yields[8,:] = 1e-5 * (2722 * II_mass**(-2.77))  # Fe
            yields[9,:] = 1e-6 * (8000 * II_mass**(-3.2))  # Ni (use Limongi 18 values for now)

            if r_process_keyword in ['typical_SN_only','both']:

                # Ba from Li+14
                yields[10,:] = 1e-12 * (1560 * II_mass**(-1.80) + 0.14 - 480*np.exp(-((II_mass-5)/5.5)**2/2)/(5.5*normpdfc))

                # Eu from Matteucci+14
                yields[11,:] = 1e-11 * (77600 * II_mass**(-4.31))  # Eu

            # Yields with parameters to vary
            yields[2,:] = 1e-5 * (100 * II_mass**(-cexp_ii))  # C
            yields[3,:] = 1e-5 * (mgnorm_ii + 13*np.exp(-((II_mass-19)/6.24)**2/2)/(6.24*normpdfc))  # Mg
            yields[5,:] += canorm_ii * 1e-6 * (40-10000*metallicity)*np.exp(-((II_mass-15)/3)**2/2)/(3*normpdfc)  # Ca

            yields /= dN_dM
            return yields

        def f_agb_metallicity(metallicity, cnorm_agb, banorm_agb=0.33, bamean_agb=1.0, eunorm_agb=0.5, eumean_agb=0.2):
            yields = np.zeros((nel,len(AGB_mass)))
            dN_dM = imf.imf(AGB_mass, imfweight)

            # Common yields
            yields[0] = 1e-1 * (1.1 * AGB_mass**(-0.9) - 0.15)  # H
            yields[1] = 1e-2 * (4 * AGB_mass**(-1.07) - 0.22)  # He
            yields[3] = 1e-5 * ((400*metallicity + 1.1) * AGB_mass**(0.08 - 340*metallicity) + (360*metallicity - 1.27))  # Mg
            yields[4] = 1e-5 * ((800*metallicity) * AGB_mass**(-0.9) - (0.03 + 80*metallicity))  # Si
            yields[5] = 1e-6 * ((-0.1 + 800*metallicity) * AGB_mass**(-0.96) - (80*metallicity))  # Ca
            yields[6] = 1e-8 * ((3400*metallicity) * AGB_mass**(-0.88) - (480*metallicity))  # Ti
            yields[7] = 1e-7 * ((1500*metallicity) * AGB_mass**(-0.95) - (160*metallicity))  # Mn
            yields[8] = 1e-5 * ((1500*metallicity) * AGB_mass**(-0.95) - (160*metallicity))  # Fe
            yields[9] = 1.e-6* ((840*metallicity) * AGB_mass**(-0.92) - (80*metallicity + 0.04)) # Ni

            # Yields with parameters to vary
            yields[2,:] = cnorm_agb * 1e-3 * (1.68-220*metallicity)*np.exp(-((AGB_mass-2)/0.6)**2/2)/(0.6*normpdfc)  # C
            yields[10,:] = banorm_agb * 1e-8 * (1000*metallicity + 0.2)*np.exp(-((AGB_mass-(3.0-bamean_agb))/(0.75-100*metallicity))**2/2)/((0.75-100*metallicity)*normpdfc)  # Ba
            yields[11,:] = eunorm_agb * 1e-11 * (3400*metallicity + 0.4)*np.exp(-((AGB_mass-(2.2-eumean_agb))/0.65)**2/2)/(0.65*normpdfc)  # Eu

            yields /= dN_dM
            return yields

    return nel, eps_sun, np.asarray(atomic_num), atomic_weight, f_ia_metallicity, f_ii_metallicity, f_agb_metallicity, f_nsm_metallicity

if __name__ == "__main__":

    #readkaryields()

    #nel, eps_sun, SN_yield, AGB_yield, M_SN, z_II, M_AGB, z_AGB = initialize_yields(r_process_keyword='none')
    #print(eps_sun, SN_yield['atomic'][9], SN_yield['Ia'][9,:])
    nel, eps_sun, atomic, weight, f_ia_metallicity, f_ii_metallicity, f_agb_metallicity = initialize_empirical()
    print(nel, atomic)
