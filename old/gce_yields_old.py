import numpy as np
import re

def check_array(array):
    nan = np.where(np.isnan(array))
    if nan[0].size != 0: 
        print('ERROR: not available (nan in weight) metallicities, masses, elements: ', nan)
        array[nan] = 0          
    return array

def load_Cristallo15(yield_path, eps_sun, eps_atomic_num, atomic_weight, verbose=False):
    #read in AGB yield file
    # returns M_AGB, z_AGB, AGB_yield
    M_cri = np.array([1.3,1.5,2.0,2.5,3.0,4.0,5.0,6.0])
    z_cri = np.array([0.001,0.002,0.003,0.006,0.008,0.01,0.014,0.02])
    atomic = np.array([1,2,6,8,12,14,20,22,26,56])
    sub_string = ['h', 'he', 'c', 'o', 'mg','si','ca','ti','fe','ba']
    cri15files = [yield_path + 'FRUITY/'+ sub_string[i] +'.txt' for i in range(len(sub_string))]
    ncri15 = len(cri15files)
    cri15 = np.zeros(len(atomic),dtype=[('atomic','float64'),('AGB','float64',(len(z_cri),len(M_cri))),('weight','float64',(len(z_cri),len(M_cri)))])
    cri15['atomic'] = atomic
    for i in range(ncri15): #for each element file
        data = np.genfromtxt(cri15files[i],names=True, encoding='utf_8')
        isotope_name = data.dtype.names[3:]
        num_isotopes = len(isotope_name)
        try:
            isotope = [int(re.findall(r'\d+',isotope_name[j])[0]) for j in range(num_isotopes)]
            if atomic[i] == 56: isotope = np.array(isotope) + 100
        except IndexError:
            isotope = []
        cri_M = data['Mass']
        cri_z = data['Metallicity']
        if num_isotopes == 1:                                         
            if isotope == []:
                isotope = cri15['atomic'][i]
            Yield = data[isotope_name[0]]
            weight = np.zeros(len(Yield))+isotope
        else:    
            iso_array = np.array(data[list(isotope_name)].tolist())
            Yield = np.sum(iso_array,axis=1)                                          #sum yields to form yield array
            weight = np.dot(iso_array,isotope)/Yield                                   #weight isotope mass by relative abundances to form weight array
        for mass_i in range(len(M_cri)):                                #save yield and weight in proper location: AGB_yield[element].___[metallicity,mass]
            mass_index = np.where(cri_M == M_cri[mass_i])
            for z_i in range(len(z_cri)): 
                z_index = np.where(cri_z[mass_index] == z_cri[z_i])
                if len(z_index) != 1:
                    print("ERROR: more than one or zero metallicity+mass pairs!", M_cri[mass_i],z_cri[z_i])
                else:
                    cri15[i]['AGB'][z_i,mass_i] = (Yield[mass_index])[z_index]
                    cri15[i]['weight'][z_i,mass_i] = (weight[mass_index])[z_index]
    check_array(cri15['weight'])
    if verbose == True:
        print('AGB_yield',cri15.dtype, cri15['atomic'])
    return M_cri, z_cri, cri15

def load_Karakas16(yield_path, eps_sun, eps_atomic_num, atomic_weight, verbose=False):
    #read in another AGB yield file: Karakas et al. 2016 and Fishlock et al. 2014
    M_kar16 = np.array([1.00, 1.25, 1.50, 1.90, 2.00, 2.10, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.50, 5.00, 5.50, 6.00, 7.00])
    #M_kar16 = np.array([1.00, 1.25, 1.50, 1.75, 1.90, 2.00, 2.10, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50, 6.00, 7.00])
    #M_kar16 = np.array([1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50, 6.00, 7.00, 7.50, 8.00])
    z_kar16 = np.array([0.001, 0.007, 0.014, 0.03])
    sub_string = ['007', '014', '03']
    kar16files = [yield_path+'kar16/yield_z'+ sub_string[i] +'.dat' for i in range(len(sub_string))]
    atomic = np.array([1,2,6,8,12,14,20,22,26,38,56])
    iso_string = np.array(['p', 'he', 'c', 'o', 'mg','si','ca','ti','fe','sr','ba']) #two p's so need to check that atomic == 1
    kar16_temp = np.zeros(len(atomic),dtype=[('atomic','float64'),('AGB','float64',(len(z_kar16),len(M_kar16))),('weight','float64',(len(z_kar16),len(M_kar16)))])
    kar16 = np.zeros(len(atomic),dtype=[('atomic','float64'),('AGB','float64',(len(z_kar16),len(M_kar16)-2)),('weight','float64',(len(z_kar16),len(M_kar16)-2))])
    kar16['atomic'] = atomic
    m_mix_masses_z007 = np.array([3,4,4.5]) #masses for which there is a m_mix choice, will look up value recommended for Z=0.014
    #m_mix_values_z007 = [1e-3, 1e-4, 1e-4]
    m_mix_masses_z014 = np.array([2, 3, 3.25, 4, 4.25, 4.5, 4.75, 5])
    m_mix_values_z014 = np.array([2e-3, 2e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 0]) # taken from Karakas 2016 Table 3
    m_mix_masses_z03 = np.array([3.25, 4.25, 4.5, 5])
    #m_mix_values_z03 = [2e-3, 1e-4, 1e-4, 0]
    Use = True
    for i in range(len(kar16files)):
        with open(kar16files[i],'r') as fi:       
            for ln in fi:
                ln_list = ln.split()
                if ln.startswith("#"):
                    if ln_list[1] == "Initial":
                        #print(ln)
                        m_init = float(ln_list[4][:-1])
                        z_in = float(ln_list[7][:-1])
                        m_mix = ln_list[13]
                        wm = np.where(M_kar16 == m_init)[0]
                        wz = np.where(z_kar16 == z_in)[0]
                        if len(ln_list)>14:
                            N_ov = float(ln_list[-1])
                            if z_in <0.02 and N_ov != 0.0:
                                Use = True
                            elif z_in > 0.02 and N_ov == 0.0:
                                Use = True
                            else: Use = False
                            m_mix = float(m_mix[:-1])
                        else: 
                            N_ov = 0.0
                            Use = True
                            m_mix = float(m_mix)
                        if Use == True:
                            if (z_in == 0.007 and ((m_init in m_mix_masses_z007) == True)) or (z_in == 0.014 and ((m_init in m_mix_masses_z014) == True)) or (z_in == 0.03 and ((m_init in m_mix_masses_z03) == True)):
                                wmmix = np.where(m_mix_masses_z014 == m_init)[0]
                                if m_mix_values_z014[wmmix] == m_mix:
                                    Use = True
                                else: Use = False
                        #if Use == True: print(z_in, m_init, N_ov, m_mix, Use)
                elif ln.startswith("   "):
                    pass
                elif Use == True:
                    elname, atom, masslost_in = ln_list[0], int(ln_list[1]), float(ln_list[-1])
                    el_index = np.where(iso_string == elname)[0]
                    if el_index == 0 and atom != 1:
                        #print('skipping',ln)
                        pass
                    elif (len(el_index) == 1) & (len(wm) == 1) & (len(wz) == 1):
                        #print(elname, atom, masslost_in)
                        #if isotope ==1: atom =1
                        wa = np.where(kar16['atomic'] == atom)[0][0]
                        #kar16[wa]['weight'][wz[0],wm[0]] = kar10[wa]['weight'][wz[0],wm[0]]+isotope*masslost_in     # add isotope mass * yield to kar10[atomic]['weight'][metallicity,mass]
                        #kar16[wa]['weight'][wz[0],wm[0]] = -1     # add isotope mass * yield to kar10[atomic]['weight'][metallicity,mass]
                        kar16_temp[wa]['AGB'][wz[0],wm[0]] = kar16_temp[wa]['AGB'][wz[0],wm[0]]+masslost_in                   # add yield to kar10[atomic]['AGB'][metallicity,mass]

    # 1.9 and 2.1 for one metallicty, so will average non-zero yields for 1.9, 2, 2.1 and cut down mass list to 2.0
    for elem in range(len(kar16['atomic'])):
        for metal in range(len(z_kar16)):
            yield_2 = kar16_temp['AGB'][elem,metal,3:6]
            mask = yield_2 > 0
            if sum(mask)>0: kar16_temp['AGB'][elem,metal,4] = np.average(yield_2[mask])
#    wmass = [0,1,2,3,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    wmass = [0,1,2,4,6,7,8,9,10,11,12,13,14,15,16,17]
    M_kar16 = M_kar16[wmass]
    kar16['AGB']=kar16_temp['AGB'][:,:,wmass]

    #print(yield_2[-1])
    #print(kar16['AGB'][0], kar16_temp['AGB'][0])

    #### load z = 0.001 from Fishlock et al. 2014
    iso_atomic = np.array([2,6,8,12,14,38,56])
    iso_string = np.array(['He', 'C', 'O', 'Mg','Si','Sr','Ba']) #two p's so need to check that atomic == 1
    M_fis1 = np.array([1, 1.25, 1.5, 2, 2.25, 2.5, 2.75, 3])
    M_fis2 = [3.25, 3.5, 4, 4.5, 5, 5.5, 6, 7]
    fis14_1file = yield_path+'fis14/table6.txt'
    fis14_2file = yield_path+'fis14/table7.txt'
    M_fis = [M_fis1,M_fis2]
    fis14_file = [fis14_1file,fis14_2file]
    
    for index in [0,1]:
        data = np.genfromtxt(fis14_file[index], names = ['elname','m1', 'm2', 'm3', 
            'm4', 'm5', 'm6', 'm7', 'm8'],dtype=None,skip_header=4,skip_footer=2, encoding='utf_8')
        mass_array = np.array(data[['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8']].tolist())
        mass_array = np.char.replace(np.char.replace(mass_array,'(','e'),')','')
        #mass_array = float(mass_array)
        elname = data['elname']
        for i in np.arange(len(elname)):
            #print(data[i])    
            wz = np.where(z_kar16 == 0.001)[0]
            #print(i,elname[i])
            el_index = np.where(iso_string == elname[i])[0]
            if (len(el_index) == 1) & (len(wz) == 1):
                atom = iso_atomic[el_index[0]]
                wa = np.where(kar16['atomic'] == atom)[0][0]
                for mass_i in range(len(M_fis[index])):
                    wm = np.where(M_kar16 == M_fis[index][mass_i])[0]
                    #print(z_kar16[wz[0]],M_kar16[wm[0]],kar16['atomic'][wa],mass_array[i,mass_i])
                    kar16[wa]['AGB'][wz[0],wm[0]] = mass_array[i,mass_i]                           
    ############## Fishlock et al. 2014 did not include H, Ca, Ti, or Fe yields - estimate them now #######
    fis14_table1_name = yield_path +'fis14/table1.txt'
    table1 = np.genfromtxt(fis14_table1_name, names = ['M_init','M_final'],
                           dtype=float,skip_header=5,skip_footer=11,usecols=(0,1), 
                           encoding='utf_8')
    atoms_to_calculate = np.array([1,20,22,26])
    eps_atoms_to_calculate = np.array([eps_sun[np.where(eps_atomic_num == atom)[0][0]] for atom in atoms_to_calculate]) + np.array([0,1,1,1])*-1.2 ## change it to [Fe/H] = -1.2
    n_atoms_total = np.sum(np.power(np.array([10]*len(eps_atoms_to_calculate)),eps_atoms_to_calculate))
    #print(atoms_to_calculate, eps_atoms_to_calculate, n_atoms_total)
    dm = table1['M_init']-table1['M_final']
    wz = np.where(z_kar16 == 0.001)[0][0]
    for i in range(len(table1)):
        wm = np.where(M_kar16 == table1['M_init'][i])[0][0]
        m_left = dm[i]-np.sum(kar16['AGB'][:,wz,wm])
        for atom in atoms_to_calculate:
            wa = np.where(kar16['atomic'] == atom)[0][0]
            wa_eps = np.where(atoms_to_calculate == atom)[0][0]
            el_yield = (m_left)*(10**eps_atoms_to_calculate[wa_eps])/n_atoms_total #????????????????
            #print M_kar16[wm],z_kar16[wz],kar16['atomic'][wa],eps_atoms_to_calculate[wa_eps],m_left,(10**eps_atoms_to_calculate[wa_eps])/n_atoms_total, el_yield
            kar16[wa]['AGB'][wz,wm] = el_yield       

    # cut down the number of elements
    wel = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
    kar16 = kar16[wel]
    # set weight to the atomic mass. Ideally, you would take into account the average mass of the isotopes created
    for j in range(len(kar16['atomic'])):
        kar16[j]['weight'] = atomic_weight[np.where(eps_atomic_num == kar16['atomic'][j])[0][0]]       # divide (isotope mass * yield) by yield
   
    if verbose == True: print('kar16 z=0.001 yields: ', kar16['AGB'][:,0,:])    

    
    #print 'kar16:', kar16['atomic']
    #print kar16['AGB']
    check_array(kar16['AGB'])
    #print 'kar16',kar16.dtype 
    #print kar16['AGB'][0]
    #print kar16['AGB'][-1]
    #if np.where(kar16['AGB']==0.0)[0].size != 0:
    #    print 'ERROR: not available (0.0 yield) metallicities, masses, elements: ', np.where(kar16['AGB']==0.0)
    #print kar16    

    print(kar16['atomic'])
    print(kar16['AGB'][0,1,:])
    return M_kar16, z_kar16, kar16

def initialize_yields_inclBa(yield_path='yields/', r_process_keyword='none', AGB_source = 'cri15', verbose = False):
    #nel, eps_sun, SN_yield, AGB_yield, M_SN, M_HN, z_II, M_AGB, z_AGB
    # r_process_keyword: 'none', 'typical_SN_only', 'rare_event_only', 'both'
    # AGB_source: 'cri15', 'kar16'

    if verbose == True:
        print("structure_name[element]['yield/weight'][metallicity,mass] or structure_name['atomic'][element]")

    eps_sun = np.array([12.00, 10.99, 3.31, 1.42, 2.88, 8.56, 8.05, 8.93, 4.56, 8.09, 6.33, 7.58, 6.47,
                        7.55, 5.45, 7.21, 5.5 , 6.56, 5.12, 6.36, 3.10, 4.99, 4.00, 5.67, 5.39, 7.52, 4.92, 6.25, 4.21, 4.60, 2.88, 3.41,
                        2.90,2.13,1.22,0.51]) 
    atomic_weight = np.array([1.00794, 4.00602, 6.941, 9.012182, 10.811, 12.0107, 14.0067, 
                     15.9994, 18.9984032, 20.1797, 22.98976928, 24.3050, 26.9815386, 
                     28.0355, 30.973762, 32.065, 35.453, 39.948, 39.0983, 40.078, 
                     44.955912, 47.957, 50.9415, 51.9951, 54.9438045, 55.845, 
                     58.933195, 58.6934, 63.546, 65.38, 69.723, 72.64, 87.62, 137.522, 138.9055, 151.964])
    eps_atomic_num = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
                               20,21,22,23,24,25,26,27,28,29,30,31,32,38,56,57,63])
    # elements that we are interested in
    wel = [0, 1, 5, 7, 11, 13, 19, 21, 25,-3]
    if verbose == True:
        print(('solar',eps_sun[wel],eps_atomic_num[wel]))
    #[H,He,C,O,Mg,Si,Ca,Ti,Fe,Ba]
    eps_sun = eps_sun[wel] 
    eps_atomic_num = eps_atomic_num[wel]
    atomic_weight = atomic_weight[wel]
    # read in AGB yield files
    if AGB_source == 'cri15':
        M_AGB, z_AGB, AGB_yield = load_Cristallo15(yield_path, eps_sun, eps_atomic_num, atomic_weight, verbose)
    elif AGB_source == 'kar16':
        M_AGB, z_AGB, AGB_yield = load_Karakas16(yield_path, eps_sun, eps_atomic_num, atomic_weight, verbose)
    else:
        print("ERROR: AGB source unknown.")
        return
    #read in SN yield files
    M_SN = np.array([13.0, 15.0, 18.0, 20.0, 25.0, 30.0, 40.0])
    M_HN = np.array([20.0, 25.0, 30.0, 40.0])
    z_II = np.array([0.0, 0.001, 0.004, 0.02]) 
    iso_string = np.array(['p','d','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si',
                           'P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge'])
    iso_atomic = np.array([1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32])
    atomic = np.unique(iso_atomic)
    if 56 not in atomic:
        atomic = np.concatenate((atomic,[56]))
    SN_yield = np.zeros(len(atomic),dtype=[('atomic','float64'),('II','float64',(len(z_II),len(M_SN))),('weight_II','float64',(len(z_II),len(M_SN))),
                                        ('HN','float64',(len(z_II),len(M_HN))),('weight_HN','float64',(len(z_II),len(M_HN))),('Ia','float64'),
                                        ('weight_Ia','float64')])
    SN_yield['atomic'] = atomic    

    #load SN II data    
    nom06snfile = yield_path+'nom06/tab1.txt'
    data = np.genfromtxt(nom06snfile, names = ['Metallicity','isotope','II_13', 
            'II_15', 'II_18', 'II_20', 'II_25', 'II_30', 'II_40'],dtype=None,
            skip_header=24, encoding='utf_8')
    mass_array = np.array(data[['II_13', 'II_15', 'II_18', 'II_20', 'II_25', 'II_30', 'II_40']].tolist())
    #print(mass_array)
    elname = np.array([data['isotope'][j].strip('0123456789-*^') for j in range(len(data['isotope']))])     #array where the atomic name only remains
    isotope_array = np.array([re.sub("\D","",data['isotope'][k]) for k in range(len(data['isotope']))])
    #print(isotope_array)
    mask = (elname != 'M_final_') & (elname != 'M_cut_')
    for i in np.arange(len(elname))[mask]:     
        wz = np.where(z_II == data['Metallicity'][i])[0]
        el_index = np.where(iso_string == elname[i])[0]
        if (len(el_index) == 1) & (len(wz) == 1):
            atom = iso_atomic[el_index[0]]
            if isotope_array[i] == '': isotope = atom
            else: isotope = int(isotope_array[i])
            wa = np.where(SN_yield['atomic'] == atom)[0][0]
            #print(isotope, atom)
            for mass_i in range(len(M_SN)):
                SN_yield[wa]['weight_II'][wz[0],mass_i] = SN_yield[wa]['weight_II'][wz[0],mass_i]+isotope*mass_array[i,mass_i]     # add isotope mass * yield to SN_yield[atomic]['weight'][metallicity,mass]
                SN_yield[wa]['II'][wz[0],mass_i] = SN_yield[wa]['II'][wz[0],mass_i]+mass_array[i,mass_i]                           # add yield to SN_yield[atomic]['AGB'][metallicity,mass]

    #load HN data
    nom06hnfile = yield_path+'nom06/tab2.txt'
    data = np.genfromtxt(nom06hnfile, names = ['Metallicity','isotope','II_20', 'II_25', 'II_30', 'II_40'],
                         dtype=None,skip_header=21, encoding='utf_8')
    mass_array = np.array(data[['II_20', 'II_25', 'II_30', 'II_40']].tolist())
    elname = np.array([data['isotope'][j].strip('0123456789-*^') for j in range(len(data['isotope']))])     #array where the atomic name only remains
    isotope_array = np.array([re.sub("\D","",data['isotope'][k]) for k in range(len(data['isotope']))])
    mask = (elname != 'M_final_') & (elname != 'M_cut_')
    for i in np.arange(len(elname))[mask]:     
        wz = np.where(z_II == data['Metallicity'][i])[0]
        el_index = np.where(iso_string == elname[i])[0]
        if (len(el_index) == 1) & (len(wz) == 1):
            atom = iso_atomic[el_index[0]]
            if isotope_array[i] == '': isotope = atom
            else: isotope = int(isotope_array[i])
            wa = np.where(SN_yield['atomic'] == atom)[0][0]
            for mass_i in range(len(M_HN)):
                SN_yield[wa]['weight_HN'][wz[0],mass_i] = SN_yield[wa]['weight_HN'][wz[0],mass_i]+isotope*mass_array[i,mass_i]     # add isotope mass * yield to SN_yield[atomic]['weight'][metallicity,mass]
                SN_yield[wa]['HN'][wz[0],mass_i] = SN_yield[wa]['HN'][wz[0],mass_i]+mass_array[i,mass_i]                           # add yield to SN_yield[atomic]['AGB'][metallicity,mass]

    #load SN Ia data
    #first load He yields from /nom06/tab3.txt - He yield is not included in iwa99
    nom06Iafile = yield_path+'nom06/tab3.txt'
    data = np.genfromtxt(nom06Iafile, names = ['isotope', 'yield'],dtype=None,
                         skip_header=16,usecols=(0,5), encoding='utf_8')
    elname = np.array([data['isotope'][j].strip('0123456789-*^') for j in range(len(data['isotope']))])     #array where the atomic name only remains
    isotope_array = np.array([re.sub("\D","",data['isotope'][k]) for k in range(len(data['isotope']))])
    for i in np.arange(len(elname)): 
        if elname[i] == 'He':    
            atom = 2
            if isotope_array[i] == '': isotope = atom
            else: isotope = int(isotope_array[i])
            wa = np.where(SN_yield['atomic'] == atom)[0][0]
            SN_yield[wa]['weight_Ia'] = SN_yield[wa]['weight_Ia']+isotope*data['yield'][i]     # add isotope mass * yield to SN_yield[atomic]['weight'][metallicity,mass]
            SN_yield[wa]['Ia'] = SN_yield[wa]['Ia']+data['yield'][i]                           # add yield to SN_yield[atomic]['AGB'][metallicity,mass]    

    #then add yields with iwa99/tab3.txt
    iwa99file = yield_path+'iwa99/tab3.txt'
    data = np.genfromtxt(iwa99file, names = ['isotope','yield'],dtype=None,
                         usecols=(0,2), encoding='utf_8')
    elname = np.array([data['isotope'][j].strip('0123456789-*^') for j in range(len(data['isotope']))])     #array where the atomic name only remains
    isotope_array = np.array([re.sub("\D","",data['isotope'][k]) for k in range(len(data['isotope']))])
    for i in np.arange(len(elname)):     
        el_index = np.where(iso_string == elname[i])[0]
        if (len(el_index) == 1):
            atom = iso_atomic[el_index[0]]
            if isotope_array[i] == '': isotope = atom
            else: isotope = int(isotope_array[i])

            wa = np.where(SN_yield['atomic'] == atom)[0][0]
            SN_yield[wa]['weight_Ia'] = SN_yield[wa]['weight_Ia']+isotope*data['yield'][i]     # add isotope mass * yield to SN_yield[atomic]['weight'][metallicity,mass]
            SN_yield[wa]['Ia'] = SN_yield[wa]['Ia']+data['yield'][i]                           # add yield to SN_yield[atomic]['AGB'][metallicity,mass]    
        
    SN_yield['weight_II'][SN_yield['II']>0] = SN_yield['weight_II'][SN_yield['II']>0]/SN_yield['II'][SN_yield['II']>0]                                                     # divide (isotope mass * yield) by yield
    SN_yield['weight_HN'][SN_yield['HN']>0] = SN_yield['weight_HN'][SN_yield['HN']>0]/SN_yield['HN'][SN_yield['HN']>0]
    SN_yield['weight_Ia'][SN_yield['Ia']>0] = SN_yield['weight_Ia'][SN_yield['Ia']>0]/SN_yield['Ia'][SN_yield['Ia']>0]
    
    wel = [0, 1, 5, 7, 11, 13, 19, 21, 25, -1]
    #[H,He,C,O,Mg,Si,Ca,Ti,Fe,Ba]
    nel = len(wel)
    SN_yield = SN_yield[wel]
    if verbose == True:
        print(('SN_yield',SN_yield.dtype, SN_yield['atomic'])) 
            
    # Add in barium abundances from SN/rare r-process event
    if r_process_keyword == 'none':
        pass
    else:
        if r_process_keyword in ['typical_SN_only','both']:
            # barium yield for weak r-process event in Li et al. 2014
            SN_yield['weight_II'][9] = np.mean(AGB_yield['weight'][9])
            li14_weakr = [1.38e-8, 2.83e-8, 5.38e-8, 6.84e-8, 9.42e-8, 0, 0 ]
            #li14_weakr[-2] = li14_weakr[-3] + (li14_weakr[-3]-li14_weakr[-4])/(M_SN[-3]-M_SN[-4])*(M_SN[-2]-M_SN[-3])
            #print li14_weakr[-2], li14_weakr[-3]*M_SN[-2]/M_SN[-3]
            #li14_weakr[-1] = li14_weakr[-3] + (li14_weakr[-3]-li14_weakr[-4])/(M_SN[-3]-M_SN[-4])*(M_SN[-1]-M_SN[-3])
            li14_weakr[-2] = li14_weakr[-3]*M_SN[-2]/M_SN[-3]
            li14_weakr[-1] = li14_weakr[-2]*M_SN[-1]/M_SN[-2]
    
                    #x_ej[len(x_ej)-1] *= M_SN_ej[len(M_SN_ej)-1]/M_SN_ej[len(M_SN_ej)-2]       #ejected mass for 100 M_sun SN (M_sun)
            for i in range(len(z_II)):
                SN_yield['II'][9][i,:] = li14_weakr 
                
            #print SN_yield['II'][9]
        if r_process_keyword in ['rare_event_only','both']:   
            SN_yield['weight_Ia'][9] = np.mean(AGB_yield['weight'][9])
            #SN_yield['Ia'][9] = 2.3e-6
            SN_yield['Ia'][9] = 2e-6
            # this is the average barium yield for a main r-process event in Li et al. 2014
            # since NSM would have a similar DTD as SN Ia, this is a proxy for including NSM. Scale barium
            # yield up and reduce rate of events proportionally to compare to models
        if r_process_keyword not in ['none','typical_SN_only','rare_event_only','both']:
            print("ERROR: INVALID R-PROCESS KEYWORD. No r-process synthesis included.")

    check_array(SN_yield['weight_II'])
    check_array(SN_yield['weight_HN'])
    check_array(SN_yield['weight_Ia'])

    #perets = np.zeros(8,dtype=[('dotIa_1','float64'),('dotIa_2','float64')])
    #perets[3:8]['dotIa_1'] = [4.4e-5, 0.001, 0.07, 0.0059, 0.00025]
    #perets[3:8]['dotIa_2'] = [2.7e-5, 0.00058, 0.052, 0.021, 0.0025]

    #format for gce.pro:
    #'atomic' =  1       2       6       8      12      14      20      22      26    56
    
    return nel, eps_sun, SN_yield, AGB_yield, M_SN, M_HN, z_II, M_AGB, z_AGB#, perets
              
if __name__ == "__main__":

    initialize_yields_inclBa(AGB_source='kar16')