"""
getdata.py

Compiles spectroscopic data from data/ folder.
"""

import numpy as np
from astropy.io import ascii

def getdata(galaxy, c=False, ba=False, mn=False):
    
    # Open files
    table = ascii.read("data/kirby10.dat").filled(-999)
    table_c = ascii.read("data/kirby15.dat").filled(-999)
    table_ba = ascii.read("data/duggan18.dat")
    table_mn = ascii.read("data/delosreyes20.dat")

    # Create tables that include all abundances and errors
    idx = np.where(table['dSph']=='Scl')
    names = table['Name'][idx]
    data = np.asarray([table['[Fe/H]'][idx], table['[Mg/Fe]'][idx], table['[Si/Fe]'][idx], table['[Ca/Fe]'][idx]]).T
    errs = np.asarray([table['e_[Fe/H]'][idx], table['e_[Mg/Fe]'][idx], table['e_[Si/Fe]'][idx], table['e_[Ca/Fe]'][idx]]).T

    # Clean the data
    finaldata = []
    finalerrs = []
    finalnames = []

    for i in range(len(names)):

        newdata = data[i]
        newerrs = errs[i]

        # Remove any rows where we don't have complete data for [Fe/H] and [alpha/Fe]
        if ~np.any(np.isclose(data[i],-999.)): 
            finaldata.append(newdata)
            finalerrs.append(newerrs)
            #finalnames.append(names[i])

        # Cross-match with carbon table if needed
        if c: 
            if names[i] in table_c['Name']:
                c_idx = np.where(table_c['Name'] == names[i])
                newdata = np.concatenate((newdata,table_c['[C/Fe]'][c_idx]))
                newerrs = np.concatenate((newerrs,table_c['e_[C/Fe]'][c_idx]))
            else:
                continue

        # Cross-match with barium table if needed
        if ba: 
            if names[i] in table_ba['Name']:
                ba_idx = np.where(table_ba['Name'] == names[i])
                newdata = np.concatenate((newdata,table_ba['[Ba/Fe]'][ba_idx]))
                newerrs = np.concatenate((newerrs,table_ba['e_[Ba/Fe]'][ba_idx]))
            else:
                continue

        # Cross-match with manganese table if needed
        if mn: 
            if names[i] in table_mn['ID']:
                mn_idx = np.where(table_mn['ID'] == names[i])
                newdata = np.concatenate((newdata,table_mn['MnFe'][mn_idx]))
                newerrs = np.concatenate((newerrs,table_mn['e_MnFe'][mn_idx]))
            else:
                continue

    data = np.asarray(finaldata).T
    errs = np.asarray(finalerrs).T
    #names = np.asarray(finalnames)

    return data, errs

if __name__ == "__main__":

    # Test to make sure script is working
    data, errs = getdata('Scl', c=False, ba=False)
    print(data.shape)