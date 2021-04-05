"""
getdata.py

Compiles spectroscopic data from data/ folder.
"""

import numpy as np
from astropy.io import ascii

# Systematic errors
# Fe/H and alpha/Fe calculated by Evan on 12/28/17
# C/Fe from Kirby+15, Ba/Fe from Duggan+18, Mn/Fe from de los Reyes+20
syserr = {'Fe':0.10103081, 'alpha':0.084143983, 'Mg':0.076933658,
        'Si':0.099193360, 'Ca':0.11088295, 'Ti':0.10586739,
        'C':0.100, 'Ba':0.100, 'Mn':0.100}

def getdata(galaxy, source='deimos', c=False, ba=False, mn=False, eu=False, outlier_reject=True):
    """Compiles observed abundances from literature tables.

    Args:
        galaxy (str): dSph galaxy to get data from (options: 'Scl')
        source (str): which set of abundances to use (options: 'deimos', 'dart')
        c, ba, mn, eu (bool): Keywords to decide which elements to include 
                        along with [Fe/H], [Mg/Fe], [Si/Fe], [Ca/Fe]
        outlier_reject (bool): If 'True', do remove high-C and high-Ba outliers

    Returns:
        data, errs (array): Observed data and errors
    """  
    
    if source=='deimos':
        # Open files
        table = ascii.read("data/kirby10.dat").filled(-999)
        table_c = ascii.read("data/kirby15.dat").filled(-999)
        table_ba = ascii.read("data/duggan18.dat")
        table_mn = ascii.read("data/delosreyes20.dat")

        # Create tables that include all abundances and errors
        idx = np.where(table['dSph']=='Scl')
        names = table['Name'][idx]
        data = np.asarray([table['[Fe/H]'][idx], table['[Mg/Fe]'][idx], table['[Si/Fe]'][idx], table['[Ca/Fe]'][idx]]).T
        errs = np.asarray([np.sqrt(table['e_[Fe/H]'][idx]**2. + syserr['Fe']**2.), 
                        np.sqrt(table['e_[Mg/Fe]'][idx]**2. + syserr['Mg']**2.),
                        np.sqrt(table['e_[Si/Fe]'][idx]**2. + syserr['Si']**2.),
                        np.sqrt(table['e_[Ca/Fe]'][idx]**2. + syserr['Ca']**2.)]).T

        # Cross-match data
        finaldata = []
        finalerrs = []

        for i in range(len(names)):

            newdata = data[i]
            newerrs = errs[i]

            # Cross-match with carbon table if needed
            if c: 
                if names[i] in table_c['Name']:
                    c_idx = np.where(table_c['Name'] == names[i])
                    newdata = np.concatenate((newdata,table_c['[C/Fe]c'][c_idx]))
                    newerrs = np.concatenate((newerrs,np.sqrt(table_c['e_[C/Fe]'][c_idx]**2. + syserr['C'])))
                else:
                    newdata = np.concatenate((newdata,[-999.]))
                    newerrs = np.concatenate((newerrs,[-999.]))

            # Cross-match with barium table if needed
            if ba: 
                if names[i] in table_ba['Name']:
                    ba_idx = np.where(table_ba['Name'] == names[i])
                    newdata = np.concatenate((newdata,table_ba['[Ba/Fe]'][ba_idx]))
                    newerrs = np.concatenate((newerrs,np.sqrt(table_ba['e_[Ba/Fe]'][ba_idx]**2. + syserr['Ba'])))
                else:
                    newdata = np.concatenate((newdata,[-999.]))
                    newerrs = np.concatenate((newerrs,[-999.]))

            # Cross-match with manganese table if needed
            if mn: 
                if names[i] in table_mn['ID']:
                    mn_idx = np.where(table_mn['ID'] == names[i])
                    newdata = np.concatenate((newdata,table_mn['MnFe'][mn_idx]))
                    newerrs = np.concatenate((newerrs,np.sqrt(table_mn['e_MnFe'][mn_idx]**2. + syserr['Mn'])))
                else:
                    newdata = np.concatenate((newdata,[-999.]))
                    newerrs = np.concatenate((newerrs,[-999.]))

            finaldata.append(newdata)
            finalerrs.append(newerrs)

        data = np.asarray(finaldata).T
        errs = np.asarray(finalerrs).T

        # Do outlier rejection
        if outlier_reject:
            for i in range(len(names)):
                if (names[i] in table_c['Name']) and (names[i] in table_ba['Name']):
                    c_idx = np.where(table_c['Name'] == names[i])
                    c_data = table_c['[C/Fe]c'][c_idx]

                    ba_idx = np.where(table_ba['Name'] == names[i])
                    ba_data = table_ba['[Ba/Fe]'][ba_idx]

                    if c_data > 0.2 and ba_data > 0.2:
                        data = np.delete(data, i, axis=1)
                        errs = np.delete(errs, i, axis=1)

    elif source=='dart':
        if galaxy=='Scl':
            # Open files
            table = ascii.read("data/hill19.dat")
            table_mn = ascii.read("data/north12.dat")

            # Create output table
            names = table['Star']
            data = np.asarray([table['[Fe/H]'], table['[Mg/Fe]'], table['[Si/Fe]'], table['[Ca/Fe]']]).T
            errs = np.asarray([table['[Fe/H]err'], table['[Mg/Fe]err'], table['[Si/Fe]err'], table['[Ca/Fe]err']]).T

            # Cross-match with carbon table if needed (currently assuming no C available)
            if c:
                c_data = -999.0 * np.ones_like(table['[Fe/H]'])
                c_data = c_data.reshape(c_data.shape[0],1)
                data = np.hstack([data, c_data])

                c_errs = -999.0 * np.ones_like(table['[Fe/H]'])
                c_errs = c_errs.reshape(c_errs.shape[0],1)
                errs = np.hstack([errs, c_errs])

            # Add barium if needed
            if ba:
                ba_data = table['[Ba/Fe]'].reshape(table['[Ba/Fe]'].shape[0],1)
                data = np.hstack([data,ba_data])

                ba_errs = table['[Ba/Fe]err'].reshape(table['[Ba/Fe]err'].shape[0],1)
                errs = np.hstack([errs,ba_errs])

            # Cross-match with manganese table if needed
            if mn:
                finaldata = []
                finalerrs = []

                for i in range(len(names)):

                    newdata = data[i]
                    newerrs = errs[i]

                    if names[i] in table_mn['Name']:
                        mn_idx = np.where(table_mn['Name'] == names[i])
                        newdata = np.concatenate((newdata,table_mn['[Mn/Fe]'][mn_idx]))
                        newerrs = np.concatenate((newerrs,np.sqrt(table_mn['error([Mn/Fe])'][mn_idx]**2.)))
                    else:
                        newdata = np.concatenate((newdata,[-999.]))
                        newerrs = np.concatenate((newerrs,[-999.]))

                    finaldata.append(newdata)
                    finalerrs.append(newerrs)

                data = np.asarray(finaldata)
                errs = np.asarray(finalerrs)

            # Add barium if needed
            if eu:
                eu_data = table['[Eu/Fe]'].reshape(table['[Eu/Fe]'].shape[0],1)
                data = np.hstack([data,eu_data])

                eu_errs = table['[Eu/Fe]err'].reshape(table['[Eu/Fe]err'].shape[0],1)
                errs = np.hstack([errs,eu_errs])

            # Mask out non-detections
            data[np.isclose(data,-99)] = -999.0
            errs[np.isclose(errs,9.9)] = -999.0

        data = data.T
        errs = errs.T

    return data, errs

if __name__ == "__main__":

    # Test to make sure script is working
    data, errs = getdata('Scl', source='dart', c=True, ba=True, mn=True, eu=True)
    print(data[:,0])