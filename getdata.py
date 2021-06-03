"""
getdata.py

Compiles spectroscopic data from data/ folder.
"""

import numpy as np
from astropy.io import ascii
from numpy.random import default_rng

# Backend for matplotlib on mahler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Do some formatting stuff with matplotlib
from matplotlib import rc
rc('font', family='serif')
rc('axes', labelsize=14) 
rc('xtick', labelsize=10)
rc('ytick', labelsize=10)
rc('xtick.major', size=8)
rc('ytick.major', size=8)
rc('legend', fontsize=12, frameon=False)
rc('text',usetex=True)
rc('xtick',direction='in')
rc('ytick',direction='in')
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

# Systematic errors
# Fe/H and alpha/Fe calculated by Evan on 12/28/17
# C/Fe from Kirby+15, Ba/Fe from Duggan+18, Mn/Fe from de los Reyes+20
syserr = {'Fe':0.10103081, 'alpha':0.084143983, 'Mg':0.076933658,
        'Si':0.099193360, 'Ca':0.11088295, 'Ti':0.10586739,
        'C':0.100, 'Ba':0.100, 'Mn':0.100}

def getdata(galaxy, source='deimos', c=False, ba=False, mn=False, eu=False, outlier_reject=True, 
    removerprocess='statistical', feh_denom=True):
    """Compiles observed abundances from literature tables.

    Args:
        galaxy (str): dSph galaxy to get data from (options: 'Scl')
        source (str): which set of abundances to use (options: 'deimos', 'dart')
        c, ba, mn, eu (bool): Keywords to decide which elements to include 
                        along with [Fe/H], [Mg/Fe], [Si/Fe], [Ca/Fe]
        outlier_reject (bool): If 'True', do remove high-C and high-Ba outliers
        removerprocess (str): If not 'None', subtract r-process contribution using Duggan+18 method
                        (options: 'individual' = individual stars; 'statistical' = fit line to [Ba/Eu] vs [Fe/H])
        feh_denom (bool): If 'True', return [Fe/H] and other elements as [X/Fe];
                        otherwise, don't return [Fe/H] and convert [X/Fe] -> [X/Mg]

    Returns:
        data, errs (array): Observed data and errors
    """  

    # Get info for r-process correction if needed
    if removerprocess=='statistical':
        # Get data from DART table
        darttable = ascii.read("data/hill19.dat")
        bafe = darttable['[Ba/Fe]'].reshape(darttable['[Ba/Fe]'].shape[0],1)
        eufe = darttable['[Eu/Fe]'].reshape(darttable['[Eu/Fe]'].shape[0],1)
        ba_errs = darttable['[Ba/Fe]err'].reshape(darttable['[Ba/Fe]err'].shape[0],1)
        eu_errs = darttable['[Eu/Fe]err'].reshape(darttable['[Eu/Fe]err'].shape[0],1)

        # Fit line to [Ba/Eu] vs [Fe/H]
        feh = darttable['[Fe/H]'].reshape(darttable['[Fe/H]'].shape[0],1)
        fe_errs = darttable['[Fe/H]err'].reshape(darttable['[Fe/H]err'].shape[0],1)

        # Use MC bootstrapping method
        idx = np.where((feh > -990.) & (bafe > -90.) & (eufe > -90.) & (fe_errs > 0.) & (ba_errs > 0.) & (eu_errs > 0.))[0]
        rng = default_rng()
        niter = 10000
        fe_iter = rng.normal(loc=feh[idx], scale=fe_errs[idx], size=(feh[idx].shape[0], niter))
        ba_iter = rng.normal(loc=bafe[idx], scale=ba_errs[idx], size=(bafe[idx].shape[0], niter))
        eu_iter = rng.normal(loc=eufe[idx], scale=eu_errs[idx], size=(eufe[idx].shape[0], niter))
        coeffs = np.zeros((2,niter))
        for i in range(niter):
            coeffs[:,i] = np.polyfit(fe_iter[:,i], ba_iter[:,i]-eu_iter[:,i], 1)
        print(np.percentile(coeffs, 50, axis=1))
        p = np.poly1d(np.percentile(coeffs, 50, axis=1))

        # Test plot
        '''
        plt.errorbar(feh[idx], bafe[idx]-eufe[idx], xerr=fe_errs[idx], yerr=np.sqrt(ba_errs[idx]**2. + eu_errs[idx]**2.), 
            mfc='white', mec=plt.cm.Set3(3), ecolor=plt.cm.Set3(3), linestyle='None', marker='o', markersize=6, linewidth=0.5)
        plt.plot([-2.5,-0.75], p([-2.5,-0.75]), ls='-', color='k', label=r'$y={:.2f}x+{:.2f}$'.format(*p))
        plt.xlim((-2.5,-0.75))
        plt.xlabel('[Fe/H]')
        plt.ylabel('[Ba/Eu]')
        plt.legend(loc='upper left')
        plt.savefig('plots/baeu_rcorrection.png', bbox_inches='tight')
        plt.show()
        '''
    
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

        # List of elements that's in data table
        elems = ['fe','mg','si','ca']

        # Cross-match data
        finaldata = []
        finalerrs = []

        for i in range(len(names)):

            newdata = data[i]
            newerrs = errs[i]

            # Cross-match with carbon table if needed
            if c: 
                elems.append('c')
                if names[i] in table_c['Name']:
                    c_idx = np.where(table_c['Name'] == names[i])
                    newdata = np.concatenate((newdata,table_c['[C/Fe]c'][c_idx]))
                    newerrs = np.concatenate((newerrs,np.sqrt(table_c['e_[C/Fe]'][c_idx]**2. + syserr['C'])))
                else:
                    newdata = np.concatenate((newdata,[-999.]))
                    newerrs = np.concatenate((newerrs,[-999.]))

            # Cross-match with barium table if needed
            if ba: 
                elems.append('ba')
                if names[i] in table_ba['Name'] and removerprocess != 'individual':
                    ba_idx = np.where(table_ba['Name'] == names[i])

                    bafe = table_ba['[Ba/Fe]'][ba_idx]
                    bafe_err = np.sqrt(table_ba['e_[Ba/Fe]'][ba_idx]**2. + syserr['Ba'])

                    if removerprocess == 'statistical':
                        # Compute fraction of r-process elements
                        baeu = p(table['[Fe/H]'][ba_idx])
                        rfrac = (10.**(-1.062)/10.**(2.209) - 10.**(-(baeu+(2.13-0.51))))/((10.**(-1.062)/10.**(2.209)) - (10.**(0.494)/10.**(1.446)))
                        if rfrac < 0.: rfrac = 0.
                        elif rfrac > 1.: rfrac = 1.

                        # Compute s-process contribution to [Ba/Fe]
                        if rfrac < 1:
                            bafe += np.log10(1.-rfrac)
                        else:
                            bafe = [-999.]
                            bafe_err = [-999.]

                    newdata = np.concatenate((newdata,bafe))
                    newerrs = np.concatenate((newerrs,bafe_err))
                else:
                    newdata = np.concatenate((newdata,[-999.]))
                    newerrs = np.concatenate((newerrs,[-999.]))

            # Cross-match with manganese table if needed
            if mn: 
                elems.append('mn')
                if names[i] in table_mn['ID']:
                    mn_idx = np.where(table_mn['ID'] == names[i])
                    newdata = np.concatenate((newdata,table_mn['MnFe'][mn_idx]))
                    newerrs = np.concatenate((newerrs,np.sqrt(table_mn['e_MnFe'][mn_idx]**2. + syserr['Mn'])))
                else:
                    newdata = np.concatenate((newdata,[-999.]))
                    newerrs = np.concatenate((newerrs,[-999.]))

            # No Eu data, so just put in an empty row for now
            if eu: 
                elems.append('eu')
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
            elems = ['fe','mg','si','ca']

            # Cross-match with carbon table if needed (currently assuming no C available)
            if c:
                elems.append('c')

                c_data = -999.0 * np.ones_like(table['[Fe/H]'])
                c_data = c_data.reshape(c_data.shape[0],1)
                data = np.hstack([data, c_data])

                c_errs = -999.0 * np.ones_like(table['[Fe/H]'])
                c_errs = c_errs.reshape(c_errs.shape[0],1)
                errs = np.hstack([errs, c_errs])

            # Add barium if needed
            if ba:
                elems.append('ba')

                if removerprocess is not None:
                    bafe = table['[Ba/Fe]'].reshape(table['[Ba/Fe]'].shape[0],1)
                    eufe = table['[Eu/Fe]'].reshape(table['[Eu/Fe]'].shape[0],1)
                    ba_errs = table['[Ba/Fe]err'].reshape(table['[Ba/Fe]err'].shape[0],1)
                    eu_errs = table['[Eu/Fe]err'].reshape(table['[Eu/Fe]err'].shape[0],1)

                    if removerprocess=='individual':
                        # Compute fraction of r-process elements
                        rfrac = (10.**(-1.062)/10.**(2.209) - 10.**(-((bafe-eufe)+(2.13-0.51))))/((10.**(-1.062)/10.**(2.209)) - (10.**(0.494)/10.**(1.446)))
                        rfrac[rfrac < 0.] = 0.
                        rfrac[rfrac > 1.] = 1.

                        # Compute s-process contribution to [Ba/Fe]
                        bafe_s = bafe + np.log10(1.-rfrac)
                        bafe_s[~np.isfinite(bafe_s)] = -999.
                        bafe_s[eufe < -90] = -999.

                    elif removerprocess=='statistical':
                        # Compute fraction of r-process elements
                        baeu = p(feh)
                        baeu[np.where(feh < -990.)] = -999.
                        rfrac = (10.**(-1.062)/10.**(2.209) - 10.**(-(baeu+(2.13-0.51))))/((10.**(-1.062)/10.**(2.209)) - (10.**(0.494)/10.**(1.446)))
                        rfrac[rfrac < 0.] = 0.
                        rfrac[rfrac > 1.] = 1.

                        # Compute s-process contribution to [Ba/Fe]
                        bafe_s = bafe + np.log10(1.-rfrac)
                        bafe_s[~np.isfinite(bafe_s)] = -999.

                    # Add to tables
                    data = np.hstack([data,bafe_s])
                    errs = np.hstack([errs,ba_errs])

                else:
                    ba_data = table['[Ba/Fe]'].reshape(table['[Ba/Fe]'].shape[0],1)
                    data = np.hstack([data,ba_data])

                    ba_errs = table['[Ba/Fe]err'].reshape(table['[Ba/Fe]err'].shape[0],1)
                    errs = np.hstack([errs,ba_errs])

            # Cross-match with manganese table if needed
            if mn:
                elems.append('mn')
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

            # Add europium if needed
            if eu:
                elems.append('eu')
                if removerprocess:
                    bafe = table['[Ba/Fe]'].reshape(table['[Ba/Fe]'].shape[0],1)
                    eufe = table['[Eu/Fe]'].reshape(table['[Eu/Fe]'].shape[0],1)
                    ba_errs = table['[Ba/Fe]err'].reshape(table['[Ba/Fe]err'].shape[0],1)
                    eu_errs = table['[Eu/Fe]err'].reshape(table['[Eu/Fe]err'].shape[0],1)

                    # Compute fraction of r-process elements
                    rfrac = (10.**(-1.062)/10.**(2.209) - 10.**(-((bafe-eufe)+(2.13-0.51))))/((10.**(-1.062)/10.**(2.209)) - (10.**(0.494)/10.**(1.446)))
                    rfrac[rfrac < 0.] = 0.
                    rfrac[rfrac > 1.] = 1.

                    # Compute s-process contribution to [Ba/Fe]
                    eufe_s = eufe + np.log10(1.-rfrac)
                    eufe_s[~np.isfinite(eufe_s)] = -999.
                    eufe_s[eufe < -90] = -999.

                    # Add to tables
                    data = np.hstack([data,eufe_s])
                    errs = np.hstack([errs,eu_errs])

                else:
                    eu_data = table['[Eu/Fe]'].reshape(table['[Eu/Fe]'].shape[0],1)
                    data = np.hstack([data,eu_data])

                    eu_errs = table['[Eu/Fe]err'].reshape(table['[Eu/Fe]err'].shape[0],1)
                    errs = np.hstack([errs,eu_errs])

            # Mask out non-detections
            data[np.isclose(data,-99)] = -999.0
            errs[np.isclose(errs,9.9)] = -999.0

        data = data.T
        errs = errs.T

    # If not using Fe, convert data to [X/Mg] and remove [Fe/H] from the equation
    if feh_denom==False:

        # Temporary (needed for plotting): add another row to data table, including [Fe/Mg]
        #data = np.vstack((data, data[0,:]))
        #errs = np.vstack((errs, errs[0,:]))

        for i in range(len(data[1,:])):

            if data[1,i] > -990:
                # Convert [X/Fe] -> [X/Mg]
                for elem in range(2,data.shape[0]):
                    data[elem,i] = data[elem,i] - data[1,i]
                    errs[elem,i] = np.sqrt(errs[elem,i]**2. + errs[1,i]**2.)

                # Temporary: convert [Fe/H] -> [Fe/Mg]
                #data[-1,i] = -data[1,i]
                #errs[-1,i] = errs[1,i]

                # Convert [Mg/Fe] -> [Mg/H]
                data[1,i] = data[1,i] + data[0,i]
                errs[1,i] = np.sqrt(errs[1,i]**2. + errs[0,i]**2.)
        
            else:
                data[:,i] = -999.


        # Remove [Fe/H] from data
        data = np.delete(data,0,axis=0)
        errs = np.delete(errs,0,axis=0)
        del elems[0]

    return data, errs, elems

if __name__ == "__main__":

    # Test to make sure script is working
    #data, errs = getdata('Scl', source='deimos', c=True, ba=True, mn=True, eu=True, feh_denom=True, removerprocess='statistical')
    elem_dart, delem_dart, elems = getdata(galaxy='Scl', source='dart', c=True, ba=True, removerprocess='statistical', feh_denom=True) #, eu=baeu)   
    print(elems)
    # print(elem_dart[-1,:])