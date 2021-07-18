"""
isochrone.py

Some test code to mess around with isochrones
"""

# Import packages for plotting - comment out if not on stravinsky
import cycler
import cmasher as cmr

#Backend for python3 on stravinsky
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

# Import other packages
import numpy as np

def readlifetimes(grid='dartmouth'):
    """ Read stellar lifetimes from isochrone grid. """

    if grid=='dartmouth':
        # Directory name
        griddir = '/home/enk/m31/photmetal/DartmouthIsochrones/sdss/'
        #griddir = ''

        # Metallicities
        metals = [-2.5,-2.0,-1.5,-1.0,-0.5,0.,0.2,0.3,0.5]  # [Fe/H]
        alphas = [-0.2,0.,0.2,0.4,0.6,0.8]

        # Array in which to save stellar lifetime info
        masses = np.zeros((len(metals), len(alphas), 16+37-1))
        ages = np.zeros((len(metals), len(alphas), 16+37-1))

        # Loop over all relevant files
        for alpha_idx, alpha in enumerate(alphas):
            alphaname = str(int(abs(alpha*10)))
            if alpha < 0:
                alphaname = 'm'+alphaname
            else:
                alphaname = 'p'+alphaname

            for metal_idx, metal in enumerate(metals):
                metalname = str(int(abs(metal*10))).zfill(2)
                if metal < 0:
                    metalname = 'm'+metalname
                else:
                    metalname = 'p'+metalname

                try:
                    # Open file with first range of ages (0.25-1 Gyr)
                    filename = griddir+'feh'+metalname+'afe'+alphaname+'.ugriz_2'
                    agecounter = 0
                    with open(filename,'r') as file:  
                        lines = file.readlines()
                        for idx, line in enumerate(lines):
                            if line != '\n':
                                ln_list = line.split()
                                if ln_list[0].startswith('#AGE='):
                                    ages[metal_idx,alpha_idx,agecounter] = ln_list[1]
                                    agecounter += 1
                            elif lines[idx-1] != '\n':
                                ln_list = lines[idx-1].split()
                                masses[metal_idx,alpha_idx,agecounter-1] = ln_list[1]

                    # Open file with second range of ages (>1 Gyr)
                    agecounter -= 1  # Don't repeat 1 Gyr
                    with open(filename[:-2],'r') as file:  
                        lines = file.readlines()
                        for idx, line in enumerate(lines):
                            if line != '\n':
                                if line.startswith('#AGE='):
                                    ages[metal_idx,alpha_idx,agecounter] = float(line[5:11])
                                    agecounter += 1
                            elif lines[idx-1] != '\n':
                                ln_list = lines[idx-1].split()
                                masses[metal_idx,alpha_idx,agecounter-1] = ln_list[1]
                        ln_list = line.split()
                        masses[metal_idx,alpha_idx,-1] = ln_list[1]
                except:
                    pass

        np.save(grid+'_ages', ages)
        np.save(grid+'_masses', masses)

    return

def plotlifetimes(grid='dartmouth'):
    """ Plot stellar lifetimes as a function of mass. """

    # Get data
    ages = np.load(grid+'_ages.npy')
    masses = np.load(grid+'_masses.npy')

    if grid=='dartmouth':
        metals = [-2.5,-2.0,-1.5,-1.0,-0.5,0.,0.2,0.3,0.5]  # [Fe/H]
        alphas = [-0.2,0.,0.2,0.4,0.6,0.8]  # [alpha/Fe]

        # Define colormaps
        metalcolor = cmr.bubblegum(np.linspace(0,1,len(metals),endpoint=True))
        alphals = [':','-','--','dashdot',(0,(1,10)),(0,(5,10))]

        # Plot metallicities at fixed alpha
        plt.figure()
        for metal_idx, metal in enumerate(metals):
            age = ages[metal_idx,1,:]
            mass = masses[metal_idx,1,:]
            if np.any(np.isclose(age, 0)) or np.any(np.isclose(mass, 0)):
                print('test', metal, alphas[1])
            else:
                plt.plot(mass, age, color=metalcolor[metal_idx], ls=alphals[1], label=r"[Fe/H] = {:.2f}".format(metal))
        legend = plt.legend(title=r"[$\alpha$/Fe] = {:.1f}".format(alphas[1]))
        plt.setp(legend.get_title(),fontsize='16')
        plt.xlabel(r'Mass ($M_{\odot}$)', fontsize=14)
        plt.ylabel(r'Lifetime (Gyr)', fontsize=14)
        plt.savefig(grid+'_lifetimes_fe.png', bbox_inches='tight')
        plt.show()

        # Plot alphas at fixed metallicity
        plt.figure()
        for alpha_idx, alpha in enumerate(alphas):
            age = ages[2,alpha_idx,:]
            mass = masses[2,alpha_idx,:]
            if np.any(np.isclose(age, 0)) or np.any(np.isclose(mass, 0)):
                print('test', metals[2], alpha)
            else:
                plt.plot(mass, age, color=metalcolor[2], ls=alphals[alpha_idx], label=r"[$\alpha$/Fe] = {:.1f}".format(alpha))
        legend = plt.legend(title=r"[Fe/H] = {:.2f}".format(metals[2]))
        plt.setp(legend.get_title(),fontsize='16')
        plt.xlabel(r'Mass ($M_{\odot}$)', fontsize=14)
        plt.ylabel(r'Lifetime (Gyr)', fontsize=14)
        plt.savefig(grid+'_lifetimes_alpha.png', bbox_inches='tight')
        plt.show()

    return

def plotbpass():
    """ Plot BPASS stellar lifetimes. """

    # Metallicities
    Z = [1.00E-05, 0.0001, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01, 0.014, 0.02, 0.03, 0.04]

    # Create array to hold data
    data = np.zeros((2, len(Z), 283))

    # Read in BPASS file
    with open('bpass.csv','r') as file:  
        lines = file.readlines()
        currentfeh = 0
        currentcounter = 0
        for idx, line in enumerate(lines):
            ls = line.split(',')
            if ls[1].startswith('LOAD') or ls[1]=='0' or ls[1].startswith('Metal'):
                pass
            elif np.any(np.isclose(Z,float(ls[1]))):
                fehidx = np.where(np.isclose(Z,float(ls[1])))[0]
                if fehidx[0]==currentfeh:
                    # Put in initial stellar mass (Msun)
                    data[0, fehidx, currentcounter] = float(ls[2])
                    # Put in H-burning lifetime (yr)
                    data[1, fehidx, currentcounter] = float(ls[3])
                    
                    # Increment current counter
                    currentcounter += 1
                else:
                    # Put in initial stellar mass (Msun)
                    data[0, fehidx, 0] = float(ls[2])
                    # Put in H-burning lifetime (yr)
                    data[1, fehidx, 0] = float(ls[3])

                    # Increment current [Fe/H]
                    currentcounter = 1
                    currentfeh += 1

        # Compute relevant metallicities
        feharray = np.log10(np.array(Z)/0.0134) # Solar metallicity from Asplund+2009
        goodidx = np.where(feharray < -0.5)[0]

        # Define colormaps
        metalcolor = cmr.gem_r(np.linspace(0,1,len(feharray[goodidx]),endpoint=True))
        metalcolor = plt.cm.coolwarm(np.linspace(0,1,len(feharray[goodidx])))

        # Plot stellar lifetimes at various metallicities
        plt.figure()
        for feh_idx, feh in enumerate(feharray[goodidx]):
            age = data[1,feh_idx,:] / 1e9  # Convert to Gyr
            mass = data[0,feh_idx,:]

            # Get relevant masses
            massidx = np.where((mass > 0.865) & (mass < 100))[0]

            plt.loglog(mass[massidx], age[massidx], color=metalcolor[feh_idx], lw=1, label=r"[Fe/H] = {:.2f}".format(feh))

        # Plot empirical equations
        masses = np.linspace(0.865,100,1000)
        t = np.zeros(len(masses))
        t = 10.** ((0.334 - np.sqrt(1.790-0.2232*(7.764-np.log10(masses))))/0.1116) 
        t[masses > 6.6] = 1.2*masses[masses > 6.6]**(-1.85) + 0.003
        #t *= 1e9  # Convert from Gyr to yr
        plt.loglog(masses, t, color='k', ls='--', lw=2, label='From equations')

        # Format plot
        legend = plt.legend()
        plt.xlabel(r'Mass ($M_{\odot}$)', fontsize=14)
        plt.ylabel(r'Lifetime (Gyr)', fontsize=14)
        plt.axvline(6.6, color='k', linestyle=':', lw=1)
        print(t[np.argmin(np.abs(masses-6.6))])
        plt.savefig('bpass_lifetimes.png', bbox_inches='tight')
        plt.show()

    return

def plotisochrones():
    '''Test function to plot isochrones of 2 different ages from one Dartmouth file.'''

    # Dartmouth isochrones
    files = ['fehm10afep2.ugriz','fehm15afep2.ugriz']
    linestyles = ['-','--']
    titles=['[Fe/H]=-1', '[Fe/H]=-1.5']
    for i in range(len(files)):
        with open(files[i],'r') as file:  
            lines = file.readlines()

            read1 = False
            read2 = False
            g1, g2, r1, r2 = [], [], [], []
            for idx, line in enumerate(lines):
                if line.startswith('#AGE'):
                    ln_list = line.split()
                    if ln_list[0].startswith('#AGE=11.0'):
                        read1 = True
                    elif ln_list[0].startswith('#AGE=13.0'):
                        read2 = True
                    else:
                        read1, read2 = False, False

                elif line.startswith('#EEP') or line=='\n':
                    pass

                else:
                    ln_list = line.split()
                    if read1:
                        g1.append(float(ln_list[6]))
                        r1.append(float(ln_list[7]))
                    if read2:
                        g2.append(float(ln_list[6]))
                        r2.append(float(ln_list[7]))
            print(g1)

        plt.plot(np.array(g1)-np.array(r1), np.array(g1), color='k', ls=linestyles[i], label='Dartmouth: 11Gyr, '+titles[i])
        plt.plot(np.array(g2)-np.array(r2), np.array(g2), color='r', ls=linestyles[i], label='Dartmouth: 13Gyr, '+titles[i])

    # Padova isochrones
    files = ['isoc_z030_sdss.dat']
    linestyles = ['--']
    titles=['[Fe/H]=-1.5']
    for i in range(len(files)):
        with open(files[i],'r') as file:  
            lines = file.readlines()

            read1 = False
            read2 = False
            g1, g2, r1, r2 = [], [], [], []
            for idx, line in enumerate(lines):
                if line.startswith('#	Isochrone'):
                    ln_list = line.split()
                    if ln_list[-2]=='1.122e+10':
                        read1 = True
                    elif ln_list[-2]=='1.259e+10':
                        read2 = True
                    else:
                        read1, read2 = False, False

                elif line.startswith('#') or line=='\n':
                    pass

                else:
                    ln_list = line.split()
                    if read1:
                        g1.append(float(ln_list[8]))
                        r1.append(float(ln_list[9]))
                    if read2:
                        g2.append(float(ln_list[8]))
                        r2.append(float(ln_list[9]))

            print(g1)

        #plt.plot(np.array(g1)-np.array(r1), np.array(g1), color='gray', ls=linestyles[i], label='Padova: 11Gyr, '+titles[i])
        plt.plot(np.array(g2)-np.array(r2), np.array(g2), color='pink', ls=linestyles[i], label='Padova: 13Gyr, '+titles[i])

    plt.gca().invert_yaxis()
    plt.legend()
    plt.xlabel(r'$g-r$')
    plt.ylabel(r'$g$')
    plt.show()

    #print(np.max(np.abs((np.array(g1)-np.array(r1)) - (np.array(g2)-np.array(r2)))))
    #print(np.max(np.abs(np.array(g1)-np.array(g2))))
    #print(np.max(np.abs(np.array(r1)-np.array(r2))))
        
    return

if __name__ == "__main__":
    #readlifetimes()
    #plotlifetimes()
    #plotbpass()
    plotisochrones()