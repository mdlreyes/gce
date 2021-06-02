"""
isochrone.py

Some test code to mess around with isochrones
"""

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

    # Import packages for plotting
    import cycler
    import cmasher as cmr

    # Get data
    ages = np.load(grid+'_ages.npy')
    masses = np.load(grid+'_masses.npy')

    if grid=='dartmouth':
        metals = [-2.5,-2.0,-1.5,-1.0,-0.5,0.,0.2,0.3,0.5]  # [Fe/H]
        alphas = [-0.2,0.,0.2,0.4,0.6,0.8]  # [alpha/Fe]

        # Define colormaps
        metalcolor = cmr.bubblegum_r(np.linspace(0,1,len(metals),endpoint=True))

        for metal_idx, metal in enumerate(metals):
            age = ages[metal_idx,0,:]
            mass = masses[metal_idx,0,:]
            if np.any(np.isclose(age, 0)) or np.any(np.isclose(mass, 0)):
                print('test', metal, alphas[0])
            plt.plot(mass, age, color=metalcolor[metal_idx], label=r"[Fe/H] = {:.2f}".format(metal))
            print(metal, mass, age)
        plt.legend()
        plt.xlabel(r'Mass ($M_{\odot}$)', fontsize=14)
        plt.ylabel(r'Lifetime (Gyr)', fontsize=14)
        plt.show()

    return

if __name__ == "__main__":
    #readlifetimes()
    plotlifetimes()