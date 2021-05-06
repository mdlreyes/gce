"""
run_gce.py

This program runs the single GCE model in gce_fast.py
"""

# Add path for testing:
import sys
sys.path.insert(1, 'old')

import gce_fast as gce
import gce_modified as gce_modified
import gce_fast_old as gce_old
import time
from line_profiler import LineProfiler

# Using names used in the paper:  pars = [A_in/1e9,  tau_in,   A_out/1000,    A_star/1e6, alpha,    M_gas_0]
scl_pars = [0.70157967, 0.26730922, 5.3575732, 0.47251228, 0.82681450, 0.49710685] #result of "restore, 'scl_sfr-law.sav'" in idl
scl_init = [2.6988, 0.27, 5.37, 4.46, 0.85, 0.]
scl_test_powell = [0.91144016, 0.19617321, 4.42241379, 4.45999299, 1.97494677, 0.17709949]
scl_test_mcmc = [1.01, 0.18, 4.30, 1.28, 0.74, 0.11]
scl_test_mcmc_fromkirby_maoz10 = [1.01, 0.17, 4.31, 1.23, 0.76, 0.21]
scl_test_mcmc_fromkirby_mannucci06 = [4.86509872, 0.05459378, 3.13738242, 4.87828528, 0.4670316, 0.17314514]
scl_fiducial1 = [0.95, 0.18, 4.34, 1.27, 0.76, 0.69]
scl_fiducialtest = [0.95, 0.18, 4.34, 1.27, 0.76, 0.]
scl_fiducial2 = [0.95, 0.18, 4.34, 2.78, 0.17, 5.24]
scl_fiducial_combined = [1.07, 0.16, 4.01, 0.89, 0.82, 0.59]
scl_fiducialcombined_powell = [1.05416462, 0.16, 4.01234049, 0.89, 0.82, 0.59145898]
#umi_pars = [1470.5896, 0.16522838, 11.038576, 1.2065735, 0.26234735, 0.53814755] #result of "restore, 'umi_sfr-law.sav'" in idl
#for_pars = [2.4642364, 0.30686976, 1.5054730, 5.0189799, 0.98204341, 14.575519] #'old/for_sfr-law.sav'
#for_pars = [2.46, 0.31, 1.51, 5.02, 0.98, 14.58] # from Kirby+11
#dra_pars = [1272.6409, 0.21561223, 9.5079511, 0.87843537, 0.34350762, 2.3213345]

def time_check(pars):
    t1=time.time()
    model, atomic = gce_old.gce_model(pars)
    print('Time for old model: %.2e'%(time.time()-t1))

    t2=time.time()
    model, atomic = gce.runmodel(pars, plot=False)
    print('Time for new model: %.2e'%(time.time()-t2))

    t3=time.time()
    model, atomic = gce.runmodel(pars, plot=False, empirical=True)
    print('Time for empirical yields model: %.2e'%(time.time()-t3))

    return

if __name__ == "__main__":

    # Time check
    #time_check(scl_fiducial1)

    # Line profiling
    '''
    lp = LineProfiler()
    lp_wrapper = lp(gce.runmodel)
    lp_wrapper(scl_fiducial1, empirical=True)
    lp.print_stats()
    '''

    # Run a single model
    model, atomic = gce.runmodel(scl_fiducial1, plot=True, title="Fiducial 1", empirical=False) #, amr="plots/amr_test")

    # Run model with input SFH
    #model, atomic = gce_modified.runmodel(scl_fiducial1, plot=True, title="Fiducial (Karakas+18)", empirical=True, amr="plots/amr_test")

    # Run a single model with empirical yield fits
    #scl_test_init = [1.07, 0.16, 4.01, 0.89, 0.82, 0.59, 0.8, 1., 1., 0., 0.6]
    #scl_test_powell = [1.07520357, 0.16244134, 4.23355874, 0.99247777, 0.69161203, 0.58384897, 0.68119825, 1.34567668, 0.60438552, 0.16198303, 3.36670239]
    #scl_test_mcmc = [0.46890192, 0.28214621, 4.53120169, 0.60097303, 0.91420782, 0.30462848, 0.52679802, 1.26303907, 0.71169272, 0.08934153, 0.76942736]
    #scl_test_nofe = [1.50066934, 0.14006733, 4.4106292, 0.88685668, 0.80094976, 1.07388209, 0.77890429, 1.06857726, 0.9905412, 0.09191467, 0.58686714]
    #scl_test_baeu = [1.02862074, 0.17523727, 3.75525287, 0.88471765, 0.70164733, 0.78237028, 0.73180173, 1.00540023, 0.99460002, 0.04271566, 0.57465244]
    #model, atomic = gce.runmodel(scl_test_nofe, plot=True, title="Empirical yield (no Fe)", empirical=True, empiricalfit=True)