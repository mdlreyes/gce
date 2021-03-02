"""
run_gce.py

This program runs the single GCE model in gce_fast.py
"""

import gce_fast as gce
import gce_fast_old as gce_old
import time
from line_profiler import LineProfiler

#   using names used in the paper:  pars = [A_in/1e9,  tau_in,   A_out/1000,    A_star/1e6, alpha,    M_gas_0]
scl_pars = [0.70157967, 0.26730922, 5.3575732, 0.47251228, 0.82681450, 0.49710685] #result of "restore, 'scl_sfr-law.sav'" in idl
scl_init = [2.6988, 0.27, 5.37, 4.46, 0.85, 0.]
scl_test_powell = [0.66951658, 0.21648284, 3.96625663, 0.17007564, 1.81059811, 1.44096689]
scl_test_mcmc = [0.79, 0.20, 4.11, 0.25, 1.65, 1.72]
#umi_pars = [1470.5896, 0.16522838, 11.038576, 1.2065735, 0.26234735, 0.53814755] #result of "restore, 'umi_sfr-law.sav'" in idl
#for_pars = [2.4642364, 0.30686976, 1.5054730, 5.0189799, 0.98204341, 14.575519] #'old/for_sfr-law.sav'
#for_pars = [2.46, 0.31, 1.51, 5.02, 0.98, 14.58] # from Kirby+11
#dra_pars = [1272.6409, 0.21561223, 9.5079511, 0.87843537, 0.34350762, 2.3213345]

def time_check(pars):
    t1=time.time()
    model, atomic = gce_old.gce_model(pars)
    print('Time for old model: %.2e'%(time.time()-t1))
    #print('mdot', model['mdot'][50:100])
    #print('test II', model['II_rate'][:50])
    #print('test Ia', model['Ia_rate'][:50])
    #print('test AGB', model['AGB_rate'][50:100])
    #print('test z', model['z'][:50])

    t2=time.time()
    model, atomic = gce.gce_model(pars)
    print('Time for new model: %.2e'%(time.time()-t2))
    #print('mdot', model['mdot'][50:100])
    #print('test II', model['II_rate'][:50])
    #print('test Ia', model['Ia_rate'][:50])
    #print('test AGB', model['AGB_rate'][50:100])
    #print('test z', model['z'][:50])

    return

if __name__ == "__main__":

    # Time check
    #time_check(scl_pars)

    # Line profiling
    #lp = LineProfiler()
    #lp_wrapper = lp(gce.gce_model)
    #lp_wrapper(scl_pars)
    #lp.print_stats()

    # Run a single model
    #testpars = [0.5, 0.25, 5., 0.5, 1., 0.5]
    #testpars = [1.07, 0.18, 5.06, 0.74, 1.06, 1.03]
    #testpars = [0.96121475, 0.2003081,  5.14680601, 0.31888703, 1.48693599, 0.65452166]
    gce.runmodel(scl_test_mcmc, plot=True)
