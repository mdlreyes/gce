import gce_fast as gce
import gce_fast_old as gce_old
import time
from line_profiler import LineProfiler

#   using names used in the paper:  pars = [A_in/1e6,  tau_in,   A_out/1000,    A_star/1e6, alpha,    M_gas_0]
scl_pars = [ 701.57967, 0.26730922, 5.3575732, 0.47251228, 0.82681450, 0.49710685] #result of "restore, 'scl_sfr-law.sav'" in idl
#umi_pars = [1470.5896, 0.16522838, 11.038576, 1.2065735, 0.26234735, 0.53814755] #result of "restore, 'umi_sfr-law.sav'" in idl
#for_pars = [2464.2364, 0.30686976, 1.5054730, 5.0189799, 0.98204341, 14.575519] #'old/for_sfr-law.sav'
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
    gce.runmodel(scl_pars, plot=True)
