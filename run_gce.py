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
    #model, atomic = gce.runmodel(scl_fiducial1, plot=True, title="Fiducial 1", empirical=False) #, amr="plots/amr_test")

    # Run model with input SFH
    #model, atomic = gce_modified.runmodel(scl_fiducial1, plot=True, title="Fiducial (Karakas+18)", empirical=True, amr="plots/amr_test")

    # Run a single model with empirical yield fits
    scl_init = [1.07, 0.16, 4.01, 0.89, 0.82, 0.59, 0.8, 1., 1., 0., 0.6, 0.33, 1.0]
    #scl_powell = [1.07520357, 0.16244134, 4.23355874, 0.99247777, 0.69161203, 0.58384897, 0.68119825, 1.34567668, 0.60438552, 0.16198303, 3.36670239]
    
    scl_mcmc_widerpriors = [0.57925062, 0.25511282, 4.73242679, 0.78293811, 0.76362412, 0.23648493, 0.5921274, 1.32518867, 0.74821858, 0.21047607, 2.0407864]
    #scl_test_nofe = [1.0668918, 0.1758647, 4.34917587, 0.8366728, 1.14919742, 0.61846894, 0.8, 0.97536312, 0.99847981, 0.0317845, 0.56556294]
    #scl_test_baeu = [0.59231386, 0.24635045, 4.68516065, 0.5798797, 0.94214962, 1.38412127, 0.58593614, 1.32270303, 0.7562073, 0.2149663, 3.06525529]
    #scl_test_noc = [0.56632929, 0.25848466, 4.73030317, 0.75939287, 0.79235297, 0.25081187, 0.58697101, 1., 0.75303671, 0.20477863, 0.6]
    #scl_test_testc = [0.70157967, 0.26730922, 5.3575732, 0.47251228, 0.82681450, 0.49710685, 0.5921274, 1.32518867, 0.74821858, 0.21047607, 2.0407864]
    #scl_test_nofe_baeu = [0.28067497, 0.33577663, 3.96807903, 2.94723046, 0.91793318, 4.440447, 0.8, 1.18306658, 1.06405578, 0.05100093, 0.77536968]
    #scl_test_baeu = [0.25865113, 0.39865368, 4.64887724, 0.61163984, 0.6056791 , 0.48667341, 0.40166305, 1.28388772, 1.59726585, 0.40740912, 0.9384386 ]
    #scl_test_dartba = [0.25545342, 0.34787294, 3.99876785, 0.67584706, 0.84561568, 4.80238424, 0.37151047, 1.36368857, 1.33738063, 0.17706523, 0.98077824]
    #scl_test_dartba = [0.34156907, 0.32722443, 4.12776806, 1.03324403, 0.99353997, 0.01626088, 0.37568131, 1.35980557, 1.42740846, 0.20203422, 1.01131155]
    scl_test_dartba_ba = [0.45488486, 0.29451152, 4.7214058, 0.57051713, 0.90860354, 0.36811399, 0.54901945, 1.31771318, 0.81434372, 0.22611351, 1.64741211, 0.93501212, 0.034125]
    scl_test_nofe_ba = [0.42794853, 0.31232367, 4.78964132, 1.82808763, 0.32359375, 0.60169182, 1.18813925, 0.87157664, 0.40730233, 2.00392269, 0.78205837, 0.08880501]
    scl_test_bothba_ba = [0.43936705, 0.30507695, 4.94212835, 0.49191929, 0.83307305, 0.40318864, 0.5627217, 1.291076, 0.85956343, 0.28562448, 1.56742018, 0.89342641, 0.01507226]
    scl_delaySFtest = [0.36963815, 0.34288084, 5.07381209, 0.73231527, 0.62864803, 0.29277844, 0.57543231, 1.2897485, 0.84635716, 0.30015869, 1.63283893, 0.93347269, 0.01408449]
    scl_iadtd_maoz17 = [0.5164266933312606,0.22668646095205333,5.083938866111591,0.6708444072138402,0.7705726151153531,0.5831520287559541,0.8952553665964144,1.3892756558045938,0.47734137612923366,0.14861562077682866,0.435341510493796,0.33493152648882035,0.013300279965722219]
    scl_iadtd_lowmin = [0.12655011964708354,0.48083816401816953,4.774355073300842,1.0142068952827288,0.13937182878786417,0.9673484547698562,0.8931868300415441,1.548247398565755,0.40718356024183644,0.037801676437251096,0.5965616973892593,0.8185482795511301,1.3007176610686921]

    #model, atomic = gce.runmodel(scl_init, plot=True, title="Sculptor dSph (Initial conditions)", empirical=True, empiricalfit=True, feh_denom=True) #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic = gce.runmodel(scl_mcmc_widerpriors, plot=True, title="Sculptor dSph", empirical=True, empiricalfit=True, feh_denom=True) #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic = gce.runmodel(scl_test_nofe, plot=True, title="Sculptor dSph (no Fe)", empirical=True, empiricalfit=True, feh_denom=False) #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic = gce.runmodel(scl_test_nofe_baeu, plot=True, title="Sculptor dSph (no Fe, with Ba)", empirical=True, empiricalfit=True, feh_denom=False) #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic = gce.runmodel(scl_test_dartba_ba, plot=True, title="Sculptor dSph (with DART Ba)", empirical=True, empiricalfit=True, feh_denom=True) #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic = gce.runmodel(scl_test_bothba_ba, plot=True, title="Sculptor dSph", empirical=True, empiricalfit=True, feh_denom=True) #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic = gce.runmodel(scl_test_nofe_ba, plot=True, title="Sculptor dSph (no Fe, with DART Ba)", empirical=True, empiricalfit=True, feh_denom=False) #, amr="plots/amr_test", sfh="plots/sfh_test")
    model, atomic = gce.runmodel(scl_test_bothba_ba, plot=True, title="Sculptor dSph", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=True) #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic = gce.runmodel(scl_delaySFtest, plot=True, title="Sculptor dSph (delayed SF)", empirical=True, empiricalfit=True, feh_denom=True, delay=True) #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic = gce.runmodel(scl_iadtd_maoz17, plot=True, title="Sculptor dSph (Maoz+17 field Ia DTD)", empirical=True, empiricalfit=True, feh_denom=True, delay=False) #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic = gce.runmodel(scl_iadtd_lowmin, plot=True, title="Sculptor dSph (min Ia delay time = 500 Myr)", empirical=True, empiricalfit=True, feh_denom=True, delay=False)