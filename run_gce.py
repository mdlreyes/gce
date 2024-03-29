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
    #model, atomic = gce.runmodel(scl_fiducial1, plot=True, title="Fiducial 1 (Lim+18)", empirical=False) #, amr="plots/amr_test")

    # Run model with input SFH
    #model, atomic = gce_modified.runmodel(scl_fiducial1, plot=True, title="Fiducial (Karakas+18)", empirical=True, amr="plots/amr_test")

    # Run a single model with empirical yield fits
    scl_init = [1.07, 0.16, 4.01, 0.89, 0.82, 0.59, 0.8, 1., 1., 0., 0.6, 0.33, 1.0]
    #scl_powell = [1.07520357, 0.16244134, 4.23355874, 0.99247777, 0.69161203, 0.58384897, 0.68119825, 1.34567668, 0.60438552, 0.16198303, 3.36670239]
    
    #scl_mcmc_widerpriors = [0.57925062, 0.25511282, 4.73242679, 0.78293811, 0.76362412, 0.23648493, 0.5921274, 1.32518867, 0.74821858, 0.21047607, 2.0407864]
    #scl_test_nofe = [1.0668918, 0.1758647, 4.34917587, 0.8366728, 1.14919742, 0.61846894, 0.8, 0.97536312, 0.99847981, 0.0317845, 0.56556294]
    #scl_test_baeu = [0.59231386, 0.24635045, 4.68516065, 0.5798797, 0.94214962, 1.38412127, 0.58593614, 1.32270303, 0.7562073, 0.2149663, 3.06525529]
    #scl_test_noc = [0.56632929, 0.25848466, 4.73030317, 0.75939287, 0.79235297, 0.25081187, 0.58697101, 1., 0.75303671, 0.20477863, 0.6]
    #scl_test_testc = [0.70157967, 0.26730922, 5.3575732, 0.47251228, 0.82681450, 0.49710685, 0.5921274, 1.32518867, 0.74821858, 0.21047607, 2.0407864]
    #scl_test_nofe_baeu = [0.28067497, 0.33577663, 3.96807903, 2.94723046, 0.91793318, 4.440447, 0.8, 1.18306658, 1.06405578, 0.05100093, 0.77536968]
    #scl_test_baeu = [0.25865113, 0.39865368, 4.64887724, 0.61163984, 0.6056791 , 0.48667341, 0.40166305, 1.28388772, 1.59726585, 0.40740912, 0.9384386 ]
    #scl_test_dartba = [0.25545342, 0.34787294, 3.99876785, 0.67584706, 0.84561568, 4.80238424, 0.37151047, 1.36368857, 1.33738063, 0.17706523, 0.98077824]
    #scl_test_dartba = [0.34156907, 0.32722443, 4.12776806, 1.03324403, 0.99353997, 0.01626088, 0.37568131, 1.35980557, 1.42740846, 0.20203422, 1.01131155]
    '''
    scl_test_dartba_ba = [0.4551175081622658,0.2945153319281185,4.722167671625599,0.5718757895875054,0.9075131641050453,0.3555618768080859,0.5486782831057839,1.3178030991867014,0.8145795259218076,0.22540348637848837,1.6438740761355892,0.9356754983097959,3-0.034871763659268494]
    scl_test_nofe_ba = [0.42794853, 0.31232367, 4.78964132, 1.82808763, 0.32359375, 0.60169182, 1.18813925, 0.87157664, 0.40730233, 2.00392269, 0.78205837, 3-0.08880501]
    scl_test_bothba_ba = [0.4389863146518289,0.305259626216913,4.942444967900384,0.4925229043278246,0.8329968649356562,0.40094641862489994,0.563019743600889,1.2909839533334972,0.8604762167017103,0.2864776957718226,1.5645763678916176,0.8939183631841486,3-0.014997329848299233]
    scl_delaySFtest = [0.36963815, 0.34288084, 5.07381209, 0.73231527, 0.62864803, 0.29277844, 0.57543231, 1.2897485, 0.84635716, 0.30015869, 1.63283893, 0.93347269, 3-0.01408449]
    scl_iadtd_maoz17 = [0.5164266933312606,0.22668646095205333,5.083938866111591,0.6708444072138402,0.7705726151153531,0.5831520287559541,0.8952553665964144,1.3892756558045938,0.47734137612923366,0.14861562077682866,0.435341510493796,0.33493152648882035,3-0.013300279965722219]
    scl_iadtd_himin = [0.12655011964708354,0.48083816401816953,4.774355073300842,1.0142068952827288,0.13937182878786417,0.9673484547698562,0.8931868300415441,1.548247398565755,0.40718356024183644,0.037801676437251096,0.5965616973892593,0.8185482795511301,3-1.3007176610686921]
    scl_iadtd_lomin = [0.5671755376691066,0.29287428128668863,5.015868834444396,0.30612159610634737,1.0034407937884338,0.4612588576531018,0.44599230385432126,1.161552777383641,1.2779361777765668,0.43394265803765714,1.326460414915983,0.9806089731602307,3-0.0059330597053563775]
    scl_iadtd_medhimin_test = [0.32165536155176916,0.3415165671993969,5.054721133238549,3.5632320350817865,1.0464037822992545,0.9464582131538404,0.6492806071357978,1.397581711321233,0.6605083959242006,0.14957742487647682,1.0149023011422285,0.46062328788695467,3-0.06506835776883141]
    scl_iadtd_medhimin = [0.2581362038956129,0.3671543880935386,4.884509919096489,0.519627246584796,0.6921089016677752,0.7093941886457517,0.667125036335381,1.4048235865635883,0.6442822761890015,0.17279690215969257,1.076567643613428,0.5756914070867104,3-0.0789755064127836]
    scl_iadtd_loindex = [0.31309403734878677,0.32698844029546187,5.119860962340789,0.5074794913085319,0.7677725611456582,0.27817942445348165,0.7747072609145225,1.3579266977743019,0.6409945773107304,0.24639512831333843,0.8835860105097602,0.5557520537168783,3-0.023556275510243575]
    scl_iadtd_hiindex = [0.5446693466306317,0.3092340505074539,4.662881112688744,0.6610461169621856,0.6648004259776421,0.22834192428764163,0.434048932393723,1.2372641358088885,1.21868854143266,0.30455907377622926,2.5503064633438433,0.9921019155833941,3-0.00552116094663595]
    scl_iadtd_cutoff = [0.3907399651848807,0.31789100855381613,4.976079316209285,0.4695236246906028,0.846066267741512,0.3848772970464857,0.5875359459715601,1.301460415128067,0.8259997983101177,0.28742136661443407,1.3484797631127226,0.7983782066064008,3-0.017047139327600602]
    scl_reioniz = [0.6411130691937341,0.24774922551128908,4.643962259917035,0.780729799230917,0.8778813431231577,0.612699567249839,0.6460839575200857,1.325818097299844,0.7336606535374587,0.26137519407263077,2.7368268789441252,0.9896010405595509,3-0.0056036256827435346]
    scl_rampressure = [0.40324345287403596,0.31993145962170993,4.678875833236568,0.4311305388839332,0.901133475075236,0.3874332761124177,0.5687645229241343,1.2899619752803073,0.8487435966713881,0.2857591805747674,1.5867499800816725,0.9139277884487471,0.013695390962180884,3-2.761601093904625]
    scl_imf_chabrier03 = [1.0680354182219103,0.29087824048307825,5.785175190841888,0.32614582504078626,0.6880109337422085,0.47419668814764776,0.8347392374670606,1.3517172298659013,0.5890139428180761,0.2739631120786506,1.7994398252473034,0.9807143044292836,3-0.011967114634611836]
    scl_imf_salpeter55 = [0.5089476125938007,0.32350548351207437,5.391102320123509,0.4003999995632118,0.7799460946453387,0.5549969145097149,0.6164885754010938,1.3299979696872426,0.7198534106528632,0.25245975628500583,2.182828951358294,0.9847378266515173,3-0.00954264476609045]
    scl_mgcheck = [0.4685350807295695,0.2930810581739879,4.947387216260614,0.5464805995104087,0.7747652694289181,0.4078297444014555,0.5739239133417713,1.2917207950984766,1.5329631705384767,0.31293591755074057,1.682734668459577,0.7616876354982305,3-0.017395358440185527]
    scl_widebapriors = [0.4521099853438738,0.2952864706519592,4.678167117614617,0.49429150574753844,1.0791120084785661,0.39783737519782053,0.5394764248347781,1.3204750735928574,1.359457919837111,0.13139217074287995,1.612782258893653,0.36473661768759114,4.402225273291386]
    scl_gaussbapriors = [0.43945705979139077,0.3020413952212676,4.727408176763617,0.46052848791458656,1.0808900211986314,0.36256994634680273,0.5375203396441758,1.315857359698602,1.401725996280066,0.15193987503658002,1.5818363678506433,0.4069203412745628,4.095657927853391]

    scl_test_bothba_ba = [0.4389863146518289,0.305259626216913,4.942444967900384,0.4925229043278246,0.8329968649356562,0.,0.563019743600889,1.2909839533334972,0.8604762167017103,0.2864776957718226,1.5645763678916176,0.8939183631841486,3-0.014997329848299233]
    scl_widebapriors = [0.4521099853438738,0.2952864706519592,4.678167117614617,0.49429150574753844,1.0791120084785661,0.,0.5394764248347781,1.3204750735928574,1.359457919837111,0.13139217074287995,1.612782258893653,0.36473661768759114,4.402225273291386]
    scl_gaussbapriors = [0.43945705979139077,0.3020413952212676,4.727408176763617,0.46052848791458656,1.0808900211986314,0.,0.5375203396441758,1.315857359698602,1.401725996280066,0.15193987503658002,1.5818363678506433,0.4069203412745628,4.095657927853391]
    scl_gaussbapriors25 = [0.42109737939397723,0.31308739823024134,4.837527469749196,0.4541058774499559,1.048019347062647,0.4110245957686604,0.537273133332133,1.3067431173812538,1.4775138184072172,0.19412055006004464,1.51505018695232,0.4627538532197125,3.6952004135922487]
    '''

    scl_final = [0.5345416815125118,0.26873964968241854,4.792520166984605,0.669974341000559,0.8076961016570472,0.,0.5788555396852141,1.3172211548419224,1.425232310196399,0.248402552325658,1.9258476702795306,1.0768034902504153,2.7983433770451396]
    scl_test_nomgas0 = [0.5396618135382528,0.2681707153106137,4.786076060415441,0.8023090276883353,0.7138939456571872,0.0,0.5817572301890092,1.3208335678363674,1.4026483028805945,0.242765941495789,1.9910917547173197,1.095006566170003,2.7833883548905374] # From restructured code
    scl_inflowconst_1 = [0., 4.63, 0.01, 0.375, 0.52, 0., 1.34, 1.60, 0.28, 1.34, 0.68, 2.85]
    scl_inflowconst_2 = [0.47, 6.49, 2.16, 1.82, 0.67, 0., 1.37, 0.95, 0.14, 0.68, 0.04, 1.77]
    scl_inflowexp = [0.06496605154954403,0.8150993638882837,4.845007136130306,1.493663884032438,0.293715730475182,0.0,0.5810620713192151,1.3270197716870957,1.3441501508542837,0.22119269118819115,2.038375621935514,0.9877440723565699,2.9129627976905557]
    scl_delaySF = [0.4385141670294113,0.31321776124241185,4.850818226490671,1.294482857279215,0.4956043540963967,0.0,0.6162743721095508,1.3331982357049064,1.3301465946600732,0.24926427379065966,2.356362120936387,1.0816722104194887,2.896019965318881]
    scl_rampressure = [0.5004475144848842,0.28014976193133256,4.560594860622903,0.7634584087926897,0.7340948173807795,0.0,0.5943831807492506,1.321766413812236,1.3977791844958891,0.2532147136111331,2.0542872982044007,1.117493993641292,2.797300482828119,3.1984896617719167]
    scl_imf_chabrier03 = [1.108118108946736,0.2840096006704498,5.6694650075134145,0.34191059182870215,0.7027291258587196,0.0,0.8609232472738231,1.3633020349319787,1.0318175595532324,0.2576929101881398,1.97387804655777,1.5108944744261965,2.611712835999786]
    scl_imf_salpeter55 =[0.5404783413198105,0.3129102870095325,5.320937484652554,0.45215716255996485,0.7580298734439346,0.0,0.6533602115066313,1.3414525941847104,1.2154187399810408,0.24681829322818655,2.5304790628863247,1.5788499974533377,2.704077988483312]
    scl_dtd_tmin200 = [0.24622522648838643,0.3792536257559661,4.81425127434149,0.575627246584796,0.6732329412228181,0.0,0.6608382723291828,1.4119642498905187,1.1924669319481056,0.17237107857841683,1.1165533126821832,1.216109930449961,2.1410963654806854]
    scl_dtd_index05 = [0.2974899305985777,0.33157342379596605,4.904629711948725,0.6035346678850428,0.7683646877884314,0.0,0.7622064730631224,1.374204559242403,1.171347119843871,0.24184697758748375,0.940535517167115,0.7624298267184866,2.487822650237609]
    scl_dtd_cutoff = [0.4379910365488767,0.2945709897315299,4.819133791981226,0.6838758109204235,0.7692711561040138,0.0,0.5948843208628403,1.326715583307327,1.3911883185668448,0.2506320725761307,1.5903178004028924,0.9151543982866176,2.784141069743188]

    scl_dtd_lomin = [0.5671755376691066,0.29287428128668863,5.015868834444396,0.30612159610634737,1.0034407937884338,0.4612588576531018,0.44599230385432126,1.161552777383641,1.2779361777765668,0.43394265803765714,1.326460414915983,0.9806089731602307,3-0.0059330597053563775]
    scl_dtd_index15 = [0.5446693466306317,0.3092340505074539,4.662881112688744,0.6610461169621856,0.6648004259776421,0.22834192428764163,0.434048932393723,1.2372641358088885,1.21868854143266,0.30455907377622926,2.5503064633438433,0.9921019155833941,3-0.00552116094663595]


    #model, atomic, _ = gce.runmodel(scl_init, plot=True, title="Sculptor dSph (Initial conditions)", empirical=True, empiricalfit=True, feh_denom=True) #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic, _ = gce.runmodel(scl_mcmc_widerpriors, plot=True, title="Sculptor dSph", empirical=True, empiricalfit=True, feh_denom=True) #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic, _ = gce.runmodel(scl_test_nofe_baeu, plot=True, title="Sculptor dSph (no Fe, with Ba)", empirical=True, empiricalfit=True, feh_denom=False) #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic, _ = gce.runmodel(scl_test_nofe_ba, plot=True, title="Sculptor dSph (no Fe, with DART Ba)", empirical=True, empiricalfit=True, feh_denom=False) #, amr="plots/amr_test", sfh="plots/sfh_test")
    
    #model, atomic, aic_orig = gce.runmodel(scl_test_bothba_ba, plot=True, title="Sculptor dSph", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False) #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic, aic_dart = gce.runmodel(scl_test_dartba_ba, plot=True, title="Sculptor dSph (DART Ba)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False) #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic, _ = gce.runmodel(scl_mgcheck, plot=True, title="Sculptor dSph (Mg check)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False, mgenhance=False)
    #model, atomic, _ = gce.runmodel(scl_delaySFtest, plot=True, title="Sculptor dSph (delayed SF)", empirical=True, empiricalfit=True, feh_denom=True, delay=True) #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic, _ = gce.runmodel(scl_iadtd_maoz17, plot=True, title="Sculptor dSph (Maoz+17 field Ia DTD)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, ia_dtd='maoz17') #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic, _ = gce.runmodel(scl_iadtd_lomin, plot=True, title="Sculptor dSph (min Ia delay time = 50 Myr)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, ia_dtd='lowmindelay')
    #model, atomic, _ = gce.runmodel(scl_iadtd_medhimin, plot=True, title="Sculptor dSph (min Ia delay time = 200 Myr)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, ia_dtd='medhidelay')
    #model, atomic, _ = gce.runmodel(scl_iadtd_loindex, plot=True, title="Sculptor dSph (Ia DTD index = -0.5)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, ia_dtd='index05')    
    #model, atomic, _ = gce.runmodel(scl_iadtd_hiindex, plot=True, title="Sculptor dSph (Ia DTD index = -1.5)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, ia_dtd='index15')    
    #model, atomic, _ = gce.runmodel(scl_reioniz, plot=True, title="Sculptor dSph (reionization test)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=True)
    #model, atomic, _ = gce.runmodel(scl_rampressure, plot=True, title="Sculptor dSph (ram pressure)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False, rampressure=True)
    #model, atomic, _ = gce.runmodel(scl_test_bothba_ba, plot=True, title="Sculptor dSph (typical SN r-process)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False, rprocess='typical_SN_only') #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic, _ = gce.runmodel(scl_iadtd_cutoff, plot=True, title="Sculptor dSph (cutoff Ia DTD)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, ia_dtd='cutoff') #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic, _ = gce.runmodel(scl_imf_chabrier03, plot=True, title="Sculptor dSph (Chabrier+03 IMF)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False, imf='chabrier03') #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic, _ = gce.runmodel(scl_imf_salpeter55, plot=True, title="Sculptor dSph (Salpeter+55 IMF)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False, imf='salpeter55') #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic, _ = gce.runmodel(scl_test_bothba_ba, plot=True, title="Sculptor dSph (best fit no Fe)", empirical=True, empiricalfit=True, feh_denom=False, delay=False, reioniz=False) #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic, _ = gce.runmodel(scl_test_bothba_ba, plot=True, title="Sculptor dSph (outflow prop to inflow)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False, outflow='inflow') #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic, _ = gce.runmodel(scl_test_bothba_ba, plot=True, title="Sculptor dSph (NSM check)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False, rprocess='rare_event_only') #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic, aic_wider = gce.runmodel(scl_widebapriors, plot=True, title="Sculptor dSph (wider Ba priors)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False, mgenhance=False) #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic, aic_gauss = gce.runmodel(scl_gaussbapriors, plot=True, title="Sculptor dSph (Gaussian Ba priors)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False, mgenhance=False) #, amr="plots/amr_test", sfh="plots/sfh_test")    
    #model, atomic, aic_gauss25 = gce.runmodel(scl_gaussbapriors25, plot=True, title="Sculptor dSph (Gaussian Ba priors)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False, mgenhance=False) #, amr="plots/amr_test", sfh="plots/sfh_test")    

    #model, atomic, aic_orig = gce.runmodel(scl_final, plot=True, title="Sculptor dSph", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False, mgenhance=False) #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic, aic_nomgas = gce.runmodel(scl_final_nomgas0, plot=True, title="Sculptor dSph (no initial gas)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False, mgenhance=False, nomgas0=True) #, amr="plots/amr_test", sfh="plots/sfh_test")
    
    model, atomic, aic = gce.runmodel(scl_test_nomgas0, plot=False, title="Sculptor dSph", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False, mgenhance=False, nomgas0=True) #, amr="plots/amr_test", sfh="plots/sfh_test")
    print(model['t'][-1], aic)
    #model, atomic, aic = gce.runmodel(scl_inflowconst_1, plot=True, title="Sculptor dSph (const inflow 1)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False, mgenhance=False, nomgas0=True, inflow='const') #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic, aic = gce.runmodel(scl_inflowconst_2, plot=False, title="Sculptor dSph (const inflow 2)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False, mgenhance=False, nomgas0=True, inflow='const') #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic, aic = gce.runmodel(scl_inflowexp, plot=False, title="Sculptor dSph (exp dec inflow)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False, mgenhance=False, nomgas0=True, inflow='expdec') #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic, aic = gce.runmodel(scl_rampressure, plot=False, title="Sculptor dSph (ram pressure)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False, mgenhance=False, nomgas0=True, inflow=None, rampressure=True) #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic, aic = gce.runmodel(scl_delaySF, plot=False, title="Sculptor dSph (delayed SF)", empirical=True, empiricalfit=True, feh_denom=True, delay=True, reioniz=False, mgenhance=False, nomgas0=True) #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic, aic = gce.runmodel(scl_reioniz, plot=True, title="Sculptor dSph (reionization)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=True, mgenhance=False, nomgas0=True) #, amr="plots/amr_test", sfh="plots/sfh_test")
    #model, atomic, aic = gce.runmodel(scl_imf_chabrier03, plot=False, title="Sculptor dSph (Chabrier+03 IMF)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False, imf='chabrier03', mgenhance=False, nomgas0=True)
    #model, atomic, aic = gce.runmodel(scl_imf_salpeter55, plot=False, title="Sculptor dSph (Salpeter+55 IMF)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, reioniz=False, imf='salpeter55', mgenhance=False, nomgas0=True)
    #model, atomic, aic = gce.runmodel(scl_dtd_tmin200, plot=False, title="Sculptor dSph (min Ia delay time = 200 Myr)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, ia_dtd='medhidelay', mgenhance=False, nomgas0=True)
    #model, atomic, aic = gce.runmodel(scl_dtd_index05, plot=False, title="Sculptor dSph (Ia DTD index = -0.5)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, ia_dtd='index05', mgenhance=False, nomgas0=True)        
    #model, atomic, aic = gce.runmodel(scl_dtd_cutoff, plot=False, title="Sculptor dSph (cutoff Ia DTD)", empirical=True, empiricalfit=True, feh_denom=True, delay=False, ia_dtd='cutoff', mgenhance=False, nomgas0=True)
    print(model['t'][-1], aic)