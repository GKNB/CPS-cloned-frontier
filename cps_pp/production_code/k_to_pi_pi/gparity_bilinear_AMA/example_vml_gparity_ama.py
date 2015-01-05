import sys
sys.path.append('../gparity_v1.0')
import generate_vml as gen


gen.Globals.set_bc(["BND_CND_GPARITY","BND_CND_PRD","BND_CND_PRD","BND_CND_APRD"])
props = gen.JobPropagatorArgs() 

#Lanczos arguments
#These parameters are optimized for a 4^4 x 2 random beta=2.25 DWF+I lattice, 80 low modes,with GPBC in the X-direction
lanc_con = gen.LanczosContainerArg(tag="lanczos", solver = "BFM_DWF",
                               lanc_arg= gen.LancArg(N_get = 80, N_use = 100, N_true_get = 80, ch_ord = 40, ch_alpha = 15, ch_beta = 1.8)
                               )

props.addLanczos(lanc_con)


m=0.01


pr0 = gen.PropagatorArg("prop_m%g_f0" % m, m , "BND_CND_APRD")
pr0.setMomentumSource(0,[1,0,0],flav=0,gauge_fix=True)
pr0.specifyFlavorPartner("prop_m%g_f1" % m)
pr0.deflateUsing("lanczos")
props.addPropagator(pr0)

pr1 = gen.PropagatorArg("prop_m%g_f1" % m, m , "BND_CND_APRD")
pr1.setMomentumSource(0,[1,0,0],flav=1,gauge_fix=True)
pr1.specifyFlavorPartner("prop_m%g_f1" % m)
pr1.deflateUsing("lanczos")
props.addPropagator(pr1)

props.write("prop_arg.vml")



mom_pi_over_l = [ 1.0/4.0, 0,0 ] 

ama = gen.GparityAMAarg(bilinear_args = [ gen.ContractionTypeAllBilinears("prop_m%g_f0" % m,"prop_m%g_f0" % m,[mom_pi_over_l],"result") ],
                                 exact_solve_timeslices=[0],
                                 fix_gauge=gen.FixGaugeArg("FIX_GAUGE_COULOMB_T",0,1,4,1e-08,10000),
                                 exact_precision=1e-8, sloppy_precision = 1e-6,
                                 config_fmt = "ckpoint_lat.%d", conf_start = 0, conf_incr = 1, conf_lessthan = 1) 

ama.write("ama_arg.vml")
                                                                      
bfm = gen.BfmArg( solver = "BFM_DWF", threads = 1 )
bfm.write("bfm_arg.vml")
