import sys
sys.path.append('../gparity_v1.0')
import generate_vml as gen


gen.Globals.set_bc(["BND_CND_GPARITY","BND_CND_GPARITY","BND_CND_PRD","BND_CND_APRD"])
props = gen.JobPropagatorArgs() 

momcombs = [ [1,1,0], [-1,-1,0], [1,-3,0], [-1,3,0] ]  #units of pi/2L for G-parity dirs. Momentum components must differ by integer multiples of 2pi/L = 4* pi/2L
momcombtags = ['110','m1m10','1m30','m130']

mass = 0.01

#Specify the Lanczos solver
# final 10^-10 solve
# double stop_rsd = 1e-10
# double qr_rsd = 1e-14
# EigenType EigenOper = DDAGD
# bool precon = 1
# int N_get = 200
# int N_use = 230
# int N_true_get = 200
# int ch_ord = 100
# double ch_alpha = 6
# double ch_beta = 0.4


# 198:0.0975313896197093
# 199:0.0978981210241576
# total number of vector inner products 52774
# total number of matrix vector products 70400
# Done : Total time = 4652.40282487869sec, Dlash time = 1743.22763109207sec, Shift time = 8.72103714942932sec. 
lanc_con = gen.LanczosContainerArg(tag="lanczos", solver = "BFM_DWF",
                                   lanc_arg= gen.LancArg(N_get = 200, N_use = 230, N_true_get = 200, 
                                                         ch_ord = 100, ch_alpha = 6, ch_beta = 0.4,
                                                         stop_rsd=1e-10, mass=mass)
                               )
props.addLanczos(lanc_con)


for f in [0,1]:
    for mc in range(len(momcombs)):
        mcvals = momcombs[mc]
        mctag = momcombtags[mc]
        
        tag = "prop_f%d_m%g_%s" % (f,mass,mctag)
        fpartner_tag = "prop_f%d_m%g_%s" % (1-f,mass,mctag)

        prop = gen.PropagatorArg(tag,mass,"BND_CND_APRD")
        prop.setMomentumSource(0,mcvals,flav=f,gauge_fix=True)
        prop.specifyFlavorPartner(fpartner_tag)
        prop.deflateUsing("lanczos")
        props.addPropagator(prop)

props.write("prop_arg.vml")

#momenta here are in units of pi consistently
p = 1.0/8
pion_momenta = [ [p,p,0], [-p,-p,0], [p,-p,0], [-p,p,0] ] 
pion_momdescr = [ "220", "m2m20", "2m20", "m220" ]
#Quark combinations to form pions with above momenta
#We form  tr A [G1]^dag B [G2]   so total momentum is -p(G1) + p(G2)
#e.g.  (2,2,0) = -(-1,-1,0) +(1,1,0)     and  (2,-2,0) = -(-1,-1,0) + (1,-3,0)
quark_mompairs = [ ['m1m10','110'], ['110','m1m10'], ['m1m10','1m30'], ['110','m130'] ]
                   
bilinear_args = []
for c in range(len(pion_momenta)):
    tag1 = "prop_f0_m%g_%s" % (mass,quark_mompairs[c][0])
    tag2 = "prop_f0_m%g_%s" % (mass,quark_mompairs[c][1])
    bil = gen.ContractionTypeAllBilinears(tag1,tag2,[pion_momenta[c]],"results/result_%s.dat" % pion_momdescr[c] )
    bilinear_args.append(bil)

ama = gen.GparityAMAarg(bilinear_args = bilinear_args,
                        exact_solve_timeslices=[0,16],
                        fix_gauge=gen.FixGaugeArg("FIX_GAUGE_COULOMB_T",0,1,32,1e-08,10000),
                        exact_precision=1e-8, sloppy_precision = 1e-4,
                        config_fmt = "configurations/ckpoint_lat.%d", conf_start = 300, conf_incr = 50, conf_lessthan = 900) 

ama.write("ama_arg.vml")
                                                                      
bfm = gen.BfmArg( solver = "BFM_DWF", threads = 1 )
bfm.write("bfm_arg.vml")


                                                                      
