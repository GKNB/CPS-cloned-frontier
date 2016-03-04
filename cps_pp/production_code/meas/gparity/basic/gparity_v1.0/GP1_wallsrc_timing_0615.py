import generate_vml


generate_vml.Globals.set_bc(["BND_CND_GPARITY","BND_CND_PRD","BND_CND_PRD","BND_CND_APRD"])
props = generate_vml.JobPropagatorArgs() 

masses = [0.032]
moms = [ [1,0,0], [-1,0,0] ]
dirstrs = ["p+", "p-"]
for m in masses:
    for momdir in range(2):
        dirstr = dirstrs[momdir]
        dirstr_opp = dirstrs[ (momdir+1) % 2 ]
        mom = moms[momdir]  #units of pi/2L
        flav = 0
        t=0
        gauge_fix = True

        pra = generate_vml.PropagatorArg("prop_m%g_APRD_%s" % (m,dirstr), m , "BND_CND_APRD")
        pra.setMomentumSource(t,mom,flav, gauge_fix)
        pra.specifyComplexConjPartner("prop_m%g_APRD_%s" % (m,dirstr_opp) )

        props.addPropagator(pra)

props.write("prop_arg.vml")



contract = generate_vml.GparityContractArg("configurations/ckpoint_lat.%d", 500, 10, 510, 
                                           generate_vml.FixGaugeArg("FIX_GAUGE_COULOMB_T",0,1,32,1e-08,10000))

#momenta here are in units of pi (*NOT* pi/L) consistently
#16^3 x 32 lattice
L = 16.0
mom_zero = [0.0,0.0,0.0]
mom_pi_by_L = [ 1.0/L, 0,0 ]

mom_pi_by_2L = [ 0.5/L, 0,0 ]
mom_mpi_by_2L = [ -0.5/L, 0,0 ]

mom_pair_zero = [mom_zero,mom_zero]
mom_pair_zero_2 = [mom_pi_by_2L,mom_mpi_by_2L]
mom_pair_zero_3 = [mom_mpi_by_2L,mom_pi_by_2L]

mom_pair_pi_by_2L = [ mom_pi_by_2L, mom_pi_by_2L ]
mom_pair_mpi_by_2L = [ mom_mpi_by_2L, mom_mpi_by_2L ]

wallsink_mompairs = [  mom_pair_zero, mom_pair_zero_2, mom_pair_zero_3,  mom_pair_pi_by_2L, mom_pair_mpi_by_2L ]

bcstubs = ["APRD"]
bccombs = ["AA"]

#Bilinears
for bc in range(1):
    for m1 in masses:
        for m2 in masses:
            prop_pp = "prop_m%g_%s_p+" % (m1,bcstubs[bc])
            prop_pm = "prop_m%g_%s_p-" % (m2,bcstubs[bc])
            cosine_sink = 0
            
            fn = "results/bilinears_%g_%g_pp_%s" % (m1,m2,bccombs[bc])
            contract.addMeas( generate_vml.ContractionTypeAllBilinears(prop_pm, prop_pp, [mom_zero, mom_pi_by_L], fn) )
            
            fn = "results/wallsink_bilinears_%g_%g_pp_%s" % (m1,m2,bccombs[bc])
            contract.addMeas( generate_vml.ContractionTypeAllWallSinkBilinearsSpecificMomentum(prop_pm, prop_pp, wallsink_mompairs, cosine_sink, fn) )


contract.write("contract_arg.vml")
                                                                      
