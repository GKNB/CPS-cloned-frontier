import generate_vml


generate_vml.Globals.set_bc(["BND_CND_GPARITY","BND_CND_PRD","BND_CND_PRD","BND_CND_APRD"])
props = generate_vml.JobPropagatorArgs() 

masses = [0.01,0.032]
moms = [ [1,0,0], [-1,0,0] ]
dirstrs = ["p+","p-"]
for m in masses:
    for momdir in range(2):
        dirstr = dirstrs[momdir]
        dirstr_opp = dirstrs[ (momdir+1) % 2 ]
        mom = moms[momdir]  #units of pi/2L
        flav = 0
        t=0
        gauge_fix = True

        prd = generate_vml.PropagatorArg("prop_m%g_PRD_%s" % (m,dirstr), m , "BND_CND_PRD")
        prd.setMomentumSource(t,mom,flav, gauge_fix)
        prd.specifyComplexConjPartner("prop_m%g_PRD_%s" % (m,dirstr_opp) )

        pra = generate_vml.PropagatorArg("prop_m%g_APRD_%s" % (m,dirstr), m , "BND_CND_APRD")
        pra.setMomentumSource(t,mom,flav, gauge_fix)
        pra.specifyComplexConjPartner("prop_m%g_APRD_%s" % (m,dirstr_opp) )
        pra.storeMidPoint()

        prf = generate_vml.PropagatorArg("prop_m%g_F_%s" % (m,dirstr), m)
        prf.setCombinationSource( prd.tag, pra.tag, "A_PLUS_B")
        prf.specifyComplexConjPartner("prop_m%g_F_%s" % (m,dirstr_opp) )
                
        prb = generate_vml.PropagatorArg("prop_m%g_B_%s" % (m,dirstr), m)
        prb.setCombinationSource( prd.tag, pra.tag, "A_MINUS_B")
        prb.specifyComplexConjPartner("prop_m%g_B_%s" % (m,dirstr_opp) )

        props.addPropagator(prd)
        props.addPropagator(pra)
        props.addPropagator(prf)
        props.addPropagator(prb)


props.write("prop_arg.vml")



contract = generate_vml.GparityContractArg("configurations/ckpoint_lat.%d", 500, 10, 1540, 
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

bcstubs = ["F","B","APRD","PRD"]
bccombs = ["FF","BB","AA","PP"]

#Bilinears
for bc in range(4):
    for m1 in masses:
        for m2 in masses:
            prop_m1_pp = "prop_m%g_%s_p+" % (m1,bcstubs[bc])
            prop_m1_pm = "prop_m%g_%s_p-" % (m1,bcstubs[bc])

            prop_m2_pp = "prop_m%g_%s_p+" % (m2,bcstubs[bc])
            prop_m2_pm = "prop_m%g_%s_p-" % (m2,bcstubs[bc])

            cosine_sink = 0
            
            #There are 2 momentum combinations that give 0 total momentum  +- (pm) and -+ (mp)
            #As the first propagator of each pair is hermitian-conjugated its momentum is swapped, hence  pm = m^dag m

            fn = "results/bilinears_%g_%g_pp_%s" % (m1,m2,bccombs[bc])
            contract.addMeas( generate_vml.ContractionTypeAllBilinears(prop_m1_pm, prop_m2_pp, [mom_zero, mom_pi_by_L], fn) ) #first prop is daggered, hence -ve source mom   #automatically adds both +p and -p (for p!=0)

            fn = "results/bilinears_%g_%g_pm_%s" % (m1,m2,bccombs[bc])
            contract.addMeas( generate_vml.ContractionTypeAllBilinears(prop_m1_pm, prop_m2_pm, [mom_zero, mom_pi_by_L], fn) )

            fn = "results/bilinears_%g_%g_mp_%s" % (m1,m2,bccombs[bc])
            contract.addMeas( generate_vml.ContractionTypeAllBilinears(prop_m1_pp, prop_m2_pp, [mom_zero, mom_pi_by_L], fn) )

            fn = "results/bilinears_%g_%g_mm_%s" % (m1,m2,bccombs[bc])
            contract.addMeas( generate_vml.ContractionTypeAllBilinears(prop_m1_pp, prop_m2_pm, [mom_zero, mom_pi_by_L], fn) )

            fn = "results/wallsink_bilinears_%g_%g_pp_%s" % (m1,m2,bccombs[bc])
            contract.addMeas( generate_vml.ContractionTypeAllWallSinkBilinearsSpecificMomentum(prop_m1_pm, prop_m2_pp, wallsink_mompairs, cosine_sink, fn) )

            fn = "results/wallsink_bilinears_%g_%g_pm_%s" % (m1,m2,bccombs[bc])
            contract.addMeas( generate_vml.ContractionTypeAllWallSinkBilinearsSpecificMomentum(prop_m1_pm, prop_m2_pm, wallsink_mompairs, cosine_sink, fn) )

            fn = "results/wallsink_bilinears_%g_%g_mp_%s" % (m1,m2,bccombs[bc])
            contract.addMeas( generate_vml.ContractionTypeAllWallSinkBilinearsSpecificMomentum(prop_m1_pp, prop_m2_pp, wallsink_mompairs, cosine_sink, fn) )

            fn = "results/wallsink_bilinears_%g_%g_mm_%s" % (m1,m2,bccombs[bc])
            contract.addMeas( generate_vml.ContractionTypeAllWallSinkBilinearsSpecificMomentum(prop_m1_pp, prop_m2_pm, wallsink_mompairs, cosine_sink, fn) )



    

#Bag parameters
#The heavy quark is the one that is daggered. We should pick momentum combinations that sum to zero
ml = masses[0]
mh = masses[1]

prop_h_F_pp = "prop_m%g_F_p+" % mh
prop_h_F_pm = "prop_m%g_F_p-" % mh

prop_h_B_pp = "prop_m%g_B_p+" % mh
prop_h_B_pm = "prop_m%g_B_p-" % mh

prop_l_F_pp = "prop_m%g_F_p+" % ml
prop_l_F_pm = "prop_m%g_F_p-" % ml

prop_l_B_pp = "prop_m%g_B_p+" % ml
prop_l_B_pm = "prop_m%g_B_p-" % ml



fn = "results/OVVpAA_%g_%g_mpmp" % (mh,ml)
contract.addMeas( generate_vml.ContractionTypeOVVpAA(prop_h_F_pp,prop_l_F_pp,prop_h_B_pp,prop_l_B_pp, fn) )
fn = "results/OVVpAA_%g_%g_mppm" % (mh,ml)
contract.addMeas( generate_vml.ContractionTypeOVVpAA(prop_h_F_pp,prop_l_F_pp,prop_h_B_pm,prop_l_B_pm, fn) )
fn = "results/OVVpAA_%g_%g_pmmp" % (mh,ml)
contract.addMeas( generate_vml.ContractionTypeOVVpAA(prop_h_F_pm,prop_l_F_pm,prop_h_B_pp,prop_l_B_pp, fn) )
fn = "results/OVVpAA_%g_%g_pmpm" % (mh,ml)
contract.addMeas( generate_vml.ContractionTypeOVVpAA(prop_h_F_pm,prop_l_F_pm,prop_h_B_pm,prop_l_B_pm, fn) )

#mres
#Just use the APRD props with mom +p
for m in masses:
    fn = "results/mres_%g" % m
    contract.addMeas( generate_vml.ContractionTypeMres("prop_m%g_APRD_p+" % m,fn) )

contract.write("contract_arg.vml")
                                                                      
