import generate_vml


generate_vml.Globals.set_bc(["BND_CND_GPARITY","BND_CND_GPARITY","BND_CND_GPARITY","BND_CND_APRD"])
props = generate_vml.JobPropagatorArgs() 

masses = [0.01,0.045]
for m in masses:
    prd = generate_vml.PropagatorArg("prop_m%g_PRD" % m, m , "BND_CND_PRD")
    prd.setMomCosSource(0,[1,1,1],0)

    pra = generate_vml.PropagatorArg("prop_m%g_APRD" % m, m , "BND_CND_APRD")
    pra.setMomCosSource(0,[1,1,1],0)

    prf = generate_vml.PropagatorArg("prop_m%g_F" % m, m)
    prf.setCombinationSource( prd.tag, pra.tag, "A_PLUS_B")
    
    prb = generate_vml.PropagatorArg("prop_m%g_B" % m, m)
    prb.setCombinationSource( prd.tag, pra.tag, "A_MINUS_B")
    
    props.addPropagator(prd)
    props.addPropagator(pra)
    props.addPropagator(prf)
    props.addPropagator(prb)


props.write("prop_arg.vml")



contract = generate_vml.GparityContractArg("../configurations/ckpoint_lat.%d", 300, 5, 910, 
                                           generate_vml.FixGaugeArg("FIX_GAUGE_COULOMB_T",0,1,32,1e-08,10000))

#momenta here are in units of pi consistently
#16^3 x 32 lattice
mom_pi_over_2l = [ 1.0/32.0, 1.0/32.0, 1.0/32.0 ]
mom_pi_over_l = [ 1.0/16.0, 1.0/16.0, 1.0/16.0 ]
mom_zero = [0.0,0.0,0.0]
mom_pair_pi_over_l = [mom_pi_over_2l,mom_pi_over_2l] #momenta add
coswall = 1 #use cosine wall sink

#LL and HH combinations
for m in masses:
    prop_f = "prop_m%g_F" % m
    prop_b = "prop_m%g_B" % m

    fn = "../results/bilinears_%g_%g_FF" % (m,m)
    contract.addMeas( generate_vml.ContractionTypeAllBilinears(prop_f, prop_f, [mom_pi_over_l, mom_zero], fn) )

    fn = "../results/bilinears_%g_%g_BB" % (m,m)
    contract.addMeas( generate_vml.ContractionTypeAllBilinears(prop_b, prop_b, [mom_pi_over_l, mom_zero], fn) )

    fn = "../results/coswallsink_bilinears_%g_%g_FF" % (m,m)
    contract.addMeas( generate_vml.ContractionTypeAllWallSinkBilinearsSpecificMomentum(prop_f, prop_f, [mom_pair_pi_over_l], coswall, fn) )

    fn = "../results/coswallsink_bilinears_%g_%g_BB" % (m,m)
    contract.addMeas( generate_vml.ContractionTypeAllWallSinkBilinearsSpecificMomentum(prop_b, prop_b, [mom_pair_pi_over_l], coswall, fn) )


#HL combination
prop_l_f = "prop_m%g_F" % masses[0]
prop_l_b = "prop_m%g_B" % masses[0]
prop_h_f = "prop_m%g_F" % masses[1]
prop_h_b = "prop_m%g_B" % masses[1]

fn = "../results/bilinears_%g_%g_FF" % (masses[1],masses[0])
contract.addMeas( generate_vml.ContractionTypeAllBilinears(prop_h_f, prop_l_f, [mom_pi_over_l, mom_zero], fn) )

fn = "../results/bilinears_%g_%g_BB" % (masses[1],masses[0])
contract.addMeas( generate_vml.ContractionTypeAllBilinears(prop_h_b, prop_l_b, [mom_pi_over_l, mom_zero], fn) )

fn = "../results/coswallsink_bilinears_%g_%g_FF" % (masses[1],masses[0])
contract.addMeas( generate_vml.ContractionTypeAllWallSinkBilinearsSpecificMomentum(prop_h_f, prop_l_f, [mom_pair_pi_over_l], coswall, fn) )

fn = "../results/coswallsink_bilinears_%g_%g_BB" % (masses[1],masses[0])
contract.addMeas( generate_vml.ContractionTypeAllWallSinkBilinearsSpecificMomentum(prop_h_b, prop_l_b, [mom_pair_pi_over_l], coswall, fn) )

#Topological charge
fn = "../results/topq"
contract.addMeas( generate_vml.ContractionTypeTopologicalCharge(20,1,1e-12,-1,0.45, fn) )


contract.write("contract_arg.vml")
                                                                      
