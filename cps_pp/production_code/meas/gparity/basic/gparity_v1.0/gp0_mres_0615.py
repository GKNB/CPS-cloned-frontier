import generate_vml

#Single gauge fixed wall source propagator with antiperiodic BCs
generate_vml.Globals.set_bc(["BND_CND_PRD","BND_CND_PRD","BND_CND_PRD","BND_CND_APRD"])
props = generate_vml.JobPropagatorArgs() 

masses = [0.01]
for m in masses:
    pra = generate_vml.PropagatorArg("prop_m%g_APRD" % m, m , "BND_CND_APRD")
    pra.setWallSource(0)
    pra.gaugeFixSource()
    pra.storeMidPoint()
    props.addPropagator(pra)


props.write("prop_arg.vml")

contract = generate_vml.GparityContractArg("../configurations/ckpoint_lat.%d", 605, 5, 610, 
                                           generate_vml.FixGaugeArg("FIX_GAUGE_COULOMB_T",0,1,32,1e-08,10000))

#Measure mres
for m in masses:
    contract.addMeas( generate_vml.ContractionTypeMres("prop_m%g_APRD" %m,"results/mres_%g" %m) )

contract.write("contract_arg.vml")
                                                                      
