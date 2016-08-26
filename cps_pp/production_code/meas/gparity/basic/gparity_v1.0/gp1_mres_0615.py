import generate_vml

#Single gauge fixed momentum source propagator with antiperiodic BCs for both flavors
generate_vml.Globals.set_bc(["BND_CND_GPARITY","BND_CND_PRD","BND_CND_PRD","BND_CND_APRD"])
props = generate_vml.JobPropagatorArgs() 

masses = [0.01]
p = [1,0,0]

for m in masses:
    for flav in [0,1]:
        prop = generate_vml.PropagatorArg("prop_f%d_m%g_APRD" % (flav,m), m , "BND_CND_APRD")
        prop.setMomentumSource(0,p,flav,True)
        prop.storeMidPoint()
        prop.specifyFlavorPartner("prop_f%d_m%g_APRD" % (1-flav,m) )
        props.addPropagator(prop)


props.write("prop_arg.vml")

contract = generate_vml.GparityContractArg("configurations/ckpoint_lat.%d", 500, 5, 505, 
                                           generate_vml.FixGaugeArg("FIX_GAUGE_COULOMB_T",0,1,1,1e-08,10000))

#Measure mres
for m in masses:
    contract.addMeas( generate_vml.ContractionTypeMres("prop_f0_m%g_APRD" %m,"results/mres_%g" %m) )

contract.write("contract_arg.vml")
                                                                      
