import generate_vml


generate_vml.Globals.set_bc(["BND_CND_GPARITY","BND_CND_GPARITY","BND_CND_PRD","BND_CND_APRD"])
props = generate_vml.JobPropagatorArgs() 

masses = [0.01]
momcombs = [ [1,1,0], [-1,-1,0], [1,-3,0], [-1,3,0] ]  #units of pi/2L for G-parity dirs. Momentum components must differ by integer multiples of 2pi/L = 4* pi/2L
momcombtags = ['pp','mm','pm','mp']

for m in masses:
    for f in [0,1]:
        for mc in range(len(momcombs)):
            mcvals = momcombs[mc]
            mctag = momcombtags[mc]

            for tbc in ['PRD','APRD']:                
                tag = "prop_f%d_m%g_t%s_%s" % (f,m,tbc,mctag)
                fpartner_tag = "prop_f%d_m%g_t%s_%s" % (1-f,m,tbc,mctag)

                tbcstr = "BND_CND_%s" % tbc
                prop = generate_vml.PropagatorArg(tag,m,tbcstr)
                prop.setMomentumSource(0,mcvals,f)
                prop.specifyFlavorPartner(fpartner_tag)
                props.addPropagator(prop)

            #Combination
            ctags = ['F','B']
            ccomb = ["A_PLUS_B","A_MINUS_B"]
            for c in range(2):
                tag = "prop_f%d_m%g_t%s_%s" % (f,m,ctags[c],mctag)
                fpartner_tag = "prop_f%d_m%g_t%s_%s" % (1-f,m,ctags[c],mctag)

                tag_prd = "prop_f%d_m%g_tPRD_%s" % (f,m,mctag)
                tag_aprd = "prop_f%d_m%g_tAPRD_%s" % (f,m,mctag)

                prop = generate_vml.PropagatorArg(tag, m)
                prop.setCombinationSource( tag_prd, tag_aprd, ccomb[c])
                prop.setGparityFlavor(f)
                prop.specifyFlavorPartner(fpartner_tag)
                props.addPropagator(prop)




props.write("prop_arg.vml")


contract = generate_vml.GparityContractArg("../configurations/ckpoint_lat.%d", 300, 5, 910, 
                                            generate_vml.FixGaugeArg("FIX_GAUGE_COULOMB_T",0,1,32,1e-08,10000))

#momenta here are in units of pi consistently
#16^3 x 32 lattice
Linv = 1.0/8
momenta = [ [Linv,Linv,0], [-Linv,-Linv,0], [Linv,-Linv,0], [-Linv,Linv,0] ] 
propcombs = [ ['mm','pp'], ['mp','pm'] ]  #  tr A [G^mm]^dag B [G^pp]  and  tr A [G^mp]^dag B [G^pm]  where A,B are arbitrary SCFmatrix. Total source momenta are pp and pm 

for m in masses:
    for fb in ['F','B']:
        for c in range(len(propcombs)):
            prop_a = "prop_f%d_m%g_t%s_%s" % (0,m,fb,propcombs[c][0])
            prop_b = "prop_f%d_m%g_t%s_%s" % (0,m,fb,propcombs[c][1])

            fn = "../results/bilinears_%g_%g_%s%s_%s_%s" % (m,m,fb,fb,propcombs[c][0],propcombs[c][1])
            contract.addMeas( generate_vml.ContractionTypeAllBilinears(prop_a, prop_b, momenta, fn) )

contract.write("contract_arg.vml")
                                                                      
