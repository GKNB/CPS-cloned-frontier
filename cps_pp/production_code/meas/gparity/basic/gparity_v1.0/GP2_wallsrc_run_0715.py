import generate_vml
import re

def momflip(s):
    m = re.search(r'p(m?\d)(m?\d)',s)
    p = [ m.group(1), m.group(2) ]
    mp = ["",""]
    for i in range(2):
        if(p[i][0] == 'm'):
            mp[i] = p[i][1]
        else:
            mp[i] = "m%c" % p[i][0] 
    out = re.sub("p%s%s" % (p[0],p[1]), "p%s%s" % (mp[0],mp[1]), s)
    return out

def parse_mom(s):
    m = re.search(r'p(m?\d)(m?\d)',s)
    p = [ m.group(1), m.group(2) ]
    for i in range(2):
        if(p[i][0] == 'm'):
            p[i] = "-%c" % p[i][1]
    pd = [ float(p[0]), float(p[1]) ]
    return pd

def ptostr(p):
    s = "p%d%d" % (p[0],p[1])
    s = re.sub(r'-','m',s)
    #print "ptostr([%d,%d]) -> %s" % ( p[0],p[1], s)
    return s
    


generate_vml.Globals.set_bc(["BND_CND_GPARITY","BND_CND_GPARITY","BND_CND_PRD","BND_CND_APRD"])
props = generate_vml.JobPropagatorArgs() 

masses = [0.01,0.032]
moms = [ [1,1,0], [-1,-1,0] ] #  , [1,-3,0], [-1,3,0] , [3,-1,0], [-3,1,0] ]

for m in masses:
    for momdir in range(len(moms)):
        dirstr = ptostr(moms[momdir]) #dirstrs[momdir]
        dirstr_opp = momflip(dirstr)

        #if(m == masses[0]):
        #    print "Flip %s -> %s" % (dirstr,dirstr_opp)

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
#output naming conventions in units of pi/L
#16^3 x 32 lattice
L = 16.0

bcstubs = ["F","B"] #,"APRD","PRD"]
bccombs = ["FF","BB"] #,"AA","PP"]

#Bilinears
for bc in range(len(bccombs)):
    for m1 in masses:
        for m2 in masses:
            for momdir1 in range(len(moms)):
                for momdir2 in range(len(moms)):
                    prop1 = "prop_m%g_%s_%s" % (m1,bcstubs[bc],ptostr(moms[momdir1]))
                    prop2 = "prop_m%g_%s_%s" % (m2,bcstubs[bc],ptostr(moms[momdir2]))

                    pstr1 = momflip(ptostr(moms[momdir1])) #first prop is daggered, which flips its source momentum
                    pstr2 = ptostr(moms[momdir2])

                    p1 = parse_mom(pstr1)
                    p2 = parse_mom(pstr2)
                    
                    psum = [ p1[0]+p2[0], p1[1]+p2[1], 0 ]
                    
                    #Skip anywhere the total momentum in either direction is > pi/L
                    if( abs(psum[0]) > 2 or abs(psum[1]) > 2):
                        continue
                    
                    #To keep the number of measurements real, just consider the total momenta where the x-component is >=0
                    #if( psum[0] < 0 ):
                    #    continue

                    #Also skip combinations where m1 < m2. These and the above can always be reconstructed using the cconj reln and g5-hermiticity
                    if( m1 < m2 ):
                        continue


                    #if(m1 == masses[0] and m2 == masses[0] and bc == 0):
                    print "dag(%s) * (%s) has total source momentum (%d,%d)" % (prop1,prop2,psum[0],psum[1])

                    #point-sink bilinears
                    fn = "results/bilinears_%g_%g_%s_%s_%s" % (m1,m2,pstr1,pstr2,bccombs[bc])
                    contract.addMeas( generate_vml.ContractionTypeAllBilinears(prop1, prop2, [ [psum[0]*0.5/L,psum[1]*0.5/L,0] ], fn) )
                    

                    #for wall-sink bilinears, where we apply a separate momentum projection to each quark, we can choose several momentum combinations that combine to give the same total momentum
                    #momenta are discretized in units of pi/2L, and we consider momentum components up to 3pi/2L
                    #5 choices of total momentum:  (2,2), (2,-2), (-2,2), (-2,-2), (0,0)
                    #do a lazy evaluation where we just loop over the allowed quark momenta and check their total
                    qmoms = [ [1,1], [-1,-1] ]#  , [1,-3], [-1,3], [-3,1], [3,-1] ]

                    mom_combs = []
                    for ps1 in qmoms:
                        for ps2 in qmoms:
                            pssum = [ ps1[0]+ps2[0], ps1[1]+ps2[1] ]
                            if(pssum[0] == psum[0] and pssum[1] == psum[1]):
                                print "For ptot = (%d,%d), found allowed mom comb (%d,%d) + (%d,%d)" % (psum[0],psum[1], ps1[0],ps1[1], ps2[0],ps2[1])
                                ps1_full = [  ps1[0]*0.5/L, ps1[1]*0.5/L, 0.0 ]
                                ps2_full = [  ps2[0]*0.5/L, ps2[1]*0.5/L, 0.0 ]

                                mom_combs.append( [ps1_full,ps2_full] )
                        
                    #print "For ptot = (%d,%d) found momcombs: "% (psum[0],psum[1])
                    #print mom_combs

                    fn = "results/wallsink_bilinears_%g_%g_%s_%s_%s" % (m1,m2,pstr1,pstr2,bccombs[bc])
                    contract.addMeas( generate_vml.ContractionTypeAllWallSinkBilinearsSpecificMomentum(prop1, prop2, mom_combs, 0, fn) )




    

# #Bag parameters
# #The heavy quark is the one that is daggered. We should pick momentum combinations that sum to zero

ml = masses[0]
mh = masses[1]

for momdir1 in range(len(moms)):
    for momdir2 in range(len(moms)):
        prop_h_F = "prop_m%g_%s_%s" % (mh,"F",ptostr(moms[momdir1]))
        prop_l_F = "prop_m%g_%s_%s" % (ml,"F",ptostr(moms[momdir2]))

        pstr1 = momflip(ptostr(moms[momdir1])) #first prop is daggered, which flips its source momentum
        pstr2 = ptostr(moms[momdir2])

        p1 = parse_mom(pstr1)
        p2 = parse_mom(pstr2)

        psum_F = [ p1[0]+p2[0], p1[1]+p2[1], 0 ]
        
        if(psum_F[0] != 0 or psum_F[1] != 0):
            continue

        for momdir3 in range(len(moms)):
            for momdir4 in range(len(moms)):
                prop_h_B = "prop_m%g_%s_%s" % (mh,"B",ptostr(moms[momdir3]))
                prop_l_B = "prop_m%g_%s_%s" % (ml,"B",ptostr(moms[momdir4]))

                pstr3 = momflip(ptostr(moms[momdir3])) #first prop is daggered, which flips its source momentum
                pstr4 = ptostr(moms[momdir4])

                p3 = parse_mom(pstr3)
                p4 = parse_mom(pstr4)

                psum_B = [ p3[0]+p4[0], p3[1]+p4[1], 0 ]

                if(psum_B[0] != 0 or psum_B[1] != 0):
                    continue        

                print "Found bag comb  %s, %s  :  %s, %s" % (prop_h_F,prop_l_F,prop_h_B,prop_l_B)

                fn = "results/OVVpAA_%g_%g_%s_%s_%s_%s" % (mh,ml,pstr1,pstr2,pstr3,pstr4)
                contract.addMeas( generate_vml.ContractionTypeOVVpAA(prop_h_F,prop_l_F,prop_h_B,prop_l_B, fn) )



#mres
#Just use the APRD props with mom 1,1
for m in masses:
    fn = "results/mres_%g" % m
    contract.addMeas( generate_vml.ContractionTypeMres("prop_m%g_APRD_p11" % m,fn) )

contract.write("contract_arg.vml")
                                                                      
