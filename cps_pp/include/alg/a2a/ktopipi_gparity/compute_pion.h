#ifndef _COMPUTE_PION_H
#define _COMPUTE_PION_H

#include<alg/a2a/mesonfield.h>

CPS_START_NAMESPACE

template<typename mf_Policies>
class ComputePion{
public:
  
  //Compute the two-point function using the pre-generated meson fields.
  //The pion is given the momentum associated with index 'pidx' in the RequiredMomentum
  //result is indexed by (tsrc, tsep)  where tsep is the source-sink separation
  template<typename PionMomentumPolicy>
  static void compute(fMatrix<typename mf_Policies::ScalarComplexType> &into, MesonFieldMomentumContainer<mf_Policies> &mf_ll_con, const PionMomentumPolicy &pion_mom, const int pidx){
    typedef typename mf_Policies::ComplexType ComplexType;
    typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
    
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    into.resize(Lt,Lt);

    ThreeMomentum p_pi_src = pion_mom.getMesonMomentum(pidx); //exp(-ipx)
    ThreeMomentum p_pi_snk = -p_pi_src; //exp(+ipx)

    assert(mf_ll_con.contains(p_pi_src));
    assert(mf_ll_con.contains(p_pi_snk));
	   
    //Construct the meson fields
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_ll_src = mf_ll_con.get(p_pi_src);
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_ll_snk = mf_ll_con.get(p_pi_snk);
#ifdef NODE_DISTRIBUTE_MESONFIELDS
    if(!UniqueID()){ printf("Gathering meson fields\n");  fflush(stdout); }
    nodeGetMany(2,&mf_ll_src,&mf_ll_snk);
    sync();
#endif

    //Compute the two-point function
    // \sum_{xsnk,xsrc} exp(ip xsnk)exp(-ipxsrc) tr( S G(xsnk, tsnk; xsrc, tsrc) S G(xsrc,tsrc; xsnk, tsnk) ) 
    //where S is the vertex spin/color/flavor structure

    //= tr( [[\sum_{xsnk} exp(ip xsnk) w^dag(xsnk,tsnk) S v(xsnk,tsnk)]] [[\sum_{xsrc} exp(-ip xsrc) w^dag(xsrc,tsrc) S v(xsrc,tsrc) ]] ) 
    if(!UniqueID()){ printf("Starting trace\n");  fflush(stdout); }
    trace(into,mf_ll_snk,mf_ll_src);
    into *= ScalarComplexType(0.5,0);
    rearrangeTsrcTsep(into); //rearrange temporal ordering
    
    sync();
    if(!UniqueID()){ printf("Finished trace\n");  fflush(stdout); }

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeDistributeMany(2,&mf_ll_src,&mf_ll_snk);
#endif
  }

};



CPS_END_NAMESPACE

#endif
