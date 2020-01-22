#ifndef _COMPUTE_SIGMA_H
#define _COMPUTE_SIGMA_H

#include<alg/a2a/mesonfield.h>

//Compute stationary sigma meson two-point function with and without GPBC
CPS_START_NAMESPACE

template<typename mf_Policies>
struct ComputeSigmaContractions{
  //Sigma has 2 terms in it's Wick contraction:    
  //0.5 * tr( G(x1,tsnk; x2, tsnk) ) * tr( G(y1, tsrc; y2, tsrc) )
  //-0.5 tr( G(y2, tsrc; x1, tsnk) * G(x2, tsnk; y1, tsrc) )
  //where x are sink locations and y src locations
  
  //We perform the Fourier transform   \Theta_x\Theta_y = \sum_{x1,x2,y1,y2} exp(-i p1_snk x1) exp(-i p2_snk x2) exp(-i p1_src y1) exp(-i p2_src y2) 
  
  
  //The first term has a vacuum subtraction so we just compute tr( G(x1,x2) ) and do the rest offline
  //\Theta_x tr( G(x1,tsnk; x2, tsnk) ) = \Theta_x tr( V(x1,tsnk) W^dag(x2,tsnk) ) =  tr( [\Theta_x W^dag(x2,tsnk) V(x1,tsnk)]  ) = tr( M(tsnk,tsnk) )

  static void computeDisconnectedBubble(fVector<typename mf_Policies::ScalarComplexType> &into, 
					std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf){
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    into.resize(Lt);

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    if(!UniqueID()){ printf("Gathering meson fields\n");  fflush(stdout); }
    nodeGetMany(1,&mf);
#endif
    
    //Distribute load over all nodes
    int work = Lt; //consider a better parallelization
    int node_work, node_off; bool do_work;
    getNodeWork(work,node_work,node_off,do_work);
    
    if(do_work){
      for(int t=node_off; t<node_off + node_work; t++){
	into(t) = trace(mf[t]);
      }
    }
    into.nodeSum();

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeDistributeMany(1,&mf);
#endif
  }

  //Tianle computes the product of the traces online and adds it to his answer. For consistency I do the same here
  static void computeDisconnectedDiagram(fMatrix<typename mf_Policies::ScalarComplexType> &into,
					 const fVector<typename mf_Policies::ScalarComplexType> &sigma_bubble_src,
					 const fVector<typename mf_Policies::ScalarComplexType> &sigma_bubble_snk){
    //0.5 * tr( G(x1,tsnk; x2, tsnk) ) * tr( G(y1, tsrc; y2, tsrc) )
    typedef typename mf_Policies::ScalarComplexType Complex;
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    into.resize(Lt,Lt);
    for(int tsrc=0; tsrc<Lt; tsrc++){
      for(int tsep=0; tsep<Lt; tsep++){
	int tsnk = (tsrc + tsep) % Lt;
	into(tsrc, tsep) = Complex(0.5) * sigma_bubble_snk(tsnk) * sigma_bubble_src(tsrc); 
      }
    }
  }

  //The second term we compute in full
  //  \Theta_x \Theta_y -0.5 tr( G(y2, tsrc; x1, tsnk) * G(x2, tsnk; y1, tsrc) )
  //= \Theta_x \Theta_y -0.5 tr( V(y2, tsrc) * W^dag(x1, tsnk) * V(x2, tsnk)*W^dag(y1, tsrc) )
  //= -0.5 tr(  [ \Theta_x W^dag(x1, tsnk) * V(x2, tsnk)] * [ \Theta_y W^dag(y1, tsrc) V(y2, tsrc) ] )
  //= -0.5 tr( M(tsnk,tsnk) * M(tsrc, tsrc) )
  static void computeConnected(fMatrix<typename mf_Policies::ScalarComplexType> &into,			       
			       std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_src,
			       std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_snk){
    typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
    
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    into.resize(Lt,Lt);

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    if(!UniqueID()){ printf("Gathering meson fields\n");  fflush(stdout); }
    nodeGetMany(2,&mf_src,&mf_snk);
    cps::sync();
#endif

    if(!UniqueID()){ printf("Starting trace\n");  fflush(stdout); }
    trace(into,mf_snk,mf_src);
    into *= typename mf_Policies::ScalarComplexType(-0.5,0);
    rearrangeTsrcTsep(into); //rearrange temporal ordering
    
    cps::sync();
    if(!UniqueID()){ printf("Finished trace\n");  fflush(stdout); }

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeDistributeMany(2,&mf_src,&mf_snk);
#endif
  }



  //The functions below extract the meson fields from their containers then call the above. There are versions for meson fields stored in MesonFieldMomentumPairContainer (indexed by both quark momenta) and MesonFieldMomentumContainer (indexed by total momentum)
  template<typename SigmaMomentumPolicy>
  static void computeDisconnectedBubble(fVector<typename mf_Policies::ScalarComplexType> &into,
					MesonFieldMomentumPairContainer<mf_Policies> &mf_sigma_con, const SigmaMomentumPolicy &sigma_mom, const int pidx
		      ){
    if(sigma_mom.nAltMom(pidx) != 1) ERR.General("ComputeSigmaContractions","computeDisconnectedBubble","Sigma with alternate momenta not implemented. Idx %d with momenta %s %s has %d alternative momenta", pidx, sigma_mom.getWdagMom(pidx).str().c_str(),  sigma_mom.getVmom(pidx).str().c_str(), sigma_mom.nAltMom(pidx) );
    
    ThreeMomentum p1 = sigma_mom.getWdagMom(pidx);
    ThreeMomentum p2 = sigma_mom.getVmom(pidx);

    assert(mf_sigma_con.contains(p1,p2));
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf = mf_sigma_con.get(p1,p2);
    computeDisconnectedBubble(into, mf);
  }
  template<typename SigmaMomentumPolicy>
  static void computeDisconnectedBubble(fVector<typename mf_Policies::ScalarComplexType> &into,
					MesonFieldMomentumContainer<mf_Policies> &mf_sigma_con, const SigmaMomentumPolicy &sigma_mom, const int pidx){
    ThreeMomentum ptot = sigma_mom.getTotalMomentum(pidx);    
    assert(mf_sigma_con.contains(ptot));
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf = mf_sigma_con.get(ptot);
    computeDisconnectedBubble(into, mf);
  }


  template<typename SigmaMomentumPolicy>
  static void computeConnected(fMatrix<typename mf_Policies::ScalarComplexType> &into,
		      MesonFieldMomentumPairContainer<mf_Policies> &mf_sigma_con, const SigmaMomentumPolicy &sigma_mom, const int pidx_src, const int pidx_snk){
    if(sigma_mom.nAltMom(pidx_snk) != 1) ERR.General("ComputeSigmaContractions","computeConnected","Sigma (sink) with alternate momenta not implemented. Idx %d with momenta %s %s has %d alternative momenta", pidx_snk, sigma_mom.getWdagMom(pidx_snk).str().c_str(),  sigma_mom.getVmom(pidx_snk).str().c_str(), sigma_mom.nAltMom(pidx_snk) );

    if(sigma_mom.nAltMom(pidx_src) != 1) ERR.General("ComputeSigmaContractions","computeConnected","Sigma (source) with alternate momenta not implemented. Idx %d with momenta %s %s has %d alternative momenta", pidx_src, sigma_mom.getWdagMom(pidx_src).str().c_str(),  sigma_mom.getVmom(pidx_src).str().c_str(), sigma_mom.nAltMom(pidx_src) );
        
    ThreeMomentum p1_src = sigma_mom.getWdagMom(pidx_src);
    ThreeMomentum p2_src = sigma_mom.getVmom(pidx_src);

    ThreeMomentum p1_snk = sigma_mom.getWdagMom(pidx_snk);
    ThreeMomentum p2_snk = sigma_mom.getVmom(pidx_snk);

    assert(mf_sigma_con.contains(p1_src,p2_src));
    assert(mf_sigma_con.contains(p1_snk,p2_snk));
	   
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_src = mf_sigma_con.get(p1_src,p2_src);
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_snk = mf_sigma_con.get(p1_snk,p2_snk);
    computeConnected(into, mf_src, mf_snk);
  }

  //psink = -psrc
  template<typename SigmaMomentumPolicy>
  static void computeConnected(fMatrix<typename mf_Policies::ScalarComplexType> &into,
			       MesonFieldMomentumContainer<mf_Policies> &mf_sigma_con, const SigmaMomentumPolicy &sigma_mom, const int pidx_src){
    ThreeMomentum ptot_src = sigma_mom.getTotalMomentum(pidx_src);
    ThreeMomentum ptot_snk = -ptot_src;

    assert(mf_sigma_con.contains(ptot_src));
    assert(mf_sigma_con.contains(ptot_snk));
	   
    //Construct the meson fields
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_src = mf_sigma_con.get(ptot_src);
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_snk = mf_sigma_con.get(ptot_snk);
    computeConnected(into, mf_src, mf_snk);
  }


};


CPS_END_NAMESPACE

#endif

