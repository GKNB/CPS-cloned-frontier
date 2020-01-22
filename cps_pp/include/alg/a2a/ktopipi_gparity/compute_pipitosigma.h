#ifndef _COMPUTE_PIPI_TO_SIGMA_H
#define _COMPUTE_PIPI_TO_SIGMA_H

#include<alg/a2a/mesonfield.h>

//Compute stationary sigma meson two-point function with and without GPBC
CPS_START_NAMESPACE

#define USE_TIANLES_CONVENTIONS
#define NODE_LOCAL true

template<typename mf_Policies>
struct ComputePiPiToSigmaContractions{
  //This is identical to the sigma->pipi up to an overall - sign, a sign change of the momenta and swapping tsrc<->tsnk. However we compute it explicitly here rather than reusing it because we want to compute only on a limited number of source timeslices
  
  //This has 2 Wick contractions. The first is disconnected, and we are able to re-use the sigma bubble from the sigma->sigma contractions and the pipi bubble from the pipi->pipi contractions
  //The second is connected, and has to be computed explicitly
  //  +sqrt(6)/2 \Theta_y \Theta_x tr( g5 s3 G(y1,tsrc; x1,tsnk) G(x2,tsnk; y3,tsrc) g5 s3 G(y4,tsrc; y2, tsrc) )
  //  +sqrt(6)/2 \Theta_y \Theta_x tr( g5 s3 V(y1,tsrc) W^dag(x1,tsnk) V(x2,tsnk) W^dag(y3,tsrc) g5 s3 V(y4,tsrc) W^dag(y2, tsrc) )
  //  +sqrt(6)/2 tr( [[\Theta_y12 W^dag(y2, tsrc) g5 s3 V(y1,tsrc)]] [[\Theta_x W^dag(x1,tsnk) V(x2,tsnk)]] [[\Theta_y34 W^dag(y3,tsrc) g5 s3 V(y4,tsrc)]]  )
  //  +sqrt(6)/2 tr( mf_piA(tsrc,tsrc) mf_sigma(tsnk,tsnk) mf_piB(tsrc,tsrc)  )

  //\Theta_x = \sum_x1,x2 exp(+i psnk2.y2) exp(+i psnk1.y1 )
  //\Theta_y = \sum_y1,y2,y3,y4  exp(-i psrc1.y1 ) exp(-i psrc2.y2) exp(-i psrc3.y3 ) exp(-i psrc4.y4)     

  //When split sources
  //  +sqrt(6)/4 tr( [mf_piB(tsrc,tsrc) mf_piA(tsrc-delta,tsrc-delta) + mf_piB(tsrc-delta,tsrc-delta) mf_piA(tsrc,tsrc) ] mf_sigma(tsnk,tsnk)   )

  //We label the inner pion as pi1, and it is this pion for which we specify the momentum

  //  +sqrt(6)/4 tr( [mf_pi1(tsrc,tsrc) mf_pi2(tsrc-delta,tsrc-delta) + mf_pi2(tsrc-delta,tsrc-delta) mf_pi1(tsrc,tsrc) ] mf_sigma(tsnk,tsnk)   )
  
  static void computeConnected(fMatrix<typename mf_Policies::ScalarComplexType> &into,
			       std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_sigma,
			       std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_pi1, //inner pion
			       std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_pi2, //outer pion
			       const int tsep_pipi, const int tstep_src
		      ){
    typedef typename mf_Policies::ComplexType ComplexType;
    typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
    
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    into.resize(Lt,Lt); into.zero();

    if(Lt % tstep_src != 0) ERR.General("ComputeSigmaToPipiContractions","computeConnected(..)","tstep_src must divide the time range exactly\n"); 

    //Distribute load over all nodes
    int work = Lt*Lt/tstep_src;
    int node_work, node_off; bool do_work;
    getNodeWork(work,node_work,node_off,do_work);

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    if(!UniqueID()){ printf("Gathering meson fields\n");  fflush(stdout); }

    //Only get the meson fields we actually need
    std::vector<bool> tslice_sigma_mask(Lt, false), tslice_pi1_mask(Lt, false), tslice_pi2_mask(Lt, false);

    if(do_work){
      for(int tt=node_off; tt<node_off + node_work; tt++){
	int tsnk = tt % Lt; //sink time
	int tsrc = tt / Lt * tstep_src; //source time
	int tsrc2 = (tsrc-tsep_pipi+Lt) % Lt;

	tslice_sigma_mask[tsnk] = tslice_pi1_mask[tsrc] = tslice_pi2_mask[tsrc2] = true;
      }
    }
    
    nodeGetMany(3,
		&mf_sigma, &tslice_sigma_mask,
		&mf_pi1, &tslice_pi1_mask,
		&mf_pi2, &tslice_pi2_mask);
    cps::sync();
#endif

    //Do the contraction  -sqrt(6)/4 tr( [mf_pi1(tsrc,tsrc) mf_pi2(tsrc-delta,tsrc-delta) + mf_pi2(tsrc-delta,tsrc-delta) mf_pi1(tsrc,tsrc) ] mf_sigma(tsnk,tsnk)   )
    //We can relate the two terms by g5-hermiticity and invoking parity and the reality of the correlation function under the ensemble average
#define PIPI_SIGMA_USE_G5_HERM

    if(do_work){
      for(int tt=node_off; tt<node_off + node_work; tt++){
	int tsnk = tt % Lt; //sink time
	int tsrc = tt / Lt * tstep_src; //source time
	int tsrc2 = (tsrc-tsep_pipi+Lt) % Lt;
	int tdis = (tsnk - tsrc + Lt) % Lt;
	
#ifdef PIPI_SIGMA_USE_G5_HERM
	A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> pi_prod;
	mult(pi_prod, mf_pi1[tsrc], mf_pi2[tsrc2],NODE_LOCAL);
	
	ScalarComplexType incr(0,0);
	incr += trace(pi_prod, mf_sigma[tsnk]);

	into(tsrc,tdis) +=  -sqrt(6.)/2. * incr; 	
#else
	A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> pi_prod_1, pi_prod_2;
	mult(pi_prod_1, mf_pi2[tsrc2], mf_pi1[tsrc],NODE_LOCAL);
	mult(pi_prod_2, mf_pi1[tsrc], mf_pi2[tsrc2],NODE_LOCAL);

	ScalarComplexType incr(0,0);
	incr += trace(pi_prod_1, mf_sigma[tsnk]);
	incr += trace(pi_prod_2, mf_sigma[tsnk]);

	into(tsrc,tdis) +=  -sqrt(6.)/4. * incr; //extra 1/2 from average of 2 topologies
#endif
      }
    }
    into.nodeSum();

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeDistributeMany(3,&mf_sigma,&mf_pi1,&mf_pi2);
#endif
  }
    
  //We provide the momentum index of the second (inner) sink pion
  template<typename SigmaMomentumPolicy, typename PionMomentumPolicy>
  static void computeConnected(fMatrix<typename mf_Policies::ScalarComplexType> &into,
			       MesonFieldMomentumPairContainer<mf_Policies> &mf_sigma_con, const SigmaMomentumPolicy &sigma_mom, const int pidx_sigma,
			       MesonFieldMomentumContainer<mf_Policies> &mf_pion_con, const PionMomentumPolicy &pion_mom, const int pidx_pi1,
			       const int tsep_pipi, const int tstep_src
		      ){
    if(sigma_mom.nAltMom(pidx_sigma) != 1) ERR.General("ComputeSigmaContractions","compute","Sigma with alternate momenta not implemented");

    //Work out the momenta we need
    ThreeMomentum p_pi1_src = pion_mom.getMesonMomentum(pidx_pi1);
    ThreeMomentum p_pi2_src = -p_pi1_src; //total zero momentum


    ThreeMomentum pWdag_sigma_snk = sigma_mom.getWdagMom(pidx_sigma);
    ThreeMomentum pV_sigma_snk = sigma_mom.getVmom(pidx_sigma);
    
    assert(mf_sigma_con.contains(pWdag_sigma_snk,pV_sigma_snk));
    assert(mf_pion_con.contains(p_pi1_src));
    assert(mf_pion_con.contains(p_pi2_src));
    
    //Gather the meson fields
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_sigma = mf_sigma_con.get(pWdag_sigma_snk, pV_sigma_snk);
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_pi1 = mf_pion_con.get(p_pi1_src); //meson field of the inner pion
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_pi2 = mf_pion_con.get(p_pi2_src);

    computeConnected(into, mf_sigma, mf_pi1, mf_pi2, tsep_pipi, tstep_src);
  }

  //Allows for moving pipi->sigma. Provide index of momenta for both pions; this then uniquely identifies the sigma momentum
  template<typename SigmaMomentumPolicy, typename PionMomentumPolicy>
  static void computeConnected(fMatrix<typename mf_Policies::ScalarComplexType> &into,
			       MesonFieldMomentumContainer<mf_Policies> &mf_sigma_con, const SigmaMomentumPolicy &sigma_mom,
			       MesonFieldMomentumContainer<mf_Policies> &mf_pion_con, const PionMomentumPolicy &pion_mom, const int pidx_pi1, const int pidx_pi2,
			       const int tsep_pipi, const int tstep_src
		      ){
    //Work out the momenta we need
    ThreeMomentum p_pi1_src = pion_mom.getMesonMomentum(pidx_pi1);
    ThreeMomentum p_pi2_src = pion_mom.getMesonMomentum(pidx_pi2);

    ThreeMomentum p_sigma_snk = -(p_pi1_src + p_pi2_src);

    assert(mf_sigma_con.contains(p_sigma_snk));
    assert(mf_pion_con.contains(p_pi1_src));
    assert(mf_pion_con.contains(p_pi2_src));
    
    //Gather the meson fields
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_sigma = mf_sigma_con.get(p_sigma_snk);
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_pi1 = mf_pion_con.get(p_pi1_src); //meson field of the inner pion
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_pi2 = mf_pion_con.get(p_pi2_src);

    computeConnected(into, mf_sigma, mf_pi1, mf_pi2, tsep_pipi, tstep_src);
  }





  //Tianle computes the product of the traces online and adds it to his answer. For consistency I do the same here
  static void computeDisconnectedDiagram(fMatrix<typename mf_Policies::ScalarComplexType> &into,
  					 const fVector<typename mf_Policies::ScalarComplexType> &sigma_bubble,
  					 const fVector<typename mf_Policies::ScalarComplexType> &pipi_bubble,  
  					 const int tstep_src){
    //-sqrt(6)/4 \Theta_y \Theta_x \tr( G(x2, tsnk; x1, tsnk) ) \tr( g5 G(y1, tsrc; y3, tsrc) g5 s3 G(y4, tsrc; y2, tsrc) s3 ) 
    //-sqrt(6)/4 \Theta_y \Theta_x \tr( V(x2, tsnk)W^dag(x1, tsnk) ) \tr( g5 V(y1, tsrc)W^dag(y3, tsrc) g5 s3 V(y4, tsrc)W^dag(y2, tsrc) s3 ) 
    //-sqrt(6)/4 \tr( [[\Theta_x W^dag(x1, tsnk) V(x2, tsnk)]] ) \tr( [[\Theta_y12 W^dag(y2, tsrc) s3 g5 V(y1, tsrc)]] [[\Theta_y34 W^dag(y3, tsrc) g5 s3 V(y4, tsrc)]] ) 
    //-sqrt(6)/4 \tr( M_sigma(tsnk,tsnk) ) \tr( M_piA(tsrc,tsrc) M_piB(tsrc,tsrc)) 

    //When separated
    //-sqrt(6)/8 \tr( M_sigma(tsnk,tsnk) ) [  \tr( M_piA(tsrc-delta,tsrc-delta) M_piB(tsrc,tsrc)) + \tr( M_piA(tsrc,tsrc) M_piB(tsrc-delta,tsrc-delta))   ] 
    //-sqrt(6)/8 \tr( M_sigma(tsnk,tsnk) ) [  \tr( M_pi2(tsrc-delta,tsrc-delta) M_pi1(tsrc,tsrc)) + \tr( M_pi1(tsrc,tsrc) M_pi2(tsrc-delta,tsrc-delta))   ] 
    //-sqrt(6)/4 \tr( M_sigma(tsnk,tsnk) ) [  M_pi1(tsrc,tsrc)) \tr( M_pi2(tsrc-delta,tsrc-delta)  ] 
    
    //\Theta_x = \sum_x1,x2 exp(+i psnk2.y2) exp(+i psnk1.y1 )
    //\Theta_y = \sum_y1,y2,y3,y4  exp(-i psrc1.y1 ) exp(-i psrc2.y2) exp(-i psrc3.y3 ) exp(-i psrc4.y4)     
    
    //As pipi total momentum = 0,    psrc1+psrc2 = -psrc3-psrc4
    
    //We have pipi_bubble(t, p) = 0.5 * tr( M_pi1(p, t) M_pi2(-p, t-delta) )  which is created by ComputePiPiGparity::computeFigureVdis
    //Thus we compute
    //-sqrt(6)/2 \tr( M_sigma(tsnk,tsnk) ) B(tsrc)
        
    int Lt = GJP.Tnodes()*GJP.TnodeSites();

    assert(sigma_bubble.size() == Lt);
    assert(pipi_bubble.size() == Lt);
	
    into.resize(Lt,Lt); into.zero();
    double coeff = -sqrt(6.)/2;
#ifdef USE_TIANLES_CONVENTIONS
    coeff *= -1.;
#endif
    
    for(int tsrc=0; tsrc<Lt; tsrc+=tstep_src){
      for(int tsep=0; tsep<Lt; tsep++){
	int tsnk = (tsrc + tsep) % Lt;
  	into(tsrc, tsep) = coeff * sigma_bubble(tsnk) * pipi_bubble(tsrc);
      }
    }
  }
};


#undef NODE_LOCAL
#undef USE_TIANLES_CONVENTIONS

CPS_END_NAMESPACE

#endif

