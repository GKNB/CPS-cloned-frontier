#ifndef _COMPUTE_PIPI_H
#define _COMPUTE_PIPI_H

#include<alg/a2a/mesonfield.h>

CPS_START_NAMESPACE

//pipi(tsrc,tdis) = < pi(tsrc + tdis + tsep)pi(tsrc + tdis) pi(tsrc)pi(tsrc - tsep) >
//Following Daiqian's conventions, label the inner pion as pi1:    pi2(tsrc + tdis + tsep)pi1(tsrc + tdis) pi1(tsrc)pi2(tsrc - tsep)
//User can specify the source and sink momentum of the first pion. We assume that the total momentum is zero so this uniquely specifies the momenta of pi2
//tsep is the pi-pi time separation at src/snk

//C = 0.5 Tr(  G(x,y) S_2 G(y,r) S_2   *   G(r,s) S_2 G(s,x) S_2 )
//D = 0.25 Tr(  G(x,y) S_2 G(y,x) S_2 )  * Tr(  G(r,s) S_2 G(s,r) S_2 )
//R = 0.5 Tr(  G(x,r) S_2 G(r,s) S_2   *   G(s,y) S_2 G(y,x) S_2 )
//V = 0.5 Tr(  G(x,r) S_2 G(r,x) S_2 ) * Tr(  G(y,s) S_2 G(s,y) S_2 )

//where x,y are the source and sink coords of the first pion and r,s the second pion

//C = 0.5 Tr( [[w^dag(y) S_2 v(y)]] [[w^dag(r) S_2 * v(r)]] [[w^dag(s) S_2 v(s)]] [[w^dag(x) S_2 v(x)]] )
//D = 0.25 Tr( [[w^dag(y) S_2 v(y)]] [[w^dag(x) S_2 v(x)]] ) * Tr( [[w^dag(s) S_2 v(s)]] [[w^dag(r) S_2 v(r)]] )
//R = 0.5 Tr( [[w^dag(r) S_2 v(r)]] [[w^dag(s) S_2 * v(s)]][[w^dag(y) S_2 v(y)]] [[w^dag(x) S_2 v(x)]] )
//V = 0.25 Tr(  [[w^dag(r) S_2 v(r)]][[w^dag(x) S_2 v(x)]] ) * Tr(  [[w^dag(s) S_2 v(s)]][[w^dag(y) S_2 v(y)]] )

#define NODE_LOCAL true

#ifdef DISABLE_PIPI_PRODUCTSTORE
#define PIPI_COMPUTE_PRODUCT(INTO, A,B) MfType INTO; mult(INTO, A,B,NODE_LOCAL)
#else
#define PIPI_COMPUTE_PRODUCT(INTO, A,B) const MfType &INTO = products.getProduct(A,B,NODE_LOCAL)
#endif

//[1] = https://rbc.phys.columbia.edu/rbc_ukqcd/individual_postings/ckelly/Gparity/contractions_v1.pdf

template<typename mf_Policies>
class ComputePiPiGparity{
  typedef A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> MfType; 
  
  //C = \sum_{x,y,r,s}  \sum_{  0.5 Tr( [[w^dag(y) S_2 v(y)]] [[w^dag(r) S_2 * v(r)]] [[w^dag(s) S_2 v(s)]] [[w^dag(x) S_2 v(x)]] )
  //  = 0.5 Tr(  mf(p_pi1_snk) mf(p_pi2_src) mf(p_pi2_snk) mf(p_pi1_src) )
  inline static void figureC(fMatrix<typename mf_Policies::ScalarComplexType> &into,
			     const std::vector<MfType >& mf_pi1_src,
			     const std::vector<MfType >& mf_pi2_src,
			     const std::vector<MfType >& mf_pi1_snk,
			     const std::vector<MfType >& mf_pi2_snk,
			     MesonFieldProductStore<mf_Policies> &products,
			     const int tsrc, const int tsnk, const int tsep, const int Lt){
    typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
    int tsrc2 = (tsrc-tsep+Lt) % Lt;
    int tsnk2 = (tsnk+tsep) % Lt;
      
    int tdis = (tsnk - tsrc + Lt) % Lt;

    //Topology 1  x4=tsrc (pi1)  y4=tsnk (pi1)  r4=tsrc2 (pi2)  s4=tsnk2 (pi2)
    //Corresponds to Eq.6 in [1]
    ScalarComplexType topo1;
    {
      PIPI_COMPUTE_PRODUCT(prod_l, mf_pi1_snk[tsnk], mf_pi2_src[tsrc2]);
      PIPI_COMPUTE_PRODUCT(prod_r, mf_pi2_snk[tsnk2], mf_pi1_src[tsrc]);
      topo1 = 0.5 * trace(prod_l, prod_r);
    }
    

#ifdef DISABLE_EVALUATION_OF_SECOND_PIPI_TOPOLOGY
    into(tsrc, tdis) += topo1;    
#else
    //Topology 2  x4=tsrc (pi1) y4=tsnk2 (pi2)  r4=tsrc2 (pi2) s4=tsnk (pi1)
    //This topology is similar to the first due to g5-hermiticity and the G-parity complex conjugate relation 
    //but g5-hermiticity swaps the quark and antiquark momenta. In the original version (reenabled with DISABLE_EVALUATION_OF_SECOND_PIPI_TOPOLOGY) 
    //we overlooked this and so combined the diagrams into one

    //Corresponds to Eq.5 in [1]
    ScalarComplexType topo2;
    {
      PIPI_COMPUTE_PRODUCT(prod_l, mf_pi1_src[tsrc], mf_pi2_snk[tsnk2]);
      PIPI_COMPUTE_PRODUCT(prod_r, mf_pi2_src[tsrc2], mf_pi1_snk[tsnk]);
      topo2 = 0.5 * trace(prod_l, prod_r);
    }
    into(tsrc,tdis) += 0.5*topo1 + 0.5*topo2;

#endif
  }
  //D = 0.25 Tr( [[w^dag(y) S_2 v(y)]] [[w^dag(x) S_2 v(x)]] ) * Tr( [[w^dag(s) S_2 v(s)]] [[w^dag(r) S_2 v(r)]] )
  //where x,y are the source and sink coords of the first pion and r,s the second pion
  //2 different topologies to average over
  inline static void figureD(fMatrix<typename mf_Policies::ScalarComplexType> &into,
			     const std::vector<MfType >& mf_pi1_src,
			     const std::vector<MfType >& mf_pi2_src,
			     const std::vector<MfType >& mf_pi1_snk,
			     const std::vector<MfType >& mf_pi2_snk,
			     const int tsrc, const int tsnk, const int tsep, const int Lt){
    typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
    int tdis = (tsnk - tsrc + Lt) % Lt;
    int tsrc2 = (tsrc-tsep+Lt) % Lt; //source position of pi2 by convention
    int tsnk2 = (tsnk+tsep) % Lt;

    ScalarComplexType tr1(0,0), tr2(0,0), incr(0,0);

    //Topology 1  x4=tsrc (pi1)  y4=tsnk (pi1)  r4=tsrc2 (pi2)  s4=tsnk2 (pi2)
    incr += trace(mf_pi1_snk[tsnk] , mf_pi1_src[tsrc])   *    trace(mf_pi2_snk[tsnk2], mf_pi2_src[tsrc2]);

    //Topology 2  x4=tsrc (pi1) y4=tsnk2 (pi2)  r4=tsrc2 (pi2) s4=tsnk (pi1)
    incr += trace(mf_pi2_snk[tsnk2] , mf_pi1_src[tsrc])   *   trace(mf_pi1_snk[tsnk] , mf_pi2_src[tsrc2]);

    incr *= ScalarComplexType(0.5*0.25); //extra factor of 0.5 from average over 2 distinct topologies
    into(tsrc, tdis) += incr;
  }

  //R = 0.5 Tr( [[w^dag(r) S_2 v(r)]] [[w^dag(s) S_2 * v(s)]][[w^dag(y) S_2 v(y)]] [[w^dag(x) S_2 v(x)]] )
  //2 different topologies to average over
  inline static void figureR(fMatrix<typename mf_Policies::ScalarComplexType> &into,
			     const std::vector<MfType >& mf_pi1_src,
			     const std::vector<MfType >& mf_pi2_src,
			     const std::vector<MfType >& mf_pi1_snk,
			     const std::vector<MfType >& mf_pi2_snk,
			     MesonFieldProductStore<mf_Policies> &products,
			     const int tsrc, const int tsnk, const int tsep, const int Lt){
    typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
    int tdis = (tsnk - tsrc + Lt) % Lt;
    int tsrc2 = (tsrc-tsep+Lt) % Lt; //source position of pi2 by convention
    int tsnk2 = (tsnk+tsep) % Lt;

    //Topology 1    x4=tsrc (pi1) y4=tsnk (pi1)   r4=tsrc2 (pi2) s4=tsnk2 (pi2)
    //Correponds to fig 9 in [1]
    ScalarComplexType topo1;
    {
      PIPI_COMPUTE_PRODUCT(prod_l_top1, mf_pi2_src[tsrc2], mf_pi2_snk[tsnk2]);
      PIPI_COMPUTE_PRODUCT(prod_r_top1, mf_pi1_snk[tsnk], mf_pi1_src[tsrc]);
      topo1 = 0.5 * trace( prod_l_top1, prod_r_top1 );
    }
    
    //Topology 2    x4=tsrc (pi1)  y4=tsnk_outer (pi2)  r4=tsrc_outer (pi2) s4=tsnk (pi1)
    //Corresponds to fig 12 in [1]
    ScalarComplexType topo2;
    {
      PIPI_COMPUTE_PRODUCT(prod_l_top2, mf_pi2_src[tsrc2], mf_pi1_snk[tsnk]);
      PIPI_COMPUTE_PRODUCT(prod_r_top2, mf_pi2_snk[tsnk2], mf_pi1_src[tsrc]);
      topo2 = 0.5 * trace( prod_l_top2, prod_r_top2 );
    }
      
#ifdef DISABLE_EVALUATION_OF_SECOND_PIPI_TOPOLOGY
    into(tsrc, tdis) += 0.5*topo1 + 0.5*topo2;
#else

    //Like with the C diagrams we previously incorrectly used g5-hermiticity to combine the pairs of similar topologies. This can be re-enabled using DISABLE_EVALUATION_OF_SECOND_PIPI_TOPOLOGY
       
    //Correponds to fig 10 in [1]
    ScalarComplexType topo3;
    {
      PIPI_COMPUTE_PRODUCT(prod_l_top3, mf_pi2_snk[tsnk2], mf_pi2_src[tsrc2]);
      PIPI_COMPUTE_PRODUCT(prod_r_top3, mf_pi1_src[tsrc], mf_pi1_snk[tsnk]);
      topo3 = 0.5 * trace( prod_l_top3, prod_r_top3 );
    }
    //Correponds to fig 11 in [1]
    ScalarComplexType topo4;
    {
      PIPI_COMPUTE_PRODUCT(prod_l_top4, mf_pi1_snk[tsnk], mf_pi2_src[tsrc2]);
      PIPI_COMPUTE_PRODUCT(prod_r_top4, mf_pi1_src[tsrc], mf_pi2_snk[tsnk2]);
      topo4 = 0.5 * trace( prod_l_top4, prod_r_top4 );
    }

    into(tsrc, tdis) += 0.25*topo1 + 0.25*topo2 + 0.25*topo3 + 0.25*topo4;
#endif
  }


public:


  //char diag is C, D, R or V
  //tsep is the pion separation in the source/sink operators
  //tstep is the source timeslice sampling frequency. Use tstep_src = 1 to compute with sources on all timeslices
  //Output matrix ordering (tsrc, tdis)  where tdis = tsnk-tsrc
  //Note, 'products' is used as a storage for products of meson fields that might potentially be re-used later

  //Calculate for general momenta
  static void compute(fMatrix<typename mf_Policies::ScalarComplexType> &into, const char diag, 
		      const ThreeMomentum &p_pi1_src, const ThreeMomentum &p_pi2_src,
		      const ThreeMomentum &p_pi1_snk, const ThreeMomentum &p_pi2_snk, 
		      const int tsep, const int tstep_src,
		      MesonFieldMomentumContainer<mf_Policies> &src_mesonfields,
		      MesonFieldMomentumContainer<mf_Policies> &snk_mesonfields,
		      MesonFieldProductStore<mf_Policies> &products
#ifdef NODE_DISTRIBUTE_MESONFIELDS
		      , bool do_redistribute_src = true, bool do_redistribute_snk = true
#endif		      
		      ){
    if(!GJP.Gparity()) ERR.General("ComputePiPiGparity","compute(..)","Implementation is for G-parity only; different contractions are needed for periodic BCs\n"); 
    const int Lt = GJP.Tnodes()*GJP.TnodeSites();

    ThreeMomentum const* mom[4] = { &p_pi1_src, &p_pi2_src, &p_pi1_snk, &p_pi2_snk };
    for(int p=0;p<2;p++)
      if(! (p < 2 ? src_mesonfields.contains(*mom[p]) : snk_mesonfields.contains(*mom[p]) ) ) 
	ERR.General("ComputePiPiGparity","compute(..)","Meson field container doesn't contain momentum %s\n",mom[p]->str().c_str());

    if(Lt % tstep_src != 0) ERR.General("ComputePiPiGparity","compute(..)","tstep_src must divide the time range exactly\n"); 
    
    //Distribute load over all nodes
    int work = Lt*Lt/tstep_src;
    int node_work, node_off; bool do_work;
    getNodeWork(work,node_work,node_off,do_work);

    std::vector<MfType >& mf_pi1_src = src_mesonfields.get(p_pi1_src);
    std::vector<MfType >& mf_pi2_src = src_mesonfields.get(p_pi2_src);
    std::vector<MfType >& mf_pi1_snk = snk_mesonfields.get(p_pi1_snk);
    std::vector<MfType >& mf_pi2_snk = snk_mesonfields.get(p_pi2_snk);
    
#ifdef NODE_DISTRIBUTE_MESONFIELDS
    //Get the meson fields we require. Only get those for the timeslices computed on this node
    std::vector<bool> tslice_src_mask(Lt, false);
    std::vector<bool> tslice_snk_mask(Lt, false);
    if(do_work){
      for(int tt=node_off; tt<node_off + node_work; tt++){
	int rem = tt;
	int tsnk = rem % Lt; rem /= Lt; //sink time
	int tsrc = rem * tstep_src; //source time

	int tsrc2 = (tsrc-tsep+Lt) % Lt;
	int tsnk2 = (tsnk+tsep) % Lt;
	
	tslice_src_mask[tsrc] = tslice_src_mask[tsrc2] = true;
	tslice_snk_mask[tsnk] = tslice_snk_mask[tsnk2] = true;
      }
    }
    nodeGetMany(4,
		&mf_pi1_src, &tslice_src_mask,
		&mf_pi2_src, &tslice_src_mask,
		&mf_pi1_snk, &tslice_snk_mask,
		&mf_pi2_snk, &tslice_snk_mask);
#endif

    into.resize(Lt,Lt); into.zero();
    
    if(do_work){
      for(int tt=node_off; tt<node_off + node_work; tt++){
	int rem = tt;
	int tsnk = rem % Lt; rem /= Lt; //sink time
	int tsrc = rem * tstep_src; //source time

	if(diag == 'C')
	  figureC(into, mf_pi1_src, mf_pi2_src, mf_pi1_snk, mf_pi2_snk, products, tsrc,tsnk, tsep,Lt);
	else if(diag == 'D')
	  figureD(into, mf_pi1_src, mf_pi2_src, mf_pi1_snk, mf_pi2_snk, tsrc,tsnk, tsep,Lt);
	else if(diag == 'R')
	  figureR(into, mf_pi1_src, mf_pi2_src, mf_pi1_snk, mf_pi2_snk, products, tsrc,tsnk, tsep,Lt);
	else ERR.General("ComputePiPiGparity","compute","Invalid diagram '%c'\n",diag);
      }
    }
    into.nodeSum();

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    //Need to take care that there's no overlap in the source and sink meson fields lest we distribute one we intend to keep
    if(do_redistribute_snk && do_redistribute_src){
      nodeDistributeMany(4,&mf_pi1_src,&mf_pi2_src,&mf_pi1_snk,&mf_pi2_snk);
    }else if(do_redistribute_snk){
      nodeDistributeUnique(mf_pi1_snk, 2, &mf_pi1_src, &mf_pi2_src);
      nodeDistributeUnique(mf_pi2_snk, 2, &mf_pi1_src, &mf_pi2_src);
    }else if(do_redistribute_src){
      nodeDistributeUnique(mf_pi1_src, 2, &mf_pi1_snk, &mf_pi2_snk);
      nodeDistributeUnique(mf_pi2_src, 2, &mf_pi1_snk, &mf_pi2_snk);
    }  
#endif
  }

  //Source and sink meson field set is the same
  static void compute(fMatrix<typename mf_Policies::ScalarComplexType> &into, const char diag, 
		      const ThreeMomentum &p_pi1_src, const ThreeMomentum &p_pi2_src,
		      const ThreeMomentum &p_pi1_snk, const ThreeMomentum &p_pi2_snk, 
		      const int tsep, const int tstep_src,
		      MesonFieldMomentumContainer<mf_Policies> &srcsnk_mesonfields,
		      MesonFieldProductStore<mf_Policies> &products
#ifdef NODE_DISTRIBUTE_MESONFIELDS
		      , bool do_redistribute_src = true, bool do_redistribute_snk = true
#endif		      
		      ){
    compute(into,diag,p_pi1_src,p_pi2_src,p_pi1_snk,p_pi2_snk,tsep,tstep_src,
	    srcsnk_mesonfields,srcsnk_mesonfields,
	    products
#ifdef NODE_DISTRIBUTE_MESONFIELDS
	    , do_redistribute_src, do_redistribute_snk
#endif	
	    );
  }

  //Calculate for p_cm=(0,0,0)
  static void compute(fMatrix<typename mf_Policies::ScalarComplexType> &into, const char diag, 
		      const ThreeMomentum &p_pi1_src, const ThreeMomentum &p_pi1_snk, 
		      const int tsep, const int tstep_src,
		      MesonFieldMomentumContainer<mf_Policies> &mesonfields, MesonFieldProductStore<mf_Policies> &products
#ifdef NODE_DISTRIBUTE_MESONFIELDS
		      , bool do_redistribute_src = true, bool do_redistribute_snk = true
#endif		      
		      ){
    ThreeMomentum p_pi2_src = -p_pi1_src;
    ThreeMomentum p_pi2_snk = -p_pi1_snk;

    compute(into, diag, p_pi1_src, p_pi2_src, p_pi1_snk, p_pi2_snk, tsep, tstep_src, mesonfields, products
#ifdef NODE_DISTRIBUTE_MESONFIELDS
		      , do_redistribute_src, do_redistribute_snk
#endif		      
	    );
  }


  //Compute the pion 'bubble'  0.5 tr( mf(t, p_pi) mf(t-tsep, p_pi2) )  [note pi2 at earlier timeslice here]
  //output into vector element  [t]

  static void computeFigureVdis(fVector<typename mf_Policies::ScalarComplexType> &into, const ThreeMomentum &p_pi, const ThreeMomentum &p_pi2, const int tsep, MesonFieldMomentumContainer<mf_Policies> &mesonfields){
    if(!GJP.Gparity()) ERR.General("ComputePiPiGparity","computeFigureVdis(..)","Implementation is for G-parity only; different contractions are needed for periodic BCs\n"); 
    const int Lt = GJP.Tnodes()*GJP.TnodeSites();

    ThreeMomentum const* mom[4] = { &p_pi, &p_pi2 };
    for(int p=0;p<2;p++)
      if(!mesonfields.contains(*mom[p])) ERR.General("ComputePiPiGparity","computeFigureVdis(..)","Meson field container doesn't contain momentum %s\n",mom[p]->str().c_str());
    
    //Get the meson fields we require
    std::vector<MfType >& mf_pi = mesonfields.get(p_pi);
    std::vector<MfType >& mf_pi2 = mesonfields.get(p_pi2);

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeGetMany(2,&mf_pi,&mf_pi2);
#endif

    into.resize(Lt); 

    //Distribute load over all nodes
    int work = Lt; //consider a better parallelization
    int node_work, node_off; bool do_work;
    getNodeWork(work,node_work,node_off,do_work);
    
    if(do_work){
      for(int t=node_off; t<node_off + node_work; t++){
	int t2 = (t-tsep+Lt) % Lt;
	into(t) = typename mf_Policies::ScalarComplexType(0.5)* trace(mf_pi[t], mf_pi2[t2]);
      }
    }
    into.nodeSum();

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeDistributeMany(2,&mf_pi,&mf_pi2);
#endif
  }

  //p_cm = (0,0,0)
  static void computeFigureVdis(fVector<typename mf_Policies::ScalarComplexType> &into, const ThreeMomentum &p_pi, const int tsep, MesonFieldMomentumContainer<mf_Policies> &mesonfields){
    ThreeMomentum p_pi2 = -p_pi;
    computeFigureVdis(into, p_pi, p_pi2, tsep, mesonfields);
  }


};



#undef NODE_LOCAL
#undef PIPI_COMPUTE_PRODUCT


CPS_END_NAMESPACE
#endif
