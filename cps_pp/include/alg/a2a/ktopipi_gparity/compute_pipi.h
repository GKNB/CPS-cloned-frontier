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
#define PIPI_COMPUTE_PRODUCT(INTO, A,B) MfType INTO = products.getProduct(A,B,NODE_LOCAL)
#endif

//[1] = https://rbc.phys.columbia.edu/rbc_ukqcd/individual_postings/ckelly/Gparity/contractions_v1.pdf

template<typename mf_Policies>
class ComputePiPiGparity{
  typedef A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> MfType; 

public:
  struct Timings{
    char diag;
    double trace;
    double prod;
    double total_compute;
    double gather;
    double distribute;
    double gsum;
    double total;
    
    Timings(char diag): trace(0), prod(0), total_compute(0), gather(0), distribute(0), gsum(0), total(0), diag(diag){}

    void reset(){ trace=prod=total=0; }
    void report(bool do_reset = true){
      std::cout << diag << ": Trace=" << trace << "s Prod=" << prod << "s Total compute=" << total_compute << "s Gather=" << gather << "s Distribute=" << distribute << "s Reduce=" << gsum << "s Total=" << total << "s" << std::endl;
      if(do_reset) reset();
    }    
  };
  inline static Timings & timingsC(){ static Timings t('C'); return t; }
  inline static Timings & timingsD(){ static Timings t('D'); return t; }
  inline static Timings & timingsR(){ static Timings t('R'); return t; }
  inline static Timings & timingsV(){ static Timings t('V'); return t; }

  inline static Timings & getTimings(char c){
    switch(c){
    case 'C':
      return timingsC();
    case 'D':
      return timingsD();
    case 'R':
      return timingsR();
    case 'V':
      return timingsV();
    }
  }
      
  struct Options{
    bool redistribute_pi1_src; //don't hold on to the pi1_src meson field after contracting
    bool redistribute_pi2_src; //don't hold on to the pi2_src meson field after contracting
    bool redistribute_pi1_snk; //don't hold on to the pi1_snk meson field after contracting
    bool redistribute_pi2_snk; //don't hold on to the pi2_snk meson field after contracting

    Options(bool redistribute = true){
      redistribute_pi1_src = redistribute;
      redistribute_pi2_src = redistribute;
      redistribute_pi1_snk = redistribute;
      redistribute_pi2_snk = redistribute;
    }
  };
      
    
private:
  
  //C = \sum_{x,y,r,s}  \sum_{  0.5 Tr( [[w^dag(y) S_2 v(y)]] [[w^dag(r) S_2 * v(r)]] [[w^dag(s) S_2 v(s)]] [[w^dag(x) S_2 v(x)]] )
  //  = 0.5 Tr(  mf(p_pi1_snk) mf(p_pi2_src) mf(p_pi2_snk) mf(p_pi1_src) )
  inline static void figureC(fMatrix<typename mf_Policies::ScalarComplexType> &into,
			     const std::vector<MfType >& mf_pi1_src,
			     const std::vector<MfType >& mf_pi2_src,
			     const std::vector<MfType >& mf_pi1_snk,
			     const std::vector<MfType >& mf_pi2_snk,
			     MesonFieldProductStore<mf_Policies> &products,
			     const int tsrc, const int tsnk, const int tsep, const int Lt){
    Timings &timings = timingsC();
    timings.total_compute -= dclock();
    
    typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
    int tsrc2 = (tsrc-tsep+Lt) % Lt;
    int tsnk2 = (tsnk+tsep) % Lt;
      
    int tdis = (tsnk - tsrc + Lt) % Lt;

    //Topology 1  x4=tsrc (pi1)  y4=tsnk (pi1)  r4=tsrc2 (pi2)  s4=tsnk2 (pi2)
    //Corresponds to Eq.6 in [1]
    ScalarComplexType topo1;
    {
      timings.prod -= dclock();
      PIPI_COMPUTE_PRODUCT(prod_l, mf_pi1_snk[tsnk], mf_pi2_src[tsrc2]);
      PIPI_COMPUTE_PRODUCT(prod_r, mf_pi2_snk[tsnk2], mf_pi1_src[tsrc]);
      timings.prod += dclock();
      
      timings.trace -= dclock();
      topo1 = 0.5 * trace(prod_l, prod_r);
      timings.trace += dclock();
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
      timings.prod -= dclock();
      PIPI_COMPUTE_PRODUCT(prod_l, mf_pi1_src[tsrc], mf_pi2_snk[tsnk2]);
      PIPI_COMPUTE_PRODUCT(prod_r, mf_pi2_src[tsrc2], mf_pi1_snk[tsnk]);
      timings.prod += dclock();

      timings.trace -= dclock();
      topo2 = 0.5 * trace(prod_l, prod_r);
      timings.trace += dclock();	    
    }
    into(tsrc,tdis) += 0.5*topo1 + 0.5*topo2;

#endif
    timings.total_compute += dclock();
  }

  //Store the products we are going to reuse
  inline static void figureCsetupProductStore(MesonFieldProductStoreComputeReuse<mf_Policies> &products,
					      const std::vector<MfType >& mf_pi1_src,
					      const std::vector<MfType >& mf_pi2_src,
					      const std::vector<MfType >& mf_pi1_snk,
					      const std::vector<MfType >& mf_pi2_snk,
					      const int tsrc, const int tsnk, const int tsep, const int Lt){
    int tsrc2 = (tsrc-tsep+Lt) % Lt;
    int tsnk2 = (tsnk+tsep) % Lt;
      
    products.addStore(mf_pi1_snk[tsnk], mf_pi2_src[tsrc2]);
    products.addStore(mf_pi2_snk[tsnk2], mf_pi1_src[tsrc]);
    
#ifndef DISABLE_EVALUATION_OF_SECOND_PIPI_TOPOLOGY
    products.addStore(mf_pi1_src[tsrc], mf_pi2_snk[tsnk2]);
    products.addStore(mf_pi2_src[tsrc2], mf_pi1_snk[tsnk]);
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
    Timings &timings = timingsD();
    timings.total_compute -= dclock();

    typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
    int tdis = (tsnk - tsrc + Lt) % Lt;
    int tsrc2 = (tsrc-tsep+Lt) % Lt; //source position of pi2 by convention
    int tsnk2 = (tsnk+tsep) % Lt;

    ScalarComplexType tr1(0,0), tr2(0,0), incr(0,0);

    timings.trace -= dclock();
    //Topology 1  x4=tsrc (pi1)  y4=tsnk (pi1)  r4=tsrc2 (pi2)  s4=tsnk2 (pi2)
    incr += trace(mf_pi1_snk[tsnk] , mf_pi1_src[tsrc])   *    trace(mf_pi2_snk[tsnk2], mf_pi2_src[tsrc2]);

    //Topology 2  x4=tsrc (pi1) y4=tsnk2 (pi2)  r4=tsrc2 (pi2) s4=tsnk (pi1)
    incr += trace(mf_pi2_snk[tsnk2] , mf_pi1_src[tsrc])   *   trace(mf_pi1_snk[tsnk] , mf_pi2_src[tsrc2]);
    timings.trace += dclock();
    
    incr *= ScalarComplexType(0.5*0.25); //extra factor of 0.5 from average over 2 distinct topologies
    into(tsrc, tdis) += incr;

    timings.total_compute += dclock();
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
    Timings &timings = timingsR();
    timings.total_compute -= dclock();
    
    typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
    int tdis = (tsnk - tsrc + Lt) % Lt;
    int tsrc2 = (tsrc-tsep+Lt) % Lt; //source position of pi2 by convention
    int tsnk2 = (tsnk+tsep) % Lt;

    //Topology 1    x4=tsrc (pi1) y4=tsnk (pi1)   r4=tsrc2 (pi2) s4=tsnk2 (pi2)
    //Correponds to fig 9 in [1]
    ScalarComplexType topo1;
    {
      timings.prod -= dclock();
      PIPI_COMPUTE_PRODUCT(prod_l_top1, mf_pi2_src[tsrc2], mf_pi2_snk[tsnk2]);
      PIPI_COMPUTE_PRODUCT(prod_r_top1, mf_pi1_snk[tsnk], mf_pi1_src[tsrc]);
      timings.prod += dclock();

      timings.trace -= dclock();
      topo1 = 0.5 * trace( prod_l_top1, prod_r_top1 );
      timings.trace += dclock();
    }
    
    //Topology 2    x4=tsrc (pi1)  y4=tsnk_outer (pi2)  r4=tsrc_outer (pi2) s4=tsnk (pi1)
    //Corresponds to fig 12 in [1]
    ScalarComplexType topo2;
    {
      timings.prod -= dclock();
      PIPI_COMPUTE_PRODUCT(prod_l_top2, mf_pi2_src[tsrc2], mf_pi1_snk[tsnk]);
      PIPI_COMPUTE_PRODUCT(prod_r_top2, mf_pi2_snk[tsnk2], mf_pi1_src[tsrc]);
      timings.prod += dclock();

      timings.trace -= dclock();
      topo2 = 0.5 * trace( prod_l_top2, prod_r_top2 );
      timings.trace += dclock();
    }
      
#ifdef DISABLE_EVALUATION_OF_SECOND_PIPI_TOPOLOGY
    into(tsrc, tdis) += 0.5*topo1 + 0.5*topo2;
#else

    //Like with the C diagrams we previously incorrectly used g5-hermiticity to combine the pairs of similar topologies. This can be re-enabled using DISABLE_EVALUATION_OF_SECOND_PIPI_TOPOLOGY
       
    //Correponds to fig 10 in [1]
    ScalarComplexType topo3;
    {
      timings.prod -= dclock();
      PIPI_COMPUTE_PRODUCT(prod_l_top3, mf_pi2_snk[tsnk2], mf_pi2_src[tsrc2]);
      PIPI_COMPUTE_PRODUCT(prod_r_top3, mf_pi1_src[tsrc], mf_pi1_snk[tsnk]);
      timings.prod += dclock();

      timings.trace -= dclock();
      topo3 = 0.5 * trace( prod_l_top3, prod_r_top3 );
      timings.trace += dclock();
    }
    //Correponds to fig 11 in [1]
    ScalarComplexType topo4;
    {
      timings.prod -= dclock();
      PIPI_COMPUTE_PRODUCT(prod_l_top4, mf_pi1_snk[tsnk], mf_pi2_src[tsrc2]);
      PIPI_COMPUTE_PRODUCT(prod_r_top4, mf_pi1_src[tsrc], mf_pi2_snk[tsnk2]);
      timings.prod += dclock();

      timings.trace -= dclock();
      topo4 = 0.5 * trace( prod_l_top4, prod_r_top4 );
      timings.trace += dclock();
    }

    into(tsrc, tdis) += 0.25*topo1 + 0.25*topo2 + 0.25*topo3 + 0.25*topo4;
#endif
    timings.total_compute += dclock();
  }


  //R = 0.5 Tr( [[w^dag(r) S_2 v(r)]] [[w^dag(s) S_2 * v(s)]][[w^dag(y) S_2 v(y)]] [[w^dag(x) S_2 v(x)]] )
  //2 different topologies to average over
  inline static void figureRsetupProductStore(MesonFieldProductStoreComputeReuse<mf_Policies> &products,
					      const std::vector<MfType >& mf_pi1_src,
					      const std::vector<MfType >& mf_pi2_src,
					      const std::vector<MfType >& mf_pi1_snk,
					      const std::vector<MfType >& mf_pi2_snk,					      
					      const int tsrc, const int tsnk, const int tsep, const int Lt){
    typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
    int tdis = (tsnk - tsrc + Lt) % Lt;
    int tsrc2 = (tsrc-tsep+Lt) % Lt; //source position of pi2 by convention
    int tsnk2 = (tsnk+tsep) % Lt;

    products.addStore(mf_pi2_src[tsrc2], mf_pi2_snk[tsnk2]);
    products.addStore(mf_pi1_snk[tsnk], mf_pi1_src[tsrc]);
    
    products.addStore(mf_pi2_src[tsrc2], mf_pi1_snk[tsnk]);
    products.addStore(mf_pi2_snk[tsnk2], mf_pi1_src[tsrc]);
      
#ifndef DISABLE_EVALUATION_OF_SECOND_PIPI_TOPOLOGY
    products.addStore(mf_pi2_snk[tsnk2], mf_pi2_src[tsrc2]);
    products.addStore(mf_pi1_src[tsrc], mf_pi1_snk[tsnk]);
    
    products.addStore(mf_pi1_snk[tsnk], mf_pi2_src[tsrc2]);
    products.addStore(mf_pi1_src[tsrc], mf_pi2_snk[tsnk2]);
#endif
  }


  static inline void mf_keep_discard(std::vector< std::vector<MfType> const* > &keep,
				     std::vector< std::vector<MfType>* > &discard,
				     std::vector<MfType> &mf,
				     bool do_discard){
    if(do_discard) discard.push_back(&mf);
    else keep.push_back(&mf);
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
		      MesonFieldProductStore<mf_Policies> &products,
		      const Options &opt = Options()   ){
    Timings &perf = getTimings(diag);
    perf.total -= dclock();
    
    if(!GJP.Gparity()) ERR.General("ComputePiPiGparity","compute(..)","Implementation is for G-parity only; different contractions are needed for periodic BCs\n"); 
    const int Lt = GJP.Tnodes()*GJP.TnodeSites();

    if(Lt % tstep_src != 0) ERR.General("ComputePiPiGparity","compute(..)","tstep_src must divide the time range exactly\n"); 
    
    //Distribute load over all nodes
    int work = Lt*Lt/tstep_src;
    int node_work, node_off; bool do_work;
    getNodeWork(work,node_work,node_off,do_work);

    std::vector<MfType >& mf_pi1_src = src_mesonfields.get(p_pi1_src);
    std::vector<MfType >& mf_pi2_src = src_mesonfields.get(p_pi2_src);
    std::vector<MfType >& mf_pi1_snk = snk_mesonfields.get(p_pi1_snk);
    std::vector<MfType >& mf_pi2_snk = snk_mesonfields.get(p_pi2_snk);

    std::cout << "Computing fig " << diag << " with meson fields:" << std::endl
	      << "pi1_src  mom=" << p_pi1_src.str() << " ptr=" << &mf_pi1_src << std::endl      
	      << "pi2_src  mom=" << p_pi2_src.str() << " ptr=" << &mf_pi2_src << std::endl
	      << "pi1_snk  mom=" << p_pi1_snk.str() << " ptr=" << &mf_pi1_snk << std::endl      
	      << "pi2_snk  mom=" << p_pi2_snk.str() << " ptr=" << &mf_pi2_snk << std::endl;
    
#ifdef NODE_DISTRIBUTE_MESONFIELDS
    perf.gather -= dclock();
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
    perf.gather += dclock();
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
    //Reduction
    perf.gsum -= dclock();
    into.nodeSum();
    perf.gsum += dclock();
    
#ifdef NODE_DISTRIBUTE_MESONFIELDS
    perf.distribute -= dclock();
    //Need to take care that there's no overlap in the source and sink meson fields lest we distribute one we intend to keep
    std::vector< std::vector<MfType> const* > keep;
    std::vector< std::vector<MfType>* > discard;
    mf_keep_discard(keep, discard, mf_pi1_src, opt.redistribute_pi1_src);
    mf_keep_discard(keep, discard, mf_pi2_src, opt.redistribute_pi2_src);
    mf_keep_discard(keep, discard, mf_pi1_snk, opt.redistribute_pi1_snk);
    mf_keep_discard(keep, discard, mf_pi2_snk, opt.redistribute_pi2_snk);
    nodeDistributeUnique(discard, keep);
    perf.distribute += dclock();
#endif

    perf.total += dclock();
  }

  //Source and sink meson field set is the same
  static void compute(fMatrix<typename mf_Policies::ScalarComplexType> &into, const char diag, 
		      const ThreeMomentum &p_pi1_src, const ThreeMomentum &p_pi2_src,
		      const ThreeMomentum &p_pi1_snk, const ThreeMomentum &p_pi2_snk, 
		      const int tsep, const int tstep_src,
		      MesonFieldMomentumContainer<mf_Policies> &srcsnk_mesonfields,
		      MesonFieldProductStore<mf_Policies> &products,
		      const Options &opt = Options()
		      ){
    compute(into,diag,p_pi1_src,p_pi2_src,p_pi1_snk,p_pi2_snk,tsep,tstep_src,
	    srcsnk_mesonfields,srcsnk_mesonfields,
	    products, opt );
  }

  //Calculate for p_cm=(0,0,0)
  static void compute(fMatrix<typename mf_Policies::ScalarComplexType> &into, const char diag, 
		      const ThreeMomentum &p_pi1_src, const ThreeMomentum &p_pi1_snk, 
		      const int tsep, const int tstep_src,
		      MesonFieldMomentumContainer<mf_Policies> &mesonfields, MesonFieldProductStore<mf_Policies> &products,
		      const Options &opt = Options()
		      ){
    ThreeMomentum p_pi2_src = -p_pi1_src;
    ThreeMomentum p_pi2_snk = -p_pi1_snk;

    compute(into, diag, p_pi1_src, p_pi2_src, p_pi1_snk, p_pi2_snk, tsep, tstep_src, mesonfields, products, opt);
  }



  //Setup in advance which products we intend to reuse
  static void setupProductStore(MesonFieldProductStoreComputeReuse<mf_Policies> &products,
				const char diag, 
				const ThreeMomentum &p_pi1_src, const ThreeMomentum &p_pi2_src,
				const ThreeMomentum &p_pi1_snk, const ThreeMomentum &p_pi2_snk, 
				const int tsep, const int tstep_src,
				MesonFieldMomentumContainer<mf_Policies> &src_mesonfields,
				MesonFieldMomentumContainer<mf_Policies> &snk_mesonfields
				){
    if(!GJP.Gparity()) ERR.General("ComputePiPiGparity","compute(..)","Implementation is for G-parity only; different contractions are needed for periodic BCs\n"); 
    const int Lt = GJP.Tnodes()*GJP.TnodeSites();

    if(Lt % tstep_src != 0) ERR.General("ComputePiPiGparity","compute(..)","tstep_src must divide the time range exactly\n"); 
    
    //Distribute load over all nodes
    int work = Lt*Lt/tstep_src;
    int node_work, node_off; bool do_work;
    getNodeWork(work,node_work,node_off,do_work);

    std::vector<MfType >& mf_pi1_src = src_mesonfields.get(p_pi1_src);
    std::vector<MfType >& mf_pi2_src = src_mesonfields.get(p_pi2_src);
    std::vector<MfType >& mf_pi1_snk = snk_mesonfields.get(p_pi1_snk);
    std::vector<MfType >& mf_pi2_snk = snk_mesonfields.get(p_pi2_snk);
        
    if(do_work){
      for(int tt=node_off; tt<node_off + node_work; tt++){
	int rem = tt;
	int tsnk = rem % Lt; rem /= Lt; //sink time
	int tsrc = rem * tstep_src; //source time

	if(diag == 'C')
	  figureCsetupProductStore(products, mf_pi1_src, mf_pi2_src, mf_pi1_snk, mf_pi2_snk, tsrc,tsnk, tsep,Lt);
	else if(diag == 'R')
	  figureRsetupProductStore(products, mf_pi1_src, mf_pi2_src, mf_pi1_snk, mf_pi2_snk, tsrc,tsnk, tsep,Lt);
      }
    }
  }

  //Source and sink meson field set is the same
  static void setupProductStore(MesonFieldProductStoreComputeReuse<mf_Policies> &products, const char diag,
				const ThreeMomentum &p_pi1_src, const ThreeMomentum &p_pi2_src,
				const ThreeMomentum &p_pi1_snk, const ThreeMomentum &p_pi2_snk, 
				const int tsep, const int tstep_src,
				MesonFieldMomentumContainer<mf_Policies> &srcsnk_mesonfields
				){
    setupProductStore(products,diag,p_pi1_src,p_pi2_src,p_pi1_snk,p_pi2_snk,tsep,tstep_src,
		      srcsnk_mesonfields,srcsnk_mesonfields);
  }

  //Calculate for p_cm=(0,0,0)
  static void setupProductStore(MesonFieldProductStoreComputeReuse<mf_Policies> &products, const char diag, 
		      const ThreeMomentum &p_pi1_src, const ThreeMomentum &p_pi1_snk, 
		      const int tsep, const int tstep_src,
		      MesonFieldMomentumContainer<mf_Policies> &mesonfields
		      ){
    ThreeMomentum p_pi2_src = -p_pi1_src;
    ThreeMomentum p_pi2_snk = -p_pi1_snk;

    setupProductStore(products, diag, p_pi1_src, p_pi2_src, p_pi1_snk, p_pi2_snk, tsep, tstep_src, mesonfields);
  }


  //Compute the pion 'bubble'  0.5 tr( mf(t, p_pi) mf(t-tsep, p_pi2) )  [note pi2 at earlier timeslice here]
  //output into vector element  [t]

  static void computeFigureVdis(fVector<typename mf_Policies::ScalarComplexType> &into, const ThreeMomentum &p_pi, const ThreeMomentum &p_pi2, const int tsep, MesonFieldMomentumContainer<mf_Policies> &mesonfields){
    Timings &timings = timingsV();
    timings.total -= dclock();
    
    if(!GJP.Gparity()) ERR.General("ComputePiPiGparity","computeFigureVdis(..)","Implementation is for G-parity only; different contractions are needed for periodic BCs\n"); 
    const int Lt = GJP.Tnodes()*GJP.TnodeSites();

    ThreeMomentum const* mom[4] = { &p_pi, &p_pi2 };
    for(int p=0;p<2;p++)
      if(!mesonfields.contains(*mom[p])) ERR.General("ComputePiPiGparity","computeFigureVdis(..)","Meson field container doesn't contain momentum %s\n",mom[p]->str().c_str());
    
    //Get the meson fields we require
    std::vector<MfType >& mf_pi = mesonfields.get(p_pi);
    std::vector<MfType >& mf_pi2 = mesonfields.get(p_pi2);

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    timings.gather -= dclock();
    nodeGetMany(2,&mf_pi,&mf_pi2);
    timings.gather += dclock();
#endif

    into.resize(Lt); 

    //Distribute load over all nodes
    int work = Lt; //consider a better parallelization
    int node_work, node_off; bool do_work;
    getNodeWork(work,node_work,node_off,do_work);

    timings.total_compute -= dclock();
    timings.trace -= dclock();
    if(do_work){
      for(int t=node_off; t<node_off + node_work; t++){
	int t2 = (t-tsep+Lt) % Lt;
	into(t) = typename mf_Policies::ScalarComplexType(0.5)* trace(mf_pi[t], mf_pi2[t2]);
      }
    }
    timings.trace += dclock();
    timings.total_compute += dclock();

    timings.gsum -= dclock();
    into.nodeSum();
    timings.gsum += dclock();

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    timings.distribute -= dclock();
    nodeDistributeMany(2,&mf_pi,&mf_pi2);
    timings.distribute += dclock();
#endif

    timings.total += dclock();
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
