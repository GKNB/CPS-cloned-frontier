#include <config.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <util/qcdio.h>
#ifdef PARALLEL
#include <comms/sysfunc_cps.h>
#endif
#include <comms/scu.h>
#include <comms/glb.h>

#include <util/lattice.h>
#include <util/time_cps.h>
#include <util/smalloc.h>

#include <util/command_line.h>

#include<unistd.h>
#include<config.h>

#include <alg/lanc_arg.h>
#include <util/spincolorflavormatrix.h>
#include <alg/propmanager.h>
#include <alg/fix_gauge_arg.h>
#include <alg/alg_gparitycontract.h>
#include <alg/prop_dft.h>

#include <alg/alg_smear.h>
#include <alg/alg_tcharge.h>
#include <alg/alg_wilsonflow.h>
#include <alg/alg_actiondensity.h>

#include <string>
#include <sstream>
#include <bitset>

#include <util/lat_cont.h>

#ifdef USE_BFM 

//CK: these are redefined by BFM (to the same values)
#undef ND
#undef SPINOR_SIZE
#undef HALF_SPINOR_SIZE
#undef GAUGE_SIZE
#endif
#include <alg/a2a/MesonField.h>

#ifdef USE_OMP
#include <omp.h>
#endif

CPS_START_NAMESPACE

#define SEPARATE_V_A  //Match the way that Daiqian writes his output
	    //Note Daiqian renamed some of his C functions such that the type-II and type-I forms look more similar
#define USE_DAIQIANS_NEW_DEFINITIONS  

//n_contract is the number of contractions
//cidx_start is the index of the first contraction that will be written into this results vector
void Gparity_KtoPiPi::setup_resultvec(const int &n_contract, const int &cidx_start,  std::vector<CorrelationFunction> &rvec){
#ifdef SEPARATE_V_A
  const static std::string GammaNames[4] = { "M_{0,V}","M_{0,A}", "M_{1,V}", "M_{1,A}" };
#else
  const static std::string GammaNames[4] = { "M_{0,V-A}","M_{0,V+A}", "M_{1,V-A}", "M_{1,V+A}" };
#endif

  int rsize = 4*4 * n_contract;

  rvec.resize(rsize);
  //#pragma omp parallel for 
  for(int i=0;i<rsize;++i){
    int cidx_rel, gidx1, gidx2;
    result_inv_map(i,cidx_rel,gidx1,gidx2);
      
    std::ostringstream os;
    os << "C" << cidx_start + cidx_rel << "( " << GammaNames[gidx1] << ", " << GammaNames[gidx2] << " )";
    std::string label = os.str();

    //printf("setup_resultvec: i=%d -> cidx=%d gidx1=%d gidx2=%d, name %s\n",i,cidx_rel,gidx1,gidx2,label.c_str() );

    rvec[i].setLabel(label.c_str());
    rvec[i].setThreadType(CorrelationFunction::THREADED);
    rvec[i].setNcontractions(1);
    rvec[i].setGlobalSumOnWrite(false); //disable automatic thread and lattice sum on write; we do it manually below
  }
}  




void Gparity_KtoPiPi::setup(const ContractionTypeKtoPiPi &args, Lattice &lat){
  if(setup_called) return;

  lattice = &lat;

  t_size = GJP.Tnodes()*GJP.TnodeSites();

  gparity_use_transconv_props = args.gparity_use_transconv_props;

  //Setup smearing functions
  pion_source = new MFBasicSource;
  MFBasicSource::set_smearing( static_cast<MFBasicSource&>(*pion_source), args.pion_source);
  pion_source->fft_src();

  kaon_source = new MFBasicSource;
  MFBasicSource::set_smearing( static_cast<MFBasicSource&>(*kaon_source), args.kaon_source);
  kaon_source->fft_src();

  //Setup propagator pointers
  prop_L = & A2ApropContainer::verify_convert(PropManager::getProp(args.prop_L),"Gparity_KtoPiPi","setup(..)").getProp(*lattice);
  if(gparity_use_transconv_props) prop_L->gparity_make_fields_transconv(); //FFT vectors are translationally covariant

  printf("Gparity_KtoPiPi::setup prop_L with tag %s has nvec=%d, nl=%d, nhits=%d\n",args.prop_L,prop_L->get_nvec(),prop_L->get_nl(),prop_L->get_nhits() );

  prop_H = & A2ApropContainer::verify_convert(PropManager::getProp(args.prop_H),"Gparity_KtoPiPi","setup(..)").getProp(*lattice);
  if(gparity_use_transconv_props) prop_H->gparity_make_fields_transconv();

  //Setup meson field spin/flavour structures
  MFqdpMatrix wdag_S2_v(MFstructure::W, MFstructure::V, true, false);
  wdag_S2_v.set_matrix(S2);   

  MFqdpMatrix wdag_g5_v(MFstructure::W, MFstructure::V, true, false);
  wdag_g5_v.set_matrix(g5);

  MFqdpMatrix wdag_w(MFstructure::W, MFstructure::W, true, false);
  wdag_w.set_matrix(unit);
  
  //Form pi1 meson field
  prop_L->set_v_momentum(args.p_qpi1.p1);
  prop_L->set_wdag_momentum(args.p_qpi1.p2);
  prop_L->fft_vw();

  wdagL_S2_vL_pi1 = new MesonField2(*prop_L,*prop_L,wdag_S2_v,*pion_source);
  printf("Gparity_KtoPiPi::setup created mesonfield wdagL_S2_vL_pi1 with %d rows and %d cols\n",wdagL_S2_vL_pi1->rows(),wdagL_S2_vL_pi1->cols());

  //Form pi2 meson field
  prop_L->set_v_momentum(args.p_qpi2.p1);
  prop_L->set_wdag_momentum(args.p_qpi2.p2);
  prop_L->fft_vw();

  wdagL_S2_vL_pi2 = new MesonField2(*prop_L,*prop_L,wdag_S2_v,*pion_source);
  printf("Gparity_KtoPiPi::setup created mesonfield wdagL_S2_vL_pi2 with %d rows and %d cols\n",wdagL_S2_vL_pi2->rows(),wdagL_S2_vL_pi2->cols());
 
  //Form kaon meson field
  prop_H->set_v_momentum(args.p_qK);
  prop_H->set_wdag_momentum(args.p_qK);
  prop_H->fft_vw();

  Float mp_qK[3] = { -args.p_qK[0], -args.p_qK[1], -args.p_qK[2] };
  prop_L->set_wdag_momentum(mp_qK);
  prop_L->fft_vw();

  wdagL_g5_vH = new MesonField2(*prop_L,*prop_H,wdag_g5_v,*kaon_source);

  //for the version where we use g5-hermiticity on the strange quark
  prop_H->set_w_momentum(args.p_qK);
  prop_H->fft_vw();

  wdagL_wH = new MesonField2(*prop_L,*prop_H,wdag_w,*kaon_source);  //  wL^dag(x) (1- sgn(pL)\sigma_2) e^{-ipLx} (1+ sgn(pH)\sigma_2) e^{-ipHx'} wH(x')
                                                                    // use pL = -p_qK   hence for a non-zero meson field we need  sgn(pH) = +1,  i.e. pH = p_qk

  //Note: For the contractions we do not need the FFT fields
#ifdef SEPARATE_V_A
  //Here we copy Daiqian an define the following (matrix index first)
  //0 M_{0,V} = F_0 \gamma_\mu
  //1 M_{0,A} = F_0 \gamma_\mu\gamma^5
  //2 M_{1,V} = -F_1 \gamma_\mu
  //3 M_{1,A} = -F_1 \gamma_\mu\gamma^5
  for(int mu=0;mu<4;mu++){
    Gamma[0][mu] = _F0 * gmu[mu];
    Gamma[1][mu] = _F0 * gmu[mu]*g5;
    Gamma[2][mu] = _F1 * gmu[mu]* -1.0;
    Gamma[3][mu] = _F1 * gmu[mu]*g5* -1.0;
  }
#else
  //0  M_{0,V-A} = F0 g^mu(1-g5)
  //1  M_{0,V+A} = F0 g^mu(1+g5)
  //2  M_{1,V-A} = -F1 g^mu(1-g5)
  //3  M_{1,V+A} = -F1 g^mu(1+g5)
  for(int mu=0;mu<4;mu++){
    Gamma[0][mu] = _F0 * gmu[mu] * ( unit - g5 );
    Gamma[1][mu] = _F0 * gmu[mu] * ( unit + g5 );
    Gamma[2][mu] = _F1 * gmu[mu] * ( unit - g5 ) * -1.0;
    Gamma[2][mu] = _F1 * gmu[mu] * ( unit + g5 ) * -1.0;
  }
#endif  

  setup_called = true;
}

void Gparity_KtoPiPi::run(const ContractionTypeKtoPiPi &args, Lattice &lat, const int &config){
  std::ostringstream file; file << args.file << "." << config;
  setup(args,lat);

  FILE *fp;
  if ((fp = Fopen(file.str().c_str(), "w")) == NULL) {
    ERR.FileW("Gparity_KtoPiPi","run(...)",file.str().c_str());
  }            
  //Type1 diagrams
  {
    std::vector<CorrelationFunction> results;
    type1(args.t_sep_pi_k, args.t_sep_pion, results);
    for(int i=0; i<results.size(); i++) results[i].write(fp);
  }
  
  Fclose(fp);
}

static bool int_in_vec(const int &what, const std::vector<int> & vec){
  for(int i=0;i<vec.size();++i) 
    if(vec[i] == what) 
      return true;
  return false;
}

//Multiply the SpinColorFlavorMatrix 'prod' by the appropriate flavor projector to make it translationally covariant in combination with the momentum-phase factor imposed elsewhere
//side = l or r for left/right respectively.
//note: w^\dagger -> w^\dagger ( 1 - sgn( p_wdag )\sigma_2 ) = w^\dagger ( 1 + sgn( p_w )\sigma_2 )
//where p_wdag is the momentum of the w^\dagger field and p_w is the momentum of w  ( p_w = -p_wdag )

static void transconv(SpinColorFlavorMatrix &prod, const Float p[3], const char side){
  int sgn_p = sign_p_gparity(p);
  static const SpinColorFlavorMatrix one(spin_unit,sigma0);
  static const SpinColorFlavorMatrix s2(spin_unit,sigma2);
  static const SpinColorFlavorMatrix _1ps2( one + s2 );
  static const SpinColorFlavorMatrix _1ms2( one - s2 );
  
  const SpinColorFlavorMatrix &use = (sgn_p == +1 ? _1ps2 : _1ms2 );

  if(side == 'l')
    prod.LeftTimesEqual(use);
  else
    prod *= use;
}

static int pp_map(const int &ppi1, const int &ppi2){  //0,1 = +,-
  //std::vector<MesonField2>*  comb_wvwv_pcomb[4] = { &comb_wvwv_pp, &comb_wvwv_pm, &comb_wvwv_mp, &comb_wvwv_mm };
  if(ppi1 == 0 && ppi2 == 0) return 0;
  else if(ppi1 == 0 && ppi2 == 1) return 1;
  else if(ppi1 == 1 && ppi2 == 0) return 2;
  else return 3;
}

//We assume the pipi total momentum is zero. Define +p as the source momentum of pi1 and thus -p is the momentum of pi2.
//There are 2 choices of sink momentum for pi1/pi2: pi1(+p)pi2(-p) and pi1(-p)pi2(+p). Thus we have an extra input parameter
//pi1_snkmomsgn = '+' or '-'  for the sign of the pi1 sink momentum relative to the sign of its source momentum
void Gparity_KtoPiPi::pipi(const int &t_sep_pion, char pi1_snkmomsgn, std::vector<CorrelationFunction> &into, std::vector<int> *tpi1_vals){
  if(!setup_called){
    if(!UniqueID()) printf("Gparity_KtoPiPi::pipi(...) : setup(..) must be called prior to running contractions!\n");
    exit(-1);
  }

  into.resize(4);
  std::string names[4] = {"C","D","R","V"};
  for(int i=0;i<4;i++){
    into[i].setLabel(names[i].c_str());
    into[i].setThreadType(CorrelationFunction::UNTHREADED);
    into[i].setNcontractions(1);
    into[i].setGlobalSumOnWrite(false); //disable automatic thread and lattice sum on write; we do it manually below
  }

  //C = 0.5 Tr(  G(x,y) S_2 G(y,r) S_2   *   G(r,s) S_2 G(s,x) S_2 )
  //D = 0.25 Tr(  G(x,y) S_2 G(y,x) S_2 )  * Tr(  G(r,s) S_2 G(s,r) S_2 )
  //R = 0.5 Tr(  G(x,r) S_2 G(r,s) S_2   *   G(s,y) S_2 G(y,x) S_2 )
  //V = 0.5 Tr(  G(x,r) S_2 G(r,x) S_2 ) * Tr(  G(y,s) S_2 G(s,y) S_2 )


  //where x,y are the source and sink coords of the first pion and r,s the second pion

  //C = 0.5 Tr( [[w^dag(y) S_2 v(y)]] [[w^dag(r) S_2 * v(r)]] [[w^dag(s) S_2 v(s)]] [[w^dag(x) S_2 v(x)]] )
  //D = 0.25 Tr( [[w^dag(y) S_2 v(y)]] [[w^dag(x) S_2 v(x)]] ) * Tr( [[w^dag(s) S_2 v(s)]] [[w^dag(r) S_2 v(r)]] )
  //R = 0.5 Tr( [[w^dag(r) S_2 v(r)]] [[w^dag(s) S_2 * v(s)]][[w^dag(y) S_2 v(y)]] [[w^dag(x) S_2 v(x)]] )
  //V = 0.25 Tr(  [[w^dag(r) S_2 v(r)]][[w^dag(x) S_2 v(x)]] ) * Tr(  [[w^dag(s) S_2 v(s)]][[w^dag(y) S_2 v(y)]] )

  //Form  [[w^dag(y) S_2 v(y)]] [[w^dag(x) S_2 v(x)]]  for all x_4. y_4 index are hidden inside the mesonfields
  //We assume one pion has momentum +p and the other momentum -p (note it doesnt matter if p is positive or negative, so we can arbitrarily choose
  //p as the momentum of pi1). 
  //We need the combination for ++,+-,-+ and --
  std::vector<MesonField2> comb_wvwv_pp(t_size);
  std::vector<MesonField2> comb_wvwv_pm(t_size);
  std::vector<MesonField2> comb_wvwv_mp(t_size);
  std::vector<MesonField2> comb_wvwv_mm(t_size);

  for(int x4=0;x4<t_size;x4++){
    RangeSpecificT fix_x4(x4);
    MesonField2::combine_mf_wv_wv(comb_wvwv_pp[x4], *wdagL_S2_vL_pi1, *wdagL_S2_vL_pi1, fix_x4);
    MesonField2::combine_mf_wv_wv(comb_wvwv_pm[x4], *wdagL_S2_vL_pi1, *wdagL_S2_vL_pi2, fix_x4);
    MesonField2::combine_mf_wv_wv(comb_wvwv_mp[x4], *wdagL_S2_vL_pi2, *wdagL_S2_vL_pi1, fix_x4);
    MesonField2::combine_mf_wv_wv(comb_wvwv_mm[x4], *wdagL_S2_vL_pi2, *wdagL_S2_vL_pi2, fix_x4);
  }

  std::vector<MesonField2>*  comb_wvwv_pcomb[4] = { &comb_wvwv_pp, &comb_wvwv_pm, &comb_wvwv_mp, &comb_wvwv_mm };

  //x,r are the pion source timeslices and y,s the sink timeslices
  
  for(int x4=0;x4<t_size;x4++){
    if(tpi1_vals != NULL && !int_in_vec(x4,*tpi1_vals) ) continue;

    for(int y4=0;y4<t_size;y4++){
      int tdiffxy = (y4-x4+t_size)%t_size;

      //2 choices for r4 and s4
      int r4_c[2] = { (x4+t_sep_pion)%t_size,  (x4-t_sep_pion+t_size)%t_size };
      int s4_d[2] = { (y4+t_sep_pion)%t_size,  (y4-t_sep_pion+t_size)%t_size };

      //Note Daiqian only considers the first case for both
      //#define AVERAGE_TPI2_PM
#ifdef AVERAGE_TPI2_PM
      int cmax = 2, dmax = 2;
      double tavg_norm = 0.25;
#else
      int cmax = 1, dmax = 1;
      double tavg_norm = 1.0;
#endif

      for(int c=0;c<cmax;c++){
	for(int d=0;d<dmax;d++){
	  int r4 = r4_c[c];
	  int s4 = s4_d[d];

	  //Loop over momentum combinations
	  int ppi1_src = 0; //+
	  int ppi2_src = 1; //-
	  int ppi1_snk = ( pi1_snkmomsgn == '+' ? 0 : 1 );
	  int ppi2_snk = (ppi1_snk+1)%2; 

	  {
	    //C = 0.5 Tr( [[w^dag(y) S_2 v(y)]] [[w^dag(r) S_2 * v(r)]] [[w^dag(s) S_2 v(s)]] [[w^dag(x) S_2 v(x)]] )		
	    MesonField2 & wyvy_wrvr = (*comb_wvwv_pcomb[ pp_map(ppi1_snk,ppi2_src) ])[r4];    //pi1_snk, pi2_src
	    MesonField2 & wsvs_wxvx = (*comb_wvwv_pcomb[ pp_map(ppi2_snk,ppi1_src) ])[x4];    //pi2_snk, pi1_src
	    
	    static bool first = true;
	    if(first){ printf("wyvy_wrvr mom %d %d -> %d,  wsvs_wxvx %d %d -> %d\n",
			      ppi1_snk,ppi2_src,pp_map(ppi1_snk,ppi2_src),
			      ppi2_snk,ppi1_src,pp_map(ppi2_snk,ppi1_src)); first = false; }

	    //extra 0.25 in coefficient from average over src/snk timeslice combinations
	    into[0](0,tdiffxy) += tavg_norm* 0.5 * MesonField2::contract_fixedt1t2(wyvy_wrvr, wsvs_wxvx, y4, s4, true); 
	  }
	  {
	    //D = 0.25 Tr( [[w^dag(y) S_2 v(y)]] [[w^dag(x) S_2 v(x)]] ) * Tr( [[w^dag(s) S_2 v(s)]] [[w^dag(r) S_2 v(r)]] )
	    MesonField2 & wyvy_wxvx = (*comb_wvwv_pcomb[ pp_map(ppi1_snk,ppi1_src) ])[x4];    //pi1_snk, pi1_src
	    MesonField2 & wsvs_wrvr = (*comb_wvwv_pcomb[ pp_map(ppi2_snk,ppi2_src) ])[r4];    //pi2_snk, pi2_src
	    
	    into[1](0,tdiffxy) += tavg_norm* 0.25 * wyvy_wxvx.trace_wv(y4,y4) * wsvs_wrvr.trace_wv(s4,s4);
	  }
	  {
	    //R = 0.5 Tr( [[w^dag(r) S_2 v(r)]] [[w^dag(s) S_2 * v(s)]][[w^dag(y) S_2 v(y)]] [[w^dag(x) S_2 v(x)]] )
	    MesonField2 & wrvr_wsvs = (*comb_wvwv_pcomb[ pp_map(ppi2_src,ppi2_snk) ])[s4];
	    MesonField2 & wyvy_wxvx = (*comb_wvwv_pcomb[ pp_map(ppi1_snk,ppi1_src) ])[x4];
	    
	    into[2](0,tdiffxy) += tavg_norm* 0.5 * MesonField2::contract_fixedt1t2(wrvr_wsvs, wyvy_wxvx, r4, y4, true); 
	  }
	  {
	    //V = 0.25 Tr(  [[w^dag(r) S_2 v(r)]][[w^dag(x) S_2 v(x)]] ) * Tr(  [[w^dag(s) S_2 v(s)]][[w^dag(y) S_2 v(y)]] )
	    MesonField2 & wrvr_wxvx = (*comb_wvwv_pcomb[ pp_map(ppi2_src,ppi1_src) ])[x4]; 
	    MesonField2 & wsvs_wyvy = (*comb_wvwv_pcomb[ pp_map(ppi2_snk,ppi1_snk) ])[y4]; 
	    
	    into[3](0,tdiffxy) += tavg_norm* 0.5 * wrvr_wxvx.trace_wv(r4,r4); //* wsvs_wyvy.trace_wv(s4,s4);
	  } 
	}
      }
    }
  }
  for(int i=0;i<into.size();i++)
  into[i].sumLattice();
}




//t_sep_pi_k is the time separatation between the kaon and the closest of the two pions
//if tK_vals = NULL it will sum over all kaon source timeslices, otherwise it will restrict the sum to the set provided
void Gparity_KtoPiPi::type1(const int &t_sep_pi_k, const int &t_sep_pion, std::vector<CorrelationFunction> &into, std::vector<int> *tK_vals){
  if(!setup_called){
    if(!UniqueID()) printf("Gparity_KtoPiPi::type1(...) : setup(..) must be called prior to running contractions!\n");
    exit(-1);
  }
  static const int &c_start = 1;
  static const int &n_con = 6;
  setup_resultvec(n_con,c_start,into); //6 contractions starting at idx 1

  //Each contraction of this type is made up of different trace combinations of two objects:
  //1) \sum_{ \vec x_pi2, \vec x_pi2' }  \Gamma_1 \prop^L(x_op,x_pi2) S_2 \prop^L(x_pi2',x_op)    where x_pi2 and x_pi2' are 4-vectors on the same time-slice
  //2) \sum_{ \vec x_pi1, \vec x_pi1', \vec x_K }  \Gamma_2 \prop^L(x_op,x_pi1) S_2 \prop^L(x_pi1',x_K) \gamma^5 \prop^H(x_K,x_op)
  
  //Part 2 in terms of v and w is
  //\sum_{ \vec x_pi1, \vec x_pi1', \vec x_K }  \Gamma_2 vL_i(x_op;t_pi1) [[ wL_i^dag(x_pi1) S_2 vL_j(x_pi1';t_K) ]] [[ wL_j^dag(x_K)\gamma^5 vH_k(x_K;top) ]] wH_k^dag(x_op)
  //where [[ ]] indicate meson fields

  //The two meson field are independent of x_op so we can pregenerate them for each t_pi1, top

  //Form contraction  con_LLLH = [[\sum_{\vec xpi1} wL_i^dag(\vec xpi1, tpi1) S2 vL_j(\vec xpi1, tpi1; tK) ]] [[\sum_{\vec xK} wL_i^dag(\vec xK, tK) g5 vH_j(\vec xK, tK; top)]]
  //for all tpi1, top, where tK = (tpi1 - t_sep_pi_k + T) % T
  RangeT1plusDelta tkrange(-t_sep_pi_k);
  MesonField2 con_LLLH;
  MesonField2::combine_mf_wv_wv(con_LLLH, *wdagL_S2_vL_pi1, *wdagL_g5_vH, tkrange);

  //Loop over xop
  int n_threads = bfmarg::threads;
  omp_set_num_threads(n_threads);
#pragma omp parallel for 
  for(int x_op_loc = 0; x_op_loc < GJP.VolNodeSites(); x_op_loc++){
    int me = omp_get_thread_num();

    int t_op_loc = x_op_loc / (GJP.VolNodeSites()/GJP.TnodeSites());
    int t_op_glob = t_op_loc + GJP.TnodeCoor()*GJP.TnodeSites();

    //Average over all tK
    for(int tK = 0; tK < t_size; tK++){
      if(tK_vals != NULL && !int_in_vec(tK,*tK_vals) ) continue;

      //We need to average over choices  1) tpi1 = tK + t_sep_pi_k, tpi2 = tpi1 + t_sep_pion  and  2) tpi2 = tK + t_sep_pi_k, tpi1 = tpi2 + t_sep_pion 
      int tpi1_c[2] = { (tK + t_sep_pi_k) % t_size              , (tK + t_sep_pi_k + t_sep_pion) % t_size };
      int tpi2_c[2] = { (tK + t_sep_pi_k + t_sep_pion) % t_size , (tK + t_sep_pi_k) % t_size              };

      for(int c=0;c<2;c++){
	//Form SpinColorFlavorMatrix prod1 = vL_i(\vec xop, top ; tpi2) [\sum_{\vec xpi2} wL_i^dag(\vec xpi2, tpi2) S2 vL_j(\vec xpi2, tpi2; top)] wL_j^dag(\vec xop,top)
	SpinColorFlavorMatrix prod1;
	MesonField2::contract_vleft_wright(prod1, *prop_L, x_op_loc, *prop_L, x_op_loc, *wdagL_S2_vL_pi2, tpi2_c[c]);

#if 0	
	if(gparity_use_transconv_props){
	  transconv(prod1,prop_L->get_v_momentum(),'l');
	  transconv(prod1,prop_L->get_w_momentum(),'r');
	}
#endif
	//Form SpinColorFlavorMatrix prod2 = vL_i(\vec xop, top ; tpi1) con_LLLH_{ij}(tpi1 ; top) wH_j^dag(\vec xop, top) 
	SpinColorFlavorMatrix prod2;
	MesonField2::contract_vleft_wright(prod2, *prop_L, x_op_loc, *prop_H, x_op_loc, con_LLLH, tpi1_c[c]);

#if 0	
	if(gparity_use_transconv_props){
	  transconv(prod2,prop_L->get_v_momentum(),'l');
	  transconv(prod2,prop_H->get_w_momentum(),'r');
	}
#endif

	for(int g1idx = 0; g1idx < 4; ++g1idx){
	  for(int g2idx = 0; g2idx < 4; ++g2idx){
	    std::complex<double>* corrs[n_con];
	    for(int cc=0;cc<n_con;cc++) corrs[cc] = & into[result_map(cc, g1idx, g2idx)](me,0,t_op_glob);

	    //I is the index of the contraction in the range [1,32]
#define C(I) *corrs[I-c_start] 

	    //Sum over mu for each \Gamma_1 and \Gamma_2	      
	    for(int mu = 0; mu < 4; ++mu){ //remember to include 0.5 from average over the swapping of the pion source timeslices
#ifdef USE_DAIQIANS_NEW_DEFINITIONS  
	      int dd[] = {1,3,4,6,2,5};
#else
	      int dd[] = {1,2,3,4,5,6};
#endif	      

	      C(dd[0]) += 0.5*Trace( Gamma[g1idx][mu], prod1 ) * Trace( Gamma[g2idx][mu], prod2 );
	      C(dd[1]) += 0.5*( SpinFlavorTrace( Gamma[g1idx][mu], prod1 ) * SpinFlavorTrace( Gamma[g2idx][mu], prod2 ) ).Trace();
	      C(dd[2]) += 0.5*Trace(Gamma[g1idx][mu]*prod1 , Gamma[g2idx][mu]*prod2);
	      C(dd[3]) += 0.5*Trace( ColorTrace( Gamma[g1idx][mu], prod1 ), ColorTrace( Gamma[g2idx][mu], prod2 ) );
	      C(dd[4]) += 0.5*( SpinFlavorTrace( Gamma[g1idx][mu], prod1 ) * Transpose(SpinFlavorTrace( Gamma[g2idx][mu], prod2 )) ).Trace();
	      C(dd[5]) += 0.5*Trace(Gamma[g1idx][mu]*prod1 , ColorTranspose(Gamma[g2idx][mu]*prod2) );
	    }

#undef C
	  }
	}
      }
    }	
  }//end of site loop
    
  for(int i=0;i<into.size();i++)
    into[i].sumLattice();

  


}




void Gparity_KtoPiPi::type1_propHg5conj(const int &t_sep_pi_k, const int &t_sep_pion, std::vector<CorrelationFunction> &into, std::vector<int> *tK_vals){
  if(!setup_called){
    if(!UniqueID()) printf("Gparity_KtoPiPi::type1_propHg5conj(...) : setup(..) must be called prior to running contractions!\n");
    exit(-1);
  }
  static const int &c_start = 1;
  static const int &n_con = 6;
  setup_resultvec(n_con,c_start,into); //6 contractions starting at idx 1

  //Each contraction of this type is made up of different trace combinations of two objects:
  //1) \sum_{ \vec x_pi2, \vec x_pi2' }  \Gamma_1 \prop^L(x_op,x_pi2) S_2 \prop^L(x_pi2',x_op)    where x_pi2 and x_pi2' are 4-vectors on the same time-slice
  //2) \sum_{ \vec x_pi1, \vec x_pi1', \vec x_K }  \Gamma_2 \prop^L(x_op,x_pi1) S_2 \prop^L(x_pi1',x_K) \gamma^5 \prop^H(x_K,x_op)
  
  //In this version we use g5-hermiticity on the strange propagator
  //2) ->  \sum_{ \vec x_pi1, \vec x_pi1', \vec x_K }  \Gamma_2 \prop^L(x_op,x_pi1) S_2 \prop^L(x_pi1',x_K)  [\prop^H(x_op,x_K)]^dag \gamma^5

  // [\prop^H(x_op,x_K)]^dag  =  [ vH^i(x_op;t_K) ( wH^i(x_K) )^dag ]^dag = wH^i(x_K) ( vH^i(x_op;t_K) )^dag
  
  //Part 2 in terms of v and w is
  //\sum_{ \vec x_pi1, \vec x_pi1', \vec x_K }  \Gamma_2 vL_i(x_op;t_pi1) [[ wL_i^dag(x_pi1) S_2 vL_j(x_pi1';t_K) ]] [[ wL_j^dag(x_K) wH_k(x_K) ]] vH_k^dag(x_op;t_K) \gamma^5
  //where [[ ]] indicate meson fields

  //The two meson field are independent of x_op so we can pregenerate them for each t_pi1 and tK

  //Form contraction  con_LLLH = [[\sum_{\vec xpi1} wL_i^dag(x_pi1) S_2 vL_j(x_pi1';t_K) ]] [[ \sum_{\vec xK} wL_j^dag(x_K) wH_k(x_K) ]]
  //for each t_pi1, t_K. The t_pi1 index is the time index of the meson field (internal) but t_K must be kept external
  std::vector< MesonField2 > con_LLLH(t_size);
  for(int t_K = 0 ; t_K < t_size; t_K++){
    if(tK_vals != NULL && !int_in_vec(t_K,*tK_vals) ) continue;
    RangeSpecificT sett2tK(t_K);
    MesonField2::combine_mf_wv_ww(con_LLLH[t_K], *wdagL_S2_vL_pi1, *wdagL_wH, sett2tK);
  }

  //Loop over xop
  int n_threads = bfmarg::threads;
  omp_set_num_threads(n_threads);
#pragma omp parallel for 
  for(int x_op_loc = 0; x_op_loc < GJP.VolNodeSites(); x_op_loc++){
    int me = omp_get_thread_num();

    int t_op_loc = x_op_loc / (GJP.VolNodeSites()/GJP.TnodeSites());
    int t_op_glob = t_op_loc + GJP.TnodeCoor()*GJP.TnodeSites();

    //Average over all tK
    for(int tK = 0; tK < t_size; tK++){
      if(tK_vals != NULL && !int_in_vec(tK,*tK_vals) ) continue;

      //We need to average over choices  1) tpi1 = tK + t_sep_pi_k, tpi2 = tpi1 + t_sep_pion  and  2) tpi2 = tK + t_sep_pi_k, tpi1 = tpi2 + t_sep_pion 
      int tpi1_c[2] = { (tK + t_sep_pi_k) % t_size              , (tK + t_sep_pi_k + t_sep_pion) % t_size };
      int tpi2_c[2] = { (tK + t_sep_pi_k + t_sep_pion) % t_size , (tK + t_sep_pi_k) % t_size              };

      for(int c=0;c<2;c++){
	//Form SpinColorFlavorMatrix prod1 = vL_i(\vec xop, top ; tpi2) [\sum_{\vec xpi2} wL_i^dag(\vec xpi2, tpi2) S2 vL_j(\vec xpi2, tpi2; top)] wL_j^dag(\vec xop,top)
	SpinColorFlavorMatrix prod1;
	MesonField2::contract_vleft_wright(prod1, *prop_L, x_op_loc, *prop_L, x_op_loc, *wdagL_S2_vL_pi2, tpi2_c[c]);

#if 0	
	if(gparity_use_transconv_props){
	  transconv(prod1,prop_L->get_v_momentum(),'l');
	  transconv(prod1,prop_L->get_w_momentum(),'r');
	}
#endif
	//Form SpinColorFlavorMatrix prod2 = vL_i(\vec xop, top ; tpi1) con_LLLH_{ij}(tpi1)[tK] vH_j^dag(\vec xop, top; t_K) \gamma^5
	SpinColorFlavorMatrix prod2; 
	MesonField2::contract_vleft_vright(prod2, *prop_L, x_op_loc, *prop_H, x_op_loc, con_LLLH[tK], tpi1_c[c], tK);
	prod2.gr(-5);

#if 0	
	if(gparity_use_transconv_props){
	  transconv(prod2,prop_L->get_v_momentum(),'l');
	  transconv(prod2,prop_H->get_w_momentum(),'r');
	}
#endif

	for(int g1idx = 0; g1idx < 4; ++g1idx){
	  for(int g2idx = 0; g2idx < 4; ++g2idx){
	    std::complex<double>* corrs[n_con];
	    for(int cc=0;cc<n_con;cc++) corrs[cc] = & into[result_map(cc, g1idx, g2idx)](me,0,t_op_glob);

	    //I is the index of the contraction in the range [1,32]
#define C(I) *corrs[I-c_start] 

	    //Sum over mu for each \Gamma_1 and \Gamma_2	      
	    for(int mu = 0; mu < 4; ++mu){ //remember to include 0.5 from average over the swapping of the pion source timeslices
#ifdef USE_DAIQIANS_NEW_DEFINITIONS  
	      int dd[] = {1,3,4,6,2,5};
#else
	      int dd[] = {1,2,3,4,5,6};
#endif	      

	      C(dd[0]) += 0.5*Trace( Gamma[g1idx][mu], prod1 ) * Trace( Gamma[g2idx][mu], prod2 );
	      C(dd[1]) += 0.5*( SpinFlavorTrace( Gamma[g1idx][mu], prod1 ) * SpinFlavorTrace( Gamma[g2idx][mu], prod2 ) ).Trace();
	      C(dd[2]) += 0.5*Trace(Gamma[g1idx][mu]*prod1 , Gamma[g2idx][mu]*prod2);
	      C(dd[3]) += 0.5*Trace( ColorTrace( Gamma[g1idx][mu], prod1 ), ColorTrace( Gamma[g2idx][mu], prod2 ) );
	      C(dd[4]) += 0.5*( SpinFlavorTrace( Gamma[g1idx][mu], prod1 ) * Transpose(SpinFlavorTrace( Gamma[g2idx][mu], prod2 )) ).Trace();
	      C(dd[5]) += 0.5*Trace(Gamma[g1idx][mu]*prod1 , ColorTranspose(Gamma[g2idx][mu]*prod2) );
	    }

#undef C
	  }
	}
      }
    }	
  }//end of site loop
    
  for(int i=0;i<into.size();i++)
    into[i].sumLattice();
}



void Gparity_KtoPiPi::type2(const int &t_sep_pi_k, const int &t_sep_pion, std::vector<CorrelationFunction> &into, std::vector<int> *tK_vals){
  if(!setup_called){
    if(!UniqueID()) printf("Gparity_KtoPiPi::type2(...) : setup(..) must be called prior to running contractions!\n");
    exit(-1);
  }
  static const int &c_start = 7;
  static const int &n_con = 6;
  setup_resultvec(n_con,c_start,into); //6 contractions starting at idx 7

  //Each contraction of this type is made up of different trace combinations of two objects (below for simplicity we ignore the fact that the two vectors in the meson fields are allowed to vary in position relative to each other):
  //1) \sum_{ \vec x_K  }  \Gamma_1 \prop^L(x_op,x_K) \gamma^5 \prop^H(x_K,x_op)    
  //2) \sum_{ \vec x_pi1, \vec x_pi2  }  \Gamma_2 \prop^L(x_op,x_pi1) S_2 \prop^L(x_pi1,x_pi2) S_2 \prop^L(x_pi2,x_op)
  
  //We use g5-hermiticity on the strange propagator
  //1) -> \sum_{ \vec x_K  }  \Gamma_1 \prop^L(x_op,x_K)  [\prop^H(x_op,x_K)]^\dagger  \gamma^5
  //    = \sum_{ \vec x_K  }  \Gamma_1 vL(x_op) wL^dag(x_K) [ vH(x_op) wH^dag(x_K) ]^dag \gamma^5
  //    = \sum_{ \vec x_K  }  \Gamma_1 vL(x_op) [[ wL^dag(x_K) wH(x_K) ]] [vH(x_op)]^dag \gamma^5 
  //  where [[ ]] indicate meson fields

  //2) In terms of v and w
  // \sum_{ \vec x_pi1, \vec x_pi2  }  \Gamma_2 vL(x_op) [[ wL^dag(x_pi1) S_2 vL(x_pi1) ]] [[ wL^dag(x_pi2) S_2 vL(x_pi2) ]] wL^dag(x_op)

  //Form the product of the two meson fields
  //con_LLL = \sum_{\vec x_pi1,\vec x_pi2} [[ wL^dag(x_pi1) S_2 vL(x_pi1) ]] [[ wL^dag(x_pi2) S_2 vL(x_pi2) ]]
  //we need both  t_pi2 = (t_pi1 + sep) % T (index 0 of array)   and t_pi2 = (t_pi1 - sep + T) % T   (index 1 of array)
  
  MesonField2 con_LLL[2];
  RangeT1plusDelta fix_tpi2_plus_sep(t_sep_pion);
  RangeT1plusDelta fix_tpi2_minus_sep(-t_sep_pion);
  MesonField2::combine_mf_wv_wv(con_LLL[0], *wdagL_S2_vL_pi1, *wdagL_S2_vL_pi2, fix_tpi2_plus_sep);
  MesonField2::combine_mf_wv_wv(con_LLL[1], *wdagL_S2_vL_pi1, *wdagL_S2_vL_pi2, fix_tpi2_minus_sep);

  //Loop over xop
  int n_threads = bfmarg::threads;
  omp_set_num_threads(n_threads);
#pragma omp parallel for 
  for(int x_op_loc = 0; x_op_loc < GJP.VolNodeSites(); x_op_loc++){
    int me = omp_get_thread_num();

    int t_op_loc = x_op_loc / (GJP.VolNodeSites()/GJP.TnodeSites());
    int t_op_glob = t_op_loc + GJP.TnodeCoor()*GJP.TnodeSites();

    //Average over all tK
    for(int tK = 0; tK < t_size; tK++){
      if(tK_vals != NULL && !int_in_vec(tK,*tK_vals) ) continue;

      //prod1 =  vL(x_op) [[ wL^dag(x_K) wH(x_K) ]] [vH(x_op)]^dag \gamma^5 
      SpinColorFlavorMatrix prod1;
      MesonField2::contract_vleft_vright(prod1, *prop_L, x_op_loc, *prop_H, x_op_loc, *wdagL_wH, tK, tK);
      prod1.gr(-5);

      //We need to average over choices  1) tpi1 = tK + t_sep_pi_k, tpi2 = tpi1 + t_sep_pion  and  2) tpi2 = tK + t_sep_pi_k, tpi1 = tpi2 + t_sep_pion 
      int tpi1_c[2] = { (tK + t_sep_pi_k) % t_size              , (tK + t_sep_pi_k + t_sep_pion) % t_size };
      int tpi2_c[2] = { (tK + t_sep_pi_k + t_sep_pion) % t_size , (tK + t_sep_pi_k) % t_size              };

      for(int c=0;c<2;c++){
	// prod2 =  vL(x_op) con_LLL[c](t_pi1) wL^dag(x_op)
	SpinColorFlavorMatrix prod2; 
	MesonField2::contract_vleft_wright(prod2, *prop_L, x_op_loc, *prop_L, x_op_loc, con_LLL[c], tpi1_c[c]);

	for(int g1idx = 0; g1idx < 4; ++g1idx){
	  for(int g2idx = 0; g2idx < 4; ++g2idx){
	    std::complex<double>* corrs[n_con];
	    for(int cc=0;cc<n_con;cc++) corrs[cc] = & into[result_map(cc, g1idx, g2idx)](me,0,t_op_glob);

	    //I is the index of the contraction in the range [1,32]
#define C(I) *corrs[I-c_start] 

	    //Sum over mu for each \Gamma_1 and \Gamma_2	      
	    for(int mu = 0; mu < 4; ++mu){ //remember to include 0.5 from average over the swapping of the pion source timeslices
   	      C(7) += 0.5*Trace( Gamma[g1idx][mu], prod1 ) * Trace( Gamma[g2idx][mu], prod2 );
	      C(8) += 0.5*( SpinFlavorTrace( Gamma[g1idx][mu], prod1 ) * Transpose(SpinFlavorTrace( Gamma[g2idx][mu], prod2 )) ).Trace();
	      C(9) += 0.5*( SpinFlavorTrace( Gamma[g1idx][mu], prod1 ) * SpinFlavorTrace( Gamma[g2idx][mu], prod2 ) ).Trace();
	      C(10) += 0.5*Trace(Gamma[g1idx][mu]*prod1 , Gamma[g2idx][mu]*prod2);
	      C(11) += 0.5*Trace(Gamma[g1idx][mu]*prod1 , ColorTranspose(Gamma[g2idx][mu]*prod2) );	      
	      C(12) += 0.5*Trace( ColorTrace( Gamma[g1idx][mu], prod1 ), ColorTrace( Gamma[g2idx][mu], prod2 ) );	      
	    }

#undef C
	  }
	}
      }
    }	
  }//end of site loop
    
  for(int i=0;i<into.size();i++)
    into[i].sumLattice();
}





void Gparity_KtoPiPi::type3(const int &t_sep_pi_k, const int &t_sep_pion, std::vector<CorrelationFunction> &into, std::vector<int> *tK_vals){
  if(!setup_called){
    if(!UniqueID()) printf("Gparity_KtoPiPi::type3(...) : setup(..) must be called prior to running contractions!\n");
    exit(-1);
  }
  static const int &c_start = 13;
  static const int &n_con = 10;
  setup_resultvec(n_con,c_start,into);

  //Each contraction of this type is made up of different trace combinations of two objects (below for simplicity we ignore the fact that the two vectors in the meson fields are allowed to vary in position relative to each other):
  //1) \prop^L(x_op,x_pi1) S_2 \prop^L(x_pi1,x_pi2) S_2 \prop^L(x_pi2,x_K) \gamma^5 \prop^H(x_K,x_op)
  //2) \prop^L(x_op,x_op)   OR   \prop^H(x_op,x_op)

  //We use g5-hermiticity on the strange prop in part 1 (but not part 2 where it appears)
  //1) \prop^L(x_op,x_pi1) S_2 \prop^L(x_pi1,x_pi2) S_2 \prop^L(x_pi2,x_K)  [ \prop^H(x_op,x_K) ]^dag \gamma^5
  // = vL(x_op) [[ wL^dag(x_pi1) S_2 vL(x_pi1) ]] [[ wL^dag(x_pi2) S_2 vL(x_pi2) ]] [[ wL^dag(x_K) wH(x_K) ]] vH^dag(x_op) \gamma^5
  //  where [[ ]] indicate meson fields
  
  //Form the product of the two meson fields
  //con_LLLL = [[ wL^dag(x_pi1) S_2 vL(x_pi1) ]] [[ wL^dag(x_pi2) S_2 vL(x_pi2) ]]
  //we need both  t_pi2 = (t_pi1 + sep) % T (index 0 of array)   and t_pi2 = (t_pi1 - sep + T) % T   (index 1 of array)

  MesonField2 con_LLLL[2];
  RangeT1plusDelta fix_tpi2_plus_sep(t_sep_pion);
  RangeT1plusDelta fix_tpi2_minus_sep(-t_sep_pion);
  MesonField2::combine_mf_wv_wv(con_LLLL[0], *wdagL_S2_vL_pi1, *wdagL_S2_vL_pi2, fix_tpi2_plus_sep);
  MesonField2::combine_mf_wv_wv(con_LLLL[1], *wdagL_S2_vL_pi1, *wdagL_S2_vL_pi2, fix_tpi2_minus_sep);

  //For each x_K calculate
  //[[ wL^dag(x_pi1) S_2 vL(x_pi1) ]] [[ wL^dag(x_pi2) S_2 vL(x_pi2) ]] [[ wL^dag(x_K) wH(x_K) ]]
  // = con_LLLL [[ wL^dag(x_K) wH(x_K) ]]

  MesonField2 con_LLLLHH[2][t_size];
  for(int pm=0;pm<2;pm++){
    for(int t_K = 0; t_K < t_size; t_K++){
      if(tK_vals != NULL && !int_in_vec(t_K,*tK_vals) ) continue;
      RangeSpecificT tKfix(t_K);
      MesonField2::combine_mf_wv_ww(con_LLLLHH[pm][t_K], con_LLLL[pm], *wdagL_wH, tKfix);
    }
  }

  //Loop over xop
  int n_threads = bfmarg::threads;
  omp_set_num_threads(n_threads);
#pragma omp parallel for 
  for(int x_op_loc = 0; x_op_loc < GJP.VolNodeSites(); x_op_loc++){
    int me = omp_get_thread_num();

    int t_op_loc = x_op_loc / (GJP.VolNodeSites()/GJP.TnodeSites());
    int t_op_glob = t_op_loc + GJP.TnodeCoor()*GJP.TnodeSites();

    //prod2 = \prop^L(x_op,x_op)   OR   \prop^H(x_op,x_op)   =    v(x_op) w^dag(x_op)
    SpinColorFlavorMatrix prod2_L, prod2_H;
    MesonField2::contract_vw(prod2_L,*prop_L,x_op_loc,*prop_L,x_op_loc);
    MesonField2::contract_vw(prod2_H,*prop_H,x_op_loc,*prop_H,x_op_loc);

    //Average over all tK
    for(int tK = 0; tK < t_size; tK++){
      if(tK_vals != NULL && !int_in_vec(tK,*tK_vals) ) continue;

      //We need to average over choices  1) tpi1 = tK + t_sep_pi_k, tpi2 = tpi1 + t_sep_pion  and  2) tpi2 = tK + t_sep_pi_k, tpi1 = tpi2 + t_sep_pion 
      int tpi1_c[2] = { (tK + t_sep_pi_k) % t_size              , (tK + t_sep_pi_k + t_sep_pion) % t_size };
      int tpi2_c[2] = { (tK + t_sep_pi_k + t_sep_pion) % t_size , (tK + t_sep_pi_k) % t_size              };

      for(int c=0;c<2;c++){
	//prod1 = vL(x_op) con_LLLLHH[c][tK](t_pi1) vH^dag(x_op) \gamma^5
	SpinColorFlavorMatrix prod1;
	MesonField2::contract_vleft_vright(prod1, *prop_L, x_op_loc, *prop_H, x_op_loc,con_LLLLHH[c][tK] , tpi1_c[c], tK);
	prod1.gr(-5);
	
	for(int g1idx = 0; g1idx < 4; ++g1idx){
	  for(int g2idx = 0; g2idx < 4; ++g2idx){
	    std::complex<double>* corrs[n_con];
	    for(int cc=0;cc<n_con;cc++) corrs[cc] = & into[result_map(cc, g1idx, g2idx)](me,0,t_op_glob);

	    //I is the index of the contraction in the range [1,32]
#define C(I) *corrs[I-c_start] 

	    //Sum over mu for each \Gamma_1 and \Gamma_2	      
	    for(int mu = 0; mu < 4; ++mu){ //remember to include 0.5 from average over the swapping of the pion source timeslices
	      C(13) += 0.5*Trace( Gamma[g1idx][mu], prod1 ) * Trace( Gamma[g2idx][mu], prod2_L );
	      C(14) += 0.5*( SpinFlavorTrace( Gamma[g1idx][mu], prod1 ) * Transpose(SpinFlavorTrace( Gamma[g2idx][mu], prod2_L )) ).Trace();
	      C(15) += 0.5*( SpinFlavorTrace( Gamma[g1idx][mu], prod1 ) * SpinFlavorTrace( Gamma[g2idx][mu], prod2_L ) ).Trace();  
	      C(16) += 0.5*Trace(Gamma[g1idx][mu]*prod1 , Gamma[g2idx][mu]*prod2_L);
	      C(17) += 0.5*Trace(Gamma[g1idx][mu]*prod1 , ColorTranspose(Gamma[g2idx][mu]*prod2_L) );
	      C(18) += 0.5*Trace( ColorTrace( Gamma[g1idx][mu], prod1 ), ColorTrace( Gamma[g2idx][mu], prod2_L ) );
	      
	      C(19) += 0.5*Trace( Gamma[g1idx][mu], prod1 ) * Trace( Gamma[g2idx][mu], prod2_H );
	      C(20) += 0.5*( SpinFlavorTrace( Gamma[g1idx][mu], prod1 ) * SpinFlavorTrace( Gamma[g2idx][mu], prod2_H ) ).Trace();
	      C(21) += 0.5*Trace(Gamma[g1idx][mu]*prod1 , Gamma[g2idx][mu]*prod2_H);
	      C(22) += 0.5*Trace( ColorTrace( Gamma[g1idx][mu], prod1 ), ColorTrace( Gamma[g2idx][mu], prod2_H ) );
	    }

#undef C
	  }
	}
      }
    }	
  }//end of site loop
    
  for(int i=0;i<into.size();i++)
    into[i].sumLattice();
}


void Gparity_KtoPiPi::type4(const int &t_sep_pi_k, const int &t_sep_pion, std::vector<CorrelationFunction> &into, std::vector<int> *tK_vals){
  if(!setup_called){
    if(!UniqueID()) printf("Gparity_KtoPiPi::type4(...) : setup(..) must be called prior to running contractions!\n");
    exit(-1);
  }
  static const int &c_start = 23;
  static const int &n_con = 10;
  setup_resultvec(n_con,c_start,into);

  //The pion blob is included in the pi-pi scattering calculation so no need to include it here
  //#define INCLUDE_PION_BLOB

#ifdef INCLUDE_PION_BLOB
  //A separate meson 'blob'
  //blob = \sum_{\vec x_pi1,\vec x_pi2} -0.5 * Tr( \prop^L(x_pi1,x_pi2) S_2 \prop^L(x_pi2,x_pi1) S_2 )
  //     = \sum_{\vec x_pi1,\vec x_pi2} -0.5 * Tr( [[ wL^dag(x_pi2) S_2 vL(x_pi2) ]] [[ wL^dag(x_pi1) S_2 vL(x_pi1) ]] )
  //     = \sum_{\vec x_pi1,\vec x_pi2} -0.5 * Tr( [[ wL^dag(x_pi1) S_2 vL(x_pi1) ]] [[ wL^dag(x_pi2) S_2 vL(x_pi2) ]] )
  //we need both  t_pi2 = (t_pi1 + sep) % T (index 0 of array)   and t_pi2 = (t_pi1 - sep + T) % T   (index 1 of array)  

  RangeT1plusDelta fix_tpi2_plus_sep(t_sep_pion);
  RangeT1plusDelta fix_tpi2_minus_sep(-t_sep_pion);
  
  CorrelationFunction blob_tpi1_plus("blob",1,CorrelationFunction::THREADED);
  MesonField2::contract_specify_t2range( *wdagL_S2_vL_pi1, *wdagL_S2_vL_pi2, 0, fix_tpi2_plus_sep, blob_tpi1_plus);
  
  CorrelationFunction blob_tpi1_minus("blob",1,CorrelationFunction::THREADED);
  MesonField2::contract_specify_t2range( *wdagL_S2_vL_pi1, *wdagL_S2_vL_pi2, 0, fix_tpi2_minus_sep, blob_tpi1_minus);

  //Sum over t_pi1 and average over pion permutations
  std::complex<double> blob(0.0);
  for(int t_pi1=0;t_pi1<t_size;t_pi1++){
    blob += blob_tpi1_plus(0,t_pi1);
    blob += blob_tpi1_minus(0,t_pi1);
  }
  blob *= -0.5*0.5;  //one factor of 1/2 from blob definition, one from average over permutations
#else
  std::complex<double> blob(0.5);
#endif

  //Each contraction of this type is made up of different trace combinations of two objects (below for simplicity we ignore the fact that the two vectors in the meson fields are allowed to vary in position relative to each other):
  //1) \prop^L(x_op,x_K) \gamma^5 \prop^H(x_K,x_op)
  //   we use g5-hermiticity on the strange prop
  //  \prop^L(x_op,x_K)  [ \prop^H(x_op,x_K) ]^dag \gamma^5
  //= vL(x_op) [[ wL^dag(x_K) wH(x_K) ]] vH^dag(x_op) \gamma_5
  
  //2) \prop^L(x_op,x_op)   OR   \prop^H(x_op,x_op)
  
  //Loop over xop
  int n_threads = bfmarg::threads;
  omp_set_num_threads(n_threads);
#pragma omp parallel for 
  for(int x_op_loc = 0; x_op_loc < GJP.VolNodeSites(); x_op_loc++){
    int me = omp_get_thread_num();

    int t_op_loc = x_op_loc / (GJP.VolNodeSites()/GJP.TnodeSites());
    int t_op_glob = t_op_loc + GJP.TnodeCoor()*GJP.TnodeSites();

    //prod2 = \prop^L(x_op,x_op)   OR   \prop^H(x_op,x_op)   =    v(x_op) w^dag(x_op)
    SpinColorFlavorMatrix prod2_L, prod2_H;
    MesonField2::contract_vw(prod2_L,*prop_L,x_op_loc,*prop_L,x_op_loc);
    MesonField2::contract_vw(prod2_H,*prop_H,x_op_loc,*prop_H,x_op_loc);

    //Average over all tK
    for(int tK = 0; tK < t_size; tK++){
      if(tK_vals != NULL && !int_in_vec(tK,*tK_vals) ) continue;

      //prod1 = vL(x_op) [[ wL^dag(x_K) wH(x_K) ]] vH^dag(x_op) \gamma_5
      SpinColorFlavorMatrix prod1;
      MesonField2::contract_vleft_vright(prod1, *prop_L, x_op_loc, *prop_H, x_op_loc, *wdagL_wH , tK, tK);
      prod1.gr(-5);

      for(int g1idx = 0; g1idx < 4; ++g1idx){
	for(int g2idx = 0; g2idx < 4; ++g2idx){
	  std::complex<double>* corrs[n_con];
	  for(int cc=0;cc<n_con;cc++) corrs[cc] = & into[result_map(cc, g1idx, g2idx)](me,0,t_op_glob);
	  
	  //I is the index of the contraction in the range [1,32]
#define C(I) *corrs[I-c_start] 
	  
	  //Sum over mu for each \Gamma_1 and \Gamma_2	      
	  for(int mu = 0; mu < 4; ++mu){ //remember to include 0.5 from average over the swapping of the pion source timeslices
	    C(23) += blob * Trace( Gamma[g1idx][mu], prod1 ) * Trace( Gamma[g2idx][mu], prod2_L );
	    C(24) += blob * ( SpinFlavorTrace( Gamma[g1idx][mu], prod1 ) * Transpose(SpinFlavorTrace( Gamma[g2idx][mu], prod2_L )) ).Trace();
	    C(25) += blob * ( SpinFlavorTrace( Gamma[g1idx][mu], prod1 ) * SpinFlavorTrace( Gamma[g2idx][mu], prod2_L ) ).Trace();  
	    C(26) += blob * Trace(Gamma[g1idx][mu]*prod1 , Gamma[g2idx][mu]*prod2_L);
	    C(27) += blob * Trace(Gamma[g1idx][mu]*prod1 , ColorTranspose(Gamma[g2idx][mu]*prod2_L) );
	    C(28) += blob * Trace( ColorTrace( Gamma[g1idx][mu], prod1 ), ColorTrace( Gamma[g2idx][mu], prod2_L ) );
	    
	    C(29) += blob * Trace( Gamma[g1idx][mu], prod1 ) * Trace( Gamma[g2idx][mu], prod2_H );
	    C(30) += blob * ( SpinFlavorTrace( Gamma[g1idx][mu], prod1 ) * SpinFlavorTrace( Gamma[g2idx][mu], prod2_H ) ).Trace();
	    C(31) += blob * Trace(Gamma[g1idx][mu]*prod1 , Gamma[g2idx][mu]*prod2_H);
	    C(32) += blob * Trace( ColorTrace( Gamma[g1idx][mu], prod1 ), ColorTrace( Gamma[g2idx][mu], prod2_H ) );
	  }
	  
#undef C
	}
      }
    }
  }//end of site loop	
    
  for(int i=0;i<into.size();i++)
    into[i].sumLattice();
}






void Gparity_KtoPiPi::psvertex_type3(const int &t_sep_pi_k, const int &t_sep_pion, std::vector<CorrelationFunction> &into, std::vector<int> *tK_vals){
  if(!setup_called){
    if(!UniqueID()) printf("Gparity_KtoPiPi::psvertex_type3(...) : setup(..) must be called prior to running contractions!\n");
    exit(-1);
  }

  //Pseudoscalar vertex with F_0 g5  and  -F_1 g5
  static const int &c_start = 13; //use the same naming as the type3 operators for ease of comparison
  static const int &n_con = 1; //only one possible trace form
  const static std::string GammaNames[2] = { "M_{0,PS}","M_{1,PS}" };

  int rsize = 2*n_con;

  into.resize(rsize);
  //#pragma omp parallel for 
  for(int c=0;c<n_con;c++){
    for(int i=0;i<2;i++){      
      std::ostringstream os;
      os << "C" << c_start + c << "( " << GammaNames[i] << " )";
      std::string label = os.str();

      into[2*c+i].setLabel(label.c_str());
      into[2*c+i].setThreadType(CorrelationFunction::THREADED);
      into[2*c+i].setNcontractions(1);
      into[2*c+i].setGlobalSumOnWrite(false); //disable automatic thread and lattice sum on write; we do it manually below
    }
  }

  //These are identical to the type3 diagrams but without the internal quark loop, and with the vertex replaced with a pseudoscalar vertex
  //We need to compute this object
  // \prop^L(x_op,x_pi1) S_2 \prop^L(x_pi1,x_pi2) S_2 \prop^L(x_pi2,x_K) \gamma^5 \prop^H(x_K,x_op)

  //We use g5-hermiticity on the strange prop
  //\prop^L(x_op,x_pi1) S_2 \prop^L(x_pi1,x_pi2) S_2 \prop^L(x_pi2,x_K)  [ \prop^H(x_op,x_K) ]^dag \gamma^5
  // = vL(x_op) [[ wL^dag(x_pi1) S_2 vL(x_pi1) ]] [[ wL^dag(x_pi2) S_2 vL(x_pi2) ]] [[ wL^dag(x_K) wH(x_K) ]] vH^dag(x_op) \gamma^5
  //  where [[ ]] indicate meson fields
  
  //Form the product of the two meson fields
  //con_LLLL = [[ wL^dag(x_pi1) S_2 vL(x_pi1) ]] [[ wL^dag(x_pi2) S_2 vL(x_pi2) ]]
  //we need both  t_pi2 = (t_pi1 + sep) % T (index 0 of array)   and t_pi2 = (t_pi1 - sep + T) % T   (index 1 of array)

  MesonField2 con_LLLL[2];
  RangeT1plusDelta fix_tpi2_plus_sep(t_sep_pion);
  RangeT1plusDelta fix_tpi2_minus_sep(-t_sep_pion);
  MesonField2::combine_mf_wv_wv(con_LLLL[0], *wdagL_S2_vL_pi1, *wdagL_S2_vL_pi2, fix_tpi2_plus_sep);
  MesonField2::combine_mf_wv_wv(con_LLLL[1], *wdagL_S2_vL_pi1, *wdagL_S2_vL_pi2, fix_tpi2_minus_sep);

  //For each x_K calculate
  //[[ wL^dag(x_pi1) S_2 vL(x_pi1) ]] [[ wL^dag(x_pi2) S_2 vL(x_pi2) ]] [[ wL^dag(x_K) wH(x_K) ]]
  // = con_LLLL [[ wL^dag(x_K) wH(x_K) ]]

  MesonField2 con_LLLLHH[2][t_size];
  for(int pm=0;pm<2;pm++){
    for(int t_K = 0; t_K < t_size; t_K++){
      if(tK_vals != NULL && !int_in_vec(t_K,*tK_vals) ) continue;
      RangeSpecificT tKfix(t_K);
      MesonField2::combine_mf_wv_ww(con_LLLLHH[pm][t_K], con_LLLL[pm], *wdagL_wH, tKfix);
    }
  }

  //Loop over xop
  int n_threads = bfmarg::threads;
  omp_set_num_threads(n_threads);
#pragma omp parallel for 
  for(int x_op_loc = 0; x_op_loc < GJP.VolNodeSites(); x_op_loc++){
    int me = omp_get_thread_num();

    int t_op_loc = x_op_loc / (GJP.VolNodeSites()/GJP.TnodeSites());
    int t_op_glob = t_op_loc + GJP.TnodeCoor()*GJP.TnodeSites();

    //Average over all tK
    for(int tK = 0; tK < t_size; tK++){
      if(tK_vals != NULL && !int_in_vec(tK,*tK_vals) ) continue;

      //We need to average over choices  1) tpi1 = tK + t_sep_pi_k, tpi2 = tpi1 + t_sep_pion  and  2) tpi2 = tK + t_sep_pi_k, tpi1 = tpi2 + t_sep_pion 
      int tpi1_c[2] = { (tK + t_sep_pi_k) % t_size              , (tK + t_sep_pi_k + t_sep_pion) % t_size };
      int tpi2_c[2] = { (tK + t_sep_pi_k + t_sep_pion) % t_size , (tK + t_sep_pi_k) % t_size              };

      for(int c=0;c<2;c++){
	//prod1 = vL(x_op) con_LLLLHH[c][tK](t_pi1) vH^dag(x_op) \gamma^5
	SpinColorFlavorMatrix prod1;
	MesonField2::contract_vleft_vright(prod1, *prop_L, x_op_loc, *prop_H, x_op_loc,con_LLLLHH[c][tK] , tpi1_c[c], tK);
	prod1.gr(-5);

	//Pseudoscalar vertex with F_0 g5  and  -F_1 g5
	static const SpinColorFlavorMatrix mstructs[2] = { SpinColorFlavorMatrix(gamma5,F0), SpinColorFlavorMatrix(gamma5,F1)*-1.0 };
	for(int gidx=0;gidx<2;gidx++){
	  std::complex<double>* corrs[n_con];
	  for(int cc=0;cc<n_con;cc++) corrs[cc] = & into[2*cc+gidx](me,0,t_op_glob);
	  
	  //I is the index of the contraction in the range [1,32]
#define C(I) *corrs[I-c_start] 
	  C(13) += 0.5*Trace( mstructs[gidx], prod1 );
#undef C
	}
      }
      
    }	
  }//end of site loop
    
  for(int i=0;i<into.size();i++)
    into[i].sumLattice();
}

void Gparity_KtoPiPi::psvertex_type4(const int &t_sep_pi_k, const int &t_sep_pion, std::vector<CorrelationFunction> &into, std::vector<int> *tK_vals){
  if(!setup_called){
    if(!UniqueID()) printf("Gparity_KtoPiPi::psvertex_type4(...) : setup(..) must be called prior to running contractions!\n");
    exit(-1);
  }
  static const int &c_start = 23;
  static const int &n_con = 1;

  const static std::string GammaNames[2] = { "M_{0,PS}","M_{1,PS}" };

  int rsize = 2*n_con;

  into.resize(rsize);
  //#pragma omp parallel for 
  for(int c=0;c<n_con;c++){
    for(int i=0;i<2;i++){      
      std::ostringstream os;
      os << "C" << c_start + c << "( " << GammaNames[i] << " )";
      std::string label = os.str();

      into[2*c+i].setLabel(label.c_str());
      into[2*c+i].setThreadType(CorrelationFunction::THREADED);
      into[2*c+i].setNcontractions(1);
      into[2*c+i].setGlobalSumOnWrite(false); //disable automatic thread and lattice sum on write; we do it manually below
    }
  }

  //The pion blob is included in the pi-pi scattering calculation so no need to include it here
  //#define INCLUDE_PION_BLOB

#ifdef INCLUDE_PION_BLOB
  //A separate meson 'blob'
  //blob = \sum_{\vec x_pi1,\vec x_pi2} -0.5 * Tr( \prop^L(x_pi1,x_pi2) S_2 \prop^L(x_pi2,x_pi1) S_2 )
  //     = \sum_{\vec x_pi1,\vec x_pi2} -0.5 * Tr( [[ wL^dag(x_pi2) S_2 vL(x_pi2) ]] [[ wL^dag(x_pi1) S_2 vL(x_pi1) ]] )
  //     = \sum_{\vec x_pi1,\vec x_pi2} -0.5 * Tr( [[ wL^dag(x_pi1) S_2 vL(x_pi1) ]] [[ wL^dag(x_pi2) S_2 vL(x_pi2) ]] )
  //we need both  t_pi2 = (t_pi1 + sep) % T (index 0 of array)   and t_pi2 = (t_pi1 - sep + T) % T   (index 1 of array)  

  RangeT1plusDelta fix_tpi2_plus_sep(t_sep_pion);
  RangeT1plusDelta fix_tpi2_minus_sep(-t_sep_pion);
  
  CorrelationFunction blob_tpi1_plus("blob",1,CorrelationFunction::THREADED);
  MesonField2::contract_specify_t2range( *wdagL_S2_vL_pi1, *wdagL_S2_vL_pi2, 0, fix_tpi2_plus_sep, blob_tpi1_plus);
  
  CorrelationFunction blob_tpi1_minus("blob",1,CorrelationFunction::THREADED);
  MesonField2::contract_specify_t2range( *wdagL_S2_vL_pi1, *wdagL_S2_vL_pi2, 0, fix_tpi2_minus_sep, blob_tpi1_minus);

  //Sum over t_pi1 and average over pion permutations
  std::complex<double> blob(0.0);
  for(int t_pi1=0;t_pi1<t_size;t_pi1++){
    blob += blob_tpi1_plus(0,t_pi1);
    blob += blob_tpi1_minus(0,t_pi1);
  }
  blob *= -0.5*0.5;  //one factor of 1/2 from blob definition, one from average over permutations
#else
  std::complex<double> blob(0.5);
#endif

  //K->operator:
  //\prop^L(x_op,x_K) \gamma^5 \prop^H(x_K,x_op)
  //   we use g5-hermiticity on the strange prop
  //  \prop^L(x_op,x_K)  [ \prop^H(x_op,x_K) ]^dag \gamma^5
  //= vL(x_op) [[ wL^dag(x_K) wH(x_K) ]] vH^dag(x_op) \gamma_5
    
  //Loop over xop
  int n_threads = bfmarg::threads;
  omp_set_num_threads(n_threads);
#pragma omp parallel for 
  for(int x_op_loc = 0; x_op_loc < GJP.VolNodeSites(); x_op_loc++){
    int me = omp_get_thread_num();

    int t_op_loc = x_op_loc / (GJP.VolNodeSites()/GJP.TnodeSites());
    int t_op_glob = t_op_loc + GJP.TnodeCoor()*GJP.TnodeSites();

    //Average over all tK
    for(int tK = 0; tK < t_size; tK++){
      if(tK_vals != NULL && !int_in_vec(tK,*tK_vals) ) continue;

      //prod1 = vL(x_op) [[ wL^dag(x_K) wH(x_K) ]] vH^dag(x_op) \gamma_5
      SpinColorFlavorMatrix prod1;
      MesonField2::contract_vleft_vright(prod1, *prop_L, x_op_loc, *prop_H, x_op_loc, *wdagL_wH , tK, tK);
      prod1.gr(-5);

      //Pseudoscalar vertex with F_0 g5  and  -F_1 g5
      static const SpinColorFlavorMatrix mstructs[2] = { SpinColorFlavorMatrix(gamma5,F0), SpinColorFlavorMatrix(gamma5,F1)*-1.0 };
      for(int gidx=0;gidx<2;gidx++){
	std::complex<double>* corrs[n_con];
	for(int cc=0;cc<n_con;cc++) corrs[cc] = & into[2*cc+gidx](me,0,t_op_glob);

#define C(I) *corrs[I-c_start] 
	  
	C(23) += blob * Trace( mstructs[gidx], prod1 );
	  
#undef C
      }
    }
  
  }//end of site loop	
    
  for(int i=0;i<into.size();i++)
    into[i].sumLattice();
}

SpinColorFlavorMatrix Gparity_KtoPiPi::S2(gamma5,sigma3);
SpinColorFlavorMatrix Gparity_KtoPiPi::_F0(spin_unit,F0);
SpinColorFlavorMatrix Gparity_KtoPiPi::_F1(spin_unit,F1);
SpinColorFlavorMatrix Gparity_KtoPiPi::g5(gamma5,sigma0);
SpinColorFlavorMatrix Gparity_KtoPiPi::unit(spin_unit,sigma0);
SpinColorFlavorMatrix Gparity_KtoPiPi::gmu[4] = { SpinColorFlavorMatrix(gamma1,sigma0), 
						  SpinColorFlavorMatrix(gamma2,sigma0),
						  SpinColorFlavorMatrix(gamma3,sigma0),
						  SpinColorFlavorMatrix(gamma4,sigma0) };


CPS_END_NAMESPACE
