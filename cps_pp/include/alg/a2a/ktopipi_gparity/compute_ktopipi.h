#ifndef _COMPUTE_KTOPIPI_H
#define _COMPUTE_KTOPIPI_H

#include <alg/a2a/mesonfield.h>
#include "compute_ktopipi_base.h"

CPS_START_NAMESPACE

//NOTE:
//Daiqian's type2 and type3 files are exactly a factor of 2 larger than mine. This is because he forgot to divide by a factor of 2 from permuting the 2 pions. In the 2015 analysis we corrected this factor at analysis
//time but I have corrected it in the code (if you disable the #define below!)

#define DAIQIAN_COMPATIBILITY_MODE  //do not divide by 2, to match Daiqian's files


//If spatial subsampling (xyzStep>1) is in use, we must randomize which sites. Daiqian used a machine-size dependent spatial sampling that was reproduced for comparison with his code
//Note xyzStep>1 is not supported for SIMD fields and so was not used for the 2020 analysis
#ifndef USE_C11_RNG //no point reproducing Daiqian's code if we don't have the same RNG!
#define DAIQIAN_EVIL_RANDOM_SITE_OFFSET //use Daiqian's machine-size dependent random spatial sampling (cf below)
#endif

//For arrays of SIMD vectorized types we must use an aligned allocator in the std::vector. This wrapper gets the aligned vector type in that case, otherwise gives a std::vector
template<typename T, typename ComplexClass>
struct getInnerVectorType{
  typedef std::vector<T> type;
};

#ifdef USE_GRID
template<typename T>
struct getInnerVectorType<T,grid_vector_complex_mark>{
  typedef typename AlignedVector<T>::type type;
};
#endif

#define TIMER_ELEMS \
  ELEM(type1_compute_mfproducts) \
  ELEM(type1_mult_vMv_setup) \
  ELEM(type1_precompute_part1_part2) \
  ELEM(contraction_time) \
  ELEM(finish_up)\
  ELEM(total)
#define TIMER Type1timings
#include "implementation/static_timer_impl.tcc"


#define TIMER_ELEMS \
  ELEM(type2_compute_mfproducts) \
  ELEM(type2_mult_vMv_setup) \
  ELEM(type2_precompute_part1_part2) \
  ELEM(contraction_time) \
  ELEM(finish_up)\
  ELEM(total)
#define TIMER Type2timings
#include "implementation/static_timer_impl.tcc"


#define TIMER_ELEMS \
  ELEM(type3_compute_mfproducts) \
  ELEM(type3_mult_vMv_setup) \
  ELEM(type3_precompute_part1) \
  ELEM(part2_calc) \
  ELEM(contraction_time) \
  ELEM(finish_up)\
  ELEM(total)
#define TIMER Type3timings
#include "implementation/static_timer_impl.tcc"


#define TIMER_ELEMS \
  ELEM(type4_mult_vMv_setup) \
  ELEM(type4_precompute_part1) \
  ELEM(contraction_time) \
  ELEM(finish_up)\
  ELEM(total)
#define TIMER Type4timings
#include "implementation/static_timer_impl.tcc"

//K->pipi requires calculation of a meson field of the form   W_l^\dagger . W_h. Here is it's momentum policy

class StandardLSWWmomentaPolicy: public MomentumAssignments{
public:
  StandardLSWWmomentaPolicy(): MomentumAssignments() {
    this->combineSameTotalMomentum(true); //momentum pairs with same total momentum will be added to same entry and treated as 'alternates' which we average together below

    const int ngp = this->nGparityDirs();
    if(ngp == 0){
      addP("(0,0,0) + (0,0,0)");      
    }else if(ngp == 1){
      addP("(-1,0,0) + (1,0,0)");
    }else if(ngp == 2){
      addP("(-1,-1,0) + (1,1,0)");
    }else if(ngp == 3){
      addP("(-1,-1,-1) + (1,1,1)");
    }else{
      ERR.General("StandardLSWWmomentaPolicy","constructor","ngp cannot be >3\n");
    }
  }
};

//Same as the above but where we reverse the momentum assignments of the W^dag and W
class ReverseLSWWmomentaPolicy: public StandardLSWWmomentaPolicy{
public:
  ReverseLSWWmomentaPolicy(): StandardLSWWmomentaPolicy() {
    this->reverseABmomentumAssignments();
  }
};

//Add Wdag, W momenta and the reverse assignment to make a symmetric combination
class SymmetricLSWWmomentaPolicy: public StandardLSWWmomentaPolicy{
public:
  SymmetricLSWWmomentaPolicy(): StandardLSWWmomentaPolicy() {
    this->symmetrizeABmomentumAssignments();
  }
};


template<typename mf_Policies>
class ComputeKtoPiPiGparity: public ComputeKtoPiPiGparityBase{  
private:
  //Determine what node timeslices are actually needed. Returns true if at least one on-node top is needed
  //Criteria are: 4-quark operator timeslice >=1 timeslice from the kaon source, and >=1 timeslice away from the inner pion sink
  inline static bool getUsedTimeslices(std::vector<bool> &node_top_used, const std::vector<int> &tsep_k_pi, const int t_pi1){
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    node_top_used.resize(GJP.TnodeSites());
    bool ret = false;

    for(int top_loc = 0; top_loc < GJP.TnodeSites(); top_loc++){
      node_top_used[top_loc] = false;
      int top_glb = top_loc  + GJP.TnodeCoor()*GJP.TnodeSites();
      for(int tkpi_idx=0;tkpi_idx<tsep_k_pi.size();tkpi_idx++){
	int t_K = modLt(t_pi1 - tsep_k_pi[tkpi_idx], Lt);
	int t_dis = modLt(top_glb - t_K, Lt);
	if(t_dis < tsep_k_pi[tkpi_idx] && t_dis > 0){
	  node_top_used[top_loc] = true;
	  ret = true;
	}
      }
    }
    return ret;
  }
  //Determine what node timeslices are actually needed and at the same time append the set of values of t_K that will be needed to t_K_all unless they are already present
  //Criteria are: 4-quark operator timeslice >=1 timeslice from the kaon source, and >=1 timeslice away from the inner pion sink
  inline static void getUsedTimeslices(std::vector<bool> &node_top_used, std::vector<int> &t_K_all, const std::vector<int> &tsep_k_pi, const int t_pi1){
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    node_top_used.resize(GJP.TnodeSites());

    for(int top_loc = 0; top_loc < GJP.TnodeSites(); top_loc++){
      node_top_used[top_loc] = false;
      int top_glb = top_loc  + GJP.TnodeCoor()*GJP.TnodeSites();
      for(int tkpi_idx=0;tkpi_idx<tsep_k_pi.size();tkpi_idx++){
	int t_K = modLt(t_pi1 - tsep_k_pi[tkpi_idx], Lt);
	if(std::find(t_K_all.begin(), t_K_all.end(), t_K) == t_K_all.end())
	  t_K_all.push_back(t_K);

	int t_dis = modLt(top_glb - t_K, Lt);
	if(t_dis < tsep_k_pi[tkpi_idx] && t_dis > 0) node_top_used[top_loc] = true;
      }
    }
  }
  inline static void getUsedTimeslicesForKaon(std::vector<bool> &node_top_used, const std::vector<int> &tsep_k_pi, const int t_K){
    //Determine what node timeslices are actually needed
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    node_top_used.resize(GJP.TnodeSites());

    for(int top_loc = 0; top_loc < GJP.TnodeSites(); top_loc++){
      node_top_used[top_loc] = false;
      int top_glb = top_loc  + GJP.TnodeCoor()*GJP.TnodeSites();

      for(int tkpi_idx=0;tkpi_idx<tsep_k_pi.size();tkpi_idx++){
	int t_dis = modLt(top_glb - t_K, Lt);
	if(t_dis < tsep_k_pi[tkpi_idx] && t_dis > 0) node_top_used[top_loc] = true;
      }
    }
  }

public:
  typedef typename mf_Policies::ComplexType ComplexType;
  typedef CPSspinColorFlavorMatrix<ComplexType> SCFmat;
  typedef typename AlignedVector<SCFmat>::type SCFmatVector;
  typedef CPSmatrixField<SCFmat> SCFmatrixField;

  typedef KtoPiPiGparityResultsContainer<typename mf_Policies::ComplexType, Aligned128AllocPolicy> ResultsContainerType;
  typedef KtoPiPiGparityMixDiagResultsContainer<typename mf_Policies::ComplexType, Aligned128AllocPolicy> MixDiagResultsContainerType;

  typedef A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> mf_WV;
  typedef A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorWfftw> mf_WW;

  typedef mult_vMv_split_lite<mf_Policies,A2AvectorV,A2AvectorWfftw,A2AvectorWfftw,A2AvectorV> vMv_split_VWWV;
  typedef mult_vMv_split_lite<mf_Policies,A2AvectorV,A2AvectorWfftw,A2AvectorVfftw,A2AvectorW> vMv_split_VWVW;

#ifdef USE_DESTRUCTIVE_FFT
  typedef A2AvectorW<mf_Policies> Wtype;
#else
  typedef const A2AvectorW<mf_Policies> Wtype;
#endif
  
  //Compute type 1,2,3,4 diagram for fixed tsep(pi->K) and tsep(pi->pi). 

  //By convention pi1 is the pion closest to the kaon

  //User provides the momentum of pi1, p_pi_1

  //tstep is the time sampling frequency for the kaon/pion

  //The kaon meson field is expected to be constructed in the W*W form

  typedef typename A2Asource<typename mf_Policies::SourcePolicies::ComplexType, typename mf_Policies::SourcePolicies::MappingPolicy, typename mf_Policies::SourcePolicies::AllocPolicy>::FieldType::InputParamType SourceParamType;
  
  //ls_WW meson fields
  template< typename Allocator, typename MomentumPolicy>
  static void generatelsWWmesonfields(std::vector<mf_WW,Allocator> &mf_ls_ww,
				      Wtype &W, Wtype &W_s, const MomentumPolicy &lsWWmom, const int kaon_rad, Lattice &lat,
				      const SourceParamType &src_params = NullObject()){
    typedef typename mf_Policies::SourcePolicies SourcePolicies;
    a2a_printf("Computing ls WW meson fields for K->pipi\n");
    double time = -dclock();

    const int Lt = GJP.Tnodes()*GJP.TnodeSites();
    mf_ls_ww.resize(Lt);

    typedef A2AflavorProjectedExpSource<SourcePolicies> SourceType;
    typedef SCFspinflavorInnerProduct<0,typename mf_Policies::ComplexType,SourceType> InnerType;
    typedef typename mf_Policies::FermionFieldType::InputParamType VWfieldInputParams;

    VWfieldInputParams fld_params = W.getFieldInputParams();

    assert(lsWWmom.nMom() == 1);
    int nalt = lsWWmom.nAltMom(0);

    for(int a=0;a<nalt;a++){
      ThreeMomentum pA = lsWWmom.getMomA(0,a);
      ThreeMomentum mpA = -pA;
      ThreeMomentum pB = lsWWmom.getMomB(0,a);
      
      SourceType fpexp(kaon_rad, pB.ptr() , src_params);
      InnerType mf_struct(sigma0,fpexp); // (1)_flav * (1)_spin 

      A2AvectorWfftw<mf_Policies> fftw_Wl_p(W.getArgs(),fld_params);
      A2AvectorWfftw<mf_Policies> fftw_Ws_p(W_s.getArgs(),fld_params);
      
#ifdef USE_DESTRUCTIVE_FFT
      fftw_Wl_p.destructiveGaugeFixTwistFFT(W, mpA.ptr(),lat); //will be daggered, swapping momentum
      fftw_Ws_p.destructiveGaugeFixTwistFFT(W_s,pB.ptr(),lat); 
#else
      fftw_Wl_p.gaugeFixTwistFFT(W, mpA.ptr(),lat); 
      fftw_Ws_p.gaugeFixTwistFFT(W_s,pB.ptr(),lat); 
#endif
      std::vector<mf_WW,Allocator> mf_ls_ww_tmp(Lt);
      mf_WW::compute(mf_ls_ww_tmp, fftw_Wl_p, mf_struct, fftw_Ws_p);

#ifdef USE_DESTRUCTIVE_FFT
      fftw_Wl_p.destructiveUnapplyGaugeFixTwistFFT(W, mpA.ptr(),lat);
      fftw_Ws_p.destructiveUnapplyGaugeFixTwistFFT(W_s, pB.ptr(),lat);
#endif

      if(a == 0)
	for(int t=0;t<Lt;t++) mf_ls_ww[t].move(mf_ls_ww_tmp[t]);
      else
	for(int t=0;t<Lt;t++) mf_ls_ww[t].plus_equals(mf_ls_ww_tmp[t]);
    }
    for(int t=0;t<Lt;t++) mf_ls_ww[t].times_equals(1./nalt);
          
    time += dclock();
    print_time("ComputeKtoPiPiGparity","ls WW meson fields",time);
  }

  //--------------------------------------------------------------------------
  //TYPE 1
  typedef CPSfield<int,1,FourDpolicy<OneFlavorPolicy> > OneFlavorIntegerField;

private:
  //Run inside threaded environment
  static void type1_contract(ResultsContainerType &result, const int t_K, const int t_dis, const int thread_id, const SCFmat part1[2], const SCFmat part2[2]);

  //Field version
  static void type1_contract(ResultsContainerType &result, const int t_K, const std::vector<SCFmatrixField> &part1, const std::vector<SCFmatrixField> &part2);

  //If spatial subsampling (xyzStep>1) is in use, we must randomize which sites. Daiqian used a machine-size dependent spatial sampling that was reproduced for comparison with his code
  //Note xyzStep>1 is not supported for SIMD fields and so was not used for the 2020 analysis. Thus this function should be considered as deprecated
  static void generateRandomOffsets(std::vector<OneFlavorIntegerField*> &random_fields, const std::vector<int> &tsep_k_pi, const int tstep, const int xyzStep);

  static void type1_compute_mfproducts(std::vector<std::vector< mf_WW > > &con_pi1_K,
				       std::vector<std::vector< mf_WW > > &con_pi2_K,
				       const std::vector<mf_WV > &mf_pi1,
				       const std::vector<mf_WV > &mf_pi2,
				       const std::vector<mf_WW > &mf_kaon, const MesonFieldMomentumContainer<mf_Policies> &mf_pions,
				       const std::vector<int> &tsep_k_pi, const int tsep_pion, const int Lt, const int ntsep_k_pi,
				       const std::vector<bool> &tpi1_mask, const std::vector<bool> &tpi2_mask);

  static void type1_mult_vMv_setup(vMv_split_VWVW &mult_vMv_split_part1_pi1,
				   vMv_split_VWVW &mult_vMv_split_part1_pi2,
				   std::vector<vMv_split_VWWV > &mult_vMv_split_part2_pi1,
				   std::vector<vMv_split_VWWV > &mult_vMv_split_part2_pi2,
				   const std::vector<std::vector< mf_WW > > &con_pi1_K,
				   const std::vector<std::vector< mf_WW > > &con_pi2_K,
				   const std::vector<mf_WV > &mf_pi1,
				   const std::vector<mf_WV > &mf_pi2,							   
				   const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
				   const A2AvectorW<mf_Policies> & wL,
				   const ModeContractionIndices<StandardIndexDilution,TimePackedIndexDilution> &i_ind_vw,
				   const ModeContractionIndices<StandardIndexDilution,FullyPackedIndexDilution> &j_ind_vw,
				   const ModeContractionIndices<TimePackedIndexDilution,StandardIndexDilution> &j_ind_wv,
				   const int top_loc, const int t_pi1, const int t_pi2, const int Lt, const std::vector<int> &tsep_k_pi, const int ntsep_k_pi, const int t_K_all[], const std::vector<bool> &node_top_used);

  static void type1_precompute_part1_part2(SCFmatVector &mult_vMv_contracted_part1_pi1,
					   SCFmatVector &mult_vMv_contracted_part1_pi2,
					   std::vector<SCFmatVector> &mult_vMv_contracted_part2_pi1,
					   std::vector<SCFmatVector> &mult_vMv_contracted_part2_pi2,
					   vMv_split_VWVW &mult_vMv_split_part1_pi1,
					   vMv_split_VWVW &mult_vMv_split_part1_pi2,
					   std::vector<vMv_split_VWWV > &mult_vMv_split_part2_pi1,
					   std::vector<vMv_split_VWWV > &mult_vMv_split_part2_pi2,
					   const int top_loc, const int Lt, const std::vector<int> &tsep_k_pi, const int ntsep_k_pi, const int t_K_all[], const std::vector<bool> &node_top_used);

public:
  //These versions overlap computation for multiple K->pi separations. Result should be an array of KtoPiPiGparityResultsContainer the same size as the vector 'tsep_k_pi'
  //xyzStep is the 3d spatial sampling frequency. If set to anything other than one it will compute the diagram
  //for a random site within the block (site, site+xyzStep) in canonical ordering. Daiqian's original implementation is machine-size dependent, but for repro I had to add an option to do it his way 

  //CPU implementation with openmp loop over site
  static void type1_omp(ResultsContainerType result[],
			     const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const int xyzStep, const ThreeMomentum &p_pi_1, 
			     const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
			     const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
			     const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH);
  
  //Field implementation both threaded and offloadable to GPU. WILL COMPILE ONLY FOR SIMD COMPLEX DATA
  static void type1_field_SIMD(ResultsContainerType result[],
		     const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const ThreeMomentum &p_pi_1, 
		     const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
		     const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
		     const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH);

  //Above version only applicable to SIMD data. For non SIMD data this version falls back to CPU version
  static void type1_field(ResultsContainerType result[],
			       const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const ThreeMomentum &p_pi_1, 
			       const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
			       const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
			       const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH);

  //Note xyzstep is ignored for field implementation
  static void type1(ResultsContainerType result[],
		    const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const int xyzStep, const ThreeMomentum &p_pi_1, 
		    const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
		    const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
		    const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH){
#if defined(GPU_VEC) || defined(FORCE_A2A_OFFLOAD)
    a2a_printf("Using type1 field implementation\n");
    type1_field(result, tsep_k_pi, tsep_pion, tstep, p_pi_1, mf_kaon, mf_pions, vL, vH, wL, wH); //falls back to CPU implementation for non-SIMD data
#else
    a2a_printf("Using type1 OMP implementation\n");
    type1_omp(result, tsep_k_pi, tsep_pion, tstep, xyzStep, p_pi_1, mf_kaon, mf_pions, vL, vH, wL, wH);
#endif
  }


  static void type1(ResultsContainerType &result,
		    const int tsep_k_pi, const int tsep_pion, const int tstep, const int xyzStep, const ThreeMomentum &p_pi_1, 
		    const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
		    const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
		    const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH){
    std::vector<int> tt(1,tsep_k_pi);
    return type1(&result,tt, tsep_pion,tstep,xyzStep,p_pi_1,
		 mf_kaon, mf_pions,
		 vL,vH,
		 wL,wH);
  }
  static void type1(std::vector<ResultsContainerType> &result,
		    const std::vector<int> tsep_k_pi, const int tsep_pion, const int tstep, const int xyzStep, const ThreeMomentum &p_pi_1, 
		    const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
		    const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
		    const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH){
    result.resize(tsep_k_pi.size());
    return type1(&result[0],tsep_k_pi, tsep_pion,tstep,xyzStep,p_pi_1,
		 mf_kaon, mf_pions,
		 vL,vH,
		 wL,wH);
  }


  //-------------------------------------------------------------------
  //TYPE 2
private:
  //Run inside threaded environment
  static void type2_contract(ResultsContainerType &result, const int t_K, const int t_dis, const int thread_id, const SCFmat &part1, const SCFmat part2[2]);

  //Field implementation
  static void type2_contract(ResultsContainerType &result, const int t_K, const SCFmatrixField &part1, const std::vector<SCFmatrixField> &part2);
 
  static void type2_compute_mfproducts(std::vector<mf_WV > &con_pi1_pi2,
				       std::vector<mf_WV > &con_pi2_pi1,							     
				       const int tsep_pion, const int tstep, const std::vector<ThreeMomentum> &p_pi_1_all,
				       MesonFieldMomentumContainer<mf_Policies> &mf_pions,
				       const int Lt, const int tpi_sampled);

  static void type2_mult_vMv_setup(std::vector<vMv_split_VWWV > &mult_vMv_split_part1,
				   std::vector<vMv_split_VWVW > &mult_vMv_split_part2_pi1_pi2,
				   std::vector<vMv_split_VWVW > &mult_vMv_split_part2_pi2_pi1,
				   const std::vector< mf_WV > &con_pi1_pi2,
				   const std::vector< mf_WV > &con_pi2_pi1,
				   const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, const A2AvectorW<mf_Policies> & wL,
				   const std::vector<mf_WW > &mf_kaon,
				   const std::vector<int> &t_K_all, const int top_loc, const int tstep, const int Lt,const int tpi_sampled,
				   const std::vector< std::vector<bool> > &node_top_used, const std::vector< std::vector<bool> > &node_top_used_kaon);

  static void type2_precompute_part1_part2(std::vector<SCFmatVector > &mult_vMv_contracted_part1,
					   std::vector<SCFmatVector > &mult_vMv_contracted_part2_pi1_pi2,
					   std::vector<SCFmatVector > &mult_vMv_contracted_part2_pi2_pi1,
					   std::vector<vMv_split_VWWV > &mult_vMv_split_part1,
					   std::vector<vMv_split_VWVW > &mult_vMv_split_part2_pi1_pi2,
					   std::vector<vMv_split_VWVW > &mult_vMv_split_part2_pi2_pi1,
					   const std::vector<int> &t_K_all, const int top_loc, const int tstep, const int Lt,const int tpi_sampled,
					   const std::vector< std::vector<bool> > &node_top_used, const std::vector< std::vector<bool> > &node_top_used_kaon);

  static void type2_compute_mfproducts(mf_WV &con_pi1_pi2,
				       mf_WV &con_pi2_pi1,
				       const int tpi1, const int tpi2,
				       const std::vector<ThreeMomentum> &p_pi_1_all,
				       MesonFieldMomentumContainer<mf_Policies> &mf_pions);

public: 
  //This version averages over multiple pion momentum configurations. Use to project onto A1 representation at run-time. Saves a lot of time!
  //This version also overlaps computation for multiple K->pi separations. Result should be an array of ResultsContainerType the same size as the vector 'tsep_k_pi'

  //CPU implementation with openmp loop over site
  static void type2_omp_v1(ResultsContainerType result[],
		       const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
		       const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
		       const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
		       const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH);
  
  //Improved CPU implementation with openmp loop over site
  static void type2_omp_v2(ResultsContainerType result[],
		       const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
		       const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
		       const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
		       const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH);

  //Field implementation both threaded and offloadable to GPU.  WILL COMPILE ONLY FOR SIMD COMPLEX DATA  
  static void type2_field_SIMD(ResultsContainerType result[],
			  const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
			  const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
			  const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
			  const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH);

  //Above version only applicable to SIMD data. For non SIMD data this version falls back to CPU version
  static void type2_field(ResultsContainerType result[],
			  const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
			  const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
			  const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
			  const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH);


  inline static void type2(ResultsContainerType result[],
			   const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
			   const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
			   const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
			   const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH){
#if defined(GPU_VEC) || defined(FORCE_A2A_OFFLOAD)
    a2a_printf("Using type2 field implementation\n");
    type2_field(result, tsep_k_pi, tsep_pion, tstep, p_pi_1_all, mf_kaon, mf_pions, vL, vH, wL, wH);
#else
    a2a_printf("Using type2 OMP implementation\n");
    //type2_v2(result, tsep_k_pi, tsep_pion, tstep, p_pi_1_all, mf_kaon, mf_pions, vL, vH, wL, wH);
    type2_omp_v2(result, tsep_k_pi, tsep_pion, tstep, p_pi_1_all, mf_kaon, mf_pions, vL, vH, wL, wH);
#endif
  }



  static void type2(ResultsContainerType &result,
		    const int tsep_k_pi, const int tsep_pion, const int tstep, const ThreeMomentum &p_pi_1, 
		    const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
		    const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
		    const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH){
    std::vector<int> tkp(1,tsep_k_pi);
    std::vector<ThreeMomentum> p(1, p_pi_1); 
    return type2(&result,tkp,tsep_pion,tstep,p,
		 mf_kaon, mf_pions,
		 vL, vH,
		 wL, wH);
  }
  template<typename MomComputePolicy>
  static void type2(ResultsContainerType &result,
		    const int tsep_k_pi, const int tsep_pion, const int tstep, const MomComputePolicy &p_pi_1_all, 
		    const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
		    const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
		    const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH){
    std::vector<int> tkp(1,tsep_k_pi);
    std::vector<ThreeMomentum> p(p_pi_1_all.nMom());
    for(int i=0;i<p_pi_1_all.nMom();i++)
      p[i] = p_pi_1_all.getMesonMomentum(i);
    return type2(&result,tkp,tsep_pion,tstep,p,
		 mf_kaon, mf_pions,
		 vL, vH,
		 wL, wH);
  }
  template<typename MomComputePolicy>
  static void type2(std::vector<ResultsContainerType> &result,
		    const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const MomComputePolicy &p_pi_1_all, 
		    const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
		    const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
		    const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH){
    result.resize(tsep_k_pi.size());
    std::vector<ThreeMomentum> p(p_pi_1_all.nMom());
    for(int i=0;i<p_pi_1_all.nMom();i++)
      p[i] = p_pi_1_all.getMesonMomentum(i);

    type2(&result[0], 
	  tsep_k_pi, tsep_pion, tstep, p,
	  mf_kaon, mf_pions,
	  vL, vH,
	  wL, wH);
  }
  
  //------------------------------------------------------------------------------------------------
  //TYPE 3 and MIX 3
private:
  //Run inside threaded environment
  static void type3_contract(ResultsContainerType &result, const int t_K, const int t_dis, const int thread_id, 
			     const SCFmat part1[2], const SCFmat &part2_L, const SCFmat &part2_H);

  //Field implementation
  static void type3_contract(ResultsContainerType &result, const int t_K, 
			     const std::vector<SCFmatrixField> &part1, const SCFmatrixField &part2_L, const SCFmatrixField &part2_H);

  static void type3_compute_mfproducts(std::vector<std::vector<mf_WW > > &con_pi1_pi2_k,
				       std::vector<std::vector<mf_WW > > &con_pi2_pi1_k,							     
				       const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
				       const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
				       const int Lt, const int tpi_sampled, const int ntsep_k_pi);
  static void type3_mult_vMv_setup(vMv_split_VWWV &mult_vMv_split_part1_pi1_pi2,
				   vMv_split_VWWV &mult_vMv_split_part1_pi2_pi1,
				   const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH,
				   const std::vector<std::vector<mf_WW > > &con_pi1_pi2_k,
				   const std::vector<std::vector<mf_WW > > &con_pi2_pi1_k,
				   const int top_loc, const int t_pi1_idx, const int tkp);

  static void type3_precompute_part1(SCFmatVector &mult_vMv_contracted_part1_pi1_pi2,
				     SCFmatVector &mult_vMv_contracted_part1_pi2_pi1,
				     vMv_split_VWWV &mult_vMv_split_part1_pi1_pi2,
				     vMv_split_VWWV &mult_vMv_split_part1_pi2_pi1);
    
  static void type3_compute_mfproducts(std::vector<mf_WW > &con_pi1_pi2_k,
				       std::vector<mf_WW > &con_pi2_pi1_k,
				       const int tpi1, const int tpi2, const std::vector<int> &tsep_k_pi, const int tsep_pion, 
				       const int tstep, const std::vector<ThreeMomentum> &p_pi_1_all,
				       const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
				       const int Lt, const int ntsep_k_pi);

  static void type3_mult_vMv_setup(vMv_split_VWWV &mult_vMv_split_part1_pi1_pi2,
				   vMv_split_VWWV &mult_vMv_split_part1_pi2_pi1,
				   const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH,
				   const std::vector<mf_WW > &con_pi1_pi2_k,
				   const std::vector<mf_WW > &con_pi2_pi1_k,
				   const int top_loc, const int tkp);

public:
  //This version averages over multiple pion momentum configurations. Use to project onto A1 representation at run-time. Saves a lot of time!
  //This version also overlaps computation for multiple K->pi separations. Result should be an array of ResultsContainerType the same size as the vector 'tsep_k_pi'

  //CPU implementation with openmp loop over site
  static void type3_omp_v1(ResultsContainerType result[], MixDiagResultsContainerType mix3[],
		    const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
		    const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
		    const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
		    const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH);

  //Improved CPU implementation with openmp loop over site
  static void type3_omp_v2(ResultsContainerType result[], MixDiagResultsContainerType mix3[],
		    const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
		    const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
		    const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
		    const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH);

  //Field implementation both threaded and offloadable to GPU.  WILL COMPILE ONLY FOR SIMD COMPLEX DATA
  static void type3_field_SIMD(ResultsContainerType result[], MixDiagResultsContainerType mix3[],
			       const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
			       const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
			       const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
			       const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH);

  //Above version only applicable to SIMD data. For non SIMD data this version falls back to CPU version
  static void type3_field(ResultsContainerType result[], MixDiagResultsContainerType mix3[],
			  const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
			  const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
			  const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
			  const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH);



  inline static void type3(ResultsContainerType result[], MixDiagResultsContainerType mix3[],
		    const std::vector<int> &tsep_k_pi, const int &tsep_pion, const int &tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
		    const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
		    const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
		    const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH){
#if defined(GPU_VEC) || defined(FORCE_A2A_OFFLOAD)
    a2a_printf("Using type3 field implementation\n");
    type3_field(result, mix3, tsep_k_pi, tsep_pion, tstep, p_pi_1_all, mf_kaon, mf_pions, vL, vH, wL, wH);
#else
    a2a_printf("Using type3 OMP implementation\n");
    //type3_v1(result, mix3, tsep_k_pi, tsep_pion, tstep, p_pi_1_all, mf_kaon, mf_pions, vH, wH, wL, wH);
    type3_omp_v2(result, mix3, tsep_k_pi, tsep_pion, tstep, p_pi_1_all, mf_kaon, mf_pions, vL, vH, wL, wH);
#endif
  }

  static void type3(ResultsContainerType &result, MixDiagResultsContainerType &mix3,
		    const int &tsep_k_pi, const int &tsep_pion, const int &tstep, const ThreeMomentum &p_pi_1, 
		    const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
		    const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
		    const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH){
    std::vector<int> tkp(1,tsep_k_pi);
    std::vector<ThreeMomentum> p(1, p_pi_1); 
    return type3(&result,&mix3,tkp,tsep_pion,tstep,p,
		 mf_kaon, mf_pions,
		 vL, vH,
		 wL, wH);
  }
  template<typename MomComputePolicy>
  static void type3(ResultsContainerType &result, MixDiagResultsContainerType &mix3,
		    const int &tsep_k_pi, const int &tsep_pion, const int &tstep, const MomComputePolicy &p_pi_1_all, 
		    const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
		    const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
		    const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH){
    std::vector<int> tkp(1,tsep_k_pi);
    std::vector<ThreeMomentum> p(p_pi_1_all.nMom());
    for(int i=0;i<p_pi_1_all.nMom();i++)
      p[i] = p_pi_1_all.getMesonMomentum(i);
    return type3(&result,&mix3,tkp,tsep_pion,tstep,p,
		 mf_kaon, mf_pions,
		 vL, vH,
		 wL, wH);
  }
  template<typename MomComputePolicy>
  static void type3(std::vector<ResultsContainerType> &result, std::vector<MixDiagResultsContainerType> &mix3,
		    const std::vector<int> &tsep_k_pi, const int &tsep_pion, const int &tstep, const MomComputePolicy &p_pi_1_all, 
		    const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
		    const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
		    const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH){
    result.resize(tsep_k_pi.size()); mix3.resize(tsep_k_pi.size());
    std::vector<ThreeMomentum> p(p_pi_1_all.nMom());
    for(int i=0;i<p_pi_1_all.nMom();i++)
      p[i] = p_pi_1_all.getMesonMomentum(i);
    return type3(&result[0],&mix3[0],tsep_k_pi,tsep_pion,tstep,p,
		 mf_kaon, mf_pions,
		 vL, vH,
		 wL, wH);
  }




  //------------------------------------------------------------------------------------------------
  //TYPE 4 and MIX 4
protected:

  static void type4_contract(ResultsContainerType &result, const int t_K, const int t_dis, const int thread_id, 
			     const SCFmat &part1, const SCFmat &part2_L, const SCFmat &part2_H);
  


  static void type4_mult_vMv_setup(std::vector<vMv_split_VWWV > &mult_vMv_split_part1,
				   const std::vector<mf_WW > &mf_kaon,
				   const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH,
				   const int top_loc, const int tstep, const int Lt);

  static void type4_precompute_part1(std::vector<SCFmatVector > &mult_vMv_contracted_part1,
				     std::vector<vMv_split_VWWV > &mult_vMv_split_part1,
				     const int top_loc, const int tstep, const int Lt);
  
  static void type4_contract(ResultsContainerType &result, const int t_K,
			     const SCFmatrixField &part1, const SCFmatrixField &part2_L, const SCFmatrixField &part2_H);


public:
  //CPU implementation with openmp loop over site
  static void type4_omp(ResultsContainerType &result, MixDiagResultsContainerType &mix4,
		    const int tstep,
		    const std::vector<mf_WW > &mf_kaon,
		    const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
		    const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH);

  //Field implementation both threaded and offloadable to GPU. WILL COMPILE ONLY FOR SIMD COMPLEX DATA
  static void type4_field_SIMD(ResultsContainerType &result, MixDiagResultsContainerType &mix4,
		     const int tstep,
		     const std::vector<mf_WW > &mf_kaon,
		     const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
		     const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH);

  //Above version only applicable to SIMD data. For non SIMD data this version falls back to CPU version
  static void type4_field(ResultsContainerType &result, MixDiagResultsContainerType &mix4,
		     const int tstep,
		     const std::vector<mf_WW > &mf_kaon,
		     const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
		     const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH);


  static void type4(ResultsContainerType &result, MixDiagResultsContainerType &mix4,
		    const int tstep,
		    const std::vector<mf_WW > &mf_kaon,
		    const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
		    const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH){
#if defined(GPU_VEC) || defined(FORCE_A2A_OFFLOAD)
    a2a_printf("Using type4 field implementation\n");
    type4_field(result, mix4, tstep, mf_kaon, vL, vH, wL, wH);
#else
    a2a_printf("Using type4 OMP implementation\n");
    type4_omp(result, mix4, tstep, mf_kaon, vL, vH, wL, wH);
#endif
  }
    

};

#include "implementation/compute_ktopipi_type1.tcc"
#include "implementation/compute_ktopipi_type2.tcc"
#include "implementation/compute_ktopipi_type3.tcc"
#include "implementation/compute_ktopipi_type4.tcc"
#include "implementation/compute_ktopipi_type1_field.tcc"
#include "implementation/compute_ktopipi_type2_field.tcc"
#include "implementation/compute_ktopipi_type3_field.tcc"
#include "implementation/compute_ktopipi_type4_field.tcc"


CPS_END_NAMESPACE

#undef DAIQIAN_COMPATIBILITY_MODE
#undef DAIQIAN_EVIL_RANDOM_SITE_OFFSET



#endif
