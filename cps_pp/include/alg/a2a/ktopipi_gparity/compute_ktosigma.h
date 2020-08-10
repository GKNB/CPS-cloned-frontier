#ifndef COMPUTE_KTOSIGMA_H_
#define COMPUTE_KTOSIGMA_H_

#include "compute_ktopipi.h"

CPS_START_NAMESPACE

template<typename mf_Policies>
class ComputeKtoSigma: public ComputeKtoPiPiGparityBase{
public:
  typedef typename mf_Policies::ComplexType ComplexType; //can be SIMD vectorized
  typedef CPSspinColorFlavorMatrix<ComplexType> SCFmat; //supports SIMD vectorization
  typedef typename getInnerVectorType<SCFmat,typename ComplexClassify<ComplexType>::type>::type SCFmatVector;
  typedef CPSmatrixField<SCFmat> SCFmatrixField;
  
  typedef KtoPiPiGparityResultsContainer<typename mf_Policies::ComplexType, typename mf_Policies::AllocPolicy> ResultsContainerType;
  typedef KtoPiPiGparityMixDiagResultsContainer<typename mf_Policies::ComplexType, typename mf_Policies::AllocPolicy> MixDiagResultsContainerType;
  typedef A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> SigmaMesonFieldType;
  typedef A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorWfftw> KaonMesonFieldType;

#ifdef KTOSIGMA_USE_SPLIT_VMV_LITE
  typedef mult_vMv_split_lite_shrbuf<mf_Policies,A2AvectorV,A2AvectorWfftw,A2AvectorWfftw,A2AvectorV> vMv_split_VWWV;
  typedef mult_vMv_split_lite_shrbuf<mf_Policies,A2AvectorV,A2AvectorWfftw,A2AvectorVfftw,A2AvectorW> vMv_split_VWVW;
  typedef vMvLiteSharedBuf<mf_Policies> vMv_split_shrbuf;
#else
  typedef mult_vMv_split<mf_Policies,A2AvectorV,A2AvectorWfftw,A2AvectorWfftw,A2AvectorV> vMv_split_VWWV;
  typedef mult_vMv_split<mf_Policies,A2AvectorV,A2AvectorWfftw,A2AvectorVfftw,A2AvectorW> vMv_split_VWVW;
#endif

private:
  const A2AvectorV<mf_Policies> & vL;
  const A2AvectorV<mf_Policies> & vH;
  const A2AvectorW<mf_Policies> & wL;
  const A2AvectorW<mf_Policies> & wH;
  
  const std::vector<KaonMesonFieldType> &mf_ls_WW;

  const std::vector<int> tsep_k_sigma;
  int tsep_k_sigma_lrg; //largest

  int Lt;
  int ntsep_k_sigma;
  int nthread;
  int size_3d;

  //---------------------------Type1/2 contractions-----------------------------------------

  //D1 = Tr( PH_KO M1 P_OS P_SO M2 P_OK G5 )
  //   = Tr( PH^dag_OK G5 M1 P_OS P_SO M2 P_OK )
  //   = Tr( WH_K VH^dag_O G5 M1 V_O W^dag_S V_S W^dag_O M2 V_O W^dag_K )
  //   = Tr( V_O [W^dag_K WH_K] VH^dag_O G5 M1 V_O [W^dag_S V_S] W^dag_O M2 )

  //D6 = Tr( PH_KO M1 P_OK G5 ) Tr( P_OS P_SO M2 )
  //   = Tr( PH^dag_OK G5 M1 P_OK ) Tr( P_OS P_SO M2 )
  //   = Tr( WH_K VH^dag_O G5 M1 V_O W^dag_K ) Tr( V_O W^dag_S V_S W^dag_O M2 )
  //   = Tr( V_O [W^dag_K WH_K] VH^dag_O G5 M1 ) Tr( V_O [W^dag_S V_S] W^dag_O M2 )
    
  //D8 = Tr( ( P_OS P_SO )_ba ( M2 P_OK G5 PH_KO M1 )_ba )
  //   = Tr( ( P_OS P_SO )_ba ( M2 P_OK PH^dag_OK G5 M1 )_ba )
  //   = Tr( ( V_O [W^dag_S V_S] W^dag_O )_ba ( M2 V_O [W^dag_K WH_K] VH^dag_O G5 M1 )_ba )

  //D11 = Tr( P_OK G5 PH_KO M1 )_ba Tr( P_OS P_SO M2 )_ba
  //    = Tr( P_OK PH^dag_OK G5 M1 )_ba Tr( P_OS P_SO M2 )_ba
  //    = Tr( V_O W^dag_K WH_K VH^dag_O G5 M1 )_ba Tr( V_O W^dag_S V_S W^dag_O M2 )_ba
  //    = Tr( V_O [W^dag_K WH_K] VH^dag_O G5 M1 )_ba Tr( V_O [W^dag_S V_S] W^dag_O M2 )_ba

  //D19 = Tr(  Tr_c( P_OS P_SO )M2 Tr_c( P_OK G5 PH_KO M1 ) )
  //    = Tr(  Tr_c( P_OS P_SO )M2 Tr_c( P_OK PH^dag_OK G5 M1 ) )
  //    = Tr(  Tr_c( V_O [W^dag_S V_S] W^dag_O ) M2 Tr_c( V_O [W^dag_K WH_K] VH^dag_O G5 M1 ) )

  inline void compute_type12_part1(SCFmat &pt1, const int tK_glb, const int top_loc, const int xop_loc){
    mult(pt1, vL, mf_ls_WW[tK_glb], vH, xop_loc, top_loc, false, true);
  }

#ifdef KTOSIGMA_USE_SPLIT_VMV_LITE 
#define BUF_ARG , vMv_split_shrbuf *shared_buf
#define BUF_PASS , shared_buf
#else
#define BUF_ARG
#define BUF_PASS 
#endif

  void setup_type12_pt1_split(std::vector<vMv_split_VWWV> &part1_split, const int top_glb, const std::vector<int> &tK_subset_map BUF_ARG){
    for(int i=0;i<tK_subset_map.size();i++){
      int tK_glb = tK_subset_map[i];
      part1_split[i].setup(vL,mf_ls_WW[tK_glb], vH,top_glb BUF_PASS);
    }
  }

  inline void compute_type12_part2(SCFmat &pt2, const int tS_glb, const int top_loc, const int xop_loc, const std::vector<SigmaMesonFieldType> &mf_S){
    mult(pt2, vL, mf_S[tS_glb], wL, xop_loc, top_loc, false, true);
  }

  void setup_type12_pt2_split(std::vector<vMv_split_VWVW> &part2_split, std::vector<SigmaMesonFieldType> &mf_S, 
			      const int top_glb, const std::vector<int> &tS_subset_map BUF_ARG){
    for(int i=0;i<tS_subset_map.size();i++){
      int tS_glb = tS_subset_map[i];
      part2_split[i].setup(vL, mf_S[tS_glb], wL, top_glb BUF_PASS);
    }
  }


  //Run inside threaded environment
  void type12_contract(ResultsContainerType &result, const int tK_glb, const int tdis_glb, const int thread_id, const SCFmat &part1, const SCFmat &part2);

  //Field implementation
  void type12_contract(ResultsContainerType &result, const int tK_glb, const SCFmatrixField &part1, const SCFmatrixField &part2);


public:

  inline void idx_t_map(std::vector<int> &map, std::vector<int> &inv_map, const std::set<int> &tset){
    int ntuse = tset.size();
    map.resize(ntuse);
    inv_map.resize(Lt);

    int map_idx = 0;
    for(int t=0;t<Lt;t++) 
      if(tset.count(t)){
	map[map_idx] = t;
	inv_map[t] = map_idx++;
      }
  }

  //CPU implementation with openmp loop over site
  void type12_omp(std::vector<ResultsContainerType> &result, std::vector<SigmaMesonFieldType> &mf_S);

  //Field implementation both threaded and offloadable to GPU. WILL COMPILE ONLY FOR SIMD COMPLEX DATA
  void type12_field_SIMD(std::vector<ResultsContainerType> &result, std::vector<SigmaMesonFieldType> &mf_S);

  //Above version only applicable to SIMD data. For non SIMD data this version falls back to CPU version
  void type12_field(std::vector<ResultsContainerType> &result, std::vector<SigmaMesonFieldType> &mf_S);

  void type12(std::vector<ResultsContainerType> &result, std::vector<SigmaMesonFieldType> &mf_S){
#ifdef GRID_NVCC
    type12_field(result, mf_S);
#else
    type12_omp(result, mf_S);
#endif
  }

private:

  //---------------------------Type3 contractions-----------------------------------------
  //D2 = Tr( PH_KO M1 P_OS P_SK G5 ) * Tr( P_OO M2 )
  //   = Tr( PH^dag_OK G5 M1 P_OS P_SK ) * Tr( P_OO M2 )
  //   = Tr( WH_K VH^dag_O G5 M1 V_O W^dag_S V_S W^dag_K ) * Tr( V_O W^dag_O M2 )
  //   = Tr( [W^dag_K WH_K] VH^dag_O G5 M1 V_O [W^dag_S V_S] ) * Tr( V_O W^dag_O M2 )

  //D3 = Tr( PH_KO M1 P_OO M2 P_OS P_SK G5 )
  //   = Tr( PH^dag_OK G5 M1 P_OO M2 P_OS P_SK )
  //   = Tr( WH_K VH^dag_O G5 M1 V_O W^dag_O M2 V_O W^dag_S V_S W^dag_K )
  //   = Tr( [W^dag_K WH_K] VH^dag_O G5 M1 V_O W^dag_O M2 V_O [W^dag_S V_S]  )

  //D7 = Tr( ( P_OS P_SK G5 PH_KO )_ab  (M1 P_OO M2)_ab )
  //   = Tr( ( P_OS P_SK PH^dag_OK G5 )_ab  (M1 P_OO M2)_ab )
  //   = Tr( ( V_O [W^dag_S V_S] [W^dag_K WH_K] VH^dag_O G5 )_ab  (M1 V_O W^dag_O M2)_ab )

  //D10 = Tr( M2 P_OO )_ab * Tr( P_OS P_SK G5 PH_KO M1 )_ab
  //    = Tr( M2 P_OO )_ab * Tr( P_OS P_SK PH^dag_OK G5 M1 )_ab
  //    = Tr( M2 V_O W^dag_O )_ab * Tr( V_O [W^dag_S V_S] [W^dag_K WH_K] VH^dag_O G5 M1 )_ab

  //D14 = Tr( P_SK G5 PH_KO M1 P_OS ) * Tr( PH_OO M2 )   //note heavy quark loop
  //    = Tr( P_SK PH^dag_OK G5 M1 P_OS ) * Tr( PH_OO M2 )
  //    = Tr( V_S W^dag_K WH_K VH^dag_O G5 M1 V_O Wdag_S ) * Tr( VH_O WH^dag_O M2 )
  //    = Tr( [W^dag_S V_S] [W^dag_K WH_K] VH^dag_O G5 M1 V_O  ) * Tr( VH_O WH^dag_O M2 )

  //D16 = Tr( P_SK G5 PH_KO M2 PH_OO M1 P_OS )
  //    = Tr( P_SK PH^dag_OK G5 M2 PH_OO M1 P_OS )
  //    = Tr( V_S W^dag_K WH_K VH^dag_O G5 M2 VH_O WH^dag_O M1 V_O W^dag_S )
  //    = Tr( [W^dag_S V_S] [W^dag_K WH_K] VH^dag_O G5 M2 VH_O WH^dag_O M1 V_O )

  //D18 = Tr( Tr_c(P_OO M2) * Tr_c( P_OS P_SK G5 PH_KO M1 ) )
  //    = Tr( Tr_c(P_OO M2) * Tr_c( P_OS P_SK PH^dag_OK G5 M1 ) )
  //    = Tr( Tr_c(V_O W^dag_O M2) * Tr_c( V_O [W^dag_S V_S] [W^dag_K WH_K] VH^dag_O G5 M1 ) )

  //D21 = Tr_c(  Tr(PH_OO M2) * Tr( P_OS P_SK G5 PH_KO M1 ) )
  //    = Tr_c(  Tr(PH_OO M2) * Tr( P_OS P_SK PH^dag_OK G5 M1 ) )
  //    = Tr_c(  Tr(VH_O WH^dag_O M2) * Tr( V_O [W^dag_S V_S] [W^dag_K WH_K] VH^dag_O G5 M1 ) )

  //D23 = Tr( Tr_c( P_OS P_SK G5 PH_KO M2 )  Tr_c( PH_OO M1 ) )
  //    = Tr( Tr_c( P_OS P_SK PH^dag_OK G5 M2 )  Tr_c( PH_OO M1 ) )
  //    = Tr( Tr_c( V_O [W^dag_S V_S] [W^dag_K WH_K] VH^dag_O G5 M2 )  Tr_c( VH_O WH^dag_O M1 ) )
  
  
  //D2  = Tr( [W^dag_K WH_K] VH^dag_O G5 M1 V_O [W^dag_S V_S] ) * Tr( V_O W^dag_O M2 )
  //D3  = Tr( [W^dag_K WH_K] VH^dag_O G5 M1 V_O W^dag_O M2 V_O [W^dag_S V_S]  )
  //D7  = Tr( ( V_O [W^dag_S V_S] [W^dag_K WH_K] VH^dag_O G5 )_ab  (M1 V_O W^dag_O M2)_ab )
  //D10 = Tr( M2 V_O W^dag_O )_ab * Tr( V_O [W^dag_S V_S] [W^dag_K WH_K] VH^dag_O G5 M1 )_ab
  //D14 = Tr( [W^dag_S V_S] [W^dag_K WH_K] VH^dag_O G5 M1 V_O  ) * Tr( VH_O WH^dag_O M2 )
  //D16 = Tr( [W^dag_S V_S] [W^dag_K WH_K] VH^dag_O G5 M2 VH_O WH^dag_O M1 V_O )
  //D18 = Tr( Tr_c(V_O W^dag_O M2) * Tr_c( V_O [W^dag_S V_S] [W^dag_K WH_K] VH^dag_O G5 M1 ) )
  //D21 = Tr_c(  Tr(VH_O WH^dag_O M2) * Tr( V_O [W^dag_S V_S] [W^dag_K WH_K] VH^dag_O G5 M1 ) )
  //D23 = Tr( Tr_c( V_O [W^dag_S V_S] [W^dag_K WH_K] VH^dag_O G5 M2 )  Tr_c( VH_O WH^dag_O M1 ) )


  //D2  = Tr( V_O [W^dag_S V_S] [W^dag_K WH_K] VH^dag_O G5 M1   ) * Tr( V_O W^dag_O M2 )
  //D3  = Tr( V_O [W^dag_S V_S] [W^dag_K WH_K] VH^dag_O G5 M1 V_O W^dag_O M2 )
  //D7  = Tr( ( V_O [W^dag_S V_S] [W^dag_K WH_K] VH^dag_O G5 M1 )_ab  ( V_O W^dag_O M2)_ab )
  //D10 = Tr( M2 V_O W^dag_O )_ab * Tr( V_O [W^dag_S V_S] [W^dag_K WH_K] VH^dag_O G5 M1 )_ab
  //D14 = Tr( V_O [W^dag_S V_S] [W^dag_K WH_K] VH^dag_O G5 M1  ) * Tr( VH_O WH^dag_O M2 )
  //D16 = Tr( V_O [W^dag_S V_S] [W^dag_K WH_K] VH^dag_O G5 M2 VH_O WH^dag_O M1  )
  //D18 = Tr( Tr_c(V_O W^dag_O M2) * Tr_c( V_O [W^dag_S V_S] [W^dag_K WH_K] VH^dag_O G5 M1 ) )
  //D21 = Tr_c(  Tr(VH_O WH^dag_O M2) * Tr( V_O [W^dag_S V_S] [W^dag_K WH_K] VH^dag_O G5 M1 ) )
  //D23 = Tr( Tr_c( V_O [W^dag_S V_S] [W^dag_K WH_K] VH^dag_O G5 M2 )  Tr_c( VH_O WH^dag_O M1 ) )

  typedef A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorWfftw> Type3MesonFieldProductType;

  inline void compute_type3_part1(SCFmat &pt1, const int top_loc, const int xop_loc, const Type3MesonFieldProductType &mf_prod){
    mult(pt1, vL, mf_prod, vH, xop_loc, top_loc, false, true);
  }

  void setup_type3_pt1_split(std::vector<vMv_split_VWWV> &part1_split, const int top_glb,
			     const std::vector<Type3MesonFieldProductType> &mf_prod, const std::vector<std::pair<int,int> > &tK_tS_idx_map BUF_ARG){
    for(int i=0;i<tK_tS_idx_map.size();i++){
      part1_split[i].setup(vL,mf_prod[i],vH, top_glb BUF_PASS);
    }
  }
  
  inline void compute_type3_part2(SCFmat &pt2_L, SCFmat &pt2_H, const int top_loc, const int xop_loc){
    mult(pt2_L, vL, wL, xop_loc, top_loc, false, true);
    mult(pt2_H, vH, wH, xop_loc, top_loc, false, true);
  }

  //Run inside threaded environment
  void type3_contract(ResultsContainerType &result, const int tK_glb, const int tdis_glb, const int thread_id, const SCFmat &part1, const SCFmat &part2_L, const SCFmat &part2_H);

  //Field implementation
  void type3_contract(ResultsContainerType &result, const int tK_glb, const SCFmatrixField &part1, const SCFmatrixField &part2_L, const SCFmatrixField &part2_H);

public:
 
  inline void idx_tpair_map(std::vector<std::pair<int,int> > &map, std::vector<std::vector<int> > &inv_map, const std::set<std::pair<int,int> > &tset){
    int ntuse = tset.size();
    map.resize(ntuse);
    inv_map.resize(Lt,std::vector<int>(Lt));

    int map_idx = 0;
    for(int t1=0;t1<Lt;t1++)
      for(int t2=0;t2<Lt;t2++)
	if(tset.count(std::pair<int,int>(t1,t2))){
	  map[map_idx] = std::pair<int,int>(t1,t2);
	  inv_map[t1][t2] = map_idx++;
	}
  }
  //CPU implementation with openmp loop over site
  void type3_omp(std::vector<ResultsContainerType> &result, std::vector<MixDiagResultsContainerType> &mix3, std::vector<SigmaMesonFieldType> &mf_S);

  //Field implementation both threaded and offloadable to GPU. WILL COMPILE ONLY FOR SIMD COMPLEX DATA
  void type3_field_SIMD(std::vector<ResultsContainerType> &result, std::vector<MixDiagResultsContainerType> &mix3, std::vector<SigmaMesonFieldType> &mf_S);

  void type3(std::vector<ResultsContainerType> &result, std::vector<MixDiagResultsContainerType> &mix3, std::vector<SigmaMesonFieldType> &mf_S){
    type3_omp(result, mix3, mf_S);
  }

private:
  //----------------------------------- type 4 -----------------------------------------------
  
  //D4 = Tr( PH_KO M1 P_OO M2 P_OK G5 ) Tr( P_SS )
  //   = Tr( PH^dag_OK G5 M1 P_OO M2 P_OK ) Tr( P_SS )
  //   = Tr( WH_K VH^dag_O G5 M1 V_O W^dag_O M2 V_O W^dag_K ) Tr( V_S W^dag_S )
  //   = Tr( V_O [W^dag_K WH_K] VH^dag_O G5 M1 V_O W^dag_O M2 ) Tr( [V_S W^dag_S] )

  //D5 = Tr( PH_KO M1 P_OK G5 ) Tr( P_OO M2 ) Tr( P_SS )
  //   = Tr( PH^dag_OK G5 M1 P_OK ) Tr( P_OO M2 ) Tr( P_SS )
  //   = Tr( WH_K VH^dag_O G5 M1 V_O W^dag_K ) Tr( V_O W^dag_O M2 ) Tr( V_S W^dag_S )
  //   = Tr( V_O [W^dag_K WH_K] VH^dag_O G5 M1 ) Tr( V_O W^dag_O M2 ) Tr( [V_S W^dag_S] )

  //D9 = Tr( (M1 P_OO)_ab (M2 P_OK G5 PH_KO )_ab ) Tr( P_SS )
  //   = Tr( (M1 P_OO)_ab (M2 P_OK PH^dag_OK G5 )_ab ) Tr( P_SS )
  //   = Tr( (M1 V_O W^dag_O)_ab (M2 V_O [W^dag_K WH_K] VH^dag_O G5 )_ab ) Tr( [V_S W^dag_S] )

  //D12 = Tr( P_OK G5 PH_KO M1 )_ab Tr( P_OO M2 )_ab Tr( P_SS )
  //    = Tr( P_OK PH^dag_OK G5 M1 )_ab Tr( P_OO M2 )_ab Tr( P_SS )
  //    = Tr( V_O [W^dag_K WH_K] VH^dag_O G5 M1 )_ab Tr( V_O W^dag_O M2 )_ab Tr( [V_S W^dag_S] )

  //D13 = Tr( G5 PH_KO M1 P_OK ) Tr( M2 PH_OO ) Tr( P_SS )
  //    = Tr( PH^dag_OK G5 M1 P_OK ) Tr( M2 PH_OO ) Tr( P_SS )
  //    = Tr( WH_K VH^dag_O G5 M1 V_O W^dag_K ) Tr( M2 VH_O WH^dag_O ) Tr( V_S W^dag_S )
  //    = Tr( V_O [W^dag_K WH_K] VH^dag_O G5 M1 ) Tr( M2 VH_O WH^dag_O ) Tr( [V_S W^dag_S] )

  //D15 = Tr( G5 PH_KO M2 PH_OO M1 P_OK ) Tr( P_SS )
  //    = Tr( PH^dag_OK G5 M2 PH_OO M1 P_OK ) Tr( P_SS )
  //    = Tr( WH_K VH^dag_O G5 M2 VH_O WH^dag_O M1 V_O W^dag_K ) Tr( V_S W^dag_S )
  //    = Tr( V_O [W^dag_K WH_K] VH^dag_O G5 M2 VH_O WH^dag_O M1  ) Tr( [V_S W^dag_S] )

  //D17 = Tr( Tr_c(P_OK G5 PH_KO M1 ) Tr_c( P_OO M2) ) Tr( P_SS )
  //    = Tr( Tr_c(P_OK PH^dag_OK G5 M1 ) Tr_c( P_OO M2) ) Tr( P_SS )
  //    = Tr( Tr_c(V_O [W^dag_K WH_K] VH^dag_O G5 M1 ) Tr_c( V_O W^dag_O M2) ) Tr( [V_S W^dag_S] )

  //D20 = Tr_c(  Tr( P_OK G5 PH_KO M1 ) Tr( PH_OO M2 ) ) Tr( P_SS ) 
  //    = Tr_c(  Tr( P_OK PH^dag_OK G5 M1 ) Tr( PH_OO M2 ) ) Tr( P_SS ) 
  //    = Tr_c(  Tr( V_O [W^dag_K WH_K] VH^dag_O G5 M1 ) Tr( VH_O WH^dag_O M2 ) ) Tr( [V_S W^dag_S] ) 

  //D22 = Tr( Tr_c( PH_OO M1 ) Tr_c( P_OK G5 PH_KO M2 ) ) Tr( P_SS ) 
  //    = Tr( Tr_c( PH_OO M1 ) Tr_c( P_OK PH^dag_OK G5 M2 ) ) Tr( P_SS ) 
  //    = Tr( Tr_c( VH_O WH^dag_O M1 ) Tr_c( V_O [W^dag_K WH_K] VH^dag_O G5 M2 ) ) Tr( [V_S W^dag_S] ) 

  //Note we don't include the sigma self-contraction in the calculation here; it should be added offline

  //pt1 = V_O [W^dag_K WH_K] VH^dag_O
  //pt2_L = V_O W^dag_O
  //pt2_H = VH_O WH^dag_O

  inline void compute_type4_part1(SCFmat &pt1, const int tK_glb, const int top_loc, const int xop_loc){
    mult(pt1, vL, mf_ls_WW[tK_glb], vH, xop_loc, top_loc, false, true);
  }

  inline void setup_type4_pt1_split(std::vector<vMv_split_VWWV> &part1_split, const int top_glb, const std::vector<int> &tK_subset_map){
    for(int i=0;i<tK_subset_map.size();i++){
      int tK_glb = tK_subset_map[i];
      part1_split[i].setup(vL,mf_ls_WW[tK_glb],vH, top_glb);
    }
  }

  inline void compute_type4_part2(SCFmat &pt2_L, SCFmat &pt2_H, const int top_loc, const int xop_loc){
    mult(pt2_L, vL, wL, xop_loc, top_loc, false, true);
    mult(pt2_H, vH, wH, xop_loc, top_loc, false, true);
  }

  void type4_contract(ResultsContainerType &result, const int tK_glb, const int tdis_glb, const int thread_id, const SCFmat &part1, const SCFmat &part2_L, const SCFmat &part2_H);


public:

  void type4_omp(ResultsContainerType &result, MixDiagResultsContainerType &mix4);

  void type4(ResultsContainerType &result, MixDiagResultsContainerType &mix4){
    type4_omp(result, mix4);
  }


  ComputeKtoSigma(const A2AvectorV<mf_Policies> & vL, const A2AvectorW<mf_Policies> & wL, const A2AvectorV<mf_Policies> & vH, const A2AvectorW<mf_Policies> & wH, 
		  const std::vector<KaonMesonFieldType> &mf_ls_WW, const std::vector<int> &tsep_k_sigma): vL(vL), wL(wL), vH(vH), wH(wH), mf_ls_WW(mf_ls_WW), tsep_k_sigma(tsep_k_sigma){
    Lt = GJP.Tnodes()*GJP.TnodeSites();
    ntsep_k_sigma = tsep_k_sigma.size();
    nthread = omp_get_max_threads();
    size_3d = vL.getMode(0).nodeSites(0)*vL.getMode(0).nodeSites(1)*vL.getMode(0).nodeSites(2); //don't use GJP; this size_3d respects the SIMD logical node packing

    assert(tsep_k_sigma.size() > 0);
    tsep_k_sigma_lrg = tsep_k_sigma[0];
    for(int i=1;i<tsep_k_sigma.size();i++)
      if(tsep_k_sigma[i] > tsep_k_sigma_lrg) 
	tsep_k_sigma_lrg = tsep_k_sigma[i];
  }

 


};

#include "implementation/compute_ktosigma.tcc"
#include "implementation/compute_ktosigma_field.tcc"

CPS_END_NAMESPACE




#endif
