#ifndef _COMPUTE_KTOPIPI_H
#define _COMPUTE_KTOPIPI_H

#include<alg/a2a/fmatrix.h>
#include<alg/a2a/required_momenta.h>
#include<alg/a2a/compute_ktopipi_base.h>

#include<memory>
#include <algorithm>

CPS_START_NAMESPACE

//NOTE:
//Daiqian's type2 and type3 files are exactly a factor of 2 larger than mine. This is because he forgot to divide by a factor of 2 from permuting the 2 pions. In the 2015 analysis we corrected this factor at analysis
//time but I have corrected it in the code (if you disable the #define below!)

#define DAIQIAN_COMPATIBILITY_MODE  //do not divide by 2, to match Daiqian's files
#define DAIQIAN_EVIL_RANDOM_SITE_OFFSET //use Daiqian's machine-size dependent random spatial sampling (cf below)


template<typename mf_Float>
class ComputeKtoPiPiGparity: public ComputeKtoPiPiGparityBase{  
private:
  inline static int modLt(int i, const int &Lt){
    while(i<0) i += Lt;
    return i % Lt;
  }

  //Determine what node timeslices are actually needed. Returns true if at least one on-node top is needed
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
  typedef std::complex<mf_Float> mf_Complex;
  
  //Compute type 1,2,3,4 diagram for fixed tsep(pi->K) and tsep(pi->pi). 

  //By convention pi1 is the pion closest to the kaon

  //User provides the momentum of pi1, p_pi_1

  //tstep is the time sampling frequency for the kaon/pion

  //The kaon meson field is expected to be constructed in the W*W form

  //ls_WW meson fields
  template< typename Allocator >
  static void generatelsWWmesonfields(std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw>,Allocator> &mf_ls_ww, const A2AvectorW<mf_Float> &W, const A2AvectorW<mf_Float> &W_s, const int kaon_rad, Lattice &lat){
    if(!UniqueID()) printf("Computing ls WW meson fields for K->pipi\n");
    double time = -dclock();

    const int Lt = GJP.Tnodes()*GJP.TnodeSites();
    mf_ls_ww.resize(Lt);

    //0 momentum kaon constructed from (-1,-1,-1) + (1,1,1)  in units of pi/2L
    int p[3];
    for(int i=0;i<3;i++) p[i] = (GJP.Bc(i) == BND_CND_GPARITY ? 1 : 0);
    
    A2AvectorWfftw<mf_Float> fftw_Wl_p(W.getArgs());
    fftw_Wl_p.gaugeFixTwistFFT(W, p,lat); //will be daggered, swapping momentum

    A2AvectorWfftw<mf_Float> fftw_Ws_p(W_s.getArgs());
    fftw_Ws_p.gaugeFixTwistFFT(W_s,p,lat); 

    A2AflavorProjectedExpSource fpexp(kaon_rad, p);
    SCFspinflavorInnerProduct<mf_Complex,A2AflavorProjectedExpSource> mf_struct(sigma0,0,fpexp); // (1)_flav * (1)_spin 

    A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw>::compute(mf_ls_ww, fftw_Wl_p, mf_struct, fftw_Ws_p);

    //for(int t=0;t<Lt;t++)
    //mf_ls_ww[t].compute(fftw_Wl_p, mf_struct, fftw_Ws_p,t);
    time += dclock();
    print_time("ComputeKtoPiPiGparity","ls WW meson fields",time);
  }

  //--------------------------------------------------------------------------
  //TYPE 1
  typedef CPSfield<int,1,FourDpolicy,OneFlavorPolicy> OneFlavorIntegerField;

private:
  //Run inside threaded environment
  static void type1_contract(KtoPiPiGparityResultsContainer &result, const int t_K, const int t_dis, const int thread_id, const SpinColorFlavorMatrix part1[2], const SpinColorFlavorMatrix part2[2]);

  static void generateRandomOffsets(std::vector<OneFlavorIntegerField*> &random_fields, const std::vector<int> &tsep_k_pi, const int tstep, const int xyzStep);

  static void type1_compute_mfproducts(std::vector<std::vector< A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > > &con_pi1_K,
				       std::vector<std::vector< A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > > &con_pi2_K,
				       const std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorVfftw> > &mf_pi1,
				       const std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorVfftw> > &mf_pi2,
				       const std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > &mf_kaon, const MesonFieldMomentumContainer<mf_Float> &mf_pions,
				       const std::vector<int> &tsep_k_pi, const int tsep_pion, const int Lt, const int ntsep_k_pi);

  static void type1_mult_vMv_setup(mult_vMv_split<mf_Float,A2AvectorV,A2AvectorWfftw,A2AvectorVfftw,A2AvectorW> &mult_vMv_split_part1_pi1,
				   mult_vMv_split<mf_Float,A2AvectorV,A2AvectorWfftw,A2AvectorVfftw,A2AvectorW> &mult_vMv_split_part1_pi2,
				   std::vector<mult_vMv_split<mf_Float,A2AvectorV,A2AvectorWfftw,A2AvectorWfftw,A2AvectorV> > &mult_vMv_split_part2_pi1,
				   std::vector<mult_vMv_split<mf_Float,A2AvectorV,A2AvectorWfftw,A2AvectorWfftw,A2AvectorV> > &mult_vMv_split_part2_pi2,
				   const std::vector<std::vector< A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > > &con_pi1_K,
				   const std::vector<std::vector< A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > > &con_pi2_K,
				   const std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorVfftw> > &mf_pi1,
				   const std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorVfftw> > &mf_pi2,							   
				   const A2AvectorV<mf_Float> & vL, const A2AvectorV<mf_Float> & vH, 
				   const A2AvectorW<mf_Float> & wL,
				   const ModeContractionIndices<StandardIndexDilution,TimePackedIndexDilution> &i_ind_vw,
				   const ModeContractionIndices<StandardIndexDilution,FullyPackedIndexDilution> &j_ind_vw,
				   const ModeContractionIndices<TimePackedIndexDilution,StandardIndexDilution> &j_ind_wv,
				   const int top_loc, const int t_pi1, const int t_pi2, const int Lt, const std::vector<int> &tsep_k_pi, const int ntsep_k_pi, const int t_K_all[], const std::vector<bool> &node_top_used);

  static void type1_precompute_part1_part2(std::vector<SpinColorFlavorMatrix> &mult_vMv_contracted_part1_pi1,
					   std::vector<SpinColorFlavorMatrix> &mult_vMv_contracted_part1_pi2,
					   std::vector<std::vector<SpinColorFlavorMatrix> > &mult_vMv_contracted_part2_pi1,
					   std::vector<std::vector<SpinColorFlavorMatrix> > &mult_vMv_contracted_part2_pi2,
					   mult_vMv_split<mf_Float,A2AvectorV,A2AvectorWfftw,A2AvectorVfftw,A2AvectorW> &mult_vMv_split_part1_pi1,
					   mult_vMv_split<mf_Float,A2AvectorV,A2AvectorWfftw,A2AvectorVfftw,A2AvectorW> &mult_vMv_split_part1_pi2,
					   std::vector<mult_vMv_split<mf_Float,A2AvectorV,A2AvectorWfftw,A2AvectorWfftw,A2AvectorV> > &mult_vMv_split_part2_pi1,
					   std::vector<mult_vMv_split<mf_Float,A2AvectorV,A2AvectorWfftw,A2AvectorWfftw,A2AvectorV> > &mult_vMv_split_part2_pi2,
					   const int top_loc, const int Lt, const std::vector<int> &tsep_k_pi, const int ntsep_k_pi, const int t_K_all[], const std::vector<bool> &node_top_used);

public:
  //xyzStep is the 3d spatial sampling frequency. If set to anything other than one it will compute the diagram
  //for a random site within the block (site, site+xyzStep) in canonical ordering. Daiqian's original implementation is machine-size dependent, but for repro I had to add an option to do it his way 

  //This version overlaps computation for multiple K->pi separations. Result should be an array of KtoPiPiGparityResultsContainer the same size as the vector 'tsep_k_pi'
  static void type1(KtoPiPiGparityResultsContainer result[],
		    const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const int xyzStep, const ThreeMomentum &p_pi_1, 
		    const std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > &mf_kaon, MesonFieldMomentumContainer<mf_Float> &mf_pions,
		    const A2AvectorV<mf_Float> & vL, const A2AvectorV<mf_Float> & vH, 
		    const A2AvectorW<mf_Float> & wL, const A2AvectorW<mf_Float> & wH);

  static void type1(KtoPiPiGparityResultsContainer &result,
		    const int tsep_k_pi, const int tsep_pion, const int tstep, const int xyzStep, const ThreeMomentum &p_pi_1, 
		    const std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > &mf_kaon, MesonFieldMomentumContainer<mf_Float> &mf_pions,
		    const A2AvectorV<mf_Float> & vL, const A2AvectorV<mf_Float> & vH, 
		    const A2AvectorW<mf_Float> & wL, const A2AvectorW<mf_Float> & wH){
    std::vector<int> tt(1,tsep_k_pi);
    return type1(&result,tt, tsep_pion,tstep,xyzStep,p_pi_1,
		 mf_kaon, mf_pions,
		 vL,vH,
		 wL,wH);
  }
  static void type1(std::vector<KtoPiPiGparityResultsContainer> &result,
		    const std::vector<int> tsep_k_pi, const int tsep_pion, const int tstep, const int xyzStep, const ThreeMomentum &p_pi_1, 
		    const std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > &mf_kaon, MesonFieldMomentumContainer<mf_Float> &mf_pions,
		    const A2AvectorV<mf_Float> & vL, const A2AvectorV<mf_Float> & vH, 
		    const A2AvectorW<mf_Float> & wL, const A2AvectorW<mf_Float> & wH){
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
  static void type2_contract(KtoPiPiGparityResultsContainer &result, const int t_K, const int t_dis, const int thread_id, const SpinColorFlavorMatrix &part1, const SpinColorFlavorMatrix part2[2]);
 
  static void type2_compute_mfproducts(std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorVfftw> > &con_pi1_pi2,
				       std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorVfftw> > &con_pi2_pi1,							     
				       const int tsep_pion, const int tstep, const std::vector<ThreeMomentum> &p_pi_1_all,
				       MesonFieldMomentumContainer<mf_Float> &mf_pions,
				       const int Lt, const int tpi_sampled);

  static void type2_mult_vMv_setup(std::vector<mult_vMv_split<mf_Float,A2AvectorV,A2AvectorWfftw,A2AvectorWfftw,A2AvectorV> > &mult_vMv_split_part1,
				   std::vector<mult_vMv_split<mf_Float,A2AvectorV,A2AvectorWfftw,A2AvectorVfftw,A2AvectorW> > &mult_vMv_split_part2_pi1_pi2,
				   std::vector<mult_vMv_split<mf_Float,A2AvectorV,A2AvectorWfftw,A2AvectorVfftw,A2AvectorW> > &mult_vMv_split_part2_pi2_pi1,
				   const std::vector< A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorVfftw> > &con_pi1_pi2,
				   const std::vector< A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorVfftw> > &con_pi2_pi1,
				   const A2AvectorV<mf_Float> & vL, const A2AvectorV<mf_Float> & vH, const A2AvectorW<mf_Float> & wL,
				   const std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > &mf_kaon,
				   const std::vector<int> &t_K_all, const int top_loc, const int tstep, const int Lt,const int tpi_sampled,
				   const std::vector< std::vector<bool> > &node_top_used, const std::vector< std::vector<bool> > &node_top_used_kaon);

  static void type2_precompute_part1_part2(std::vector<std::vector<SpinColorFlavorMatrix> > &mult_vMv_contracted_part1,
					   std::vector<std::vector<SpinColorFlavorMatrix> > &mult_vMv_contracted_part2_pi1_pi2,
					   std::vector<std::vector<SpinColorFlavorMatrix> > &mult_vMv_contracted_part2_pi2_pi1,
					   std::vector<mult_vMv_split<mf_Float,A2AvectorV,A2AvectorWfftw,A2AvectorWfftw,A2AvectorV> > &mult_vMv_split_part1,
					   std::vector<mult_vMv_split<mf_Float,A2AvectorV,A2AvectorWfftw,A2AvectorVfftw,A2AvectorW> > &mult_vMv_split_part2_pi1_pi2,
					   std::vector<mult_vMv_split<mf_Float,A2AvectorV,A2AvectorWfftw,A2AvectorVfftw,A2AvectorW> > &mult_vMv_split_part2_pi2_pi1,
					   const std::vector<int> &t_K_all, const int top_loc, const int tstep, const int Lt,const int tpi_sampled,
					   const std::vector< std::vector<bool> > &node_top_used, const std::vector< std::vector<bool> > &node_top_used_kaon);

public: 
  //This version averages over multiple pion momentum configurations. Use to project onto A1 representation at run-time. Saves a lot of time!
  //This version also overlaps computation for multiple K->pi separations. Result should be an array of KtoPiPiGparityResultsContainer the same size as the vector 'tsep_k_pi'
  static void type2(KtoPiPiGparityResultsContainer result[],
		    const std::vector<int> &tsep_k_pi, const int &tsep_pion, const int &tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
		    const std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > &mf_kaon, MesonFieldMomentumContainer<mf_Float> &mf_pions,
		    const A2AvectorV<mf_Float> & vL, const A2AvectorV<mf_Float> & vH, 
		    const A2AvectorW<mf_Float> & wL, const A2AvectorW<mf_Float> & wH);

  static void type2(KtoPiPiGparityResultsContainer &result,
		    const int &tsep_k_pi, const int &tsep_pion, const int &tstep, const ThreeMomentum &p_pi_1, 
		    const std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > &mf_kaon, MesonFieldMomentumContainer<mf_Float> &mf_pions,
		    const A2AvectorV<mf_Float> & vL, const A2AvectorV<mf_Float> & vH, 
		    const A2AvectorW<mf_Float> & wL, const A2AvectorW<mf_Float> & wH){
    std::vector<int> tkp(1,tsep_k_pi);
    std::vector<ThreeMomentum> p(1, p_pi_1); 
    return type2(&result,tkp,tsep_pion,tstep,p,
		 mf_kaon, mf_pions,
		 vL, vH,
		 wL, wH);
  }
  template<typename MomComputePolicy>
  static void type2(KtoPiPiGparityResultsContainer &result,
		    const int &tsep_k_pi, const int &tsep_pion, const int &tstep, const RequiredMomentum<MomComputePolicy> &p_pi_1_all, 
		    const std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > &mf_kaon, MesonFieldMomentumContainer<mf_Float> &mf_pions,
		    const A2AvectorV<mf_Float> & vL, const A2AvectorV<mf_Float> & vH, 
		    const A2AvectorW<mf_Float> & wL, const A2AvectorW<mf_Float> & wH){
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
  static void type2(std::vector<KtoPiPiGparityResultsContainer> &result,
		    const std::vector<int> &tsep_k_pi, const int &tsep_pion, const int &tstep, const RequiredMomentum<MomComputePolicy> &p_pi_1_all, 
		    const std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > &mf_kaon, MesonFieldMomentumContainer<mf_Float> &mf_pions,
		    const A2AvectorV<mf_Float> & vL, const A2AvectorV<mf_Float> & vH, 
		    const A2AvectorW<mf_Float> & wL, const A2AvectorW<mf_Float> & wH){
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
  static void type3_contract(KtoPiPiGparityResultsContainer &result, const int t_K, const int t_dis, const int thread_id, 
			     const SpinColorFlavorMatrix part1[2], const SpinColorFlavorMatrix &part2_L, const SpinColorFlavorMatrix &part2_H);

  static void type3_compute_mfproducts(std::vector<std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > > &con_pi1_pi2_k,
				       std::vector<std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > > &con_pi2_pi1_k,							     
				       const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
				       const std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > &mf_kaon, MesonFieldMomentumContainer<mf_Float> &mf_pions,
				       const int Lt, const int tpi_sampled, const int ntsep_k_pi);
  static void type3_mult_vMv_setup(mult_vMv_split<mf_Float,A2AvectorV,A2AvectorWfftw,A2AvectorWfftw,A2AvectorV> &mult_vMv_split_part1_pi1_pi2,
				   mult_vMv_split<mf_Float,A2AvectorV,A2AvectorWfftw,A2AvectorWfftw,A2AvectorV> &mult_vMv_split_part1_pi2_pi1,
				   const A2AvectorV<mf_Float> & vL, const A2AvectorV<mf_Float> & vH,
				   const std::vector<std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > > &con_pi1_pi2_k,
				   const std::vector<std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > > &con_pi2_pi1_k,
				   const int top_loc, const int t_pi1_idx, const int tkp);

  static void type3_precompute_part1(std::vector<SpinColorFlavorMatrix> &mult_vMv_contracted_part1_pi1_pi2,
				     std::vector<SpinColorFlavorMatrix> &mult_vMv_contracted_part1_pi2_pi1,
				     mult_vMv_split<mf_Float,A2AvectorV,A2AvectorWfftw,A2AvectorWfftw,A2AvectorV> &mult_vMv_split_part1_pi1_pi2,
				     mult_vMv_split<mf_Float,A2AvectorV,A2AvectorWfftw,A2AvectorWfftw,A2AvectorV> &mult_vMv_split_part1_pi2_pi1,
				     const int top_loc, const int t_pi1_idx, const int tkp);
    
public:
  //This version averages over multiple pion momentum configurations. Use to project onto A1 representation at run-time. Saves a lot of time!
  //This version also overlaps computation for multiple K->pi separations. Result should be an array of KtoPiPiGparityResultsContainer the same size as the vector 'tsep_k_pi'
  static void type3(KtoPiPiGparityResultsContainer result[], KtoPiPiGparityMixDiagResultsContainer mix3[],
		    const std::vector<int> &tsep_k_pi, const int &tsep_pion, const int &tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
		    const std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > &mf_kaon, MesonFieldMomentumContainer<mf_Float> &mf_pions,
		    const A2AvectorV<mf_Float> & vL, const A2AvectorV<mf_Float> & vH, 
		    const A2AvectorW<mf_Float> & wL, const A2AvectorW<mf_Float> & wH);

  static void type3(KtoPiPiGparityResultsContainer &result, KtoPiPiGparityMixDiagResultsContainer &mix3,
		    const int &tsep_k_pi, const int &tsep_pion, const int &tstep, const ThreeMomentum &p_pi_1, 
		    const std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > &mf_kaon, MesonFieldMomentumContainer<mf_Float> &mf_pions,
		    const A2AvectorV<mf_Float> & vL, const A2AvectorV<mf_Float> & vH, 
		    const A2AvectorW<mf_Float> & wL, const A2AvectorW<mf_Float> & wH){
    std::vector<int> tkp(1,tsep_k_pi);
    std::vector<ThreeMomentum> p(1, p_pi_1); 
    return type3(&result,&mix3,tkp,tsep_pion,tstep,p,
		 mf_kaon, mf_pions,
		 vL, vH,
		 wL, wH);
  }
  template<typename MomComputePolicy>
  static void type3(KtoPiPiGparityResultsContainer &result, KtoPiPiGparityMixDiagResultsContainer &mix3,
		    const int &tsep_k_pi, const int &tsep_pion, const int &tstep, const RequiredMomentum<MomComputePolicy> &p_pi_1_all, 
		    const std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > &mf_kaon, MesonFieldMomentumContainer<mf_Float> &mf_pions,
		    const A2AvectorV<mf_Float> & vL, const A2AvectorV<mf_Float> & vH, 
		    const A2AvectorW<mf_Float> & wL, const A2AvectorW<mf_Float> & wH){
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
  static void type3(std::vector<KtoPiPiGparityResultsContainer> &result, std::vector<KtoPiPiGparityMixDiagResultsContainer> &mix3,
		    const std::vector<int> &tsep_k_pi, const int &tsep_pion, const int &tstep, const RequiredMomentum<MomComputePolicy> &p_pi_1_all, 
		    const std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > &mf_kaon, MesonFieldMomentumContainer<mf_Float> &mf_pions,
		    const A2AvectorV<mf_Float> & vL, const A2AvectorV<mf_Float> & vH, 
		    const A2AvectorW<mf_Float> & wL, const A2AvectorW<mf_Float> & wH){
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
private:

  static void type4_contract(KtoPiPiGparityResultsContainer &result, const int t_K, const int t_dis, const int thread_id, 
			     const SpinColorFlavorMatrix &part1, const SpinColorFlavorMatrix &part2_L, const SpinColorFlavorMatrix &part2_H);
  


  static void type4_mult_vMv_setup(std::vector<mult_vMv_split<mf_Float,A2AvectorV,A2AvectorWfftw,A2AvectorWfftw,A2AvectorV> > &mult_vMv_split_part1,
				   const std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > &mf_kaon,
				   const A2AvectorV<mf_Float> & vL, const A2AvectorV<mf_Float> & vH,
				   const int top_loc, const int tstep, const int Lt);

  static void type4_precompute_part1(std::vector<std::vector<SpinColorFlavorMatrix> > &mult_vMv_contracted_part1,
				     std::vector<mult_vMv_split<mf_Float,A2AvectorV,A2AvectorWfftw,A2AvectorWfftw,A2AvectorV> > &mult_vMv_split_part1,
				     const int top_loc, const int tstep, const int Lt);
public:

  static void type4(KtoPiPiGparityResultsContainer &result, KtoPiPiGparityMixDiagResultsContainer &mix4,
		    const int &tstep,
		    const std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorWfftw> > &mf_kaon,
		    const A2AvectorV<mf_Float> & vL, const A2AvectorV<mf_Float> & vH, 
		    const A2AvectorW<mf_Float> & wL, const A2AvectorW<mf_Float> & wH);




};

CPS_END_NAMESPACE


#include<alg/a2a/compute_ktopipi_type1.h>
#include<alg/a2a/compute_ktopipi_type2.h>
#include<alg/a2a/compute_ktopipi_type3.h>
#include<alg/a2a/compute_ktopipi_type4.h>

#undef DAIQIAN_COMPATIBILITY_MODE
#undef DAIQIAN_EVIL_RANDOM_SITE_OFFSET



#endif


//       //Daiqian thinned the spatial sampling by computing only on a random site within a linearized site index block
//       NullObject n;
//       CPSfield<int,1,FourDpolicy,OneFlavorPolicy> randoff(n);
//       randoff.zero();
      
//       LRG.SetInterval(1.0,0.0);
//       for(int t=0;t<GJP.TnodeSites();t++){
// 	if(!node_top_used[t]) continue;

// 	// int top_glb = t  + GJP.TnodeCoor()*GJP.TnodeSites();
// 	// int t_dis = modLt(top_glb - t_K_all[0], Lt); //FIXME
// 	// if(t_dis >= tsep_k_pi[0] || t_dis == 0) continue; //FIXME ! //don't bother generating random numbers for timeslices we're not going to use

// 	if(!UniqueID()) printf("Generating random field for t_pi_1 = %d and t_lcl = %d, a random number = %12e\n",t_pi1,t,LRG.Urand(FOUR_D));

// 	for(int i = 0; i < size_3d / xyzStep; i++){
// 	  int x = i*xyzStep + GJP.VolNodeSites()/GJP.TnodeSites()*t;

// #ifndef DAIQIAN_EVIL_RANDOM_SITE_OFFSET
// 	  LRG.AssignGenerator(x);
// #else
// 	  //Generate in same order as Daiqian. He doesn't set the hypercube RNG so it uses whichever one was used last! I know this sucks.
// #endif

// 	  *(randoff.site_ptr(x)) = ((int)(LRG.Urand(FOUR_D) * xyzStep)) % xyzStep;
// 	}
//       }
