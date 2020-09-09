#ifndef _COMPUTE_KTOPIPI_TYPE1_FIELD_H
#define _COMPUTE_KTOPIPI_TYPE1_FIELD_H


#define TIMER_ELEMS \
  ELEM(piK_mfproducts) \
  ELEM(part1) \
  ELEM(part2) \
  ELEM(contraction_time) \
  ELEM(finish_up)\
  ELEM(total)
#define TIMER Type1FieldTimings
#include "static_timer_impl.tcc"


//Expect part1 and part2 to be length=2 vectors, the first with pi1 connected directly to the 4-quark op, and the second with pi2
template<typename mf_Policies>
void ComputeKtoPiPiGparity<mf_Policies>::type1_contract(ResultsContainerType &result, const int t_K, const std::vector<SCFmatrixField> &part1, const std::vector<SCFmatrixField> &part2){
#ifndef MEMTEST_MODE
  static const int n_contract = 6; //six type1 diagrams
  static const int con_off = 1; //index of first contraction in set
  for(int pt1_pion=0; pt1_pion<2; pt1_pion++){ //which pion is associated with part 1?
    int pt2_pion = (pt1_pion + 1) % 2;

    for(int mu=0;mu<4;mu++){ //sum over mu here
      for(int gcombidx=0;gcombidx<8;gcombidx++){
	auto G1_pt1 = multGammaLeft(part1[pt1_pion],1,gcombidx,mu);
	auto G2_pt2 = multGammaLeft(part2[pt2_pion],2,gcombidx,mu);

	auto tr_sf_G1_pt1 = SpinFlavorTrace(G1_pt1);

	auto tr_sf_G2_pt2 = SpinFlavorTrace(G2_pt2);

	auto ctrans_G2_pt2 = TransposeColor(G2_pt2);

	add(1, result, t_K, gcombidx, con_off, 
	  Trace(G1_pt1) * Trace(G2_pt2)
	    );
	add(2, result, t_K, gcombidx, con_off, 
	    Trace( tr_sf_G1_pt1, Transpose(tr_sf_G2_pt2) )
	    );
	add(3, result, t_K, gcombidx, con_off, 
	    Trace( tr_sf_G1_pt1 , tr_sf_G2_pt2 )
	    );
	add(4, result, t_K, gcombidx, con_off, 
	    Trace( G1_pt1, G2_pt2 )
	    );
	add(5, result, t_K, gcombidx, con_off, 
	    Trace( G1_pt1, ctrans_G2_pt2 ) 
	    );
	add(6, result, t_K, gcombidx, con_off, 
	    Trace( ColorTrace(G1_pt1) , ColorTrace(G2_pt2) )
	    );
      }
    }
  }
#endif
}



template<typename mf_Policies>
void ComputeKtoPiPiGparity<mf_Policies>::type1_field_SIMD(ResultsContainerType result[],
						     const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const ThreeMomentum &p_pi_1, 
						     const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
						     const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
						     const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH){
  Type1FieldTimings::timer().reset();
  Type1FieldTimings::timer().total -= dclock();

  auto field_params = vL.getMode(0).getDimPolParams();  
  
  //Precompute mode mappings
  ModeContractionIndices<StandardIndexDilution,TimePackedIndexDilution> i_ind_vw(vL);
  ModeContractionIndices<StandardIndexDilution,FullyPackedIndexDilution> j_ind_vw(wL);
  ModeContractionIndices<TimePackedIndexDilution,StandardIndexDilution> j_ind_wv(vH);

  const int Lt = GJP.Tnodes()*GJP.TnodeSites();
  const int ntsep_k_pi = tsep_k_pi.size();
    
  const ThreeMomentum p_pi_2 = -p_pi_1;

  static const int n_contract = 6; //six type1 diagrams
  static const int con_off = 1; //index of first contraction in set

  for(int i=0;i<ntsep_k_pi;i++)
    result[i].resize(n_contract); //Resize also zeroes 'result'
    
  std::vector<mf_WV > &mf_pi1 = mf_pions.get(p_pi_1); //*mf_pi1_ptr;
  std::vector<mf_WV > &mf_pi2 = mf_pions.get(p_pi_2); //*mf_pi2_ptr;

  //Compute which pion timeslices are involved in the calculation on this node
  std::vector<bool> pi1_tslice_mask(Lt,false);
  std::vector<bool> pi2_tslice_mask(Lt,false);
  for(int t_pi1_lin = 1; t_pi1_lin <= Lt; t_pi1_lin += tstep){
    int t_pi1 = modLt(t_pi1_lin,Lt);
    int t_pi2 = modLt(t_pi1 + tsep_pion, Lt);
    pi1_tslice_mask[t_pi1] = true;
    pi2_tslice_mask[t_pi2] = true;
  }
#ifdef NODE_DISTRIBUTE_MESONFIELDS
  if(!UniqueID()) printf("Memory prior to fetching meson fields type1 K->pipi:\n");    
  printMem();
  nodeGetMany(2,
	      &mf_pi1,&pi1_tslice_mask,
	      &mf_pi2,&pi2_tslice_mask);
  if(!UniqueID()) printf("Memory after fetching meson fields type1 K->pipi:\n");
  printMem();
#endif

  //The two meson field are independent of x_op so we can pregenerate them for each y_4, top    
  //Form contraction  con_pi_K(y_4, t_K) =   [[ wL_i^dag(y_4) S_2 vL_j(y_4) ]] [[ wL_j^dag(t_K) wH_k(t_K) ) ]]
  //y_4 = t_K + tsep_k_pi
  //Compute contraction for each K->pi separation. Try to reuse as there will be some overlap.
  Type1FieldTimings::timer().piK_mfproducts -= dclock();
  std::vector<std::vector< mf_WW > > con_pi1_K(Lt); //[tpi][tsep_k_pi]
  std::vector<std::vector< mf_WW > > con_pi2_K(Lt);
    
  type1_compute_mfproducts(con_pi1_K,con_pi2_K,mf_pi1,mf_pi2,mf_kaon,mf_pions,tsep_k_pi,tsep_pion,Lt,ntsep_k_pi,pi1_tslice_mask,pi2_tslice_mask);
  Type1FieldTimings::timer().piK_mfproducts += dclock();

  if(!UniqueID()) printf("Memory after computing mfproducts type1 K->pipi:\n");
  printMem();

  //Compute data for which operator lies on this node
  //Determine which t_K and t_pi1 combinations this node contributes to given that t_op > t_K && t_op < t_pi1
  
  std::map<int, std::vector<int> > map_used_tpi1_lin_to_tsep_k_pi;  //tpi1_lin -> vector of tsep_k_pi index

  std::stringstream ss;
  ss << "Node " << UniqueID() << " doing (t_K, t_pi1) = {";

  //for(int t_pi1 = 0; t_pi1 < Lt; t_pi1 += tstep){ //my sensible ordering
  for(int t_pi1_lin = 1; t_pi1_lin <= Lt; t_pi1_lin += tstep){ //Daiqian's weird ordering
    int t_pi1 = modLt(t_pi1_lin,Lt);
    
    //Using the pion timeslices, get tK for each separation
    for(int tkpi_idx=0;tkpi_idx<ntsep_k_pi;tkpi_idx++){
      int t_K = modLt(t_pi1 - tsep_k_pi[tkpi_idx], Lt);
      
      for(int top=GJP.TnodeCoor()*GJP.TnodeSites(); top < (GJP.TnodeCoor()+1)*GJP.TnodeSites(); top++){
	int t_dis = modLt(top - t_K, Lt);
	if(t_dis > 0 && t_dis < tsep_k_pi[tkpi_idx]){
	  map_used_tpi1_lin_to_tsep_k_pi[t_pi1_lin].push_back(tkpi_idx);
	  ss << " (" << t_K << "," << t_pi1 << ")";
	  break; //only need one timeslice in window
	}
      }
    }
  }
  ss << "}\n";
  printf("%s",ss.str().c_str());
	  
  for(auto t_it = map_used_tpi1_lin_to_tsep_k_pi.begin(); t_it != map_used_tpi1_lin_to_tsep_k_pi.end(); ++t_it){
    int t_pi1_lin = t_it->first;
    
    int t_pi1 = modLt(t_pi1_lin,Lt);
    int t_pi2 = modLt(t_pi1 + tsep_pion, Lt);

    //Determine what node timeslices are actually needed
    //std::vector<bool> node_top_used;
    //getUsedTimeslices(node_top_used,tsep_k_pi,t_pi1);

    //Compute part1 and part2
    //Construct part 1 (no dependence on t_K):
    //\Gamma_1 vL_i(x_op; x_4) [[\sum_{\vec x} wL_i^dag(x) S_2 vL_j(x;top)]] wL_j^dag(x_op)
    Type1FieldTimings::timer().part1 -= dclock();
    std::vector<SCFmatrixField> part1(2, SCFmatrixField(field_params)); //part1 goes from insertion to pi1, pi2 (x_4 = t_pi1, t_pi2)
    mult(part1[0], vL, mf_pi1[t_pi1], wL, false, true);
    mult(part1[1], vL, mf_pi2[t_pi2], wL, false, true);
    Type1FieldTimings::timer().part1 += dclock();    

    int ntsep_k_pi_do = t_it->second.size();

    for(int ii =0; ii< ntsep_k_pi_do; ii++){
      int tkpi_idx = t_it->second[ii];
      int t_K = modLt(t_pi1 - tsep_k_pi[tkpi_idx], Lt);

      //Construct part 2:
      //\Gamma_2 vL_i(x_op;y_4) [[ wL_i^dag(y) S_2 vL_j(y;t_K) ]] [[ wL_j^dag(x_K)\gamma^5 \gamma^5 wH_k(x_K) ) ]] vH_k^\dagger(x_op;t_K)\gamma^5
      Type1FieldTimings::timer().part2 -= dclock();
      std::vector<SCFmatrixField> part2(2, SCFmatrixField(field_params)); //part2 has pi1, pi2 at y_4 = t_pi1, t_pi2
      mult(part2[0], vL, con_pi1_K[t_pi1][tkpi_idx], vH, false, true);
      mult(part2[1], vL, con_pi2_K[t_pi2][tkpi_idx], vH, false, true);
      gr(part2[0], -5); //right multiply by g5
      gr(part2[1], -5);
      Type1FieldTimings::timer().part2 += dclock();

      Type1FieldTimings::timer().contraction_time -= dclock();     
      type1_contract(result[tkpi_idx],t_K,part1,part2);
      Type1FieldTimings::timer().contraction_time += dclock();     
    }//end of loop over k->pi seps
  }//tpi loop

  if(!UniqueID()) printf("Memory before finishing up type1 K->pipi:\n");
  printMem();


  if(!UniqueID()){ printf("Type 1 finishing up results\n"); fflush(stdout); }
  Type1FieldTimings::timer().finish_up -= dclock();
  for(int tkpi_idx =0; tkpi_idx< ntsep_k_pi; tkpi_idx++){
    result[tkpi_idx].nodeSum();
    result[tkpi_idx] *= Float(0.5); //coefficient of 0.5 associated with average over pt1_pion loop
  }

  //For comparison with old code, zero out data not within the K->pi time region (later optimization may render this unnecessary)
  for(int tkpi_idx=0; tkpi_idx<ntsep_k_pi; tkpi_idx++){
    for(int t_dis=0;t_dis<Lt;t_dis++){
      if( !(t_dis < tsep_k_pi[tkpi_idx] && t_dis > 0) ){      

	for(int t_K=0;t_K<Lt;t_K++)
	  for(int conidx=0;conidx<n_contract;conidx++)
	    for(int gcombidx=0;gcombidx<8;gcombidx++)
	      result[tkpi_idx](t_K,t_dis,conidx,gcombidx) = 0;
      }
    }
  }

  Type1FieldTimings::timer().finish_up += dclock();

  if(!UniqueID()) printf("Memory after finishing up type1 K->pipi:\n");
  printMem();


#ifdef NODE_DISTRIBUTE_MESONFIELDS
  nodeDistributeMany(2,&mf_pi1,&mf_pi2);
  if(!UniqueID()) printf("Memory after redistributing meson fields type1 K->pipi:\n");
  printMem();
#endif
  Type1FieldTimings::timer().total += dclock();
  Type1FieldTimings::timer().report();
}



//Field version only applicable to SIMD data. For non SIMD data we should fall back to CPU version
template<typename mf_Policies, typename complexClass>
struct _type1_field_wrap{};

template<typename mf_Policies>
struct _type1_field_wrap<mf_Policies, grid_vector_complex_mark>{
  typedef typename ComputeKtoPiPiGparity<mf_Policies>::ResultsContainerType ResultsContainerType;  
  typedef typename ComputeKtoPiPiGparity<mf_Policies>::mf_WW mf_WW;  
  static void calc(ResultsContainerType result[],
		   const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const ThreeMomentum &p_pi_1, 
		   const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
		   const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
		   const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH){
    ComputeKtoPiPiGparity<mf_Policies>::type1_field_SIMD(result, tsep_k_pi, tsep_pion, tstep, p_pi_1, mf_kaon, mf_pions, vL, vH, wL, wH);
  }
};

template<typename mf_Policies>
struct _type1_field_wrap<mf_Policies, complex_double_or_float_mark>{
  typedef typename ComputeKtoPiPiGparity<mf_Policies>::ResultsContainerType ResultsContainerType;  
  typedef typename ComputeKtoPiPiGparity<mf_Policies>::mf_WW mf_WW;  
  static void calc(ResultsContainerType result[],
		   const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const ThreeMomentum &p_pi_1, 
		   const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
		   const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
		   const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH){
    if(!UniqueID()) printf("Type1 field implementation falling back to OMP implementation due to non-SIMD data\n");
    ComputeKtoPiPiGparity<mf_Policies>::type1_omp(result, tsep_k_pi, tsep_pion, tstep, 1, p_pi_1, mf_kaon, mf_pions, vL, vH, wL, wH);
  }
};


template<typename mf_Policies>
void ComputeKtoPiPiGparity<mf_Policies>::type1_field(ResultsContainerType result[],
							  const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const ThreeMomentum &p_pi_1, 
							  const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
							  const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
							  const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH){
  _type1_field_wrap<mf_Policies, typename ComplexClassify<ComplexType>::type>::calc(result, tsep_k_pi, tsep_pion, tstep, p_pi_1, mf_kaon, mf_pions, vL, vH, wL, wH);
}
  


#endif
