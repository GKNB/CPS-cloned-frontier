#ifndef _COMPUTE_KTOPIPI_TYPE2_FIELD_H
#define _COMPUTE_KTOPIPI_TYPE2_FIELD_H

//Expect part2 to be a length=2 vector where for the first elements, pi1 enters first in part2, and in the second element, pi2
template<typename mf_Policies>
void ComputeKtoPiPiGparity<mf_Policies>::type2_contract(ResultsContainerType &result, const int t_K, const SCFmatrixField &part1, const std::vector<SCFmatrixField> &part2){
#ifndef MEMTEST_MODE
  static const int n_contract = 6; //six type2 diagrams
  static const int con_off = 7; //index of first contraction in set
  for(int mu=0;mu<4;mu++){ //sum over mu here
    for(int gcombidx=0;gcombidx<8;gcombidx++){
      auto G1_pt1 = part1;
      multGammaLeft(G1_pt1,1,gcombidx,mu);

      auto tr_sf_G1_pt1 = SpinFlavorTrace(G1_pt1);
      
      for(int pt2_pion=0; pt2_pion<2; pt2_pion++){ //which pion comes first in part 2?
	auto G2_pt2 = part2[pt2_pion]; 
	multGammaLeft(G2_pt2,2,gcombidx,mu);

	auto tr_sf_G2_pt2 = SpinFlavorTrace(G2_pt2);
		
	auto ctrans_G2_pt2 = TransposeColor(G2_pt2);//speedup by transposing part 1
		
	add(7, result, t_K, gcombidx, con_off, 
	    Trace(G1_pt1) * Trace(G2_pt2)
	    );
	add(8, result, t_K, gcombidx, con_off, 
	    Trace( tr_sf_G1_pt1 , Transpose(tr_sf_G2_pt2) )
	    );
	add(9, result, t_K, gcombidx, con_off, 
	    Trace( tr_sf_G1_pt1 , tr_sf_G2_pt2 )
	    );
	add(10, result, t_K, gcombidx, con_off, 
	    Trace( G1_pt1 , G2_pt2 )
	    );
	add(11, result, t_K, gcombidx, con_off, 
	    Trace( G1_pt1, ctrans_G2_pt2 )
	    );
	add(12, result, t_K, gcombidx, con_off, 
	  Trace( ColorTrace(G1_pt1) , ColorTrace(G2_pt2) )
	    );
      }
    }
  }
#endif
}


template<typename mf_Policies>
void ComputeKtoPiPiGparity<mf_Policies>::type2_field_SIMD(ResultsContainerType result[],
						     const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
						     const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
						     const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
						     const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH){
  Type2timings::timer().reset();
  Type2timings::timer().total -= dclock();
    
  const int Lt = GJP.Tnodes()*GJP.TnodeSites();
  assert(Lt % tstep == 0);
  const int tpi_sampled = Lt/tstep;

  static const int n_contract = 6; //six type2 diagrams
  static const int con_off = 7; //index of first contraction in set
  
  auto field_params = vL.getMode(0).getDimPolParams();  
   
  //Compile some information about which timeslices are involved in the calculation such that we can minimize work by skipping unused timeslices
  std::vector< std::vector<bool> > node_top_used(tpi_sampled); //Which local operator timeslices are used for a given pi1 index
  std::vector<int> t_K_all;   //Which kaon timeslices we need overall
  for(int t_pi1_lin = 1; t_pi1_lin <= Lt; t_pi1_lin += tstep){ //Daiqian's weird ordering
    int t_pi1 = modLt(t_pi1_lin,Lt);   int t_pi1_idx = t_pi1 / tstep;
    getUsedTimeslices(node_top_used[t_pi1_idx],t_K_all,tsep_k_pi,t_pi1);
  }
  std::vector< std::vector<bool> > node_top_used_kaon(t_K_all.size()); //Which local operator timeslices are used for a given kaon index
  std::vector<int> tkidx_map(Lt,-1);

  for(int tkidx=0;tkidx<t_K_all.size();tkidx++){
    getUsedTimeslicesForKaon(node_top_used_kaon[tkidx],tsep_k_pi,t_K_all[tkidx]);
    tkidx_map[t_K_all[tkidx]] = tkidx; //allow us to map into the storage given a value of t_K
  }    

  for(int tkp=0;tkp<tsep_k_pi.size();tkp++)
    result[tkp].resize(n_contract); //Resize zeroes output. Result will be thread-reduced before this method ends 


  //Compute part 1	  
  //    = \sum_{ \vec x_K  }   vL(x_op) [[ wL^dag(x_K) wH(x_K) ]] [vH(x_op)]^dag \gamma^5 
  //Part 1 does not care about the location of the pion, only that of the kaon. It may be used multiple times if we have multiple K->pi seps, so compute it separately
  std::vector<SCFmatrixField> part1_storage(t_K_all.size(), SCFmatrixField(field_params));
  for(int tkidx=0; tkidx < t_K_all.size(); tkidx++){    
    int t_K = t_K_all[tkidx];
    mult(part1_storage[tkidx], vL, mf_kaon[t_K], vH, false, true);
    gr(part1_storage[tkidx], -5);
  }
  
  //Compute pi1<->pi2 contractions
  mf_WV con_pi1_pi2, con_pi2_pi1;
  std::vector<SCFmatrixField> part2(2, SCFmatrixField(field_params));

  //for(int t_pi1 = 0; t_pi1 < Lt; t_pi1 += tstep){ //my sensible ordering
  for(int t_pi1_lin = 1; t_pi1_lin <= Lt; t_pi1_lin += tstep){ //Daiqian's weird ordering
    int t_pi1 = modLt(t_pi1_lin,Lt);
    int t_pi1_idx = t_pi1 / tstep;
    
    int t_pi2 = modLt(t_pi1 + tsep_pion, Lt);
    
    //Form the product of the two meson fields
    //con_*_* = \sum_{\vec y,\vec z} [[ wL^dag(y) S_2 vL(y) ]] [[ wL^dag(z) S_2 vL(z) ]]
    type2_compute_mfproducts(con_pi1_pi2, con_pi2_pi1, t_pi1, t_pi2, p_pi_1_all, mf_pions);

    //Construct part 2 (this doesn't involve the kaon):
    // \sum_{ \vec y, \vec z  }  vL(x_op) [[ wL^dag(y) S_2 vL(y) ]] [[ wL^dag(z) S_2 vL(z) ]] wL^dag(x_op)
    mult(part2[0], vL, con_pi1_pi2, wL, false, true); //part2 goes from insertion to pi1 to pi2 and back to insertion
    mult(part2[1], vL, con_pi2_pi1, wL, false, true); //part2 goes from insertion to pi2 to pi1 and back to insertion

    for(int tkpi_idx = 0; tkpi_idx < tsep_k_pi.size(); tkpi_idx++){
      int t_K = modLt(t_pi1 - tsep_k_pi[tkpi_idx], Lt);
      
      const SCFmatrixField &part1 = part1_storage[tkidx_map[t_K]];
      Type2timings::timer().contraction_time -= dclock();
      type2_contract(result[tkpi_idx],t_K,part1,part2);
      Type2timings::timer().contraction_time += dclock();
    }
  }

  Type2timings::timer().finish_up -= dclock();
  for(int tkp=0;tkp<tsep_k_pi.size();tkp++){
    result[tkp].nodeSum();
#ifndef DAIQIAN_COMPATIBILITY_MODE
    result[tkp] *= Float(0.5); //coefficient of 0.5 associated with average over pt2 pion ordering
#endif
  }

  //For comparison with old code, zero out data not within the K->pi time region (later optimization may render this unnecessary)
  for(int tkpi_idx=0; tkpi_idx<tsep_k_pi.size(); tkpi_idx++){
    for(int t_dis=0;t_dis<Lt;t_dis++){
      if( !(t_dis < tsep_k_pi[tkpi_idx] && t_dis > 0) ){      

	for(int t_K=0;t_K<Lt;t_K++)
	  for(int conidx=0;conidx<n_contract;conidx++)
	    for(int gcombidx=0;gcombidx<8;gcombidx++)
	      result[tkpi_idx](t_K,t_dis,conidx,gcombidx) = 0;
      }
    }
  }

  Type2timings::timer().finish_up += dclock();
  Type2timings::timer().total += dclock();
  Type2timings::timer().report();
}


//Field version only applicable to SIMD data. For non SIMD data we should fall back to CPU version
template<typename mf_Policies, typename complexClass>
struct _type2_field_wrap{};

template<typename mf_Policies>
struct _type2_field_wrap<mf_Policies, grid_vector_complex_mark>{
  typedef typename ComputeKtoPiPiGparity<mf_Policies>::ResultsContainerType ResultsContainerType;  
  typedef typename ComputeKtoPiPiGparity<mf_Policies>::mf_WW mf_WW;  
  static void calc(ResultsContainerType result[],
		   const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
		   const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
		   const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
		   const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH){
    ComputeKtoPiPiGparity<mf_Policies>::type2_field_SIMD(result, tsep_k_pi, tsep_pion, tstep, p_pi_1_all, mf_kaon, mf_pions, vL, vH, wL, wH);
  }
};

template<typename mf_Policies>
struct _type2_field_wrap<mf_Policies, complex_double_or_float_mark>{
  typedef typename ComputeKtoPiPiGparity<mf_Policies>::ResultsContainerType ResultsContainerType;  
  typedef typename ComputeKtoPiPiGparity<mf_Policies>::mf_WW mf_WW;  
  static void calc(ResultsContainerType result[],
		   const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
		   const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
		   const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
		   const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH){
    if(!UniqueID()) printf("Type2 field implementation falling back to OMP implementation due to non-SIMD data\n");
    ComputeKtoPiPiGparity<mf_Policies>::type2_omp_v2(result, tsep_k_pi, tsep_pion, tstep, p_pi_1_all, mf_kaon, mf_pions, vL, vH, wL, wH);
  }
};


template<typename mf_Policies>
void ComputeKtoPiPiGparity<mf_Policies>::type2_field(ResultsContainerType result[],
						     const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
						     const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
						     const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
						     const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH){
  _type2_field_wrap<mf_Policies, typename ComplexClassify<ComplexType>::type>::calc(result, tsep_k_pi, tsep_pion, tstep, p_pi_1_all, mf_kaon, mf_pions, vL, vH, wL, wH);
}



#endif
