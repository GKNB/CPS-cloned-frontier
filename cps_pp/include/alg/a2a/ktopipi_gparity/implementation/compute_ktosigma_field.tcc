template<typename mf_Policies>
void ComputeKtoSigma<mf_Policies>::type12_contract(ResultsContainerType &result, const int tK_glb, const SCFmatrixField &part1, const SCFmatrixField &part2){
#ifndef MEMTEST_MODE

  //D1   = Tr( [pt1] G5 M1 [pt2] M2 )
  //D6   = Tr( [pt1] G5 M1 ) Tr( [pt2] M2 )
  //D8   = Tr( ( [pt2] M2 )_ba ( [pt1] G5 M1 )_ba )
  //D11  = Tr( [pt1] G5 M1 )_ba Tr( [pt2] M2 )_ba
  //D19  = Tr(  Tr_c( [pt2] M2 ) Tr_c( [pt1] G5 M1 ) )  

  for(int mu=0;mu<4;mu++){ //sum over mu here
    for(int gcombidx=0;gcombidx<8;gcombidx++){
      auto pt1_G5_M1 = part1;       
      gr(pt1_G5_M1, -5);
      multGammaRight(pt1_G5_M1, 1, gcombidx,mu);

      auto pt2_M2 = part2;
      multGammaRight(pt2_M2, 2, gcombidx,mu);

      auto ctrans_pt1_G5_M1 = TransposeColor(pt1_G5_M1);

      auto tr_sf_pt1_G5_M1 = SpinFlavorTrace(pt1_G5_M1);
      auto tr_sf_p2_M2 = SpinFlavorTrace(pt2_M2);
       
      int c = 0;
      add(c++, result, tK_glb, gcombidx, 0,
	  Trace( pt1_G5_M1, pt2_M2 ) 
	  ); //D1
      add(c++, result, tK_glb, gcombidx, 0,
	  Trace(pt1_G5_M1) * Trace(pt2_M2)
	  ); //D6
      add(c++, result, tK_glb, gcombidx, 0,
	  Trace( pt2_M2, ctrans_pt1_G5_M1 )
	  ); //D8
      add(c++, result, tK_glb, gcombidx, 0,
	  Trace( tr_sf_pt1_G5_M1, Transpose(tr_sf_p2_M2) )
	  ); //D11
      add(c++, result, tK_glb, gcombidx, 0,
	  Trace( ColorTrace(pt1_G5_M1), ColorTrace(pt2_M2) )
	  ); //D19
    }
  }
#endif
}



template<typename mf_Policies>
void ComputeKtoSigma<mf_Policies>::type12_field_SIMD(std::vector<ResultsContainerType> &result, std::vector<SigmaMesonFieldType> &mf_S){  
  if(!UniqueID()) printf("Starting type 1/2 K->sigma contractions (field version)\n");
  double total_time = dclock();
       
  auto field_params = vL.getMode(0).getDimPolParams();  

  double time;

  //Compute which tS and tK we need
  std::set<int> tS_use, tK_use;
  for(int top_loc = 0; top_loc < GJP.TnodeSites(); top_loc++){
    const int top_glb = top_loc  + GJP.TnodeCoor()*GJP.TnodeSites();
    for(int tdis=0; tdis< tsep_k_sigma_lrg; tdis++){
      int tK_glb = modLt(top_glb - tdis, Lt);
      tK_use.insert(tK_glb);
      for(int i=0;i<ntsep_k_sigma;i++){
	if(tdis > tsep_k_sigma[i]) continue;
	int tS_glb = modLt(tK_glb + tsep_k_sigma[i], Lt);
	tS_use.insert(tS_glb);
      }
    }
  }   

  //Map an index to tS and tK
  int ntS = tS_use.size(), ntK = tK_use.size();
  std::vector<int> tS_subset_map, tS_subset_inv_map, tK_subset_map, tK_subset_inv_map;
  idx_t_map(tS_subset_map, tS_subset_inv_map, tS_use);
  idx_t_map(tK_subset_map, tK_subset_inv_map, tK_use);

  //Gather
#ifdef NODE_DISTRIBUTE_MESONFIELDS
  time = dclock();
  std::vector<bool> gather_tslice_mask(Lt,false);
  for(std::set<int>::const_iterator it = tS_use.begin(); it != tS_use.end(); it++) gather_tslice_mask[*it] = true;
  nodeGetMany(1, &mf_S, &gather_tslice_mask);
  print_time("ComputeKtoSigma","type12 mf gather",dclock()-time);     
#endif
 
  //Start main loop
  result.resize(ntsep_k_sigma); for(int i=0;i<ntsep_k_sigma;i++) result[i].resize(5,1);

  double vmv_setup_time = 0;
  double pt1_time = 0;
  double pt2_time = 0;
  double contract_time = 0;

  //Precompute part2   ( vL(xop) mf_S(tS) wL(xop) )  -  no tK dependence
  std::vector<SCFmatrixField> pt2_store(ntS, SCFmatrixField(field_params));
  for(int tS_idx=0;tS_idx<ntS;tS_idx++){
    int tS_glb = tS_subset_map[tS_idx];
    pt2_time -= dclock();
    mult(pt2_store[tS_idx], vL, mf_S[tS_glb], wL, false, true); //result is field in xop
    pt2_time += dclock();
  }

  SCFmatrixField pt1(field_params);  

  //Start loop over tK
  for(int tK_glb=0;tK_glb< Lt; tK_glb++){
    pt1_time -= dclock();
    mult(pt1, vL, mf_ls_WW[tK_glb], vH, false, true);
    pt1_time += dclock();

    //loop over K->sigma seps, reuse precomputed pt2
    for(int i=0;i<ntsep_k_sigma;i++){
      int tS_glb = modLt(tK_glb + tsep_k_sigma[i], Lt);
      
      const SCFmatrixField &pt2 = pt2_store[tS_subset_inv_map[tS_glb]];

      contract_time -= dclock();
      type12_contract(result[i], tK_glb, pt1, pt2);
      contract_time += dclock();
    }
  }//tdis

  print_time("ComputeKtoSigma","type12 vMv setup",vmv_setup_time);     
  print_time("ComputeKtoSigma","type12 pt1 compute",pt1_time);     
  print_time("ComputeKtoSigma","type12 pt2 compute",pt2_time);     
  print_time("ComputeKtoSigma","type12 contract",contract_time);     

  time = dclock();
  for(int i=0;i<ntsep_k_sigma;i++){ result[i].nodeSum(); }

  //For comparison with old code, zero out data not within the K->pi time region (later optimization may render this unnecessary)
  int n_contract = 5;

  for(int i=0; i<ntsep_k_sigma; i++){
    for(int t_dis=0;t_dis<Lt;t_dis++){
      if( !(t_dis <= tsep_k_sigma[i] && t_dis >= 0) ){      

	for(int t_K=0;t_K<Lt;t_K++)
	  for(int conidx=0;conidx<n_contract;conidx++)
	    for(int gcombidx=0;gcombidx<8;gcombidx++)
	      result[i](t_K,t_dis,conidx,gcombidx) = 0;
      }
    }
  }

  print_time("ComputeKtoSigma","type12 accum",dclock()-time);     

#ifdef NODE_DISTRIBUTE_MESONFIELDS
  time = dclock();
  nodeDistributeMany(1,&mf_S);
  print_time("ComputeKtoSigma","type12 mf distribute",dclock()-time);     
#endif

  print_time("ComputeKtoSigma","type12 total",dclock()-total_time); 
}

//Field version only applicable to SIMD data. For non SIMD data we should fall back to CPU version
template<typename mf_Policies, typename complexClass>
struct _ktosigma_type12_field_wrap{};

template<typename mf_Policies>
struct _ktosigma_type12_field_wrap<mf_Policies, grid_vector_complex_mark>{
  typedef typename ComputeKtoSigma<mf_Policies>::ResultsContainerType ResultsContainerType;  
  typedef typename ComputeKtoSigma<mf_Policies>::SigmaMesonFieldType SigmaMesonFieldType;

  static void calc(std::vector<ResultsContainerType> &result, std::vector<SigmaMesonFieldType> &mf_S, ComputeKtoSigma<mf_Policies> &compute){  
    compute.type12_field_SIMD(result, mf_S);
  }
};

template<typename mf_Policies>
struct _ktosigma_type12_field_wrap<mf_Policies, complex_double_or_float_mark>{
  typedef typename ComputeKtoSigma<mf_Policies>::ResultsContainerType ResultsContainerType;  
  typedef typename ComputeKtoSigma<mf_Policies>::SigmaMesonFieldType SigmaMesonFieldType;

  static void calc(std::vector<ResultsContainerType> &result, std::vector<SigmaMesonFieldType> &mf_S, ComputeKtoSigma<mf_Policies> &compute){  
    if(!UniqueID()) printf("K->sigma type1/2 field implementation falling back to OMP implementation due to non-SIMD data\n");
    compute.type12_omp(result, mf_S);
  }
};


template<typename mf_Policies>
void ComputeKtoSigma<mf_Policies>::type12_field(std::vector<ResultsContainerType> &result, std::vector<SigmaMesonFieldType> &mf_S){  
  _ktosigma_type12_field_wrap<mf_Policies, typename ComplexClassify<ComplexType>::type>::calc(result, mf_S, *this);
}



