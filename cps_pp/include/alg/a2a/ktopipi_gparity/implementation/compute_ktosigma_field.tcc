///////////////////////////////////////////////   TYPE 1/2   ///////////////////////////////////////////////////////////////////////


template<typename mf_Policies>
void ComputeKtoSigma<mf_Policies>::type12_contract(ResultsContainerType &result, const int tK_glb, const SCFmatrixField &part1, const SCFmatrixField &part2){
#ifndef MEMTEST_MODE

  //D1   = Tr( [pt1] G5 M1 [pt2] M2 )
  //D6   = Tr( [pt1] G5 M1 ) Tr( [pt2] M2 )
  //D8   = Tr( ( [pt2] M2 )_ba ( [pt1] G5 M1 )_ba )
  //D11  = Tr( [pt1] G5 M1 )_ba Tr( [pt2] M2 )_ba
  //D19  = Tr(  Tr_c( [pt2] M2 ) Tr_c( [pt1] G5 M1 ) )  
  
  auto pt1_G5 = gr_r(part1, -5);
  
  for(int mu=0;mu<4;mu++){ //sum over mu here
    for(int gcombidx=0;gcombidx<8;gcombidx++){
      auto pt1_G5_M1 = multGammaRight(pt1_G5, 1, gcombidx,mu);
      auto pt2_M2 = multGammaRight(part2, 2, gcombidx,mu);

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
    int t_range_start = modLt(tS_glb - tsep_k_sigma_lrg, Lt); //no point computing outside of range between kaon and sigma operator
    
    pt2_time -= dclock();
    std::cout << Grid::GridLogMessage << "Part2 vMv " << t_range_start << "->" << tS_glb << std::endl;
    mult(pt2_store[tS_idx], vL, mf_S[tS_glb], wL, false, true, t_range_start, tsep_k_sigma_lrg); //result is field in xop
    pt2_time += dclock();
  }

  SCFmatrixField pt1(field_params);  
  int node_tstart = GJP.TnodeCoor()*GJP.TnodeSites();
  int node_tlessthan = (GJP.TnodeCoor()+1)*GJP.TnodeSites();
  
  //Start loop over tK
  for(int tK_glb=0;tK_glb< Lt; tK_glb++){
    if( tK_glb >= node_tlessthan ) continue;
    
    int t_range_end = modLt(tK_glb + tsep_k_sigma_lrg, Lt); //no point computing outside of range between kaon and sigma operator
    
    pt1_time -= dclock();
    std::cout << Grid::GridLogMessage << "Part1 t_K=" << tK_glb << " vMv " << tK_glb << "->" << t_range_end << std::endl;
    mult(pt1, vL, mf_ls_WW[tK_glb], vH, false, true, tK_glb, tsep_k_sigma_lrg);
    pt1_time += dclock();

    //loop over K->sigma seps, reuse precomputed pt2
    for(int i=0;i<ntsep_k_sigma;i++){
      int tS_glb = modLt(tK_glb + tsep_k_sigma[i], Lt);

      if( tS_glb  < node_tstart ) continue;
      
      const SCFmatrixField &pt2 = pt2_store[tS_subset_inv_map[tS_glb]];

      std::cout << Grid::GridLogMessage << "Contract t_K=" << tK_glb << " t_sigma=" << tS_glb << std::endl;
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
      if(t_dis > tsep_k_sigma[i] || (tsep_k_sigma[i] == tsep_k_sigma_lrg && t_dis == tsep_k_sigma[i])){ //annoying to have to reproduce typos from old code!
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



///////////////////////////////////////////////   TYPE 3   ///////////////////////////////////////////////////////////////////////



template<typename mf_Policies>
void ComputeKtoSigma<mf_Policies>::type3_contract(ResultsContainerType &result, const int tK_glb, const SCFmatrixField &part1, 
						  const SCFmatrixField &part2_L, const SCFmatrixField &part2_H){
#ifndef MEMTEST_MODE

  //D2  = Tr( [pt1] G5 M1   ) * Tr( [pt2_L] M2 )
  //D3  = Tr( [pt1] G5 M1 [pt2_L] M2 )
  //D7  = Tr( ( [pt1] G5 M1 )_ab  ( [pt2_L] M2)_ab )
  //D10 = Tr( M2 [pt2_L] )_ab * Tr( [pt1] G5 M1 )_ab
  //D14 = Tr( [pt1] G5 M1  ) * Tr( [pt2_H] M2 )
  //D16 = Tr([pt1] G5 M2 [pt2_H] M1  )
  //D18 = Tr( Tr_c([pt2_L] M2) * Tr_c( [pt1] G5 M1 ) )
  //D21 = Tr_c(  Tr( [pt2_H] M2) * Tr( [pt1] G5 M1 ) )
  //D23 = Tr( Tr_c( [pt1] G5 M2 )  Tr_c( [pt2_H] M1 ) )

  auto pt1_G5 = gr_r(part1, -5);

  for(int mu=0;mu<4;mu++){ //sum over mu here
    for(int gcombidx=0;gcombidx<8;gcombidx++){
      auto pt1_G5_M1 = multGammaRight(pt1_G5, 1, gcombidx,mu);
      auto pt1_G5_M2 = multGammaRight(pt1_G5, 2, gcombidx,mu);
      auto pt2L_M2 = multGammaRight(part2_L, 2, gcombidx,mu);
      auto pt2H_M1 = multGammaRight(part2_H, 1, gcombidx,mu);
      auto pt2H_M2 = multGammaRight(part2_H, 2, gcombidx,mu);

      auto ctrans_pt2L_M2 = TransposeColor(pt2L_M2);

      int c = 0;
	
      add(c++, result, tK_glb, gcombidx, 0,
	  Trace(pt1_G5_M1) * Trace(pt2L_M2)
	  ); //D2
      add(c++, result, tK_glb, gcombidx, 0,
	  Trace( pt1_G5_M1, pt2L_M2 )
	  ); //D3
      add(c++, result, tK_glb, gcombidx, 0,
	  Trace( pt1_G5_M1, ctrans_pt2L_M2 )
	  ); //D7
      add(c++, result, tK_glb, gcombidx, 0,
	  Trace( SpinFlavorTrace(pt2L_M2), Transpose(SpinFlavorTrace(pt1_G5_M1)) ) 
	  ); //D10
      add(c++, result, tK_glb, gcombidx, 0,
	  Trace(pt1_G5_M1) * Trace(pt2H_M2)
	  ); //D14
      add(c++, result, tK_glb, gcombidx, 0,
	  Trace( pt1_G5_M2, pt2H_M1 )
	  ); //D16
      add(c++, result, tK_glb, gcombidx, 0,
	  Trace( ColorTrace(pt2L_M2), ColorTrace(pt1_G5_M1) )
	  ); //D18
      add(c++, result, tK_glb, gcombidx, 0,
	  Trace( SpinFlavorTrace(pt2H_M2), SpinFlavorTrace(pt1_G5_M1) )
	  ); //D21
      add(c++, result, tK_glb, gcombidx, 0,
	  Trace( ColorTrace(pt1_G5_M2), ColorTrace(pt2H_M1) )
	  ); //D23

    }
  }
#endif
}



template<typename mf_Policies>
void ComputeKtoSigma<mf_Policies>::type3_field_SIMD(std::vector<ResultsContainerType> &result, std::vector<MixDiagResultsContainerType> &mix3, std::vector<SigmaMesonFieldType> &mf_S){   
  if(!UniqueID()) printf("Starting type 3 K->sigma contractions (field implementation)\n");
  double total_time = dclock();
  double time;

  auto field_params = vL.getMode(0).getDimPolParams();  
    
  //Determine tK,tS pairings needed for node, and also tS values
  std::set<std::pair<int,int> > tK_tS_use;
  std::set<int> tS_use, tK_use;
  for(int top_loc = 0; top_loc < GJP.TnodeSites(); top_loc++){
    const int top_glb = top_loc  + GJP.TnodeCoor()*GJP.TnodeSites();
    for(int tdis=0; tdis< tsep_k_sigma_lrg; tdis++){
      int tK_glb = modLt(top_glb - tdis, Lt);
      tK_use.insert(tK_glb);
      for(int i=0;i<ntsep_k_sigma;i++){
	if(tdis > tsep_k_sigma[i]) continue;
	int tS_glb = modLt(tK_glb + tsep_k_sigma[i], Lt);
	tK_tS_use.insert(std::pair<int,int>(tK_glb,tS_glb));
	tS_use.insert(tS_glb);
      }
    }
  }
    
  //Make a mapping between an int and a tK,tS pair
  int ntK_tS = tK_tS_use.size();
  std::vector<std::pair<int,int> > tK_tS_idx_map;
  std::vector<std::vector<int> > tK_tS_idx_inv_map;
  idx_tpair_map(tK_tS_idx_map, tK_tS_idx_inv_map, tK_tS_use);

  //Gather meson fields
#ifdef NODE_DISTRIBUTE_MESONFIELDS
  std::vector<bool> gather_tslice_mask(Lt,false);
  for(std::set<int>::const_iterator it = tS_use.begin(); it != tS_use.end(); it++) gather_tslice_mask[*it] = true;
  time = dclock();
  nodeGetMany(1, &mf_S, &gather_tslice_mask);
  print_time("ComputeKtoSigma","type3 mf gather",dclock()-time);  
#endif
  printMem("Memory after type3 mf gather");
  
  //Setup output
  std::cout << Grid::GridLogMessage << "Resize output" << std::endl;
  result.resize(ntsep_k_sigma); 
  mix3.resize(ntsep_k_sigma);
  for(int i=0;i<ntsep_k_sigma;i++){
    result[i].resize(9,1);
    mix3[i].resize(1);
  }

  std::cout << Grid::GridLogMessage << "mix3_Gamma" << std::endl;
  SCFmat mix3_Gamma[2];
  mix3_Gamma[0].unit().pr(F0).gr(-5);
  mix3_Gamma[1].unit().pr(F1).gr(-5).timesMinusOne();

  double vmv_setup_time = 0;
  double pt1_time = 0;
  double pt2_time = 0;
  double contract_time = 0;

  //Compute pt2 (loop): (vL,wL) and (vH,wH)
  pt2_time -= dclock();
  SCFmatrixField pt2_L(field_params), pt2_H(field_params);
  mult(pt2_L, vL, wL, false, true);
  mult(pt2_H, vH, wH, false, true);
  pt2_time += dclock();

  int node_tstart = GJP.TnodeCoor()*GJP.TnodeSites();
  int node_tlessthan = (GJP.TnodeCoor()+1)*GJP.TnodeSites();
  
  SCFmatrixField pt1(field_params);
  Type3MesonFieldProductType mf_prod;
  
  for(int tK_glb=0;tK_glb<Lt;tK_glb++){
    if( tK_glb >= node_tlessthan ) continue;
    
    for(int i=0;i<ntsep_k_sigma;i++){
      int tS_glb = modLt(tK_glb + tsep_k_sigma[i], Lt);

      if( tS_glb  < node_tstart ) continue;
      
      pt1_time -= dclock();
      mult(mf_prod, mf_S[tS_glb], mf_ls_WW[tK_glb], true);      //node local because the tK,tS pairings are specific to this node
      std::cout << Grid::GridLogMessage << "vMv "<< tK_glb << "->" << tS_glb << std::endl; 
      mult(pt1, vL, mf_prod, vH, false, true, tK_glb, tsep_k_sigma[i]); //only compute between kaon and sigma
      pt1_time += dclock();
      
      contract_time -= dclock();
      type3_contract(result[i], tK_glb, pt1, pt2_L, pt2_H);

      //Compute mix3 diagram
      //These are identical to the type3 diagrams but without the internal quark loop, and with the vertex replaced with a pseudoscalar vertex
      auto pt1_G5 = pt1;
      gr(pt1_G5, -5);
      for(int mix3_gidx=0; mix3_gidx<2; mix3_gidx++){
#ifndef MEMTEST_MODE	  
	add(mix3[i], tK_glb, mix3_gidx, 
	    Trace( pt1_G5, mix3_Gamma[mix3_gidx] )	      
	    );
#endif
      }
      contract_time += dclock();
    }//i
  }//tK_glb

  print_time("ComputeKtoSigma","type3 vMv setup",vmv_setup_time);     
  print_time("ComputeKtoSigma","type3 pt1 compute",pt1_time);     
  print_time("ComputeKtoSigma","type3 pt2 compute",pt2_time);     
  print_time("ComputeKtoSigma","type3 contract",contract_time);  

  time = dclock();
  for(int i=0;i<ntsep_k_sigma;i++){ 
    result[i].nodeSum(); 
    mix3[i].nodeSum();
  }
  print_time("ComputeKtoSigma","type3 accum",dclock()-time);  

  //For comparison with old code, zero out data not within the K->pi time region (later optimization may render this unnecessary)
  int n_contract = 9;

  for(int i=0; i<ntsep_k_sigma; i++){
    for(int t_dis=0;t_dis<Lt;t_dis++){
      if(t_dis > tsep_k_sigma[i] || (tsep_k_sigma[i] == tsep_k_sigma_lrg && t_dis == tsep_k_sigma[i])){ //annoying to have to reproduce typos from old code!
	for(int t_K=0;t_K<Lt;t_K++){
	  for(int conidx=0;conidx<n_contract;conidx++)
	    for(int gcombidx=0;gcombidx<8;gcombidx++)
	      result[i](t_K,t_dis,conidx,gcombidx) = 0;

	  for(int midx=0;midx<2;midx++){
	    mix3[i](t_K,t_dis,midx) = 0;	  
	  }
	}
      }
    }
  }


#ifdef NODE_DISTRIBUTE_MESONFIELDS
  time = dclock();
  nodeDistributeMany(1,&mf_S);
  print_time("ComputeKtoSigma","type3 mf distribute",dclock()-time);  
#endif

  print_time("ComputeKtoSigma","type3 total",dclock()-total_time); 
}


//Field version only applicable to SIMD data. For non SIMD data we should fall back to CPU version
template<typename mf_Policies, typename complexClass>
struct _ktosigma_type3_field_wrap{};

template<typename mf_Policies>
struct _ktosigma_type3_field_wrap<mf_Policies, grid_vector_complex_mark>{
  typedef typename ComputeKtoSigma<mf_Policies>::ResultsContainerType ResultsContainerType;  
  typedef typename ComputeKtoSigma<mf_Policies>::MixDiagResultsContainerType MixDiagResultsContainerType;  
  typedef typename ComputeKtoSigma<mf_Policies>::SigmaMesonFieldType SigmaMesonFieldType;

  static void calc(std::vector<ResultsContainerType> &result, std::vector<MixDiagResultsContainerType> &mix3,
		   std::vector<SigmaMesonFieldType> &mf_S, ComputeKtoSigma<mf_Policies> &compute){  
    compute.type3_field_SIMD(result, mix3, mf_S);
  }
};

template<typename mf_Policies>
struct _ktosigma_type3_field_wrap<mf_Policies, complex_double_or_float_mark>{
  typedef typename ComputeKtoSigma<mf_Policies>::ResultsContainerType ResultsContainerType;  
  typedef typename ComputeKtoSigma<mf_Policies>::MixDiagResultsContainerType MixDiagResultsContainerType;  
  typedef typename ComputeKtoSigma<mf_Policies>::SigmaMesonFieldType SigmaMesonFieldType;

  static void calc(std::vector<ResultsContainerType> &result, std::vector<MixDiagResultsContainerType> &mix3,
		   std::vector<SigmaMesonFieldType> &mf_S, ComputeKtoSigma<mf_Policies> &compute){  
    if(!UniqueID()) printf("K->sigma type3 field implementation falling back to OMP implementation due to non-SIMD data\n");
    compute.type3_omp(result, mix3, mf_S);
  }
};


template<typename mf_Policies>
void ComputeKtoSigma<mf_Policies>::type3_field(std::vector<ResultsContainerType> &result, std::vector<MixDiagResultsContainerType> &mix3, std::vector<SigmaMesonFieldType> &mf_S){  
  _ktosigma_type3_field_wrap<mf_Policies, typename ComplexClassify<ComplexType>::type>::calc(result, mix3, mf_S, *this);
}


///////////////////////////////////////////////   TYPE 4   ///////////////////////////////////////////////////////////////////////


template<typename mf_Policies>
void ComputeKtoSigma<mf_Policies>::type4_contract(ResultsContainerType &result, const int tK_glb, const SCFmatrixField &part1, const SCFmatrixField &part2_L, const SCFmatrixField &part2_H){
#ifndef MEMTEST_MODE

  //D4   = Tr( V_O [W^dag_K WH_K] VH^dag_O G5 M1 V_O W^dag_O M2 ) Tr( [V_S W^dag_S] )
  //D5   = Tr( V_O [W^dag_K WH_K] VH^dag_O G5 M1 ) Tr( V_O W^dag_O M2 ) Tr( [V_S W^dag_S] )
  //D9   = Tr( (M1 V_O W^dag_O)_ab (M2 V_O [W^dag_K WH_K] VH^dag_O G5 )_ab ) Tr( [V_S W^dag_S] )
  //D12  = Tr( V_O [W^dag_K WH_K] VH^dag_O G5 M1 )_ab Tr( V_O W^dag_O M2 )_ab Tr( [V_S W^dag_S] )
  //D13  = Tr( V_O [W^dag_K WH_K] VH^dag_O G5 M1 ) Tr( M2 VH_O WH^dag_O ) Tr( [V_S W^dag_S] )
  //D15  = Tr( V_O [W^dag_K WH_K] VH^dag_O G5 M2 VH_O WH^dag_O M1  ) Tr( [V_S W^dag_S] )
  //D17  = Tr( Tr_c(V_O [W^dag_K WH_K] VH^dag_O G5 M1 ) Tr_c( V_O W^dag_O M2) ) Tr( [V_S W^dag_S] )
  //D20  = Tr_c(  Tr( V_O [W^dag_K WH_K] VH^dag_O G5 M1 ) Tr( VH_O WH^dag_O M2 ) ) Tr( [V_S W^dag_S] ) 
  //D22  = Tr( Tr_c( VH_O WH^dag_O M1 ) Tr_c( V_O [W^dag_K WH_K] VH^dag_O G5 M2 ) ) Tr( [V_S W^dag_S] ) 
    
  //D4   = Tr( [pt1] G5 M1 [pt2_L] M2 ) 
  //D5   = Tr( [pt1] G5 M1 ) Tr( [pt2_L] M2 )
  //D9   = Tr( (M1 [pt2_L])_ab (M2 [pt1] G5 )_ab )
  //D12  = Tr( [pt1] G5 M1 )_ab Tr( [pt2_L] M2 )_ab
  //D13  = Tr( [pt1] G5 M1 ) Tr( M2 [pt2_H] )
  //D15  = Tr( [pt1] G5 M2 [pt2_H] M1  )
  //D17  = Tr( Tr_c([pt1] G5 M1 ) Tr_c( [pt2_L] M2) )
  //D20  = Tr_c(  Tr( [pt1] G5 M1 ) Tr( [pt2_H] M2 ) ) 
  //D22  = Tr( Tr_c( [pt2_H] M1 ) Tr_c( [pt1] G5 M2 ) )

  auto pt1_G5 = gr_r(part1, -5);

  for(int mu=0;mu<4;mu++){ //sum over mu here
    for(int gcombidx=0;gcombidx<8;gcombidx++){
      auto pt1_G5_M1 = multGammaRight(pt1_G5, 1, gcombidx,mu);
      auto pt1_G5_M2 = multGammaRight(pt1_G5, 2, gcombidx,mu);
      auto pt2L_M2 = multGammaRight(part2_L, 2, gcombidx,mu);
      auto pt2H_M1 = multGammaRight(part2_H, 1, gcombidx,mu);
      auto pt2H_M2 = multGammaRight(part2_H, 2, gcombidx,mu);

      auto ctrans_pt2L_M2 = TransposeColor(pt2L_M2);
	
      int c = 0;

      add(c++, result, tK_glb, gcombidx, 0,
	  Trace( pt1_G5_M1, pt2L_M2 )
	  ); //D4
      add(c++, result, tK_glb, gcombidx, 0,
	  Trace(pt1_G5_M1) * Trace(pt2L_M2)
	  ); //D5
      add(c++, result, tK_glb, gcombidx, 0,
	  Trace( pt1_G5_M1, ctrans_pt2L_M2 )
	  ); //D9
      add(c++, result, tK_glb, gcombidx, 0,
	  Trace( SpinFlavorTrace(pt1_G5_M1), Transpose( SpinFlavorTrace(pt2L_M2) ) )
	  ); //D12
      add(c++, result, tK_glb, gcombidx, 0,
	  Trace(pt1_G5_M1) * Trace(pt2H_M2)
	  ); //D13
      add(c++, result, tK_glb, gcombidx, 0,
	  Trace( pt1_G5_M2, pt2H_M1 )
	  ); //D15
      add(c++, result, tK_glb, gcombidx, 0,
	  Trace( ColorTrace(pt1_G5_M1), ColorTrace(pt2L_M2) )
	  ); //D17
      add(c++, result, tK_glb, gcombidx, 0,
	  Trace( SpinFlavorTrace(pt1_G5_M1), SpinFlavorTrace(pt2H_M2) )
	  ); //D20
      add(c++, result, tK_glb, gcombidx, 0,
	  Trace( ColorTrace(pt1_G5_M2), ColorTrace(pt2H_M1) )
	  ); //D22
    }
  }
#endif
}



template<typename mf_Policies>
void ComputeKtoSigma<mf_Policies>::type4_field_SIMD(ResultsContainerType &result, MixDiagResultsContainerType &mix4){
  if(!UniqueID()) printf("Starting type 4 K->sigma contractions (field implementation)\n");
  double total_time = dclock();
       
  double time;

  auto field_params = vL.getMode(0).getDimPolParams();  

  result.resize(9, 1);
  mix4.resize(1);

  //Compute which tS and tK we need
  //K<----tdis------>Q<------tsep_k_sigma-tdis------- sigma
  //Q is constrained to lie on this node and the largest tdis = tsep_k_sigma (such that the sigma is on top of the Q)
  //No point computing with kaon outside this range
  std::set<int> tK_use;
  for(int top_loc = 0; top_loc < GJP.TnodeSites(); top_loc++){
    const int top_glb = top_loc  + GJP.TnodeCoor()*GJP.TnodeSites();
    for(int tdis=0; tdis< tsep_k_sigma_lrg; tdis++){
      int tK_glb = modLt(top_glb - tdis, Lt);
      tK_use.insert(tK_glb);
    }
  }   

  //Map an index to  tK
  int ntK = tK_use.size();
  std::vector<int> tK_subset_map, tK_subset_inv_map;
  idx_t_map(tK_subset_map, tK_subset_inv_map, tK_use);

  SCFmat mix4_Gamma[2];
  mix4_Gamma[0].unit().pr(F0).gr(-5);
  mix4_Gamma[1].unit().pr(F1).gr(-5).timesMinusOne();

  double vmv_setup_time = 0;
  double pt1_time = 0;
  double pt2_time = 0;
  double contract_time = 0;

  //Compute part2 =  (vL wL) and (vH, wH)
  pt2_time -= dclock();
  SCFmatrixField pt2_L(field_params), pt2_H(field_params);
  mult(pt2_L, vL, wL, false, true);
  mult(pt2_H, vH, wH, false, true);
  pt2_time += dclock();

  for(int tK_idx=0; tK_idx< tK_subset_map.size(); tK_idx++){
    int tK_glb = tK_subset_map[tK_idx];

    //Compute part1 = (vL mf_K vH) evaluated with vL, vH at Q
    pt1_time -= dclock();
    SCFmatrixField pt1(field_params);	
    mult(pt1, vL, mf_ls_WW[tK_glb], vH, false, true);
    pt1_time += dclock();


    contract_time -= dclock();
    type4_contract(result, tK_glb, pt1, pt2_L, pt2_H);
	  
    //Compute mix4 diagram
    //These are identical to the type4 diagrams but without the quark loop, and with the vertex replaced with a pseudoscalar vertex
    SCFmatrixField pt1_G5 = pt1; 
    gr(pt1_G5,-5);
    for(int mix4_gidx=0; mix4_gidx<2; mix4_gidx++){
#ifndef MEMTEST_MODE
      add(mix4, tK_glb, mix4_gidx, 
	Trace( pt1_G5 , mix4_Gamma[mix4_gidx] )
      );
#endif
    }
    contract_time += dclock();
  }//tK_idx

  print_time("ComputeKtoSigma","type4 vMv setup",vmv_setup_time);     
  print_time("ComputeKtoSigma","type4 pt1 compute",pt1_time);     
  print_time("ComputeKtoSigma","type4 pt2 compute",pt2_time);     
  print_time("ComputeKtoSigma","type4 contract",contract_time);  

  time = dclock();
  result.nodeSum();
  mix4.nodeSum();

  print_time("ComputeKtoSigma","type4 accum",dclock()-time);  
  print_time("ComputeKtoSigma","type4 total",dclock()-total_time);  

  //For comparison with old code, zero out data not within the K->sigma time region for any of the specified K->sigma tseps (later optimization may render this unnecessary)
  int n_contract = 9;

  for(int t_dis=tsep_k_sigma_lrg;t_dis<Lt;t_dis++){
    for(int t_K=0;t_K<Lt;t_K++){
      for(int conidx=0;conidx<n_contract;conidx++)
	for(int gcombidx=0;gcombidx<8;gcombidx++)
	  result(t_K,t_dis,conidx,gcombidx) = 0;

      for(int midx=0;midx<2;midx++){
	mix4(t_K,t_dis,midx) = 0;	  
      }
    }
  }

}

//Field version only applicable to SIMD data. For non SIMD data we should fall back to CPU version
template<typename mf_Policies, typename complexClass>
struct _ktosigma_type4_field_wrap{};

template<typename mf_Policies>
struct _ktosigma_type4_field_wrap<mf_Policies, grid_vector_complex_mark>{
  typedef typename ComputeKtoSigma<mf_Policies>::ResultsContainerType ResultsContainerType;  
  typedef typename ComputeKtoSigma<mf_Policies>::MixDiagResultsContainerType MixDiagResultsContainerType;  

  static void calc(ResultsContainerType &result, MixDiagResultsContainerType &mix4,
		   ComputeKtoSigma<mf_Policies> &compute){  
    compute.type4_field_SIMD(result, mix4);
  }
};

template<typename mf_Policies>
struct _ktosigma_type4_field_wrap<mf_Policies, complex_double_or_float_mark>{
  typedef typename ComputeKtoSigma<mf_Policies>::ResultsContainerType ResultsContainerType;  
  typedef typename ComputeKtoSigma<mf_Policies>::MixDiagResultsContainerType MixDiagResultsContainerType;  

  static void calc(ResultsContainerType &result, MixDiagResultsContainerType &mix4,
		   ComputeKtoSigma<mf_Policies> &compute){  
    if(!UniqueID()) printf("K->sigma type4 field implementation falling back to OMP implementation due to non-SIMD data\n");
    compute.type4_omp(result, mix4);
  }
};


template<typename mf_Policies>
void ComputeKtoSigma<mf_Policies>::type4_field(ResultsContainerType &result, MixDiagResultsContainerType &mix4){  
  _ktosigma_type4_field_wrap<mf_Policies, typename ComplexClassify<ComplexType>::type>::calc(result, mix4, *this);
}
