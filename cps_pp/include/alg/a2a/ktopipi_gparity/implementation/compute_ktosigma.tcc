template<typename mf_Policies>
void ComputeKtoSigma<mf_Policies>::type12_contract(ResultsContainerType &result, const int tK_glb, const int tdis_glb, const int thread_id, const SCFmat &part1, const SCFmat &part2){
#ifndef MEMTEST_MODE

  //D1   = Tr( [pt1] G5 M1 [pt2] M2 )
  //D6   = Tr( [pt1] G5 M1 ) Tr( [pt2] M2 )
  //D8   = Tr( ( [pt2] M2 )_ba ( [pt1] G5 M1 )_ba )
  //D11  = Tr( [pt1] G5 M1 )_ba Tr( [pt2] M2 )_ba
  //D19  = Tr(  Tr_c( [pt2] M2 ) Tr_c( [pt1] G5 M1 ) )  

  for(int mu=0;mu<4;mu++){ //sum over mu here
    for(int gcombidx=0;gcombidx<8;gcombidx++){
      SCFmat pt1_G5_M1 = part1;       
      pt1_G5_M1.gr(-5);
      multGammaRight(pt1_G5_M1, 1, gcombidx,mu);

      SCFmat pt2_M2 = part2;
      multGammaRight(pt2_M2, 2, gcombidx,mu);

      SCFmat ctrans_pt1_G5_M1 = pt1_G5_M1.TransposeColor();

      CPScolorMatrix<ComplexType> tr_sf_pt1_G5_M1 = pt1_G5_M1.SpinFlavorTrace();
      CPScolorMatrix<ComplexType> tr_sf_p2_M2 = pt2_M2.SpinFlavorTrace();
       
#define D(IDX) result(tK_glb,tdis_glb,c++,gcombidx,thread_id)	      
      int c = 0;

      D(1) += Trace( pt1_G5_M1, pt2_M2 );
      D(6) += pt1_G5_M1.Trace() * pt2_M2.Trace();
      D(8) += Trace( pt2_M2, ctrans_pt1_G5_M1 );
      D(11) += Trace( tr_sf_pt1_G5_M1, Transpose(tr_sf_p2_M2) );
      D(19) += Trace( pt1_G5_M1.ColorTrace(), pt2_M2.ColorTrace() );
#undef D

    }
  }
#endif
}


template<typename mf_Policies>
void ComputeKtoSigma<mf_Policies>::type12_omp(std::vector<ResultsContainerType> &result, std::vector<SigmaMesonFieldType> &mf_S){  
  if(!UniqueID()) printf("Starting type 1/2 K->sigma contractions\n");
  double total_time = dclock();
       
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
  result.resize(ntsep_k_sigma); for(int i=0;i<ntsep_k_sigma;i++) result[i].resize(5,nthread);

  double vmv_setup_time = 0;
  double pt1_time = 0;
  double pt2_time = 0;
  double contract_time = 0;

  std::vector<typename AlignedVector<SCFmat>::type > pt2_store(nthread, typename AlignedVector<SCFmat>::type(ntS));
    
  vMv_split_shrbuf shared_buf_inst; 

  for(int top_loc = 0; top_loc < GJP.TnodeSites(); top_loc++){
    const int top_glb = top_loc  + GJP.TnodeCoor()*GJP.TnodeSites();

#ifndef DISABLE_KTOSIGMA_TYPE12_SPLIT_VMV   
    time = dclock();
    std::vector<vMv_split_VWWV> part1_split(ntK);
    setup_type12_pt1_split(part1_split,top_glb, tK_subset_map, &shared_buf_inst);

    std::vector<vMv_split_VWVW> part2_split(ntS);
    setup_type12_pt2_split(part2_split,mf_S,top_glb, tS_subset_map, &shared_buf_inst);
    vmv_setup_time += dclock() - time;
#endif

#pragma omp parallel for schedule(static)
    for(int xop3d_loc = 0; xop3d_loc < size_3d; xop3d_loc++){
      int thread_id = omp_get_thread_num();
      double ttime;

      //Precompute part2
      for(int tS_idx=0;tS_idx<ntS;tS_idx++){
	int tS = tS_subset_map[tS_idx];
	ttime = dclock();
#ifndef DISABLE_KTOSIGMA_TYPE12_SPLIT_VMV   
	part2_split[tS_idx].contract(pt2_store[thread_id][tS_idx],xop3d_loc, false, true);
#else
	compute_type12_part2(pt2_store[thread_id][tS_idx], tS, top_loc, xop3d_loc, mf_S);
#endif
	if(!thread_id) pt2_time += dclock() - ttime;	  
      }

      //Start loop over tdis
      for(int tdis=0; tdis< tsep_k_sigma_lrg; tdis++){ //note for tsep_k_sigma==tsep_k_sigma_lrg this is inconsistent with skipping tdis>tsep_k_sigma below; annoying but keep for repro purposes
	int tK_glb = modLt(top_glb - tdis, Lt);
	    
	SCFmat pt1;
	ttime = dclock();
#ifndef DISABLE_KTOSIGMA_TYPE12_SPLIT_VMV   
	part1_split[tK_subset_inv_map[tK_glb]].contract(pt1, xop3d_loc, false, true);
#else
	compute_type12_part1(pt1, tK_glb, top_loc, xop3d_loc);
#endif
	if(!thread_id) pt1_time += dclock() - ttime;
	    	 
	for(int i=0;i<ntsep_k_sigma;i++){
	  if(tdis > tsep_k_sigma[i]) continue;
	  int tS_glb = modLt(tK_glb + tsep_k_sigma[i], Lt);

	  const SCFmat &pt2 = pt2_store[thread_id][tS_subset_inv_map[tS_glb]];

	  ttime = dclock();
	  type12_contract(result[i], tK_glb, tdis, thread_id, pt1, pt2);
	  if(!thread_id) contract_time += dclock() - ttime;
	}
      }//tdis
    }//xop
  }//top

  print_time("ComputeKtoSigma","type12 vMv setup",vmv_setup_time);     
  print_time("ComputeKtoSigma","type12 pt1 compute",pt1_time);     
  print_time("ComputeKtoSigma","type12 pt2 compute",pt2_time);     
  print_time("ComputeKtoSigma","type12 contract",contract_time);     

  time = dclock();
  for(int i=0;i<ntsep_k_sigma;i++){ result[i].threadSum(); result[i].nodeSum(); }
  print_time("ComputeKtoSigma","type12 accum",dclock()-time);     

#ifdef NODE_DISTRIBUTE_MESONFIELDS
  time = dclock();
  nodeDistributeMany(1,&mf_S);
  print_time("ComputeKtoSigma","type12 mf distribute",dclock()-time);     
#endif

  print_time("ComputeKtoSigma","type12 total",dclock()-total_time); 
}


template<typename mf_Policies>
void ComputeKtoSigma<mf_Policies>::type3_contract(ResultsContainerType &result, const int tK_glb, const int tdis_glb, const int thread_id, const SCFmat &part1, const SCFmat &part2_L, const SCFmat &part2_H){
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

  for(int mu=0;mu<4;mu++){ //sum over mu here
    for(int gcombidx=0;gcombidx<8;gcombidx++){
      SCFmat pt1_G5_M1 = part1;
      pt1_G5_M1.gr(-5);
      multGammaRight(pt1_G5_M1, 1, gcombidx,mu);

      SCFmat pt2L_M2 = part2_L;
      multGammaRight(pt2L_M2, 2, gcombidx,mu);

      SCFmat pt2H_M2 = part2_H;
      multGammaRight(pt2H_M2, 2, gcombidx,mu);

      SCFmat ctrans_pt2L_M2 = pt2L_M2.TransposeColor();

      SCFmat pt1_G5_M2 = part1;
      pt1_G5_M2.gr(-5);
      multGammaRight(pt1_G5_M2, 2, gcombidx,mu);

      SCFmat pt2H_M1 = part2_H;
      multGammaRight(pt2H_M1, 1, gcombidx,mu);

      int c = 0;
	
#define D(IDX) result(tK_glb,tdis_glb,c++,gcombidx,thread_id)	
	
      D(2) += pt1_G5_M1.Trace() * pt2L_M2.Trace();
      D(3) += Trace( pt1_G5_M1, pt2L_M2 );
      D(7) += Trace( pt1_G5_M1, ctrans_pt2L_M2 );	
      D(10) += Trace( pt2L_M2.SpinFlavorTrace(), Transpose(pt1_G5_M1.SpinFlavorTrace()) );
      D(14) += pt1_G5_M1.Trace() * pt2H_M2.Trace();
      D(16) += Trace( pt1_G5_M2, pt2H_M1 );
      D(18) += Trace( pt2L_M2.ColorTrace(), pt1_G5_M1.ColorTrace() );
      D(21) += Trace( pt2H_M2.SpinFlavorTrace(), pt1_G5_M1.SpinFlavorTrace() );
      D(23) += Trace( pt1_G5_M2.ColorTrace(), pt2H_M1.ColorTrace() );

#undef D
    }
  }
#endif
}



template<typename mf_Policies>
void ComputeKtoSigma<mf_Policies>::type3_omp(std::vector<ResultsContainerType> &result, std::vector<MixDiagResultsContainerType> &mix3, std::vector<SigmaMesonFieldType> &mf_S){   
  if(!UniqueID()) printf("Starting type 3 K->sigma contractions\n");
  double total_time = dclock();
  double time;
    
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

  //Setup output
  result.resize(ntsep_k_sigma); 
  mix3.resize(ntsep_k_sigma);
  for(int i=0;i<ntsep_k_sigma;i++){
    result[i].resize(9,nthread);
    mix3[i].resize(nthread);
  }

  SCFmat mix3_Gamma[2];
  mix3_Gamma[0].unit().pr(F0).gr(-5);
  mix3_Gamma[1].unit().pr(F1).gr(-5).timesMinusOne();

  //Precompute meson field products but only for tK,tS pairs we actually need
  time = dclock();
  std::vector<Type3MesonFieldProductType> mf_prod(ntK_tS);
  for(int i=0;i<ntK_tS;i++){
    int tK=tK_tS_idx_map[i].first, tS = tK_tS_idx_map[i].second;
    mult(mf_prod[i], mf_S[tS], mf_ls_WW[tK],true); //node local because the tK,tS pairings are specific to this node
  }
  print_time("ComputeKtoSigma","type3 mf product",dclock()-time); 

  double vmv_setup_time = 0;
  double pt1_time = 0;
  double pt2_time = 0;
  double contract_time = 0;

  vMv_split_shrbuf shared_buf_inst; 

  std::vector< SCFmatVector > pt1_store_allthr(omp_get_max_threads(), SCFmatVector(ntK_tS));

  for(int top_loc = 0; top_loc < GJP.TnodeSites(); top_loc++){
    const int top_glb = top_loc  + GJP.TnodeCoor()*GJP.TnodeSites();

#ifndef DISABLE_KTOSIGMA_TYPE3_SPLIT_VMV   
    time = dclock();
    std::vector<vMv_split_VWWV> part1_split(ntK_tS);
    setup_type3_pt1_split(part1_split,top_glb,mf_prod,tK_tS_idx_map, &shared_buf_inst);
    vmv_setup_time += dclock() - time;
#endif

#pragma omp parallel for schedule(static)
    for(int xop3d_loc = 0; xop3d_loc < size_3d; xop3d_loc++){
      int thread_id = omp_get_thread_num();
  
      double ttime = dclock();

      SCFmat pt2_L, pt2_H;
      compute_type3_part2(pt2_L, pt2_H, top_loc, xop3d_loc);
      if(!thread_id) pt2_time += dclock() - ttime;

      //Precompute part1
      SCFmat* pt1_store = pt1_store_allthr[thread_id].data();
      for(int i=0;i<ntK_tS;i++){
	ttime = dclock();
#ifndef DISABLE_KTOSIGMA_TYPE3_SPLIT_VMV   
	part1_split[i].contract(pt1_store[i], xop3d_loc, false, true);
#else
	compute_type3_part1(pt1_store[i], top_loc, xop3d_loc, mf_prod[i]);
#endif
	if(!thread_id) pt1_time += dclock() - ttime;
      }

      for(int tdis=0; tdis< tsep_k_sigma_lrg; tdis++){
	int tK_glb = modLt(top_glb - tdis, Lt);

	for(int i=0;i<ntsep_k_sigma;i++){
	  if(tdis > tsep_k_sigma[i]) continue;

	  int tS_glb = modLt(tK_glb + tsep_k_sigma[i], Lt);

	  const SCFmat &pt1 = pt1_store[ tK_tS_idx_inv_map[tK_glb][tS_glb] ];

	  ttime = dclock();
	  type3_contract(result[i], tK_glb, tdis, thread_id, pt1, pt2_L, pt2_H);

	  //Compute mix3 diagram
	  //These are identical to the type3 diagrams but without the internal quark loop, and with the vertex replaced with a pseudoscalar vertex
	  SCFmat pt1_G5 = pt1; pt1_G5.gr(-5);
	  for(int mix3_gidx=0; mix3_gidx<2; mix3_gidx++){
#ifndef MEMTEST_MODE
#define M mix3[i](tK_glb,tdis,mix3_gidx,thread_id)
	    M += Trace( pt1_G5, mix3_Gamma[mix3_gidx] );
#undef M
#endif
	  }
	  if(!thread_id) contract_time += dclock() - ttime;	    
	}
      }
    }//xop3d
  }//top

  print_time("ComputeKtoSigma","type3 vMv setup",vmv_setup_time);     
  print_time("ComputeKtoSigma","type3 pt1 compute",pt1_time);     
  print_time("ComputeKtoSigma","type3 pt2 compute",pt2_time);     
  print_time("ComputeKtoSigma","type3 contract",contract_time);  

  time = dclock();
  for(int i=0;i<ntsep_k_sigma;i++){ 
    result[i].threadSum(); result[i].nodeSum(); 
    mix3[i].threadSum(); mix3[i].nodeSum();
  }
  print_time("ComputeKtoSigma","type3 accum",dclock()-time);  

#ifdef NODE_DISTRIBUTE_MESONFIELDS
  time = dclock();
  nodeDistributeMany(1,&mf_S);
  print_time("ComputeKtoSigma","type3 mf distribute",dclock()-time);  
#endif

  print_time("ComputeKtoSigma","type3 total",dclock()-total_time); 
}


template<typename mf_Policies>
void ComputeKtoSigma<mf_Policies>::type4_contract(ResultsContainerType &result, const int tK_glb, const int tdis_glb, const int thread_id, const SCFmat &part1, const SCFmat &part2_L, const SCFmat &part2_H){
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


  for(int mu=0;mu<4;mu++){ //sum over mu here
    for(int gcombidx=0;gcombidx<8;gcombidx++){
      SCFmat pt1_G5_M1 = part1;
      pt1_G5_M1.gr(-5);
      multGammaRight(pt1_G5_M1, 1, gcombidx,mu);

      SCFmat pt2L_M2 = part2_L;
      multGammaRight(pt2L_M2, 2, gcombidx,mu);

      SCFmat pt2H_M2 = part2_H;
      multGammaRight(pt2H_M2, 2, gcombidx,mu);

      SCFmat ctrans_pt2L_M2 = pt2L_M2.TransposeColor();
	
      SCFmat pt1_G5_M2 = part1;
      pt1_G5_M2.gr(-5);
      multGammaRight(pt1_G5_M2, 2, gcombidx,mu);

      SCFmat pt2H_M1 = part2_H;
      multGammaRight(pt2H_M1, 1, gcombidx,mu);

      int c = 0;

#define D(IDX) result(tK_glb,tdis_glb,c++,gcombidx,thread_id)	
	
      D(4) += Trace( pt1_G5_M1, pt2L_M2 );
      D(5) += pt1_G5_M1.Trace() * pt2L_M2.Trace();
      D(9) += Trace( pt1_G5_M1, ctrans_pt2L_M2 );
      D(12) += Trace( pt1_G5_M1.SpinFlavorTrace(), Transpose( pt2L_M2.SpinFlavorTrace() ) );
      D(13) += pt1_G5_M1.Trace() * pt2H_M2.Trace();
      D(15) += Trace( pt1_G5_M2, pt2H_M1 );
      D(17) += Trace( pt1_G5_M1.ColorTrace(), pt2L_M2.ColorTrace() );
      D(20) += Trace( pt1_G5_M1.SpinFlavorTrace(), pt2H_M2.SpinFlavorTrace() );
      D(22) += Trace( pt1_G5_M2.ColorTrace(), pt2H_M1.ColorTrace() );

#undef D
    }
  }
#endif
}



template<typename mf_Policies>
void ComputeKtoSigma<mf_Policies>::type4_omp(ResultsContainerType &result, MixDiagResultsContainerType &mix4){
  if(!UniqueID()) printf("Starting type 4 K->sigma contractions\n");
  double total_time = dclock();
       
  double time;

  result.resize(9, nthread);
  mix4.resize(nthread);


  //Compute which tS and tK we need
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

  for(int top_loc = 0; top_loc < GJP.TnodeSites(); top_loc++){
    const int top_glb = top_loc  + GJP.TnodeCoor()*GJP.TnodeSites();

#ifndef DISABLE_KTOSIGMA_TYPE4_SPLIT_VMV
    time = dclock();
    std::vector<vMv_split_VWWV> part1_split(ntK);
    setup_type4_pt1_split(part1_split,top_glb,tK_subset_map);
    vmv_setup_time += dclock() - time;
#endif
           
#pragma omp parallel for schedule(static)
    for(int xop3d_loc = 0; xop3d_loc < size_3d; xop3d_loc++){
      int thread_id = omp_get_thread_num();
  
      double ttime;

      ttime = dclock();
      SCFmat pt2_L, pt2_H;
      compute_type4_part2(pt2_L, pt2_H, top_loc, xop3d_loc);
      if(!thread_id) pt2_time += dclock() - ttime;

      for(int tdis=0; tdis< tsep_k_sigma_lrg; tdis++){
	int tK_glb = modLt(top_glb - tdis, Lt);

	ttime = dclock();
	SCFmat pt1;
#ifndef DISABLE_KTOSIGMA_TYPE4_SPLIT_VMV 
	part1_split[ tK_subset_inv_map[tK_glb] ].contract(pt1,xop3d_loc,false,true);
#else
	compute_type4_part1(pt1, tK_glb, top_loc, xop3d_loc);
#endif
	if(!thread_id) pt1_time += dclock() - ttime;

	ttime = dclock();
	type4_contract(result, tK_glb, tdis, thread_id, pt1, pt2_L, pt2_H);
	  
	//Compute mix4 diagram
	//These are identical to the type4 diagrams but without the quark loop, and with the vertex replaced with a pseudoscalar vertex
	SCFmat pt1_G5 = pt1; pt1_G5.gr(-5);
	for(int mix4_gidx=0; mix4_gidx<2; mix4_gidx++){
#ifndef MEMTEST_MODE
#define M mix4(tK_glb,tdis,mix4_gidx,thread_id)
	  M += Trace( pt1_G5 , mix4_Gamma[mix4_gidx] );
#undef M
#endif
	}
	if(!thread_id) contract_time += dclock() - ttime;
      }
    }//xop3d
  }//top

  print_time("ComputeKtoSigma","type4 vMv setup",vmv_setup_time);     
  print_time("ComputeKtoSigma","type4 pt1 compute",pt1_time);     
  print_time("ComputeKtoSigma","type4 pt2 compute",pt2_time);     
  print_time("ComputeKtoSigma","type4 contract",contract_time);  

  time = dclock();
  result.threadSum(); result.nodeSum();
  mix4.threadSum(); mix4.nodeSum();
  print_time("ComputeKtoSigma","type4 accum",dclock()-time);  

  print_time("ComputeKtoSigma","type4 total",dclock()-total_time);  
}
