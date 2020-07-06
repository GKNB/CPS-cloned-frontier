#ifndef _COMPUTE_KTOPIPI_TYPE2_H
#define _COMPUTE_KTOPIPI_TYPE2_H

//TYPE 2
//Each contraction of this type is made up of different trace combinations of two objects (below [but not in the code!] for simplicity we ignore the fact that the two vectors in 
//the meson fields are allowed to vary in position relative to each other):
//1) \sum_{ \vec x_K  }  \Gamma_1 \prop^L(x_op,x_K) \gamma^5 \prop^H(x_K,x_op)    
//2) \sum_{ \vec y, \vec z  }  \Gamma_2 \prop^L(x_op,y) S_2 \prop^L(y,z) S_2 \prop^L(z,x_op)
    
//We use g5-hermiticity on the strange propagator
//1) -> \sum_{ \vec x_K  }  \Gamma_1 \prop^L(x_op,x_K)  [\prop^H(x_op,x_K)]^\dagger  \gamma^5
//    = \sum_{ \vec x_K  }  \Gamma_1 vL(x_op) wL^dag(x_K) [ vH(x_op) wH^dag(x_K) ]^dag \gamma^5
//    = \sum_{ \vec x_K  }  \Gamma_1 vL(x_op) [[ wL^dag(x_K) wH(x_K) ]] [vH(x_op)]^dag \gamma^5 
//  where [[ ]] indicate meson fields
    
//2) In terms of v and w
// \sum_{ \vec y, \vec z  }  \Gamma_2 vL(x_op) [[ wL^dag(y) S_2 vL(y) ]] [[ wL^dag(z) S_2 vL(z) ]] wL^dag(x_op)

//Run inside threaded environment
template<typename mf_Policies>
void ComputeKtoPiPiGparity<mf_Policies>::type2_contract(ResultsContainerType &result, const int t_K, const int t_dis, const int thread_id, const SCFmat &part1, const SCFmat part2[2]){
#ifndef MEMTEST_MODE
  static const int n_contract = 6; //six type2 diagrams
  static const int con_off = 7; //index of first contraction in set
  for(int mu=0;mu<4;mu++){ //sum over mu here
    for(int gcombidx=0;gcombidx<8;gcombidx++){
      SCFmat G1_pt1 = part1; //= G1*part1;
      multGammaLeft(G1_pt1,1,gcombidx,mu);

      CPScolorMatrix<ComplexType> tr_sf_G1_pt1 = G1_pt1.SpinFlavorTrace();
      
      for(int pt2_pion=0; pt2_pion<2; pt2_pion++){ //which pion comes first in part 2?
	SCFmat G2_pt2 = part2[pt2_pion]; //= G2*part2[pt2_pion];
	multGammaLeft(G2_pt2,2,gcombidx,mu);

	CPScolorMatrix<ComplexType> tr_sf_G2_pt2 = G2_pt2.SpinFlavorTrace();
		
	SCFmat ctrans_G2_pt2 = G2_pt2.TransposeColor();//speedup by transposing part 1
		
#define C(IDX) result(t_K,t_dis,IDX-con_off,gcombidx,thread_id)	      
	C(7) += G1_pt1.Trace() * G2_pt2.Trace();
	C(8) += Trace( tr_sf_G1_pt1 , Transpose(tr_sf_G2_pt2) );
	C(9) += Trace( tr_sf_G1_pt1 , tr_sf_G2_pt2 );
	C(10) += Trace( G1_pt1 , G2_pt2 );
	C(11) += Trace( G1_pt1, ctrans_G2_pt2 );
	C(12) += Trace( G1_pt1.ColorTrace() , G2_pt2.ColorTrace() );
#undef C	     
      }
    }
  }
#endif
}


template<typename mf_Policies>
void ComputeKtoPiPiGparity<mf_Policies>::type2_compute_mfproducts(std::vector<mf_WV > &con_pi1_pi2,
							       std::vector<mf_WV > &con_pi2_pi1,							     
							       const int tsep_pion, const int tstep, const std::vector<ThreeMomentum> &p_pi_1_all,
							       MesonFieldMomentumContainer<mf_Policies> &mf_pions,
							       const int Lt, const int tpi_sampled){
  Type2timings::timer().type2_compute_mfproducts -= dclock();
  con_pi1_pi2.resize(tpi_sampled); //y is associated with pi1, z with pi2
  con_pi2_pi1.resize(tpi_sampled); //y is associated with pi2, z with pi1

  mf_WV tmp;
    
  if(!UniqueID()){ printf("Computing con_*_*\n"); fflush(stdout); }

#ifndef DISABLE_TYPE2_PRODUCTSTORE
  //Some of these mults are quite likely duplicates, so use the product store to maximize reuse
  MesonFieldProductStore<mf_Policies> products;
#endif
  
  int nmom = p_pi_1_all.size();
  for(int pidx=0;pidx<nmom;pidx++){
    const ThreeMomentum &p_pi_1 = p_pi_1_all[pidx];
    ThreeMomentum p_pi_2 = -p_pi_1;
    
    std::vector<mf_WV > &mf_pi1 = mf_pions.get(p_pi_1);
    std::vector<mf_WV > &mf_pi2 = mf_pions.get(p_pi_2);
#ifdef NODE_DISTRIBUTE_MESONFIELDS
    std::vector<bool> pi1_tslice_mask(Lt,false);
    std::vector<bool> pi2_tslice_mask(Lt,false);
    
    for(int t_pi1_lin = 1; t_pi1_lin <= Lt; t_pi1_lin += tstep){ //Daiqian's weird ordering
      int tpi1 = modLt(t_pi1_lin,Lt);
      int tpi2 = modLt(tpi1  + tsep_pion, Lt);
      pi1_tslice_mask[tpi1] = true;
      pi2_tslice_mask[tpi2] = true;
    }    
    nodeGetMany(2,&mf_pi1,&pi1_tslice_mask, &mf_pi2, &pi2_tslice_mask);
#endif

    //for(int tpi1=0;tpi1<Lt;tpi1 += tstep){ //my sensible ordering
    for(int t_pi1_lin = 1; t_pi1_lin <= Lt; t_pi1_lin += tstep){ //Daiqian's weird ordering
      int tpi1 = modLt(t_pi1_lin,Lt);

      int tpi1_idx = tpi1 / tstep;
      int tpi2 = modLt(tpi1  + tsep_pion, Lt);

      if(pidx==0){
#ifndef DISABLE_TYPE2_PRODUCTSTORE
	con_pi1_pi2[tpi1_idx] = products.getProduct(mf_pi1[tpi1], mf_pi2[tpi2]); //node distributed
	con_pi2_pi1[tpi1_idx] = products.getProduct(mf_pi2[tpi2], mf_pi1[tpi1]);
#else
	mult(con_pi1_pi2[tpi1_idx], mf_pi1[tpi1], mf_pi2[tpi2]);
	mult(con_pi2_pi1[tpi1_idx], mf_pi2[tpi2], mf_pi1[tpi1]);
#endif
      }else{	
#ifndef DISABLE_TYPE2_PRODUCTSTORE
	tmp = products.getProduct(mf_pi1[tpi1], mf_pi2[tpi2]);
#else
	mult(tmp, mf_pi1[tpi1], mf_pi2[tpi2]);
#endif
	con_pi1_pi2[tpi1_idx].plus_equals(tmp, true);

#ifndef DISABLE_TYPE2_PRODUCTSTORE	
	tmp = products.getProduct(mf_pi2[tpi2], mf_pi1[tpi1]);
#else
	mult(tmp, mf_pi2[tpi2], mf_pi1[tpi1]);   
#endif
	con_pi2_pi1[tpi1_idx].plus_equals(tmp, true);
      }
      //NB time coordinate of con_*_* is the time coordinate of pi1 (that closest to the kaon)
    }
#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeDistributeMany(2,&mf_pi1,&mf_pi2);
#endif
  }
  if(nmom > 1)
    for(int t=0;t<tpi_sampled;t++){
      con_pi1_pi2[t].times_equals(1./nmom);  con_pi2_pi1[t].times_equals(1./nmom);
    }
  Type2timings::timer().type2_compute_mfproducts += dclock();
}





template<typename mf_Policies>
void ComputeKtoPiPiGparity<mf_Policies>::type2_mult_vMv_setup(std::vector<vMv_split_VWWV> &mult_vMv_split_part1,
							      std::vector<vMv_split_VWVW> &mult_vMv_split_part2_pi1_pi2,
							      std::vector<vMv_split_VWVW> &mult_vMv_split_part2_pi2_pi1,
							      const std::vector< mf_WV > &con_pi1_pi2,
							      const std::vector< mf_WV > &con_pi2_pi1,
							      const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, const A2AvectorW<mf_Policies> & wL,
							      const std::vector<mf_WW > &mf_kaon,
							      const std::vector<int> &t_K_all, const int top_loc, const int tstep, const int Lt,const int tpi_sampled,
							      const std::vector< std::vector<bool> > &node_top_used, const std::vector< std::vector<bool> > &node_top_used_kaon){
  Type2timings::timer().type2_mult_vMv_setup -= dclock();
  //Split the vector-mesonfield outer product into two stages where in the first we reorder the mesonfield to optimize cache hits
  mult_vMv_split_part1.resize(t_K_all.size());
  mult_vMv_split_part2_pi1_pi2.resize(tpi_sampled);
  mult_vMv_split_part2_pi2_pi1.resize(tpi_sampled);

  int top_glb = top_loc  + GJP.TnodeCoor()*GJP.TnodeSites();

  {
  //Part 1
#pragma omp parallel for
  for(int tkidx=0; tkidx < t_K_all.size(); tkidx++){
    if(!node_top_used_kaon[tkidx][top_loc]) continue;
    int t_K = t_K_all[tkidx];
    mult_vMv_split_part1[tkidx].setup(vL,mf_kaon[t_K],vH,top_glb);
  }
  }

  {
  //Part 2
#pragma omp parallel for
  for(int t_pi1_lin = 1; t_pi1_lin <= Lt; t_pi1_lin += tstep){ //Daiqian's weird ordering
    int t_pi1 = modLt(t_pi1_lin,Lt);
    int t_pi1_idx = t_pi1 / tstep;
    if(!node_top_used[t_pi1_idx][top_loc]) continue; //can be better parallelized!
    
    mult_vMv_split_part2_pi1_pi2[t_pi1_idx].setup(vL,con_pi1_pi2[t_pi1_idx],wL, top_glb);
    mult_vMv_split_part2_pi2_pi1[t_pi1_idx].setup(vL,con_pi2_pi1[t_pi1_idx],wL, top_glb);
  }
  }
  Type2timings::timer().type2_mult_vMv_setup += dclock();
}

template<typename mf_Policies>
void ComputeKtoPiPiGparity<mf_Policies>::type2_precompute_part1_part2(std::vector<SCFmatVector > &mult_vMv_contracted_part1,
								      std::vector<SCFmatVector > &mult_vMv_contracted_part2_pi1_pi2,
								      std::vector<SCFmatVector > &mult_vMv_contracted_part2_pi2_pi1,
								      std::vector<vMv_split_VWWV> &mult_vMv_split_part1,
								      std::vector<vMv_split_VWVW> &mult_vMv_split_part2_pi1_pi2,
								      std::vector<vMv_split_VWVW > &mult_vMv_split_part2_pi2_pi1,
								      const std::vector<int> &t_K_all, const int top_loc, const int tstep, const int Lt,const int tpi_sampled,
								      const std::vector< std::vector<bool> > &node_top_used, const std::vector< std::vector<bool> > &node_top_used_kaon){
  Type2timings::timer().type2_precompute_part1_part2 -= dclock();
  mult_vMv_contracted_part1.resize(t_K_all.size());
  mult_vMv_contracted_part2_pi1_pi2.resize(tpi_sampled);
  mult_vMv_contracted_part2_pi2_pi1.resize(tpi_sampled);

  for(int tkidx=0; tkidx < t_K_all.size(); tkidx++){
    if(!node_top_used_kaon[tkidx][top_loc]) continue;
    int t_K = t_K_all[tkidx];
    mult_vMv_split_part1[tkidx].contract(mult_vMv_contracted_part1[tkidx], false, true);
    mult_vMv_split_part1[tkidx].free_mem();
  }

  for(int t_pi1_lin = 1; t_pi1_lin <= Lt; t_pi1_lin += tstep){ //Daiqian's weird ordering
    int t_pi1 = modLt(t_pi1_lin,Lt);
    int t_pi1_idx = t_pi1 / tstep;
    if(!node_top_used[t_pi1_idx][top_loc]) continue; 
    
    mult_vMv_split_part2_pi1_pi2[t_pi1_idx].contract(mult_vMv_contracted_part2_pi1_pi2[t_pi1_idx], false,true);
    mult_vMv_split_part2_pi1_pi2[t_pi1_idx].free_mem();
    
    mult_vMv_split_part2_pi2_pi1[t_pi1_idx].contract(mult_vMv_contracted_part2_pi2_pi1[t_pi1_idx], false,true);
    mult_vMv_split_part2_pi2_pi1[t_pi1_idx].free_mem();
  }
  Type2timings::timer().type2_precompute_part1_part2 += dclock();
}


//This version averages over multiple pion momentum configurations. Use to project onto A1 representation at run-time. Saves a lot of time!
//This version also overlaps computation for multiple K->pi separations. Result should be an array of ResultsContainerType the same size as the vector 'tsep_k_pi'
template<typename mf_Policies>
void ComputeKtoPiPiGparity<mf_Policies>::type2_v1(ResultsContainerType result[],
		  const std::vector<int> &tsep_k_pi, const int &tsep_pion, const int &tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
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
  const int nthread = omp_get_max_threads();    
  const int size_3d = vL.getMode(0).nodeSites(0)*vL.getMode(0).nodeSites(1)*vL.getMode(0).nodeSites(2);

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

  //Form the product of the two meson fields
  //con_*_* = \sum_{\vec y,\vec z} [[ wL^dag(y) S_2 vL(y) ]] [[ wL^dag(z) S_2 vL(z) ]]
  std::vector< mf_WV > con_pi1_pi2;//(tpi_sampled); //y is associated with pi1, z with pi2
  std::vector< mf_WV > con_pi2_pi1; //(tpi_sampled); //y is associated with pi2, z with pi1
  type2_compute_mfproducts(con_pi1_pi2,con_pi2_pi1,tsep_pion,tstep,p_pi_1_all,mf_pions, Lt, tpi_sampled);

  for(int tkp=0;tkp<tsep_k_pi.size();tkp++)
    result[tkp].resize(n_contract,nthread); //Resize zeroes output. Result will be thread-reduced before this method ends 

  for(int top_loc = 0; top_loc < GJP.TnodeSites(); top_loc++){
    const int top_glb = top_loc  + GJP.TnodeCoor()*GJP.TnodeSites();

#ifndef DISABLE_TYPE2_SPLIT_VMV
    //Split the vector-mesonfield outer product into two stages where in the first we reorder the mesonfield to optimize cache hits
    std::vector<vMv_split_VWWV > mult_vMv_split_part1; //[t_K_all.size()];

    std::vector<vMv_split_VWVW > mult_vMv_split_part2_pi1_pi2; //[tpi_sampled];
    std::vector<vMv_split_VWVW > mult_vMv_split_part2_pi2_pi1; //[tpi_sampled];
    type2_mult_vMv_setup(mult_vMv_split_part1,mult_vMv_split_part2_pi1_pi2,mult_vMv_split_part2_pi2_pi1,con_pi1_pi2,con_pi2_pi1,vL,vH,wL,mf_kaon,t_K_all,top_loc,tstep, Lt,tpi_sampled,node_top_used,node_top_used_kaon);

# ifndef DISABLE_TYPE2_PRECOMPUTE
    //Contract on all 3d sites on this node with fixed operator time coord top_glb into a canonically ordered output vector
    std::vector<SCFmatVector > mult_vMv_contracted_part1; //[t_K_all.size()][x3d];
    
    std::vector<SCFmatVector > mult_vMv_contracted_part2_pi1_pi2; //[tpi_sampled][x3d];
    std::vector<SCFmatVector > mult_vMv_contracted_part2_pi2_pi1; //[tpi_sampled][x3d];
    
    type2_precompute_part1_part2(mult_vMv_contracted_part1,mult_vMv_contracted_part2_pi1_pi2,mult_vMv_contracted_part2_pi2_pi1,
				 mult_vMv_split_part1,mult_vMv_split_part2_pi1_pi2,mult_vMv_split_part2_pi2_pi1,
				 t_K_all,top_loc,tstep,Lt,tpi_sampled,node_top_used,node_top_used_kaon);
# endif
#endif

    //Now loop over Q_i insertion location. Each node naturally has its own sublattice to work on. Thread over sites in usual way
    Type2timings::timer().contraction_time -= dclock();
#pragma omp parallel for schedule(static)
    for(int xop3d_loc = 0; xop3d_loc < size_3d; xop3d_loc++){
      int thread_id = omp_get_thread_num();

      //Part 1 does not care about the location of the pion, only that of the kaon. It may be used multiple times if we have multiple K->pi seps, so compute it separately.
      SCFmatVector part1_storage(t_K_all.size());
      for(int tkidx=0; tkidx < t_K_all.size(); tkidx++){
	if(!node_top_used_kaon[tkidx][top_loc]) continue;
	int t_K = t_K_all[tkidx];

	//Compute part 1	  
	//    = \sum_{ \vec x_K  }  \Gamma_1 vL(x_op) [[ wL^dag(x_K) wH(x_K) ]] [vH(x_op)]^dag \gamma^5 
	SCFmat &part1 = part1_storage[tkidx];
#if defined(DISABLE_TYPE2_SPLIT_VMV)
	mult(part1, vL, mf_kaon[t_K], vH, xop3d_loc, top_loc, false, true);
#elif defined(DISABLE_TYPE2_PRECOMPUTE)
	mult_vMv_split_part1[tkidx].contract(part1,xop3d_loc,false,true);
#else
	part1 = mult_vMv_contracted_part1[tkidx][xop3d_loc];
#endif
	part1.gr(-5); //right multiply by g5
      }
    
      //for(int t_pi1 = 0; t_pi1 < Lt; t_pi1 += tstep){ //my sensible ordering
      for(int t_pi1_lin = 1; t_pi1_lin <= Lt; t_pi1_lin += tstep){ //Daiqian's weird ordering
	int t_pi1 = modLt(t_pi1_lin,Lt);
	int t_pi1_idx = t_pi1 / tstep;

	int t_pi2 = modLt(t_pi1 + tsep_pion, Lt);

	if(!node_top_used[t_pi1_idx][top_loc]) continue; //skip unused timeslices
      
	//Construct part 2 (this doesn't involve the kaon):
	// \sum_{ \vec y, \vec z  }  \Gamma_2 vL(x_op) [[ wL^dag(y) S_2 vL(y) ]] [[ wL^dag(z) S_2 vL(z) ]] wL^dag(x_op)
	SCFmat part2[2]; 

#if defined(DISABLE_TYPE2_SPLIT_VMV)
	mult(part2[0], vL, con_pi1_pi2[t_pi1_idx], wL, xop3d_loc, top_loc, false, true); //part2 goes from insertion to pi1 to pi2 and back to insertion
	mult(part2[1], vL, con_pi2_pi1[t_pi1_idx], wL, xop3d_loc, top_loc, false, true); //part2 goes from insertion to pi2 to pi1 and back to insertion
#elif defined(DISABLE_TYPE2_PRECOMPUTE)
	mult_vMv_split_part2_pi1_pi2[t_pi1_idx].contract(part2[0],xop3d_loc,false,true);
	mult_vMv_split_part2_pi2_pi1[t_pi1_idx].contract(part2[1],xop3d_loc,false,true);
#else
	part2[0] = mult_vMv_contracted_part2_pi1_pi2[t_pi1_idx][xop3d_loc];
	part2[1] = mult_vMv_contracted_part2_pi2_pi1[t_pi1_idx][xop3d_loc];
#endif

	for(int tkpi_idx = 0; tkpi_idx < tsep_k_pi.size(); tkpi_idx++){
	  int t_K = modLt(t_pi1 - tsep_k_pi[tkpi_idx], Lt);
	  int t_dis = modLt(top_glb - t_K, Lt); //distance between kaon and operator is the output time coordinate

	  if(t_dis >= tsep_k_pi[tkpi_idx] || t_dis == 0) continue; //don't bother computing operator insertion locations outside of the region between the kaon and first pion or on top of either operator
	  
	  const SCFmat &part1 = part1_storage[tkidx_map[t_K]];
	  type2_contract(result[tkpi_idx],t_K,t_dis,thread_id,part1,part2);
	}

      }//tpi1 loop
    }//xop3d_loc loop
    Type2timings::timer().contraction_time += dclock();
    
  }//top_loc loop

  Type2timings::timer().finish_up -= dclock();
  for(int tkp=0;tkp<tsep_k_pi.size();tkp++){
    result[tkp].threadSum();
    result[tkp].nodeSum();
#ifndef DAIQIAN_COMPATIBILITY_MODE
    result[tkp] *= Float(0.5); //coefficient of 0.5 associated with average over pt2 pion ordering
#endif
  }
  Type2timings::timer().finish_up += dclock();
  Type2timings::timer().total += dclock();
  Type2timings::timer().report();
}







template<typename mf_Policies>
void ComputeKtoPiPiGparity<mf_Policies>::type2_compute_mfproducts(mf_WV &con_pi1_pi2,
								  mf_WV &con_pi2_pi1,
								  const int tpi1, const int tpi2,
								  const std::vector<ThreeMomentum> &p_pi_1_all,
								  MesonFieldMomentumContainer<mf_Policies> &mf_pions){
  Type2timings::timer().type2_compute_mfproducts -= dclock();

  mf_WV tmp;
    
  if(!UniqueID()){ printf("Computing con_*_* for tpi1=%d\n",tpi1); fflush(stdout); }
 
#ifdef NODE_DISTRIBUTE_MESONFIELDS
  void *gather_buf_1, *gather_buf_2;
  size_t gather_buf_sz;
#endif

  int nmom = p_pi_1_all.size();
  for(int pidx=0;pidx<nmom;pidx++){
    const ThreeMomentum &p_pi_1 = p_pi_1_all[pidx];
    ThreeMomentum p_pi_2 = -p_pi_1;
    
    std::vector<mf_WV > &mf_pi1 = mf_pions.get(p_pi_1);
    std::vector<mf_WV > &mf_pi2 = mf_pions.get(p_pi_2);

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    if(pidx == 0){
      gather_buf_sz = mf_pi1[tpi1].byte_size();
      gather_buf_1 = memalign_check(128,gather_buf_sz);
      gather_buf_2 = memalign_check(128,gather_buf_sz);
    }
    mf_pi1[tpi1].enableExternalBuffer(gather_buf_1,gather_buf_sz,128);
    mf_pi2[tpi2].enableExternalBuffer(gather_buf_2,gather_buf_sz,128);
    
    mf_pi1[tpi1].nodeGet();
    mf_pi2[tpi2].nodeGet();
#endif

    if(pidx==0){
      mult(con_pi1_pi2, mf_pi1[tpi1], mf_pi2[tpi2]);
      mult(con_pi2_pi1, mf_pi2[tpi2], mf_pi1[tpi1]);
    }else{	
      mult(tmp, mf_pi1[tpi1], mf_pi2[tpi2]);
#ifndef MEMTEST_MODE
      con_pi1_pi2.plus_equals(tmp, true);
#endif

      mult(tmp, mf_pi2[tpi2], mf_pi1[tpi1]);   

#ifndef MEMTEST_MODE
      con_pi2_pi1.plus_equals(tmp, true);
#endif
    }

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    mf_pi1[tpi1].nodeDistribute();
    mf_pi2[tpi2].nodeDistribute();
    mf_pi1[tpi1].disableExternalBuffer();
    mf_pi2[tpi2].disableExternalBuffer();
#endif
  }
  
#ifdef NODE_DISTRIBUTE_MESONFIELDS
  free(gather_buf_1); free(gather_buf_2);
#endif

  if(nmom > 1){
#ifndef MEMTEST_MODE
    con_pi1_pi2.times_equals(1./nmom);  con_pi2_pi1.times_equals(1./nmom);
#endif
  }
  Type2timings::timer().type2_compute_mfproducts += dclock();
}


template<typename mf_Policies>
void ComputeKtoPiPiGparity<mf_Policies>::type2_v2(ResultsContainerType result[],
		  const std::vector<int> &tsep_k_pi, const int &tsep_pion, const int &tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
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
  const int nthread = omp_get_max_threads();    
  const int size_3d = vL.getMode(0).nodeSites(0)*vL.getMode(0).nodeSites(1)*vL.getMode(0).nodeSites(2);

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
    result[tkp].resize(n_contract,nthread); //Resize zeroes output. Result will be thread-reduced before this method ends 

  //Part 1 does not care about the location of the pion, only that of the kaon. It may be used multiple times if we have multiple K->pi seps, so compute it separately
  SCFmatVector part1_storage[t_K_all.size()][GJP.TnodeSites()];  //[tkidx][top_loc][xop3d_loc]
  for(int tkidx=0;tkidx<t_K_all.size();tkidx++)
    for(int t=0;t<GJP.TnodeSites();t++)
      part1_storage[tkidx][t].resize(size_3d);

  for(int top_loc = 0; top_loc < GJP.TnodeSites(); top_loc++){
    const int top_glb = top_loc  + GJP.TnodeCoor()*GJP.TnodeSites();

    for(int tkidx=0; tkidx < t_K_all.size(); tkidx++){
      if(!node_top_used_kaon[tkidx][top_loc]) continue;
      int t_K = t_K_all[tkidx];
      
#ifndef DISABLE_TYPE2_SPLIT_VMV
      vMv_split_VWWV mult_vMv_split_part1;
      mult_vMv_split_part1.setup(vL, mf_kaon[t_K], vH, top_glb);
#endif

#pragma omp parallel for schedule(static)
      for(int xop3d_loc = 0; xop3d_loc < size_3d; xop3d_loc++){
	int thread_id = omp_get_thread_num();
	
	//Compute part 1	  
	//    = \sum_{ \vec x_K  }  \Gamma_1 vL(x_op) [[ wL^dag(x_K) wH(x_K) ]] [vH(x_op)]^dag \gamma^5 
	SCFmat &part1 = part1_storage[tkidx][top_loc][xop3d_loc];
#ifdef DISABLE_TYPE2_SPLIT_VMV
	mult(part1, vL, mf_kaon[t_K], vH, xop3d_loc, top_loc, false, true);
#else
	mult_vMv_split_part1.contract(part1,xop3d_loc,false,true);
#endif
	part1.gr(-5); //right multiply by g5
      }//xop3d_loc
    }//tkidx
  }//top_loc

  mf_WV con_pi1_pi2, con_pi2_pi1;

  //for(int t_pi1 = 0; t_pi1 < Lt; t_pi1 += tstep){ //my sensible ordering
  for(int t_pi1_lin = 1; t_pi1_lin <= Lt; t_pi1_lin += tstep){ //Daiqian's weird ordering
    int t_pi1 = modLt(t_pi1_lin,Lt);
    int t_pi1_idx = t_pi1 / tstep;
    
    int t_pi2 = modLt(t_pi1 + tsep_pion, Lt);
    
    //Form the product of the two meson fields
    //con_*_* = \sum_{\vec y,\vec z} [[ wL^dag(y) S_2 vL(y) ]] [[ wL^dag(z) S_2 vL(z) ]]
    type2_compute_mfproducts(con_pi1_pi2, con_pi2_pi1, t_pi1, t_pi2, p_pi_1_all, mf_pions);

    for(int top_loc = 0; top_loc < GJP.TnodeSites(); top_loc++){
      if(!node_top_used[t_pi1_idx][top_loc]) continue; //skip unused timeslices

      const int top_glb = top_loc  + GJP.TnodeCoor()*GJP.TnodeSites();
      
#ifndef DISABLE_TYPE2_SPLIT_VMV
      //Split the vector-mesonfield outer product into two stages where in the first we reorder the mesonfield to optimize cache hits    
      Type2timings::timer().type2_mult_vMv_setup -= dclock();
      vMv_split_VWVW mult_vMv_split_part2_pi1_pi2;
      vMv_split_VWVW mult_vMv_split_part2_pi2_pi1;

      mult_vMv_split_part2_pi1_pi2.setup(vL,con_pi1_pi2,wL, top_glb);
      mult_vMv_split_part2_pi2_pi1.setup(vL,con_pi2_pi1,wL, top_glb);
      Type2timings::timer().type2_mult_vMv_setup += dclock();
#endif

    //Now loop over Q_i insertion location. Each node naturally has its own sublattice to work on. Thread over sites in usual way
      Type2timings::timer().contraction_time -= dclock();
#pragma omp parallel for schedule(static)
      for(int xop3d_loc = 0; xop3d_loc < size_3d; xop3d_loc++){
	int thread_id = omp_get_thread_num();
   
	//Construct part 2 (this doesn't involve the kaon):
	// \sum_{ \vec y, \vec z  }  \Gamma_2 vL(x_op) [[ wL^dag(y) S_2 vL(y) ]] [[ wL^dag(z) S_2 vL(z) ]] wL^dag(x_op)
	SCFmat part2[2];

#ifdef DISABLE_TYPE2_SPLIT_VMV
	mult(part2[0], vL, con_pi1_pi2, wL, xop3d_loc, top_loc, false, true); //part2 goes from insertion to pi1 to pi2 and back to insertion
	mult(part2[1], vL, con_pi2_pi1, wL, xop3d_loc, top_loc, false, true); //part2 goes from insertion to pi2 to pi1 and back to insertion
#else
	mult_vMv_split_part2_pi1_pi2.contract(part2[0],xop3d_loc,false,true);
	mult_vMv_split_part2_pi2_pi1.contract(part2[1],xop3d_loc,false,true);
#endif

	for(int tkpi_idx = 0; tkpi_idx < tsep_k_pi.size(); tkpi_idx++){
	  int t_K = modLt(t_pi1 - tsep_k_pi[tkpi_idx], Lt);
	  int t_dis = modLt(top_glb - t_K, Lt); //distance between kaon and operator is the output time coordinate
	
	  if(t_dis >= tsep_k_pi[tkpi_idx] || t_dis == 0) continue; //don't bother computing operator insertion locations outside of the region between the kaon and first pion or on top of either operator
	
	  const SCFmat &part1 = part1_storage[tkidx_map[t_K]][top_loc][xop3d_loc];
	  type2_contract(result[tkpi_idx],t_K,t_dis,thread_id,part1,part2);
	}
      
      }//xop3d

      Type2timings::timer().contraction_time += dclock();
    
    }//top_loc loop
  }//tpi1_lin

  Type2timings::timer().finish_up -= dclock();
  for(int tkp=0;tkp<tsep_k_pi.size();tkp++){
    result[tkp].threadSum();
    result[tkp].nodeSum();
#ifndef DAIQIAN_COMPATIBILITY_MODE
    result[tkp] *= Float(0.5); //coefficient of 0.5 associated with average over pt2 pion ordering
#endif
  }
  Type2timings::timer().finish_up += dclock();
  Type2timings::timer().total += dclock();
  Type2timings::timer().report();
}



#endif
