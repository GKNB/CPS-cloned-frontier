#ifndef _COMPUTE_KTOPIPI_TYPE1_FIELD_H
#define _COMPUTE_KTOPIPI_TYPE1_FIELD_H


#define TIMER_ELEMS \
  ELEM(piK_mfproducts) \
  ELEM(part1) \
  ELEM(part2) \
  ELEM(total_contraction_time) \
  ELEM(contraction_time_geninputs_multgamma) \
  ELEM(contraction_time_geninputs_other) \
  ELEM(contraction_time_opcon) \
  ELEM(finish_up)\
  ELEM(total)
#define TIMER Type1FieldTimings
#include "static_timer_impl.tcc"


//Expect part1 and part2 to be length=2 vectors, the first with pi1 connected directly to the 4-quark op, and the second with pi2
template<typename mf_Policies>
void ComputeKtoPiPiGparity<mf_Policies>::type1_contract(ResultsContainerType &result, const int t_K, const std::vector<SCFmatrixField> &part1, const std::vector<SCFmatrixField> &part2){
#ifndef MEMTEST_MODE
  std::cout << "Starting K->pipi type 1 contractions with tK=" << t_K << std::endl;
  static const int n_contract = 6; //six type1 diagrams
  static const int con_off = 1; //index of first contraction in set
  auto dimpol = part1[0].getDimPolParams();
  SCFmatrixField G1_pt1(dimpol);
  SCFmatrixField G2_pt2(dimpol);
  SCFmatrixField ctrans_G2_pt2(dimpol);
  CPSmatrixField<CPScolorMatrix<ComplexType> > tr_sf_G1_pt1(dimpol);
  CPSmatrixField<CPScolorMatrix<ComplexType> > tr_sf_G2_pt2(dimpol);
  CPSmatrixField<CPSspinMatrix<CPSflavorMatrix<ComplexType> > > tr_c_G1_pt1(dimpol);
  CPSmatrixField<CPSspinMatrix<CPSflavorMatrix<ComplexType> > > tr_c_G2_pt2(dimpol);

  device_profile_start();
  
  for(int pt1_pion=0; pt1_pion<2; pt1_pion++){ //which pion is associated with part 1?
    int pt2_pion = (pt1_pion + 1) % 2;

    for(int mu=0;mu<4;mu++){ //sum over mu here
      for(int gcombidx=0;gcombidx<8;gcombidx++){

	Type1FieldTimings::timer().contraction_time_geninputs_multgamma -= dclock();     

	multGammaLeft(G1_pt1, part1[pt1_pion],1,gcombidx,mu);
	multGammaLeft(G2_pt2, part2[pt2_pion],2,gcombidx,mu);

	Type1FieldTimings::timer().contraction_time_geninputs_multgamma += dclock();     

	Type1FieldTimings::timer().contraction_time_geninputs_other -= dclock();     

	SpinFlavorTrace(tr_sf_G1_pt1, G1_pt1);
	SpinFlavorTrace(tr_sf_G2_pt2, G2_pt2);

	TransposeColor(ctrans_G2_pt2, G2_pt2);

	ColorTrace(tr_c_G1_pt1, G1_pt1);
	ColorTrace(tr_c_G2_pt2, G2_pt2);

	Type1FieldTimings::timer().contraction_time_geninputs_other += dclock();     

	Type1FieldTimings::timer().contraction_time_opcon -= dclock();     

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
	    Trace( tr_c_G1_pt1 , tr_c_G2_pt2 )
	    );

	Type1FieldTimings::timer().contraction_time_opcon += dclock();     
      }
    }
  }

  device_profile_stop();
  
  std::cout << "Finished K->pipi type 1 contractions with tK=" << t_K << std::endl;
#endif
}



template<typename mf_Policies>
void ComputeKtoPiPiGparity<mf_Policies>::type1_field_SIMD(ResultsContainerType result[],
						     const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const ThreeMomentum &p_pi_1, 
						     const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_Policies> &mf_pions,
						     const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
						     const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH){
  Type1FieldTimings::timer().reset();
  timerStart(Type1FieldTimings::timer().total,"Start");

  auto field_params = vL.getMode(0).getDimPolParams();  
  
  //Precompute mode mappings
  ModeContractionIndices<StandardIndexDilution,TimePackedIndexDilution> i_ind_vw(vL);
  ModeContractionIndices<StandardIndexDilution,FullyPackedIndexDilution> j_ind_vw(wL);
  ModeContractionIndices<TimePackedIndexDilution,StandardIndexDilution> j_ind_wv(vH);

  const int Lt = GJP.Tnodes()*GJP.TnodeSites();
  const int ntsep_k_pi = tsep_k_pi.size();
  int tsep_k_pi_largest = 0;
  for(int sep: tsep_k_pi) tsep_k_pi_largest = std::max(tsep_k_pi_largest, sep);
  
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

  //Determine which t_K and t_pi1 combinations this node contributes to given that t_op > t_K && t_op < t_pi1 
  std::map<int, std::vector<int> > map_used_tpi1_lin_to_tsep_k_pi;  //tpi1_lin -> vector of tsep_k_pi index

  std::stringstream ss;
  ss << "Node " << UniqueID() << " doing (t_K, t_pi1) = {";

  //for(int t_pi1 = 0; t_pi1 < Lt; t_pi1 += tstep){ //my sensible ordering
  for(int t_pi1_lin = 1; t_pi1_lin <= Lt; t_pi1_lin += tstep){ //Daiqian's weird ordering
    int t_pi1 = modLt(t_pi1_lin,Lt);
    int t_pi2 = modLt(t_pi1 + tsep_pion, Lt);
    
    //Using the pion timeslices, get tK for each separation
    for(int tkpi_idx=0;tkpi_idx<ntsep_k_pi;tkpi_idx++){
      int t_K = modLt(t_pi1 - tsep_k_pi[tkpi_idx], Lt);

      //Loop over timeslices on this node
      for(int top=GJP.TnodeCoor()*GJP.TnodeSites(); top < (GJP.TnodeCoor()+1)*GJP.TnodeSites(); top++){
	int t_dis = modLt(top - t_K, Lt);
	if(t_dis > 0 && t_dis < tsep_k_pi[tkpi_idx]){
	  pi1_tslice_mask[t_pi1] = true;
	  pi2_tslice_mask[t_pi2] = true;
	  map_used_tpi1_lin_to_tsep_k_pi[t_pi1_lin].push_back(tkpi_idx);
	  ss << " (" << t_K << "," << t_pi1 << ")";
	  break; //only need one timeslice in window
	}
      }
    }
  }
  ss << "}\n";
  a2a_printf("%s",ss.str().c_str());

  //Get the meson fields
#ifdef NODE_DISTRIBUTE_MESONFIELDS
  LOGA2A << "Memory prior to fetching meson fields type1 K->pipi:" << std::endl;
  printMem();
  nodeGetMany(2,
	      &mf_pi1,&pi1_tslice_mask,
	      &mf_pi2,&pi2_tslice_mask);
  LOGA2A << "Memory after fetching meson fields type1 K->pipi:" << std::endl;
  printMem();
#endif

  std::vector<SCFmatrixField> part1(2, SCFmatrixField(field_params)); //part1 goes from insertion to pi1, pi2 (x_4 = t_pi1, t_pi2)
  std::vector<SCFmatrixField> part2(2, SCFmatrixField(field_params)); //part2 has pi1, pi2 at y_4 = t_pi1, t_pi2
  mf_WW con_pi1_K, con_pi2_K;
  
  int titer = 0;
  for(auto t_it = map_used_tpi1_lin_to_tsep_k_pi.begin(); t_it != map_used_tpi1_lin_to_tsep_k_pi.end(); ++t_it, ++titer){
    _Type1FieldTimings ttimer; //timings for this loop iteration
    timerStart(ttimer.total, "Loop iteration start");

    int t_pi1_lin = t_it->first;
    
    int t_pi1 = modLt(t_pi1_lin,Lt);
    int t_pi2 = modLt(t_pi1 + tsep_pion, Lt);

    //Check the meson fields are available!
    assert(mf_pi1[t_pi1].isOnNode());
    assert(mf_pi2[t_pi2].isOnNode());

    //Get lowest value of t_K, i.e. from largest sep
    int tK_min = modLt(t_pi1 - tsep_k_pi_largest, Lt);
    
    //Compute part1 and part2
    //Construct part 1 (no dependence on t_K):
    //\Gamma_1 vL_i(x_op; x_4) [[\sum_{\vec x} wL_i^dag(x) S_2 vL_j(x;top)]] wL_j^dag(x_op)
    timerStart(ttimer.part1, "Part 1");
    //Only compute in window between kaon and inner pion to save computation
    mult(part1[0], vL, mf_pi1[t_pi1], wL, false, true, tK_min, tsep_k_pi_largest);
    mult(part1[1], vL, mf_pi2[t_pi2], wL, false, true, tK_min, tsep_k_pi_largest);
    timerEnd(ttimer.part1, "Part 1");

    for(int tkpi_idx : t_it->second){
      int t_K = modLt(t_pi1 - tsep_k_pi[tkpi_idx], Lt);     
      assert(mf_kaon[t_K].isOnNode());
      
      //Form contraction  con_pi_K(y_4, t_K) =   [[ wL_i^dag(y_4) S_2 vL_j(y_4) ]] [[ wL_j^dag(t_K) wH_k(t_K) ) ]]
      //y_4 = t_K + tsep_k_pi
      timerStart(ttimer.piK_mfproducts, "MFproducts");
      mult(con_pi1_K, mf_pi1[t_pi1], mf_kaon[t_K], true); //node local
      mult(con_pi2_K, mf_pi2[t_pi2], mf_kaon[t_K], true);
      timerEnd(ttimer.piK_mfproducts, "MFproducts");

      //Construct part 2:
      //\Gamma_2 vL_i(x_op;y_4) [[ wL_i^dag(y) S_2 vL_j(y;t_K) ]] [[ wL_j^dag(x_K)\gamma^5 \gamma^5 wH_k(x_K) ) ]] vH_k^\dagger(x_op;t_K)\gamma^5
      timerStart(ttimer.part2, "Part 2");
      mult(part2[0], vL, con_pi1_K, vH, false, true, t_K, tsep_k_pi[tkpi_idx]);
      mult(part2[1], vL, con_pi2_K, vH, false, true, t_K, tsep_k_pi[tkpi_idx]);
      gr(part2[0], -5); //right multiply by g5
      gr(part2[1], -5);
      timerEnd(ttimer.part2, "Part 2");

      timerStart(ttimer.total_contraction_time, "Contractions");
      type1_contract(result[tkpi_idx],t_K,part1,part2);
      timerEnd(ttimer.total_contraction_time, "Contractions");
    }//end of loop over k->pi seps

    //For development, timings during outer tpi1 loop
    timerEnd(ttimer.total, "Loop iteration end");
    LOGA2A << "Type 1 end of tpi1=" << t_pi1 << "(iter " << (titer+1) << "/" << map_used_tpi1_lin_to_tsep_k_pi.size() << ") report:" << std::endl;
    ttimer.report();

    ttimer.total = 0; Type1FieldTimings::timer() += ttimer;    
  }//tpi loop

  LOGA2A << "Memory before finishing up type1 K->pipi:" << std::endl;
  printMem();


  LOGA2A << "Type 1 finishing up results" << std::endl;
  timerStart(Type1FieldTimings::timer().finish_up, "Finish up");
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
  timerEnd(Type1FieldTimings::timer().finish_up, "Finish up");

  LOGA2A << "Memory after finishing up type1 K->pipi:" << std::endl;
  printMem();


#ifdef NODE_DISTRIBUTE_MESONFIELDS
  nodeDistributeMany(2,&mf_pi1,&mf_pi2);
  LOGA2A << "Memory after redistributing meson fields type1 K->pipi:" << std::endl;
  printMem();
#endif
  timerEnd(Type1FieldTimings::timer().total,"End");
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
    LOGA2A << "Type1 field implementation falling back to OMP implementation due to non-SIMD data" << std::endl;
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
