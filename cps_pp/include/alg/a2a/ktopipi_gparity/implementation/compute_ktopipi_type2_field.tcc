#ifndef _COMPUTE_KTOPIPI_TYPE2_FIELD_H
#define _COMPUTE_KTOPIPI_TYPE2_FIELD_H

#define TIMER_ELEMS \
  ELEM(mfproducts) \
  ELEM(part1) \
  ELEM(part2) \
  ELEM(contraction_time) \
  ELEM(finish_up)\
  ELEM(total)
#define TIMER Type2FieldTimings
#include "static_timer_impl.tcc"


//Expect part2 to be a length=2 vector where for the first elements, pi1 enters first in part2, and in the second element, pi2
template<typename mf_Policies>
void ComputeKtoPiPiGparity<mf_Policies>::type2_contract(ResultsContainerType &result, const int t_K, const SCFmatrixField &part1, const std::vector<SCFmatrixField> &part2){
#ifndef MEMTEST_MODE
  static const int n_contract = 6; //six type2 diagrams
  static const int con_off = 7; //index of first contraction in set

  auto dimpol = part1.getDimPolParams();
  SCFmatrixField G1_pt1(dimpol);
  SCFmatrixField G2_pt2(dimpol);
  SCFmatrixField ctrans_G2_pt2(dimpol);
  
  CPSmatrixField<CPScolorMatrix<ComplexType> > tr_sf_G1_pt1(dimpol);
  CPSmatrixField<CPScolorMatrix<ComplexType> > tr_sf_G2_pt2(dimpol);

  CPSmatrixField<CPSspinMatrix<CPSflavorMatrix<ComplexType> > > tr_c_G1_pt1(dimpol);
  CPSmatrixField<CPSspinMatrix<CPSflavorMatrix<ComplexType> > > tr_c_G2_pt2(dimpol);
  
  for(int mu=0;mu<4;mu++){ //sum over mu here
    for(int gcombidx=0;gcombidx<8;gcombidx++){
      multGammaLeft(G1_pt1,part1,1,gcombidx,mu);
      
      SpinFlavorTrace(tr_sf_G1_pt1,G1_pt1);
      ColorTrace(tr_c_G1_pt1,G1_pt1);
      
      for(int pt2_pion=0; pt2_pion<2; pt2_pion++){ //which pion comes first in part 2?
	multGammaLeft(G2_pt2,part2[pt2_pion],2,gcombidx,mu);
	
	SpinFlavorTrace(tr_sf_G2_pt2,G2_pt2);	
	TransposeColor(ctrans_G2_pt2,G2_pt2);//speedup by transposing part 1
	ColorTrace(tr_c_G2_pt2,G2_pt2);

	
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
	  Trace( tr_c_G1_pt1 , tr_c_G2_pt2 )
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
  Type2FieldTimings::timer().reset();
  timerStart(Type2FieldTimings::timer().total,"Start");
    
  const int Lt = GJP.Tnodes()*GJP.TnodeSites();
  assert(Lt % tstep == 0);
  const int tpi_sampled = Lt/tstep;

  static const int n_contract = 6; //six type2 diagrams
  static const int con_off = 7; //index of first contraction in set

  for(int tkp=0;tkp<tsep_k_pi.size();tkp++)
    result[tkp].resize(n_contract); //Resize zeroes output
  
  auto field_params = vL.getMode(0).getDimPolParams();  

  //Meson fields
  int nmom = p_pi_1_all.size();
  std::vector< std::vector<mf_WV >* > mf_pi1(nmom), mf_pi2(nmom);
  for(int i=0;i<nmom;i++){
    const ThreeMomentum &p_pi_1 = p_pi_1_all[i];
    ThreeMomentum p_pi_2 = -p_pi_1;
    mf_pi1[i] = &mf_pions.get(p_pi_1);
    mf_pi2[i] = &mf_pions.get(p_pi_2);
  }
   
  //Determine which t_K and t_pi1 combinations this node contributes to given that t_op > t_K && t_op < t_pi1 
  int ntsep_k_pi = tsep_k_pi.size();
  std::map<int, std::vector<int> > map_used_tpi1_lin_to_tsep_k_pi;  //tpi1_lin -> vector of tsep_k_pi index
  std::set<int> t_K_all;
  std::vector<bool> pi1_tslice_mask(Lt,false), pi2_tslice_mask(Lt,false);

  int tsep_k_pi_largest = 0;
  for(int sep: tsep_k_pi) tsep_k_pi_largest = std::max(tsep_k_pi_largest, sep);
  
  std::stringstream ss;
  ss << "Node " << UniqueID() << " doing (t_K, t_pi1) = {";

  //for(int t_pi1 = 0; t_pi1 < Lt; t_pi1 += tstep){ //my sensible ordering
  for(int t_pi1_lin = 1; t_pi1_lin <= Lt; t_pi1_lin += tstep){ //Daiqian's weird ordering
    int t_pi1 = modLt(t_pi1_lin,Lt);
    int t_pi2 = modLt(t_pi1 + tsep_pion, Lt);

    //Using the pion timeslices, get tK for each separation
    for(int tkpi_idx=0;tkpi_idx<ntsep_k_pi;tkpi_idx++){
      int t_K = modLt(t_pi1 - tsep_k_pi[tkpi_idx], Lt);
      
      for(int top=GJP.TnodeCoor()*GJP.TnodeSites(); top < (GJP.TnodeCoor()+1)*GJP.TnodeSites(); top++){
	int t_dis = modLt(top - t_K, Lt);
	if(t_dis > 0 && t_dis < tsep_k_pi[tkpi_idx]){
	  map_used_tpi1_lin_to_tsep_k_pi[t_pi1_lin].push_back(tkpi_idx);
	  t_K_all.insert(t_K);
	  pi1_tslice_mask[t_pi1] = true;
	  pi2_tslice_mask[t_pi2] = true;
	  ss << " (" << t_K << "," << t_pi1 << ")";
	  break; //only need one timeslice in window
	}
      }
    }
  }
  ss << "}\n";
  printf("%s",ss.str().c_str());

  //Gather the meson fields we need
#ifdef NODE_DISTRIBUTE_MESONFIELDS
  for(int i=0;i<nmom;i++)
    nodeGetMany(2,mf_pi1[i],&pi1_tslice_mask, mf_pi2[i], &pi2_tslice_mask);
#endif

  //Compute part 1	  
  //    = \sum_{ \vec x_K  }   vL(x_op) [[ wL^dag(x_K) wH(x_K) ]] [vH(x_op)]^dag \gamma^5 
  //Part 1 does not care about the location of the pion, only that of the kaon. It may be used multiple times if we have multiple K->pi seps, so compute it separately
  timerStart(Type2FieldTimings::timer().part1,"Part 1");
  std::map<int,std::unique_ptr<SCFmatrixField> > part1_storage;
  for(auto it=t_K_all.begin(); it != t_K_all.end(); ++it){
    int t_K = *it;
    part1_storage[t_K].reset(new SCFmatrixField(field_params));
    SCFmatrixField &into = *part1_storage[t_K];
    assert(mf_kaon[t_K].isOnNode());

    //Get maximum t_pi1 given t_K
    //int t_pi1_max = modLt(t_K + tsep_k_pi_largest, Lt);    
    
    mult(into, vL, mf_kaon[t_K], vH, false, true, t_K, tsep_k_pi_largest);
    gr(into, -5);
  }
  timerEnd(Type2FieldTimings::timer().part1,"Part 1");
  
  //Compute pi1<->pi2 contractions
  mf_WV con_pi1_pi2, con_pi2_pi1, tmp;
  std::vector<SCFmatrixField> part2(2, SCFmatrixField(field_params));

  for(auto t_pair : map_used_tpi1_lin_to_tsep_k_pi){
    int t_pi1_lin = t_pair.first;
    
    int t_pi1 = modLt(t_pi1_lin,Lt);
    int t_pi2 = modLt(t_pi1 + tsep_pion, Lt);
    
    //Form the product of the two meson fields
    //con_*_* = \sum_{\vec y,\vec z} [[ wL^dag(y) S_2 vL(y) ]] [[ wL^dag(z) S_2 vL(z) ]]
    //Average over all momenta provided (i.e. to the A1 projection)
    timerStart(Type2FieldTimings::timer().mfproducts,"MFproducts");
    mult(con_pi1_pi2, mf_pi1[0]->at(t_pi1), mf_pi2[0]->at(t_pi2), true); //node local
    mult(con_pi2_pi1, mf_pi2[0]->at(t_pi2), mf_pi1[0]->at(t_pi1), true);
    for(int pp=1;pp<nmom;pp++){
      mult(tmp, mf_pi1[pp]->at(t_pi1), mf_pi2[pp]->at(t_pi2), true); con_pi1_pi2.plus_equals(tmp); //pp??
      mult(tmp, mf_pi2[pp]->at(t_pi2), mf_pi1[pp]->at(t_pi1), true); con_pi2_pi1.plus_equals(tmp);
    }
    if(nmom > 1){
      con_pi1_pi2.times_equals(1./nmom);  con_pi2_pi1.times_equals(1./nmom);
    }
    timerEnd(Type2FieldTimings::timer().mfproducts,"MFproducts");

    //Construct part 2 (this doesn't involve the kaon):
    // \sum_{ \vec y, \vec z  }  vL(x_op) [[ wL^dag(y) S_2 vL(y) ]] [[ wL^dag(z) S_2 vL(z) ]] wL^dag(x_op)
    int tK_min = modLt(t_pi1 - tsep_k_pi_largest, Lt); //only compute for timeslices between kaon and inner pion
    
    timerStart(Type2FieldTimings::timer().part2,"Part 2");
    mult(part2[0], vL, con_pi1_pi2, wL, false, true, tK_min, tsep_k_pi_largest); //part2 goes from insertion to pi1 to pi2 and back to insertion
    mult(part2[1], vL, con_pi2_pi1, wL, false, true, tK_min, tsep_k_pi_largest); //part2 goes from insertion to pi2 to pi1 and back to insertion
    timerEnd(Type2FieldTimings::timer().part2,"Part 2");

    for(int tkpi_idx : t_pair.second){
      int t_K = modLt(t_pi1 - tsep_k_pi[tkpi_idx], Lt);
      
      const SCFmatrixField &part1 = *part1_storage[t_K];
      timerStart(Type2FieldTimings::timer().contraction_time,"Contractions");
      type2_contract(result[tkpi_idx],t_K,part1,part2);
      timerEnd(Type2FieldTimings::timer().contraction_time,"Contractions");
    }
  }

  timerStart(Type2FieldTimings::timer().finish_up,"Finish up");

#ifdef NODE_DISTRIBUTE_MESONFIELDS
  for(int i=0;i<nmom;i++)
    nodeDistributeMany(2,mf_pi1[i],mf_pi2[i]);
  if(!UniqueID()) printf("Memory after redistributing meson fields type2 K->pipi:\n");
  printMem();  
#endif

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

  timerEnd(Type2FieldTimings::timer().finish_up,"Finish up");
  timerEnd(Type2FieldTimings::timer().total,"End");
  Type2FieldTimings::timer().report();
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
