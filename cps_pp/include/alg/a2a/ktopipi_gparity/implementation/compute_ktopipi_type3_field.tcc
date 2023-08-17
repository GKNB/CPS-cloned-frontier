#ifndef _COMPUTE_KTOPIPI_TYPE3_FIELD_H
#define _COMPUTE_KTOPIPI_TYPE3_FIELD_H

#define TIMER_ELEMS \
  ELEM(mfproducts) \
  ELEM(part1) \
  ELEM(part2) \
  ELEM(contraction_time) \
  ELEM(finish_up)\
  ELEM(total)
#define TIMER Type3FieldTimings
#include "static_timer_impl.tcc"


template<typename Vtype, typename Wtype>
void ComputeKtoPiPiGparity<Vtype,Wtype>::type3_contract(ResultsContainerType &result, const int t_K, 
							const std::vector<SCFmatrixField> &part1, const SCFmatrixField &part2_L, const SCFmatrixField &part2_H){
#ifndef MEMTEST_MODE
  static const int con_off = 13; //index of first contraction in set

  auto dimpol = part1[0].getDimPolParams();
  SCFmatrixField G1_pt1(dimpol);
  SCFmatrixField G2_pt2_L(dimpol);
  SCFmatrixField G2_pt2_H(dimpol);
  SCFmatrixField ctrans_G2_pt2_L(dimpol);
  
  CPSmatrixField<CPScolorMatrix<ComplexType> > tr_sf_G1_pt1(dimpol);
  CPSmatrixField<CPScolorMatrix<ComplexType> > tr_sf_G2_pt2_L(dimpol);
  CPSmatrixField<CPScolorMatrix<ComplexType> > tr_sf_G2_pt2_H(dimpol);

  CPSmatrixField<CPSspinMatrix<CPSflavorMatrix<ComplexType> > > tr_c_G1_pt1(dimpol);
  CPSmatrixField<CPSspinMatrix<CPSflavorMatrix<ComplexType> > > tr_c_G2_pt2_L(dimpol);
  CPSmatrixField<CPSspinMatrix<CPSflavorMatrix<ComplexType> > > tr_c_G2_pt2_H(dimpol);
    
  for(int mu=0;mu<4;mu++){ //sum over mu here
    for(int gcombidx=0;gcombidx<8;gcombidx++){
      for(int pt1_pion=0; pt1_pion<2; pt1_pion++){  //which pion comes first in part 1?
	multGammaLeft(G1_pt1, part1[pt1_pion],1,gcombidx,mu);
	multGammaLeft(G2_pt2_L, part2_L,2,gcombidx,mu);
	multGammaLeft(G2_pt2_H, part2_H,2,gcombidx,mu);

	SpinFlavorTrace(tr_sf_G1_pt1, G1_pt1);
	SpinFlavorTrace(tr_sf_G2_pt2_L, G2_pt2_L);
	SpinFlavorTrace(tr_sf_G2_pt2_H, G2_pt2_H);

	TransposeColor(ctrans_G2_pt2_L, G2_pt2_L); //speedup by transposing part 1
		
	ColorTrace(tr_c_G1_pt1, G1_pt1);
	ColorTrace(tr_c_G2_pt2_L, G2_pt2_L);
	ColorTrace(tr_c_G2_pt2_H, G2_pt2_H);
	
	//First 6 have a light-quark loop
	add(13, result, t_K, gcombidx, con_off, 
	    Trace(G1_pt1) * Trace(G2_pt2_L)
	    );
	add(14, result, t_K, gcombidx, con_off, 
	    Trace( tr_sf_G1_pt1 , Transpose(tr_sf_G2_pt2_L) )
	    );
	add(15, result, t_K, gcombidx, con_off, 
	    Trace( tr_sf_G1_pt1 , tr_sf_G2_pt2_L )
	    );
	add(16, result, t_K, gcombidx, con_off, 
	    Trace( G1_pt1 , G2_pt2_L )
	    );
	add(17, result, t_K, gcombidx, con_off, 
	    Trace( G1_pt1 , ctrans_G2_pt2_L )
	    );
	add(18, result, t_K, gcombidx, con_off, 
	    Trace( tr_c_G1_pt1 , tr_c_G2_pt2_L )
	    );
	      
	//Second 4 have strange loop
	add(19, result, t_K, gcombidx, con_off, 
	    Trace(G1_pt1) * Trace(G2_pt2_H)
	    );
	add(20, result, t_K, gcombidx, con_off, 
	    Trace( tr_sf_G1_pt1 , tr_sf_G2_pt2_H )
	    );
	add(21, result, t_K, gcombidx, con_off, 
	    Trace( G1_pt1 , G2_pt2_H )
	    );
	add(22, result, t_K, gcombidx, con_off, 
	    Trace( tr_c_G1_pt1 , tr_c_G2_pt2_H )
	    );
      }
    }
  }
#endif
}



template<typename Vtype, typename Wtype>
void ComputeKtoPiPiGparity<Vtype,Wtype>::type3_field_SIMD(ResultsContainerType result[], MixDiagResultsContainerType mix3[],
						     const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
						     const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_WV> &mf_pions,
						     const Vtype & vL, const Vtype & vH, 
						     const Wtype & wL, const Wtype & wH){
  Type3FieldTimings::timer().reset();
  timerStart(Type3FieldTimings::timer().total,"Start");
  SCFmat mix3_Gamma[2];
  mix3_Gamma[0].unit().pr(F0).gr(-5);
  mix3_Gamma[1].unit().pr(F1).gr(-5).timesMinusOne();
						   
  const int Lt = GJP.Tnodes()*GJP.TnodeSites();
  const int tpi_sampled = Lt/tstep;
  const int ntsep_k_pi = tsep_k_pi.size();

  int tsep_k_pi_largest = 0;
  for(int sep: tsep_k_pi) tsep_k_pi_largest = std::max(tsep_k_pi_largest, sep);
  
  static const int n_contract = 10; //ten type3 diagrams
  static const int con_off = 13; //index of first contraction in set

  auto field_params = vL.getMode(0).getDimPolParams();  
  
  for(int tkp=0;tkp<tsep_k_pi.size();tkp++)
    result[tkp].resize(n_contract);


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
  std::map<int, std::vector<int> > map_used_tpi1_lin_to_tsep_k_pi;  //tpi1_lin -> vector of tsep_k_pi index
  std::vector<bool> pi1_tslice_mask(Lt,false), pi2_tslice_mask(Lt,false);

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
	  pi1_tslice_mask[t_pi1] = true;
	  pi2_tslice_mask[t_pi2] = true;
	  ss << " (" << t_K << "," << t_pi1 << ")";
	  break; //only need one timeslice in window
	}
      }
    }
  }
  ss << "}\n";
  a2a_printf("%s",ss.str().c_str());

  //Gather the meson fields we need
#ifdef NODE_DISTRIBUTE_MESONFIELDS
  for(int i=0;i<nmom;i++)
    nodeGetMany(2,mf_pi1[i],&pi1_tslice_mask, mf_pi2[i], &pi2_tslice_mask);
#endif

  //Construct part 2 (independent of kaon position):
  //vL(x_op) wL^dag(x_op)   or  vH(x_op) wH^dag(x_op)
  timerStart(Type3FieldTimings::timer().part2,"Part 2");
  SCFmatrixField part2_L(field_params), part2_H(field_params);
  mult(part2_L, vL, wL, false, true);
  mult(part2_H, vH, wH, false, true);
  timerEnd(Type3FieldTimings::timer().part2,"Part 2");

  mf_WV con_pi1_pi2, con_pi2_pi1, tmp_WV;
  mf_WW con_pi1_pi2_K, con_pi2_pi1_K;
  std::vector<SCFmatrixField> part1(2, SCFmatrixField(field_params));
 
  for(auto t_pair : map_used_tpi1_lin_to_tsep_k_pi){
    int t_pi1_lin = t_pair.first;
    int t_pi1 = modLt(t_pi1_lin,Lt);   
    int t_pi2 = modLt(t_pi1 + tsep_pion, Lt);
    
    //Form the product of the three meson fields
    //con_*_*_k = [[ wL^dag(y) S_2 vL(y) ]] [[ wL^dag(z) S_2 vL(z) ]] [[ wL^dag(x_K) wH(x_K) ]]

    //Compute pion MF product first and average over momenta to project onto appropriate rotational rep
    timerStart(Type3FieldTimings::timer().mfproducts,"MFproducts");
    mult(con_pi1_pi2, mf_pi1[0]->at(t_pi1), mf_pi2[0]->at(t_pi2), true); //node local
    mult(con_pi2_pi1, mf_pi2[0]->at(t_pi2), mf_pi1[0]->at(t_pi1), true);
    for(int pp=1;pp<nmom;pp++){
      mult(tmp_WV, mf_pi1[pp]->at(t_pi1), mf_pi2[pp]->at(t_pi2), true); con_pi1_pi2.plus_equals(tmp_WV);
      mult(tmp_WV, mf_pi2[pp]->at(t_pi2), mf_pi1[pp]->at(t_pi1), true); con_pi2_pi1.plus_equals(tmp_WV);
    }
    if(nmom > 1){
      con_pi1_pi2.times_equals(1./nmom);  con_pi2_pi1.times_equals(1./nmom);
    }
    timerEnd(Type3FieldTimings::timer().mfproducts,"MFproducts");

    for(int tkp = 0; tkp < ntsep_k_pi; tkp++){	  
      int t_K = modLt(t_pi1 - tsep_k_pi[tkp], Lt);

      timerStart(Type3FieldTimings::timer().mfproducts,"MFproducts tkP");
      mult(con_pi1_pi2_K, con_pi1_pi2, mf_kaon[t_K], true);
      mult(con_pi2_pi1_K, con_pi2_pi1, mf_kaon[t_K], true);
      timerEnd(Type3FieldTimings::timer().mfproducts,"MFproducts tkP");

      //Construct part 1:
      // = vL(x_op) [[ wL^dag(x_pi1) S_2 vL(x_pi1) ]] [[ wL^dag(x_pi2) S_2 vL(x_pi2) ]] [[ wL^dag(x_K) wH(x_K) ]] vH^dag(x_op) \gamma^5
      timerStart(Type3FieldTimings::timer().part1,"Part 1");
      mult(part1[0], vL, con_pi1_pi2_K, vH, false, true, t_K, tsep_k_pi[tkp]);
      mult(part1[1], vL, con_pi2_pi1_K, vH, false, true, t_K, tsep_k_pi[tkp]);
      gr(part1[0], -5);
      gr(part1[1], -5);
      timerEnd(Type3FieldTimings::timer().part1,"Part 1");
	  
      timerStart(Type3FieldTimings::timer().contraction_time,"Contractions");
      type3_contract(result[tkp], t_K, part1, part2_L, part2_H);
      
      //Compute mix3 diagram
      //These are identical to the type3 diagrams but without the internal quark loop, and with the vertex replaced with a pseudoscalar vertex
      for(int mix3_gidx=0; mix3_gidx<2; mix3_gidx++){
	for(int pt1_pion=0; pt1_pion<2; pt1_pion++){
#ifndef MEMTEST_MODE	  
	  add(mix3[tkp],t_K,mix3_gidx, 
	      Trace( part1[pt1_pion] , mix3_Gamma[mix3_gidx] )
	      );
#endif
	}
      }
      timerEnd(Type3FieldTimings::timer().contraction_time,"Contractions");
    }//tkp
  }//tpi1_lin	
  
  timerStart(Type3FieldTimings::timer().finish_up,"Finish up");
  for(int tkp = 0; tkp < ntsep_k_pi; tkp++){
    result[tkp].nodeSum();
    mix3[tkp].nodeSum();
      
#ifndef DAIQIAN_COMPATIBILITY_MODE
    result[tkp] *= Float(0.5); //coefficient of 0.5 associated with average over pt2 pion ordering
    mix3[tkp] *= Float(0.5);
#endif
  }
				    
  //For comparison with old code, zero out data not within the K->pi time region (later optimization may render this unnecessary)
  for(int tkpi_idx=0; tkpi_idx<tsep_k_pi.size(); tkpi_idx++){
    for(int t_dis=0;t_dis<Lt;t_dis++){
      if( !(t_dis < tsep_k_pi[tkpi_idx] && t_dis > 0) ){      

	for(int t_K=0;t_K<Lt;t_K++){
	  for(int conidx=0;conidx<n_contract;conidx++)
	    for(int gcombidx=0;gcombidx<8;gcombidx++)
	      result[tkpi_idx](t_K,t_dis,conidx,gcombidx) = 0;
	  for(int midx=0;midx<2;midx++){
	    mix3[tkpi_idx](t_K,t_dis,midx) = 0;	      
	  }
	}
      }
    }
  }
				    
  timerEnd(Type3FieldTimings::timer().finish_up,"Finish up");
  timerEnd(Type3FieldTimings::timer().total,"End");
  Type3FieldTimings::timer().report();  
} 




//Field version only applicable to SIMD data. For non SIMD data we should fall back to CPU version
template<typename Vtype, typename Wtype, typename complexClass>
struct _type3_field_wrap{};

template<typename Vtype, typename Wtype>
struct _type3_field_wrap<Vtype,Wtype, grid_vector_complex_mark>{
  typedef ComputeKtoPiPiGparity<Vtype,Wtype> Compute;

  static void calc(typename Compute::ResultsContainerType result[], typename Compute::MixDiagResultsContainerType mix3[],
	    const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
	    const std::vector<typename Compute::mf_WW > &mf_kaon, MesonFieldMomentumContainer<typename Compute::mf_WV> &mf_pions,
	    const Vtype & vL, const Vtype & vH, 
	    const Wtype & wL, const Wtype & wH){
    Compute::type3_field_SIMD(result, mix3, tsep_k_pi, tsep_pion, tstep, p_pi_1_all, mf_kaon, mf_pions, vL, vH, wL, wH);
  }
};

template<typename Vtype, typename Wtype>
struct _type3_field_wrap<Vtype,Wtype, complex_double_or_float_mark>{
  typedef ComputeKtoPiPiGparity<Vtype,Wtype> Compute;

  static void calc(typename Compute::ResultsContainerType result[], typename Compute::MixDiagResultsContainerType mix3[],
	    const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
	    const std::vector<typename Compute::mf_WW > &mf_kaon, MesonFieldMomentumContainer<typename Compute::mf_WV> &mf_pions,
	    const Vtype & vL, const Vtype & vH, 
	    const Wtype & wL, const Wtype & wH){
    LOGA2A << "Type3 field implementation falling back to OMP implementation due to non-SIMD data" << std::endl;
    Compute::type3_omp_v2(result, mix3, tsep_k_pi, tsep_pion, tstep, p_pi_1_all, mf_kaon, mf_pions, vL, vH, wL, wH);
  }
};

template<typename Vtype, typename Wtype>
void ComputeKtoPiPiGparity<Vtype,Wtype>::type3_field(ResultsContainerType result[], MixDiagResultsContainerType mix3[],
						     const std::vector<int> &tsep_k_pi, const int tsep_pion, const int tstep, const std::vector<ThreeMomentum> &p_pi_1_all, 
						     const std::vector<mf_WW > &mf_kaon, MesonFieldMomentumContainer<mf_WV> &mf_pions,
						     const Vtype & vL, const Vtype & vH, 
						     const Wtype & wL, const Wtype & wH){
  _type3_field_wrap<Vtype,Wtype, typename ComplexClassify<ComplexType>::type>::calc(result, mix3, tsep_k_pi, tsep_pion, tstep, p_pi_1_all, mf_kaon, mf_pions, vL, vH, wL, wH);
}  

#endif
