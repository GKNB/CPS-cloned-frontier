#ifndef _COMPUTE_KTOPIPI_TYPE4_FIELD_H
#define _COMPUTE_KTOPIPI_TYPE4_FIELD_H

#define TIMER_ELEMS \
  ELEM(part1) \
  ELEM(part2) \
  ELEM(total_contraction_time) \
  ELEM(contraction_time_geninputs_multgamma) \
  ELEM(contraction_time_geninputs_other) \
  ELEM(contraction_time_opcon) \
  ELEM(finish_up)\
  ELEM(total)
#define TIMER Type4FieldTimings
#include "static_timer_impl.tcc"




template<typename Vtype, typename Wtype>
void ComputeKtoPiPiGparity<Vtype,Wtype>::type4_contract(ResultsContainerType &result, const int t_K, 
							const SCFmatrixField &part1, const SCFmatrixField &part2_L, const SCFmatrixField &part2_H){

#ifndef MEMTEST_MODE
  static const int con_off = 23; //index of first contraction in set

  auto dimpol = part1.getDimPolParams();
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
      Type4FieldTimings::timer().contraction_time_geninputs_multgamma -= dclock();

      multGammaLeft(G1_pt1, part1,1,gcombidx,mu);
      multGammaLeft(G2_pt2_L, part2_L,2,gcombidx,mu);
      multGammaLeft(G2_pt2_H, part2_H,2,gcombidx,mu);

      Type4FieldTimings::timer().contraction_time_geninputs_multgamma += dclock();

      Type4FieldTimings::timer().contraction_time_geninputs_other -= dclock();

      SpinFlavorTrace(tr_sf_G1_pt1, G1_pt1);
      SpinFlavorTrace(tr_sf_G2_pt2_L, G2_pt2_L);
      SpinFlavorTrace(tr_sf_G2_pt2_H, G2_pt2_H);
	    
      TransposeColor(ctrans_G2_pt2_L, G2_pt2_L);  //speedup by transposing part 1
	
      ColorTrace(tr_c_G1_pt1, G1_pt1);
      ColorTrace(tr_c_G2_pt2_L, G2_pt2_L);
      ColorTrace(tr_c_G2_pt2_H, G2_pt2_H);

      Type4FieldTimings::timer().contraction_time_geninputs_other += dclock();

      Type4FieldTimings::timer().contraction_time_opcon -= dclock();
      
      //First 6 have a light-quark loop
      add(23, result, t_K, gcombidx, con_off, 
	  Trace(G1_pt1) * Trace(G2_pt2_L)
	  );

      add(24, result, t_K, gcombidx, con_off, 
	  Trace( tr_sf_G1_pt1 , Transpose(tr_sf_G2_pt2_L) ) 
	  );

      add(25, result, t_K, gcombidx, con_off,       
	  Trace( tr_sf_G1_pt1 , tr_sf_G2_pt2_L ) 
	  );
      add(26, result, t_K, gcombidx, con_off,       
	  Trace( G1_pt1 , G2_pt2_L ) 
	  );
      add(27, result, t_K, gcombidx, con_off,       
	  Trace( G1_pt1 , ctrans_G2_pt2_L ) 
	  );
      add(28, result, t_K, gcombidx, con_off,       
	  Trace( tr_c_G1_pt1 , tr_c_G2_pt2_L ) 
	  );
	      
      //Second 4 have strange loop
      add(29, result, t_K, gcombidx, con_off,       
	  Trace(G1_pt1) * Trace(G2_pt2_H) 
	  );	      
      add(30, result, t_K, gcombidx, con_off,       
	  Trace( tr_sf_G1_pt1 , tr_sf_G2_pt2_H ) 
	  );
      add(31, result, t_K, gcombidx, con_off,       
	  Trace( G1_pt1 , G2_pt2_H ) 
	  );
      add(32, result, t_K, gcombidx, con_off,       
	  Trace( tr_c_G1_pt1 , tr_c_G2_pt2_H ) 
	  );

      Type4FieldTimings::timer().contraction_time_opcon += dclock();

#undef C	     	    
    }
  }
#endif
}



template<typename Vtype, typename Wtype>
void ComputeKtoPiPiGparity<Vtype,Wtype>::type4_field_SIMD(ResultsContainerType &result, MixDiagResultsContainerType &mix4,
						     const int tstep,
						     const std::vector<mf_WW > &mf_kaon,
						     const Vtype & vL, const Vtype & vH, 
						     const Wtype & wL, const Wtype & wH){
  
  Type4FieldTimings::timer().reset();
  timerStart(Type4FieldTimings::timer().total,"Start");
  SCFmat mix4_Gamma[2];
  mix4_Gamma[0].unit().pr(F0).gr(-5);
  mix4_Gamma[1].unit().pr(F1).gr(-5).timesMinusOne();
  
  //CK: the loop term could be re-used from type3
  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  assert(Lt % tstep == 0);
  
  static const int n_contract = 10; //ten type4 diagrams
  static const int con_off = 23; //index of first contraction in set

  result.resize(n_contract); //it will be thread-reduced before this method ends
  mix4.resize(1);

  auto field_params = vL.getMode(0).getDimPolParams();

  //Construct part 2 (doesn't care about kaon timeslice):
  //vL(x_op) wL^dag(x_op)   or  vH(x_op) wH^dag(x_op)  (CK: should re-use these from type-3)
  timerStart(Type4FieldTimings::timer().part2,"Part 2");
  SCFmatrixField part2_L(field_params), part2_H(field_params);
  mult(part2_L, vL, wL, false, true);
  mult(part2_H, vH, wH, false, true);
  timerEnd(Type4FieldTimings::timer().part2,"Part 2");

  SCFmatrixField part1(field_params);
  
  for(int t_K = 0; t_K < Lt; t_K += tstep){ //global times
    //Construct part 1:
    // = vL(x_op) [[ wL^dag(x_K) wH(x_K) ]] vH^dag(x_op) \gamma_5
    timerStart(Type4FieldTimings::timer().part1,"Part 1");
    mult(part1, vL,mf_kaon[t_K],vH, false,true);
    gr(part1, -5);
    timerEnd(Type4FieldTimings::timer().part1,"Part 1");
    
    timerStart(Type4FieldTimings::timer().total_contraction_time,"Contractions");
    type4_contract(result,t_K,part1,part2_L,part2_H);

    //Compute mix4 diagram
    //These are identical to the type4 diagrams but without the quark loop, and with the vertex replaced with a pseudoscalar vertex
    for(int mix4_gidx=0; mix4_gidx<2; mix4_gidx++){
#ifndef MEMTEST_MODE
      add(mix4,t_K,mix4_gidx, 
	  Trace( part1 , mix4_Gamma[mix4_gidx] )
	  );
#endif
    }
    timerEnd(Type4FieldTimings::timer().total_contraction_time,"Contractions");

  }//t_K loop

  timerStart(Type4FieldTimings::timer().finish_up,"Finish up");
  result.nodeSum();
  mix4.nodeSum();
  timerEnd(Type4FieldTimings::timer().finish_up,"Finish up");
  timerEnd(Type4FieldTimings::timer().total,"End");
  Type4FieldTimings::timer().report();  
}


//Field version only applicable to SIMD data. For non SIMD data we should fall back to CPU version
template<typename Vtype, typename Wtype, typename complexClass>
struct _type4_field_wrap{};

template<typename Vtype, typename Wtype>
struct _type4_field_wrap<Vtype,Wtype, grid_vector_complex_mark>{
  typedef ComputeKtoPiPiGparity<Vtype,Wtype> Compute;

  static void calc(typename Compute::ResultsContainerType &result, typename Compute::MixDiagResultsContainerType &mix4,
		   const int tstep,
		   const std::vector<typename Compute::mf_WW > &mf_kaon,
		   const Vtype & vL, const Vtype & vH, 
		   const Wtype & wL, const Wtype & wH){
    Compute::type4_field_SIMD(result, mix4, tstep, mf_kaon, vL, vH, wL, wH);
  }
};

template<typename Vtype, typename Wtype>
struct _type4_field_wrap<Vtype,Wtype, complex_double_or_float_mark>{
  typedef ComputeKtoPiPiGparity<Vtype,Wtype> Compute;

  static void calc(typename Compute::ResultsContainerType &result, typename Compute::MixDiagResultsContainerType &mix4,
		   const int tstep,
		   const std::vector<typename Compute::mf_WW > &mf_kaon,
		   const Vtype & vL, const Vtype & vH, 
		   const Wtype & wL, const Wtype & wH){
    LOGA2A << "Type4 field implementation falling back to OMP implementation due to non-SIMD data" << std::endl;
    Compute::type4_omp(result, mix4, tstep, mf_kaon, vL, vH, wL, wH);
  }
};


template<typename Vtype, typename Wtype>
void ComputeKtoPiPiGparity<Vtype,Wtype>::type4_field(ResultsContainerType &result, MixDiagResultsContainerType &mix4,
						     const int tstep,
						     const std::vector<mf_WW > &mf_kaon,
						     const Vtype & vL, const Vtype & vH, 
						     const Wtype & wL, const Wtype & wH){
  _type4_field_wrap<Vtype,Wtype, typename ComplexClassify<ComplexType>::type>::calc(result, mix4, tstep, mf_kaon, vL, vH, wL, wH);
}  


#endif
