#ifndef _COMPUTE_KTOPIPI_TYPE4_FIELD_H
#define _COMPUTE_KTOPIPI_TYPE4_FIELD_H

template<typename mf_Policies>
void ComputeKtoPiPiGparity<mf_Policies>::type4_contract(ResultsContainerType &result, const int t_K, 
							const SCFmatrixField &part1, const SCFmatrixField &part2_L, const SCFmatrixField &part2_H){

#ifndef MEMTEST_MODE
  static const int con_off = 23; //index of first contraction in set

  for(int mu=0;mu<4;mu++){ //sum over mu here
    for(int gcombidx=0;gcombidx<8;gcombidx++){
      auto G1_pt1 = part1;
      multGammaLeft(G1_pt1,1,gcombidx,mu);

      auto tr_sf_G1_pt1 = SpinFlavorTrace(G1_pt1);

      auto G2_pt2_L = part2_L;
      multGammaLeft(G2_pt2_L,2,gcombidx,mu);

      auto tr_sf_G2_pt2_L = SpinFlavorTrace(G2_pt2_L);

      auto G2_pt2_H = part2_H;
      multGammaLeft(G2_pt2_H,2,gcombidx,mu);

      auto tr_sf_G2_pt2_H = SpinFlavorTrace(G2_pt2_H);
	    
      auto ctrans_G2_pt2_L = TransposeColor(G2_pt2_L);  //speedup by transposing part 1
	
      auto tr_c_G1_pt1 = ColorTrace(G1_pt1);
	
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
	  Trace( tr_c_G1_pt1 , ColorTrace(G2_pt2_L) ) 
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
	  Trace( tr_c_G1_pt1 , ColorTrace(G2_pt2_H) ) 
	  );

#undef C	     	    
    }
  }
#endif
}



template<typename mf_Policies>
void ComputeKtoPiPiGparity<mf_Policies>::type4_field(ResultsContainerType &result, MixDiagResultsContainerType &mix4,
						     const int &tstep,
						     const std::vector<mf_WW > &mf_kaon,
						     const A2AvectorV<mf_Policies> & vL, const A2AvectorV<mf_Policies> & vH, 
						     const A2AvectorW<mf_Policies> & wL, const A2AvectorW<mf_Policies> & wH){
  
  Type4timings::timer().reset();
  Type4timings::timer().total -= dclock();
  SCFmat mix4_Gamma[2];
  mix4_Gamma[0].unit().pr(F0).gr(-5);
  mix4_Gamma[1].unit().pr(F1).gr(-5).timesMinusOne();
  
  //CK: the loop term could be re-used from type3
  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  assert(Lt % tstep == 0);
  
  static const int n_contract = 10; //ten type4 diagrams
  static const int con_off = 23; //index of first contraction in set
  int nthread = omp_get_max_threads();

  result.resize(n_contract,nthread); //it will be thread-reduced before this method ends
  mix4.resize(nthread);

  auto field_params = vL.getMode(0).getDimPolParams();

  //Construct part 2 (doesn't care about kaon timeslice):
  //vL(x_op) wL^dag(x_op)   or  vH(x_op) wH^dag(x_op)  (CK: should re-use these from type-3)
  SCFmatrixField part2_L(field_params), part2_H(field_params);
  mult(part2_L, vL, wL, false, true);
  mult(part2_H, vH, wH, false, true);
   
  for(int t_K = 0; t_K < Lt; t_K += tstep){ //global times

    //Construct part 1:
    // = vL(x_op) [[ wL^dag(x_K) wH(x_K) ]] vH^dag(x_op) \gamma_5
    SCFmatrixField part1(field_params);
    mult(part1, vL,mf_kaon[t_K],vH, false,true);
    gr(part1, -5);
    
    Type4timings::timer().contraction_time -= dclock();
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
    Type4timings::timer().contraction_time += dclock();

  }//t_K loop

  Type4timings::timer().finish_up -= dclock();
  result.nodeSum();
  mix4.nodeSum();
  Type4timings::timer().finish_up += dclock();
  Type4timings::timer().total += dclock();  
  Type4timings::timer().report();  
}



#endif
