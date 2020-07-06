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



#endif
