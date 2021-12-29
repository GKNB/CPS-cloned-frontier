template<typename A2Apolicies, typename FermionOperatorTypeD>
void computeVWlow(A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W, const EvecInterface<typename FermionOperatorTypeD::FermionField> &evecs,  
		  const A2AlowModeCompute<FermionOperatorTypeD> &impl){  
  int nl = V.getNl();
  if(nl == 0) return;
  assert(nl <= evecs.nEvecs());
  
  typedef typename FermionOperatorTypeD::FermionField GridFermionFieldD;

  FermionOperatorTypeD & OpD = impl.getOp();
  Grid::GridBase *EvecGrid = evecs.getEvecGridD(); //may be regular or checkerboarded Grid
  Grid::GridBase *UGrid = OpD.GaugeGrid();

  GridFermionFieldD evec(EvecGrid);
  GridFermionFieldD tmp_full_4d(UGrid);

  //See section 1.3 of 
  //https://rbc.phys.columbia.edu/rbc_ukqcd/individual_postings/ckelly/Gparity/note_a2a_v5.pdf
#ifndef MEMTEST_MODE
  for(size_t i = 0; i < nl; i++) {
    //Step 1) Compute Vl
    Float eval = evecs.getEvec(evec,i);
    impl.computeVl(tmp_full_4d, evec, eval);
    V.getVl(i).importGridField(tmp_full_4d);
        
    //Step 2) Compute Wl
    impl.computeWl(tmp_full_4d, evec);
    W.getWl(i).importGridField(tmp_full_4d);
  }
#endif
}




#ifdef USE_GRID_LANCZOS
#endif


#ifdef USE_BFM_LANCZOS
//Compute the low mode part of the W and V vectors. In the Lanczos class you can choose to store the vectors in single precision (despite the overall precision, which is fixed to double here)
//Set 'singleprec_evecs' if this has been done
template< typename Policies>
void computeVWlow(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, BFM_Krylov::Lanczos_5d<double> &eig, bfm_evo<double> &dwf, bool singleprec_evecs, const CGcontrols &cg_controls){
  EvecInterfaceBFM<Policies> ev(eig,dwf,lat,singleprec_evecs);
  return computeVWlow(V,W,lat,ev,dwf.mass,cg_controls);
}
#endif
