template<typename GridFermionFieldD, typename Vtype, typename Wtype>
void computeVWlow(Vtype &V, Wtype &W, const EvecInterface<GridFermionFieldD> &evecs,  
		  const A2AlowModeCompute<GridFermionFieldD> &impl){  
  int nl = V.getNl();
  if(nl == 0) return;
  assert(nl <= evecs.nEvecs());
  
  Grid::GridBase *EvecGrid = evecs.getEvecGridD(); //may be regular or checkerboarded Grid
  Grid::GridBase *UGrid = impl.get4Dgrid();

  GridFermionFieldD evec(EvecGrid);
  GridFermionFieldD tmp_full_4d(UGrid);

  //See section 1.3 of 
  //https://rbc.phys.columbia.edu/rbc_ukqcd/individual_postings/ckelly/Gparity/note_a2a_v5.pdf
#ifndef MEMTEST_MODE
  for(size_t i = 0; i < nl; i++) {
    //Step 1) Compute Vl
    Float eval = evecs.getEvecD(evec,i);
    impl.computeVl(tmp_full_4d, evec, eval);
    V.getVl(i).importGridField(tmp_full_4d);
        
    //Step 2) Compute Wl
    impl.computeWl(tmp_full_4d, evec);
    W.getWl(i).importGridField(tmp_full_4d);
  }
#endif
}

