template<typename GridFermionFieldD, typename Vtype, typename Wtype>
void computeVWlow(Vtype &V, Wtype &W, const EvecInterface<GridFermionFieldD> &evecs,  
		  const A2AlowModeCompute<GridFermionFieldD> &impl){
  LOGA2A << "Computing V,W low modes" << std::endl;
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
    printMem("getEvecD");
    Float eval = evecs.getEvecD(evec,i);
    printMem("computeVl");
    impl.computeVl(tmp_full_4d, evec, eval);
    printMem("importVl");
    V.getVl(i).importGridField(tmp_full_4d);
        
    //Step 2) Compute Wl
    printMem("computeWl");
    impl.computeWl(tmp_full_4d, evec);
    printMem("importWl");
    W.getWl(i).importGridField(tmp_full_4d);
  }
  printMem("Completed V,W low modes");
#endif
  LOGA2A << "Finished V,W low modes" << std::endl;
}

