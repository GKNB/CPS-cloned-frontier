//block_size is the number of sources deflated simultaneously, and if the inverter supports it, inverted concurrently
template<typename A2Apolicies, typename GridFermionFieldD>
void computeVWhigh(A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W, 
		   const A2AhighModeSource<A2Apolicies> &Wsrc_impl,
		   const EvecInterface<GridFermionFieldD> &evecs,  
		   const A2AhighModeCompute<GridFermionFieldD> &impl,
		   size_t block_size){  
  Grid::GridBase* UGrid = impl.get4Dgrid(); //should be the same for both operators

  //cf https://rbc.phys.columbia.edu/rbc_ukqcd/individual_postings/ckelly/Gparity/note_a2a_v4.pdf  section D
#ifndef MEMTEST_MODE
  if(!W.WhRandPerformed()) Wsrc_impl.setHighModeSources(W); //set the sources if not previously done
#endif
  size_t nh = W.getNh();
  size_t nl = W.getNl();
  size_t nhits = W.getNhits();

  std::vector<GridFermionFieldD> grid_src_v(block_size,UGrid);
  std::vector<GridFermionFieldD> grid_sol_v(block_size,UGrid);

  int Nblocks = (nh + block_size - 1)/block_size;
  LOGA2A << "Computing V,W high mode contribution in " << Nblocks << " blocks of " << block_size << std::endl;
  printMem("Prior to V,W high mode calculation");
  
#ifndef MEMTEST_MODE
  for(size_t b=0;b<Nblocks;b++){
    size_t istart = b*block_size;
    size_t ilessthan = std::min( (b+1)*block_size, nh );
    size_t ni = ilessthan - istart;
    if(b<Nblocks-1) assert(ni == block_size);
    else if(ni < block_size){
      grid_src_v.resize(ni,UGrid); //reduce size only on last step
      grid_sol_v.resize(ni,UGrid);
    }

    for(size_t ii=0;ii<ni;ii++){
      //Get the diluted W vector to invert upon
      Wsrc_impl.get4DinverseSource(grid_src_v[ii],istart+ii,W);
      grid_sol_v[ii] = Grid::Zero();      
    }
    //Compute the high mode contribution
    impl.highModeContribution4D(grid_sol_v, grid_src_v, evecs, nl); 

    //Normalize and export
    for(size_t ii=0;ii<ni;ii++){
      grid_sol_v[ii] = Grid::RealD(1. / nhits) * grid_sol_v[ii]; //include factor of 1/Nhits into V such that it doesn't need to be explicitly included      
      V.getVh(istart+ii).importGridField(grid_sol_v[ii]);
    }
  }

  Wsrc_impl.solutionPostOp(V);
#endif
  LOGA2A << "Finished V,W high modes" << std::endl;    
}

