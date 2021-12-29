//Allow the operator used for the high mode inversions (1) to differ from that used for the low mode contribution, (2) eg for MADWF
//block_size is the number of sources deflated simultaneously, and if the inverter supports it, inverted concurrently
template<typename A2Apolicies, typename FermionOperatorTypeD1, typename FermionOperatorTypeD2>
void computeVWhigh(A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W, 
		   const EvecInterface<typename FermionOperatorTypeD2::FermionField> &evecs,  
		   const A2AlowModeCompute<FermionOperatorTypeD2> &impl,
		   const A2Ainverter4dBase<FermionOperatorTypeD1> &inverter,
		   size_t block_size){  
  Grid::GridBase* UGrid = impl.getOp().GaugeGrid(); //should be the same for both operators
  typedef typename FermionOperatorTypeD1::FermionField GridFermionFieldD;

  //cf https://rbc.phys.columbia.edu/rbc_ukqcd/individual_postings/ckelly/Gparity/note_a2a_v4.pdf  section D
#ifndef MEMTEST_MODE
  W.setWhRandom();
#endif
  size_t nh = W.getNh();
  size_t nl = W.getNl();
  size_t nhits = W.getNhits();

  std::vector<GridFermionFieldD> grid_src_v(block_size,UGrid);
  std::vector<GridFermionFieldD> grid_sol_v(block_size,UGrid);
  std::vector<GridFermionFieldD> lowmode_contrib_v(block_size,UGrid);

  CPSfermion4D<typename A2Apolicies::ComplexTypeD,typename A2Apolicies::FermionFieldType::FieldMappingPolicy, 
	       typename A2Apolicies::FermionFieldType::FieldAllocPolicy> v4dfield(V.getFieldInputParams());

  int Nblocks = (nh + block_size - 1)/block_size;
  std::cout << Grid::GridLogMessage << "Computing V,W high mode contribution in " << Nblocks << " blocks of " << block_size << std::endl;

#ifndef MEMTEST_MODE
  for(size_t b=0;b<Nblocks;b++){
    size_t istart = b*block_size;
    size_t ilessthan = std::min( (b+1)*block_size, nh );
    size_t ni = ilessthan - istart;
    if(b<Nblocks-1) assert(ni == block_size);
    else if(ni < block_size){
      grid_src_v.resize(ni,UGrid); //reduce size only on last step
      grid_sol_v.resize(ni,UGrid);
      lowmode_contrib_v.resize(ni,UGrid);
    }

    for(size_t ii=0;ii<ni;ii++){
      //Step 1) Get the diluted W vector to invert upon
      W.getDilutedSource(v4dfield, istart+ii);

      //Step 2) Export to Grid field
      v4dfield.exportGridField(grid_src_v[ii]);

      grid_sol_v[ii] = Grid::Zero();      
    }
    //Step 3) Perform (deflated) inversion from *4D->4D*
    inverter.invert4Dto4D(grid_sol_v, grid_src_v);
    
    //Step 4) Subtract low-mode part in *4D* space   
    impl.lowModeContribution4D(lowmode_contrib_v, grid_src_v, evecs, nl);

    for(size_t ii=0;ii<ni;ii++){
      grid_sol_v[ii] = grid_sol_v[ii] - lowmode_contrib_v[ii];      
      grid_sol_v[ii] = Grid::RealD(1. / nhits) * grid_sol_v[ii]; //include factor of 1/Nhits into V such that it doesn't need to be explicitly included      
      V.getVh(istart+ii).importGridField(grid_sol_v[ii]);
    }
  }
#endif
}



#ifdef USE_GRID_LANCZOS
#endif

#ifdef USE_BFM_LANCZOS
template< typename Policies>
void computeVWhigh(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, BFM_Krylov::Lanczos_5d<double> &eig, bool singleprec_evecs, Lattice &lat, const CGcontrols &cg_controls, bfm_evo<double> &dwf_d, bfm_evo<float> *dwf_fp){
  bool mixed_prec_cg = dwf_fp != NULL; 
  if(mixed_prec_cg){
    //NOT IMPLEMENTED YET
    ERR.General("A2AvectorW","computeVWhigh","No grid implementation of mixed precision CG with BFM evecs\n");
  }

  if(mixed_prec_cg && !singleprec_evecs){ ERR.General("A2AvectorW","computeVWhigh","If using mixed precision CG, input eigenvectors must be stored in single precision"); }

  EvecInterfaceBFM<Policies> ev(eig,dwf_d,lat,singleprec_evecs);
  return computeVWhigh(V,W,lat,ev,dwf_d.mass,cg_controls);
}
#endif
