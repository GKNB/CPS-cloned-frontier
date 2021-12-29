template<typename A2Apolicies, typename FermionOperatorTypeD>
void computeVWlow(A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W, const EvecInterface<typename FermionOperatorTypeD::FermionField> &evecs,  
		  const A2AlowModeCompute<FermionOperatorTypeD> &impl){  
  int nl = V.getNl();
  if(nl == 0) return;
  assert(nl <= evecs.nEvecs());
  
  typedef typename FermionOperatorTypeD::FermionField GridFermionFieldD;

  FermionOperatorTypeD & OpD = impl.getOp();
  Grid::GridBase *EvecGrid = evecs.getEvecGridD(); //may be regular or checkerboarded Grid
  Grid::GridBase *FGrid = OpD.FermionGrid();
  Grid::GridBase *UGrid = OpD.GaugeGrid();

  assert(FGrid->Nd() == 5);
  int Ls = FGrid->GlobalDimensions()[0]; //5th dim is dimension 0!

  GridFermionFieldD evec(EvecGrid);
  GridFermionFieldD tmp_full(FGrid);
  GridFermionFieldD tmp_full_4d(UGrid);

  //See section 1.3 of 
  //https://rbc.phys.columbia.edu/rbc_ukqcd/individual_postings/ckelly/Gparity/note_a2a_v5.pdf
#ifndef MEMTEST_MODE
  for(size_t i = 0; i < nl; i++) {
    //Step 1) Compute V
    Float eval = evecs.getEvec(evec,i);
    impl.evecTransformVl(tmp_full, evec);

    //Get 4D part and poke into a
    //Recall that D^{-1} = <v w^\dagger> = <q \bar q>.  v therefore transforms like a spinor. For spinors \psi(x) = P_R \psi(x,Ls-1) + P_L \psi(x,0),  i.e. s_u=Ls-1 and s_l=0 for CPS gamma5

    DomainWallFiveToFour(tmp_full_4d, tmp_full, Ls-1,0);
    tmp_full_4d = Grid::RealD(1./eval) * tmp_full_4d;
    V.getVl(i).importGridField(tmp_full_4d);
        
    //Step 2) Compute Wl. As the Dirac preconditioned Dirac operators is abstracted out, there are no differences between the two preconditioning schemes here
    impl.evecTransformWl(tmp_full, evec);

    //Get 4D part, poke onto a then copy into wl
    //Recall that D^{-1} = <v w^\dagger> = <q \bar q>.  w (and w^\dagger) therefore transforms like a conjugate spinor. For spinors \bar\psi(x) =  \bar\psi(x,0) P_R +  \bar\psi(x,Ls-1) P_L,  i.e. s_u=0 and s_l=Ls-1 for CPS gamma5
    DomainWallFiveToFour(tmp_full_4d, tmp_full, 0, Ls-1);
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
