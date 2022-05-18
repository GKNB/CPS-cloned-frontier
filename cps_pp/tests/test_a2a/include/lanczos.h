#pragma once

#include <Grid/algorithms/iterative/LocalCoherenceLanczos.h>

CPS_START_NAMESPACE

#ifdef USE_GRID

template<typename GridA2Apolicies>
void testLanczosIO(typename GridA2Apolicies::FgridGFclass &lattice){
  LancArg lanc_arg;
  lanc_arg.mass = 0.01;
  lanc_arg.stop_rsd = 1e-08;
  lanc_arg.N_true_get = 50;
  GridLanczosWrapper<GridA2Apolicies> lanc;
  lanc.randomizeEvecs(lanc_arg,lattice);

  lanc.writeParallel("lanc");

  {
    GridLanczosWrapper<GridA2Apolicies> lanc2;
    lanc2.readParallel("lanc");

    assert(lanc2.evec_f.size() == 0);
    assert(lanc.evec.size() == lanc2.evec.size());
    assert(lanc.eval.size() == lanc2.eval.size());

    CPSfermion5Dcb4Dodd<cps::ComplexD> c_odd_d_1;
    CPSfermion5Dcb4Dodd<cps::ComplexD> c_odd_d_2;
  
    for(int i=0;i<lanc.eval.size();i++){
      assert(lanc.eval[i] == lanc2.eval[i]);
      c_odd_d_1.importGridField(lanc.evec[i]);
      c_odd_d_2.importGridField(lanc2.evec[i]);
      
      assert( c_odd_d_1.equals( c_odd_d_2 ) );

      auto view = lanc.evec[i].View(Grid::CpuRead);
      auto view2 = lanc2.evec[i].View(Grid::CpuRead);

      for(int s=0;s<lanc.evec[i].Grid()->oSites();s++)
	assert( GridTensorEquals(view[s] , view2[s]) );
      
      
    }
  }

  lanc.toSingle();
  lanc.writeParallel("lanc");
  
  {
    GridLanczosWrapper<GridA2Apolicies> lanc2;
    lanc2.readParallel("lanc");

    assert(lanc2.evec.size() == 0);
    assert(lanc.evec_f.size() == lanc2.evec_f.size());
    assert(lanc.eval.size() == lanc2.eval.size());

    CPSfermion5Dcb4Dodd<cps::ComplexF> c_odd_f_1;
    CPSfermion5Dcb4Dodd<cps::ComplexF> c_odd_f_2;
  
    for(int i=0;i<lanc.eval.size();i++){
      assert(lanc.eval[i] == lanc2.eval[i]);
      c_odd_f_1.importGridField(lanc.evec_f[i]);
      c_odd_f_2.importGridField(lanc2.evec_f[i]);
      
      assert( c_odd_f_1.equals( c_odd_f_2 ) );

      auto view = lanc.evec_f[i].View(Grid::CpuRead);
      auto view2 = lanc2.evec_f[i].View(Grid::CpuRead);

      for(int s=0;s<lanc.evec_f[i].Grid()->oSites();s++)
	assert( GridTensorEquals(view[s] , view2[s]) );
    }
  }
}


template<typename GridA2Apolicies>
void testCompressedEvecInterface(typename GridA2Apolicies::FgridGFclass &lattice, double tol){
  Grid::GridCartesian *UGridD = lattice.getUGrid();
  Grid::GridCartesian *UGridF = lattice.getUGridF();

  Grid::GridCartesian *FGridD = lattice.getFGrid();
  Grid::GridCartesian *FGridF = lattice.getFGridF();
  
  const Grid::Coordinate &fineLatt = UGridD->FullDimensions();
  int Ls = FGridD->FullDimensions()[0];

  std::vector<int> blockSize(5,2);
  
  Grid::Coordinate coarseLatt(4);
  for (int d=0;d<4;d++){
    coarseLatt[d] = fineLatt[d]/blockSize[d];    
    assert(coarseLatt[d] % 2 == 0);
    assert(coarseLatt[d]*blockSize[d]==fineLatt[d]);
  }

  std::cout << "5d coarse lattice is ";
  for (int i=0;i<4;i++){
    std::cout << coarseLatt[i]<<"x";
  }
  int cLs = Ls/blockSize[4]; assert(cLs*blockSize[4]==Ls);
  std::cout << cLs<<std::endl;

  Grid::GridCartesian         * CoarseGrid4D    = Grid::SpaceTimeGrid::makeFourDimGrid(coarseLatt, UGridD->_simd_layout, UGridD->_processors);
  Grid::GridRedBlackCartesian * CoarseGrid4rbD  = Grid::SpaceTimeGrid::makeFourDimRedBlackGrid(CoarseGrid4D);
  Grid::GridCartesian         * CoarseGrid5D  = Grid::SpaceTimeGrid::makeFiveDimGrid(cLs,CoarseGrid4D);

  Grid::GridCartesian         * CoarseGrid4F    = Grid::SpaceTimeGrid::makeFourDimGrid(coarseLatt, UGridF->_simd_layout, UGridF->_processors);
  Grid::GridRedBlackCartesian * CoarseGrid4rbF  = Grid::SpaceTimeGrid::makeFourDimRedBlackGrid(CoarseGrid4F);
  Grid::GridCartesian         * CoarseGrid5F  = Grid::SpaceTimeGrid::makeFiveDimGrid(cLs,CoarseGrid4F);


  constexpr int basis_size = 20;
  
  typedef typename GridA2Apolicies::GridFermionField GridFermionFieldD;
  typedef typename GridA2Apolicies::GridFermionFieldF GridFermionFieldF;

  //For testing we use the same number of evecs as basis vecs and make the coarse evecs unit vectors. 
  //We make it slightly nontrivial we multiply the basis vecs by a number and set the coefficient (coarse evecs) to remove that factor
  std::vector<GridFermionFieldD> fineEvecsD(basis_size, FGridD);
  std::vector<GridFermionFieldF> fineEvecsF(basis_size, FGridF);
  std::vector<double> evals(basis_size,3.0);
  
  Grid::GridParallelRNG fineRNGD(FGridD);
  for(int i=0;i<basis_size;i++)
    random(fineRNGD, fineEvecsD[i]);
  
  typedef Grid::GparityWilsonImplD::SiteSpinor SiteSpinorD;
  typedef Grid::LocalCoherenceLanczos<SiteSpinorD,Grid::vTComplexD,basis_size> LCLD;
  typedef LCLD::CoarseSiteVector CoarseSiteVectorD;
  typedef LCLD::CoarseField CoarseFieldD;
  typedef LCLD::CoarseScalar CoarseScalarD;
  typedef CoarseSiteVectorD::scalar_object CoarseSiteVectorSD;

  typedef Grid::GparityWilsonImplF::SiteSpinor SiteSpinorF;
  typedef Grid::LocalCoherenceLanczos<SiteSpinorF,Grid::vTComplexF,basis_size> LCLF;
  typedef LCLF::CoarseSiteVector CoarseSiteVectorF;
  typedef LCLF::CoarseField CoarseFieldF;
  typedef LCLF::CoarseScalar CoarseScalarF;
  typedef CoarseSiteVectorF::scalar_object CoarseSiteVectorSF;
  
 //We need to block orthogonalize the basis vectors for the math to work out
  CoarseScalarD InnerProd(CoarseGrid5D);
  std::cout << "Gramm-Schmidt pass 1"<<std::endl;
  Grid::blockOrthogonalise(InnerProd,fineEvecsD);
  std::cout << "Gramm-Schmidt pass 2"<<std::endl;
  Grid::blockOrthogonalise(InnerProd,fineEvecsD);

  for(int i=0;i<basis_size;i++)
    precisionChange(fineEvecsF[i],fineEvecsD[i]);

  std::vector<CoarseFieldD> coarseEvecsD(basis_size, CoarseGrid5D);
  std::vector<CoarseFieldF> coarseEvecsF(basis_size, CoarseGrid5F);

  CoarseSiteVectorSD cbase = Grid::Zero();
  std::vector<GridFermionFieldD> basisD(fineEvecsD);
  std::vector<GridFermionFieldF> basisF(fineEvecsF);
  for(int i=0;i<basis_size;i++){
    basisD[i] = basisD[i] * 2.0; //multiply by factor
    CoarseSiteVectorSD c(cbase);
    c(i) = 0.5;
    coarseEvecsD[i] = c; //copy to all sites

    precisionChange(basisF[i],basisD[i]);
    precisionChange(coarseEvecsF[i],coarseEvecsD[i]);
  }
  
  EvecInterfaceCompressedMixedDoublePrec<GridFermionFieldD,GridFermionFieldF,basis_size> ifaceD(coarseEvecsD,basisD,evals,FGridD,FGridF);
  EvecInterfaceCompressedSinglePrec<GridFermionFieldD,GridFermionFieldF,basis_size> ifaceF(coarseEvecsF,basisF,evals,FGridD,FGridF);
  
  //Test uncompression
  GridFermionFieldD tmpD(FGridD), tmp2D(FGridD);
  GridFermionFieldF tmpF(FGridF), tmp2F(FGridF);

  std::cout << "Testing uncompression" << std::endl;
  for(int i=0;i<basis_size;i++){
    //DD
    double eval = ifaceD.getEvecD(tmpD,i);
    assert(eval == evals[i]);

    tmp2D = tmpD - fineEvecsD[i];
    Grid::RealD nDD = Grid::norm2(tmp2D);

    //DF
    eval = ifaceD.getEvecF(tmpF,i);
    assert(eval == evals[i]);

    tmp2F = tmpF - fineEvecsF[i];
    Grid::RealD nDF = Grid::norm2(tmp2F);

    //FD
    eval = ifaceF.getEvecD(tmpD,i);
    assert(eval == evals[i]);

    tmp2D = tmpD - fineEvecsD[i];
    Grid::RealD nFD = Grid::norm2(tmp2D);

    //FF
    eval = ifaceF.getEvecF(tmpF,i);
    assert(eval == evals[i]);

    tmp2F = tmpF - fineEvecsF[i];
    Grid::RealD nFF = Grid::norm2(tmp2F);

    std::cout << i << " DD:" << nDD << " DF:" << nDF << " FD:" << nFD << " FF:" << nFF << std::endl;
    assert(fabs(nDD) < tol);
    assert(fabs(nDF) < tol);
    assert(fabs(nFD) < tol);
    assert(fabs(nFF) < tol);
  }
  
  //Test deflation
  int ndeflate=2;
  std::vector<GridFermionFieldD> defl_in_D(ndeflate, FGridD);
  std::vector<GridFermionFieldF> defl_in_F(ndeflate, FGridF);
  for(int i=0;i<ndeflate;i++){
    random(fineRNGD, defl_in_D[i]);
    precisionChange(defl_in_F[i],defl_in_D[i]);
  }

  std::vector<GridFermionFieldD> defl_out_1_D(ndeflate, FGridD);
  std::vector<GridFermionFieldD> defl_out_2_D(ndeflate, FGridD);
  std::vector<GridFermionFieldF> defl_out_1_F(ndeflate, FGridF);
  std::vector<GridFermionFieldF> defl_out_2_F(ndeflate, FGridF);

  EvecInterfaceMixedDoublePrec<GridFermionFieldD,GridFermionFieldF> iface_origD(fineEvecsD,evals,FGridD,FGridF);
  EvecInterfaceSinglePrec<GridFermionFieldD,GridFermionFieldF> iface_origF(fineEvecsF,evals,FGridD,FGridF);

  std::cout << "Testing DD deflation" << std::endl;
  iface_origD.deflatedGuessD(defl_out_1_D,defl_in_D);
  ifaceD.deflatedGuessD(defl_out_2_D,defl_in_D);
  
  for(int i=0;i<ndeflate;i++){
    tmpD = defl_out_1_D[i] - defl_out_2_D[i];
    Grid::RealD n = Grid::norm2(tmpD);
    std::cout << i << " " << n << std::endl;
    assert(fabs(n) < tol);
  }
  std::cout << "Testing DF deflation" << std::endl;
  iface_origD.deflatedGuessF(defl_out_1_F,defl_in_F);
  ifaceD.deflatedGuessF(defl_out_2_F,defl_in_F);
  
  for(int i=0;i<ndeflate;i++){
    tmpF = defl_out_1_F[i] - defl_out_2_F[i];
    Grid::RealD n = Grid::norm2(tmpF);
    std::cout << i << " " << n << std::endl;
    assert(fabs(n) < tol);
  }


  std::cout << "Testing FD deflation" << std::endl;
  iface_origF.deflatedGuessD(defl_out_1_D,defl_in_D);
  ifaceF.deflatedGuessD(defl_out_2_D,defl_in_D);
  
  for(int i=0;i<ndeflate;i++){
    tmpD = defl_out_1_D[i] - defl_out_2_D[i];
    Grid::RealD n = Grid::norm2(tmpD);
    std::cout << i << " " << n << std::endl;
    assert(fabs(n) < tol);
  }
  std::cout << "Testing FF deflation" << std::endl;
  iface_origF.deflatedGuessF(defl_out_1_F,defl_in_F);
  ifaceF.deflatedGuessF(defl_out_2_F,defl_in_F);
  
  for(int i=0;i<ndeflate;i++){
    tmpF = defl_out_1_F[i] - defl_out_2_F[i];
    Grid::RealD n = Grid::norm2(tmpF);
    std::cout << i << " " << n << std::endl;
    assert(fabs(n) < tol);
  }

}




#endif //USE_GRID


CPS_END_NAMESPACE
