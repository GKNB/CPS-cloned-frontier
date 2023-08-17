#pragma once

CPS_START_NAMESPACE

template<typename A2Apolicies_std, typename A2Apolicies_grid>
void testPionContractionGridStd(A2AvectorV<A2Apolicies_std> &V_std, A2AvectorW<A2Apolicies_std> &W_std,
			 A2AvectorV<A2Apolicies_grid> &V_grid, A2AvectorW<A2Apolicies_grid> &W_grid,
			 typename A2Apolicies_grid::FgridGFclass &lattice,
			 typename SIMDpolicyBase<3>::ParamType simd_dims_3d,
			 double tol){
  std::cout << "Starting testPionContractionGridStd" << std::endl;
  StandardPionMomentaPolicy momenta;
  typedef A2AvectorV<A2Apolicies_std> Vstd;
  typedef A2AvectorW<A2Apolicies_std> Wstd;
  typedef A2AvectorV<A2Apolicies_grid> Vgrid;
  typedef A2AvectorW<A2Apolicies_grid> Wgrid; 
  typedef getMesonFieldType<Wgrid,Vgrid> mf_WV_grid;
  typedef getMesonFieldType<Wstd,Vstd> mf_WV_std;

  MesonFieldMomentumContainer<mf_WV_std> mf_ll_con_std;
  MesonFieldMomentumContainer<mf_WV_grid> mf_ll_con_grid;

  std::cout << "Computing non-SIMD meson fields" << std::endl;
  computeGparityLLmesonFields1s<Vstd,Wstd,StandardPionMomentaPolicy,15,sigma3>::computeMesonFields(mf_ll_con_std,momenta,W_std,V_std,2.0,lattice);
  std::cout << "Computing SIMD meson fields" << std::endl;
  computeGparityLLmesonFields1s<Vgrid,Wgrid,StandardPionMomentaPolicy,15,sigma3>::computeMesonFields(mf_ll_con_grid,momenta,W_grid,V_grid,2.0,lattice,simd_dims_3d);

  std::cout << "Computing non-SIMD pion 2pt" << std::endl;
  fMatrix<typename A2Apolicies_std::ScalarComplexType> fmat_std;
  ComputePion<Vstd,Wstd>::compute(fmat_std, mf_ll_con_std, momenta, 0);

  std::cout << "Computing SIMD pion 2pt" << std::endl;
  fMatrix<typename A2Apolicies_grid::ScalarComplexType> fmat_grid;
  ComputePion<Vgrid,Wgrid>::compute(fmat_grid, mf_ll_con_grid, momenta, 0);

  bool fail = false;
  for(int r=0;r<fmat_std.nRows();r++){
    for(int c=0;c<fmat_std.nCols();c++){
      double rdiff = fmat_std(r,c).real() - fmat_grid(r,c).real();
      double idiff = fmat_std(r,c).imag() - fmat_grid(r,c).imag();
      if(rdiff > tol|| idiff > tol){
	printf("Fail Pion %d %d : (%f,%f) (%f,%f) diff (%g,%g)\n",r,c, fmat_std(r,c).real(),  fmat_std(r,c).imag(), fmat_grid(r,c).real(), fmat_grid(r,c).imag(), rdiff, idiff);	
	fail = true;
      }
    }
  }
  if(fail)ERR.General("","","Standard vs Grid implementation pion test failed\n");
  std::cout << "testPionContractionGridStd passed" << std::endl;
}


CPS_END_NAMESPACE
