#pragma once

CPS_START_NAMESPACE

template<typename A2Apolicies_std, typename A2Apolicies_grid>
void testKaonContractionGridStd(A2AvectorV<A2Apolicies_std> &V_std, A2AvectorW<A2Apolicies_std> &W_std,
				A2AvectorV<A2Apolicies_grid> &V_grid, A2AvectorW<A2Apolicies_grid> &W_grid,
				typename A2Apolicies_grid::FgridGFclass &lattice,
				typename SIMDpolicyBase<3>::ParamType simd_dims_3d,
				double tol){
  std::cout << "Starting testKaonContractionGridStd" << std::endl;

  StationaryKaonMomentaPolicy kaon_mom;

  typedef A2AvectorW<A2Apolicies_std> WtypeStd;
  typedef A2AvectorV<A2Apolicies_std> VtypeStd;
  typedef A2AmesonField<A2Apolicies_std,WtypeStd::template FFTvectorTemplate, VtypeStd::template FFTvectorTemplate> MesonFieldTypeStd;

  typedef A2AvectorW<A2Apolicies_grid> WtypeGrid;
  typedef A2AvectorV<A2Apolicies_grid> VtypeGrid;
  typedef A2AmesonField<A2Apolicies_grid,WtypeGrid::template FFTvectorTemplate, VtypeGrid::template FFTvectorTemplate> MesonFieldTypeGrid;
  
  std::vector<MesonFieldTypeStd> mf_ls_std;
  std::vector<MesonFieldTypeStd> mf_sl_std;
  ComputeKaon<VtypeStd, WtypeStd>::computeMesonFields(mf_ls_std, mf_sl_std,
						   W_std, V_std,
						   W_std, V_std,
						   kaon_mom,
						   2.0, lattice);

  std::vector<MesonFieldTypeGrid> mf_ls_grid;
  std::vector<MesonFieldTypeGrid> mf_sl_grid;
  ComputeKaon<VtypeGrid, WtypeGrid>::computeMesonFields(mf_ls_grid, mf_sl_grid,
						    W_grid, V_grid,
						    W_grid, V_grid,
						    kaon_mom,
						    2.0, lattice, simd_dims_3d);

  fMatrix<typename A2Apolicies_std::ScalarComplexType> fmat_std;
  ComputeKaon<VtypeStd,WtypeStd>::compute(fmat_std, mf_ls_std, mf_sl_std);

  fMatrix<typename A2Apolicies_grid::ScalarComplexType> fmat_grid;
  ComputeKaon<VtypeGrid,WtypeGrid>::compute(fmat_grid, mf_ls_grid, mf_sl_grid);
  
  bool fail = false;
  for(int r=0;r<fmat_std.nRows();r++){
    for(int c=0;c<fmat_std.nCols();c++){
      double rdiff = fmat_std(r,c).real() - fmat_grid(r,c).real();
      double idiff = fmat_std(r,c).imag() - fmat_grid(r,c).imag();
      if(rdiff > tol|| idiff > tol){
	printf("Fail Kaon %d %d : (%f,%f) (%f,%f) diff (%g,%g)\n",r,c, fmat_std(r,c).real(),  fmat_std(r,c).imag(), fmat_grid(r,c).real(), fmat_grid(r,c).imag(), rdiff, idiff);
	fail = true;
      }
    }
  }
  if(fail)ERR.General("","","Standard vs Grid implementation kaon test failed\n");
  std::cout << "testKaonContractionGridStd passed" << std::endl;
}


CPS_END_NAMESPACE
