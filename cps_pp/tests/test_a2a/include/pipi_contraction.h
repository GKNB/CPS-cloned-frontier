#pragma once

CPS_START_NAMESPACE

template<typename A2Apolicies_std, typename A2Apolicies_grid>
void testPiPiContractionGridStd(A2AvectorV<A2Apolicies_std> &V_std, A2AvectorW<A2Apolicies_std> &W_std,
				A2AvectorV<A2Apolicies_grid> &V_grid, A2AvectorW<A2Apolicies_grid> &W_grid,
				typename A2Apolicies_grid::FgridGFclass &lattice,
				typename SIMDpolicyBase<3>::ParamType simd_dims_3d,
				double tol){
  std::cout << "Starting testPiPiContractionGridStd" << std::endl;
  
  ThreeMomentum p_plus( GJP.Bc(0)==BND_CND_GPARITY? 1 : 0,
			GJP.Bc(1)==BND_CND_GPARITY? 1 : 0,
			GJP.Bc(2)==BND_CND_GPARITY? 1 : 0 );
  ThreeMomentum p_minus = -p_plus;

  ThreeMomentum p_pi_plus = p_plus * 2;
  
  StandardPionMomentaPolicy momenta;
  typedef getMesonFieldType<A2AvectorW<A2Apolicies_std>, A2AvectorV<A2Apolicies_std>> mf_WV_std;
  typedef getMesonFieldType<A2AvectorW<A2Apolicies_grid>, A2AvectorV<A2Apolicies_grid>> mf_WV_grid;
  
  MesonFieldMomentumContainer<mf_WV_std> mf_ll_con_std;
  MesonFieldMomentumContainer<mf_WV_grid> mf_ll_con_grid;
  
  computeGparityLLmesonFields1s<A2AvectorV<A2Apolicies_std>, A2AvectorW<A2Apolicies_std>,StandardPionMomentaPolicy,15,sigma3>::computeMesonFields(mf_ll_con_std,momenta,W_std,V_std,2.0,lattice);
  computeGparityLLmesonFields1s<A2AvectorV<A2Apolicies_grid>, A2AvectorW<A2Apolicies_grid>,StandardPionMomentaPolicy,15,sigma3>::computeMesonFields(mf_ll_con_grid,momenta,W_grid,V_grid,2.0,lattice,simd_dims_3d);

  char diags[] = {'C','D','R'};
  for(int d=0;d<3;d++){
    fMatrix<typename A2Apolicies_std::ScalarComplexType> fmat_std;
    MesonFieldProductStore<mf_WV_std> products_std;
    ComputePiPiGparity<A2AvectorV<A2Apolicies_std>, A2AvectorW<A2Apolicies_std>>::compute(fmat_std, diags[d], p_pi_plus, p_pi_plus, 2, 1, mf_ll_con_std, products_std);

    fMatrix<typename A2Apolicies_grid::ScalarComplexType> fmat_grid;
    MesonFieldProductStore<mf_WV_grid> products_grid;
    ComputePiPiGparity<A2AvectorV<A2Apolicies_grid>, A2AvectorW<A2Apolicies_grid>>::compute(fmat_grid, diags[d], p_pi_plus, p_pi_plus, 2, 1, mf_ll_con_grid, products_grid);

    bool fail = false;
    for(int r=0;r<fmat_std.nRows();r++){
      for(int c=0;c<fmat_std.nCols();c++){
	double rdiff = 2*( fmat_std(r,c).real() - fmat_grid(r,c).real() ) / ( fmat_std(r,c).real() + fmat_grid(r,c).real() );
	double idiff = 2*( fmat_std(r,c).imag() - fmat_grid(r,c).imag() ) / ( fmat_std(r,c).imag() + fmat_grid(r,c).imag() );
	if(rdiff > tol|| idiff > tol){
	  printf("Fail Pipi fig %c elem %d %d : (%f,%f) (%f,%f) reldiff (%g,%g)\n",diags[d],r,c, fmat_std(r,c).real(),  fmat_std(r,c).imag(), fmat_grid(r,c).real(), fmat_grid(r,c).imag(), rdiff, idiff);
	  fail = true;
	}
      }
    }
    if(fail)ERR.General("","","Standard vs Grid implementation pipi fig %c test failed\n",diags[d]);
    printf("Pipi fig %c pass\n",diags[d]);    
  }
    
    
  {
    fVector<typename A2Apolicies_std::ScalarComplexType> pipi_figV_std;
    fVector<typename A2Apolicies_grid::ScalarComplexType> pipi_figV_grid;
    
    ComputePiPiGparity<A2AvectorV<A2Apolicies_std>, A2AvectorW<A2Apolicies_std>>::computeFigureVdis(pipi_figV_std, p_pi_plus, 1, mf_ll_con_std);
    ComputePiPiGparity<A2AvectorV<A2Apolicies_grid>, A2AvectorW<A2Apolicies_grid>>::computeFigureVdis(pipi_figV_grid, p_pi_plus, 1, mf_ll_con_grid);

    bool fail = false;
    for(int r=0;r<pipi_figV_std.size();r++){
      double rdiff = 2*( pipi_figV_std(r).real() - pipi_figV_grid(r).real() ) / ( pipi_figV_std(r).real() + pipi_figV_grid(r).real() );
      double idiff = 2*( pipi_figV_std(r).imag() - pipi_figV_grid(r).imag() ) / ( pipi_figV_std(r).imag() + pipi_figV_grid(r).imag() ) ;
      if(rdiff > tol|| idiff > tol){
	printf("Fail Pipi fig V elem %d : (%f,%f) (%f,%f) reldiff (%g,%g)\n",r, pipi_figV_std(r).real(),  pipi_figV_std(r).imag(), pipi_figV_grid(r).real(), pipi_figV_grid(r).imag(), rdiff, idiff);
	fail = true;
      }      
    }
    if(fail)ERR.General("","","Standard vs Grid implementation pipi fig V test failed\n");
    printf("Pipi fig V pass\n");

    std::cout << "testPiPiContractionGridStd passed" << std::endl;
  }
}


CPS_END_NAMESPACE
