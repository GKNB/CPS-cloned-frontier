#ifndef USE_GRID
#error "Requires Grid"
#endif
#define A2A_PREC_SIMD_DOUBLE
#define USE_GRID_LANCZOS
#define USE_GRID_A2A

#include<alg/a2a/mesonfield.h>

#include <Hadrons/A2AMatrix.hpp>

using namespace cps;

#include "main.h"

int main(int argc,char *argv[])
{
  Start(&argc, &argv);
  
  bool verbose = true;
  int ngp = 0;
#ifdef USE_GRID_GPARITY
  ngp = 3;
#endif

  int size[5] = {4,4,4,4,4};

  CommonArg common_arg;
  DoArg do_arg;  setupDoArg(do_arg,size,ngp,verbose);

  GJP.Initialize(do_arg);

  typedef A2ApoliciesSIMDdoubleAutoAlloc A2Apolicies;

  //Setup the lattice
  FgridParams fgp; fgp.epsilon = 0.; fgp.mobius_scale = 32./12.;
  typename A2Apolicies::FgridGFclass lattice(fgp);
  lattice.SetGfieldDisOrd();

  A2AArg a2a_args;
  a2a_args.nl = 100;
  a2a_args.nhits = 1;
  a2a_args.rand_type = UONE;
  a2a_args.src_width = 1;
  
  //This is the CPSfield equivalent of setting up Grid's Grids
  typedef A2Apolicies::FermionFieldType::InputParamType simd_params_4d;
  typedef A2Apolicies::SourcePolicies::MappingPolicy::ParamType simd_params_3d;
  typedef typename A2Asource<typename A2Apolicies::SourcePolicies::ComplexType, typename A2Apolicies::SourcePolicies::MappingPolicy, 
			     typename A2Apolicies::SourcePolicies::AllocPolicy>::FieldType SourceFieldType;

  simd_params_4d sp4;
  setupFieldParams<typename A2Apolicies::FermionFieldType>(sp4);

  simd_params_3d sp3;
  setupFieldParams<SourceFieldType>(sp3);

  A2AvectorW<A2Apolicies> W(a2a_args, sp4);
  W.testRandom();

  A2AvectorV<A2Apolicies> V(a2a_args, sp4);
  V.testRandom();

  //Do the conversion
  std::vector<typename A2Apolicies::GridFermionField> W_grid;
  convertToGrid(W_grid, W, lattice.getUGrid());

  std::vector<typename A2Apolicies::GridFermionField> V_grid;
  convertToGrid(V_grid, V, lattice.getUGrid());
  
  //FFT the V,W vectors
  A2AvectorVfftw<A2Apolicies> Vfft(a2a_args, sp4);
  A2AvectorWfftw<A2Apolicies> Wfft(a2a_args, sp4);
  
  Vfft.fft(V); //I don't really care about gauge fixing or applying quark momentum for this test
  Wfft.fft(W);

  //Make some meson fields
  typedef A2AexpSource<typename A2Apolicies::SourcePolicies> SourceType;
  SourceType source(2.0, sp3);
  SCFspinflavorInnerProductCT<15, sigma3, typename A2Apolicies::SourcePolicies::ComplexType, SourceType> g5_s3(source);
  SCFspinflavorInnerProductCT<1, sigma3, typename A2Apolicies::SourcePolicies::ComplexType, SourceType> g1_s3(source);


  typedef A2AmesonField<A2Apolicies, A2AvectorWfftw, A2AvectorVfftw> MesonFieldType;

  std::vector<MesonFieldType> mf_g5_s3;
  std::vector<MesonFieldType> mf_g1_s3;

  MesonFieldType::compute(mf_g5_s3, Wfft, g5_s3, Vfft);
  MesonFieldType::compute(mf_g1_s3, Wfft, g1_s3, Vfft);

  int Lt = mf_g5_s3.size();

  std::vector<Grid::Hadrons::A2AMatrix<Grid::ComplexD> > mf_g5_s3_grid(Lt), mf_g1_s3_grid(Lt);
  for(int t=0;t<Lt;t++){
    convertToGrid(mf_g5_s3_grid[t], mf_g5_s3[t]);
    convertToGrid(mf_g1_s3_grid[t], mf_g1_s3[t]);
  }  

  //Try some operation so we can make sure we get the same result
  Grid::ComplexD grid_tr(0.);
  Grid::Hadrons::A2AContraction::accTrMul(grid_tr, mf_g5_s3_grid[0], mf_g1_s3_grid[0]);  //tr( A * B )

  ComplexD cps_tr = trace( mf_g5_s3[0], mf_g1_s3[0] );

  std::cout << "Mesonfield check  CPS: " << cps_tr << " Grid: " << grid_tr << std::endl;  

  if(!UniqueID()){ printf("Finished\n"); fflush(stdout); }
  End();
  
  return 0;
}
