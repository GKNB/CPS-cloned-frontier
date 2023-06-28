#include <alg/a2a/ktopipi_gparity.h>
#include <alg/a2a/a2a_fields.h>

using namespace cps;

#include "test_a2a.h"

inline int toInt(const char* a){
  std::stringstream ss; ss << a; int o; ss >> o;
  return o;
}

void setupDoArg(DoArg &do_arg, int size[5], int ngp, bool verbose = true){
  do_arg.x_sites = size[0];
  do_arg.y_sites = size[1];
  do_arg.z_sites = size[2];
  do_arg.t_sites = size[3];
  do_arg.s_sites = size[4];
  do_arg.x_node_sites = 0;
  do_arg.y_node_sites = 0;
  do_arg.z_node_sites = 0;
  do_arg.t_node_sites = 0;
  do_arg.s_node_sites = 0;
  do_arg.x_nodes = 0;
  do_arg.y_nodes = 0;
  do_arg.z_nodes = 0;
  do_arg.t_nodes = 0;
  do_arg.s_nodes = 0;
  do_arg.updates = 0;
  do_arg.measurements = 0;
  do_arg.measurefreq = 0;
  do_arg.cg_reprod_freq = 10;
  do_arg.x_bc = BND_CND_PRD;
  do_arg.y_bc = BND_CND_PRD;
  do_arg.z_bc = BND_CND_PRD;
  do_arg.t_bc = BND_CND_APRD;
  do_arg.start_conf_kind = START_CONF_ORD;
  do_arg.start_conf_load_addr = 0x0;
  do_arg.start_seed_kind = START_SEED_FIXED;
  do_arg.start_seed_filename = "../rngs/ckpoint_rng.0";
  do_arg.start_conf_filename = "../configurations/ckpoint_lat.0";
  do_arg.start_conf_alloc_flag = 6;
  do_arg.wfm_alloc_flag = 2;
  do_arg.wfm_send_alloc_flag = 2;
  do_arg.start_seed_value = 83209;
  do_arg.beta =   2.25;
  do_arg.c_1 =   -3.3100000000000002e-01;
  do_arg.u0 =   1.0000000000000000e+00;
  do_arg.dwf_height =   1.8000000000000000e+00;
  do_arg.dwf_a5_inv =   1.0000000000000000e+00;
  do_arg.power_plaq_cutoff =   0.0000000000000000e+00;
  do_arg.power_plaq_exponent = 0;
  do_arg.power_rect_cutoff =   0.0000000000000000e+00;
  do_arg.power_rect_exponent = 0;
  do_arg.verbose_level = -1202; //VERBOSE_DEBUG_LEVEL; //-1202;
  do_arg.checksum_level = 0;
  do_arg.exec_task_list = 0;
  do_arg.xi_bare =   1.0000000000000000e+00;
  do_arg.xi_dir = 3;
  do_arg.xi_v =   1.0000000000000000e+00;
  do_arg.xi_v_xi =   1.0000000000000000e+00;
  do_arg.clover_coeff =   0.0000000000000000e+00;
  do_arg.clover_coeff_xi =   0.0000000000000000e+00;
  do_arg.xi_gfix =   1.0000000000000000e+00;
  do_arg.gfix_chkb = 1;
  do_arg.asqtad_KS =   0.0000000000000000e+00;
  do_arg.asqtad_naik =   0.0000000000000000e+00;
  do_arg.asqtad_3staple =   0.0000000000000000e+00;
  do_arg.asqtad_5staple =   0.0000000000000000e+00;
  do_arg.asqtad_7staple =   0.0000000000000000e+00;
  do_arg.asqtad_lepage =   0.0000000000000000e+00;
  do_arg.p4_KS =   0.0000000000000000e+00;
  do_arg.p4_knight =   0.0000000000000000e+00;
  do_arg.p4_3staple =   0.0000000000000000e+00;
  do_arg.p4_5staple =   0.0000000000000000e+00;
  do_arg.p4_7staple =   0.0000000000000000e+00;
  do_arg.p4_lepage =   0.0000000000000000e+00;

  if(verbose) do_arg.verbose_level = VERBOSE_DEBUG_LEVEL;

  BndCndType* bc[3] = { &do_arg.x_bc, &do_arg.y_bc, &do_arg.z_bc };
  for(int i=0;i<ngp;i++){ 
    *(bc[i]) = BND_CND_GPARITY;
  }
}

struct Options{
  double gfix_alpha;
  bool do_init_gauge_fix;
  Options(){
    gfix_alpha = 0.01;
    do_init_gauge_fix = false;
  }
};

void testGparity(CommonArg &common_arg, A2AArg &a2a_arg, FixGaugeArg &fix_gauge_arg, LancArg &lanc_arg, DoArg &do_arg, int ntests, int nthreads, double tol, const Options &opt){
  //Setup types
  typedef A2ApoliciesSIMDdoubleAutoAllocGparity A2Apolicies_grid;
  typedef A2ApoliciesDoubleAutoAllocGparity A2Apolicies_std; 

  typedef A2Apolicies_std::ComplexType mf_Complex;
  typedef A2Apolicies_grid::ComplexType grid_Complex;
  typedef A2Apolicies_grid::FgridGFclass LatticeType;  
  typedef A2Apolicies_grid::GridFermionField GridFermionField;

  //Setup SIMD info
  int nsimd = grid_Complex::Nsimd();
  typename SIMDpolicyBase<4>::ParamType simd_dims;
  SIMDpolicyBase<4>::SIMDdefaultLayout(simd_dims,nsimd,2); //only divide over spatial directions
  
  printf("Nsimd = %d, SIMD dimensions:\n", nsimd);
  for(int i=0;i<4;i++)
    printf("%d ", simd_dims[i]);
  printf("\n");

  typename SIMDpolicyBase<3>::ParamType simd_dims_3d;
  SIMDpolicyBase<3>::SIMDdefaultLayout(simd_dims_3d,nsimd);
  
  //Setup lattice
  FgridParams grid_params; 
  grid_params.mobius_scale = 2.0;
  LatticeType lattice(grid_params);
  lattice.ImportGauge(); //lattice -> Grid  

  doGaugeFix(lattice,!opt.do_init_gauge_fix,fix_gauge_arg);
  
  A2AvectorV<A2Apolicies_grid> V_grid(a2a_arg,simd_dims), Vh_grid(a2a_arg,simd_dims);
  A2AvectorW<A2Apolicies_grid> W_grid(a2a_arg,simd_dims), Wh_grid(a2a_arg,simd_dims);

  LatRanGen lrg_bak = LRG;
  randomizeVW<A2Apolicies_grid>(V_grid,W_grid);
  randomizeVW<A2Apolicies_grid>(Vh_grid,Wh_grid);

  A2AvectorV<A2Apolicies_std> V_std(a2a_arg), Vh_std(a2a_arg);
  A2AvectorW<A2Apolicies_std> W_std(a2a_arg), Wh_std(a2a_arg);

  LRG = lrg_bak;
  randomizeVW<A2Apolicies_std>(V_std,W_std);
  randomizeVW<A2Apolicies_std>(Vh_std,Wh_std);


  //Sanity check to ensure Grid and non-Grid V,W are the same
  for(int i=0;i<V_grid.getNmodes();i++){
    double v = fabs(V_grid.getMode(i).norm2() - V_std.getMode(i).norm2());
    if(v > tol){
      std::cout << "Error in V setup, field " << i << " norm2 diff " << v << std::endl;
      exit(1);
    }
  }

  for(int i=0;i<W_grid.getNl();i++){
    double v = fabs(W_grid.getWl(i).norm2() - W_std.getWl(i).norm2());
    if(v > tol){
      std::cout << "Error in Wl setup, field " << i << " norm2 diff " << v << std::endl;
      exit(1);
    }
  }

  for(int i=0;i<W_grid.getNhits();i++){
    double v = fabs(W_grid.getWh(i).norm2() - W_std.getWh(i).norm2());
    if(v > tol){
      std::cout << "Error in Wh setup, field " << i << " norm2 diff " << v << std::endl;
      exit(1);
    }
  }

  
  std::cout << "OPENMP threads is " << omp_get_max_threads() << std::endl;
  std::cout << "Starting tests" << std::endl;

  /*
  if(1) testPoolAllocator();
  if(1) testAsyncTransferManager();
  if(1) testHolisticPoolAllocator();
  if(1) testMesonFieldViews<A2Apolicies_grid>(a2a_arg);
  
  if(1) testCPSfieldDeviceCopy<A2Apolicies_grid>();
  if(1) testAutoView();
  if(1) testViewArray();
  if(1) testCPSfieldArray<A2Apolicies_grid>();

  if(1) testMemoryStorageBase();
  if(1) testBurstBufferMemoryStorage();
  if(1) testDistributedStorage();
  if(1) testDistributedStorageOneSided();
  //if(1) testMmapMemoryStorage(); //mmap is not implemented on crusher, apparently
  
  if(1) testA2AfieldAccess<A2Apolicies_grid>();
  if(1) testCPSfieldDeviceCopy<A2Apolicies_grid>();
  if(1) testMultiSourceDeviceCopy<A2Apolicies_grid>();
  
  if(1) testCPSsquareMatrix();
  if(1) testCPSspinColorMatrix();

  if(1) checkCPSfieldGridImpex5Dcb<A2Apolicies_grid>(lattice);

#ifdef USE_GRID
#ifdef GRID_SYCL
  if(1) testOneMKLwrapper();
#endif
#endif

  if(1) testFlavorProjectedSourceView<A2Apolicies_grid>();
  
  if(1) testMFmult<A2Apolicies_std>(a2a_arg,tol);
#ifdef USE_GRID
  if(1) testMFmult<A2Apolicies_grid>(a2a_arg,tol);
#endif

 
  if(1) testMFmultTblock<A2Apolicies_std>(a2a_arg,tol);
#ifdef USE_GRID
  if(1) testMFmultTblock<A2Apolicies_grid>(a2a_arg,tol);
#endif
  */
  //if(1) testCshiftCconjBc();
  //if(1) testCshiftCconjBcMatrix(simd_dims);
  //if(1) testGfixCPSmatrixField(lattice, simd_dims);

  if(1) testGridGaugeFix(lattice, opt.gfix_alpha, simd_dims);
  /*
  if(1) testGaugeFixAndPhasingGridStd<A2Apolicies_std, A2Apolicies_grid>(simd_dims,lattice);
   
  if(1) testFlavorMatrixSCcontractStd<A2Apolicies_std>(tol);
  if(1) testGparityInnerProduct<A2Apolicies_std>(tol);

  if(1) testA2AfieldGetFlavorDilutedVect<A2Apolicies_std>(a2a_arg, tol);

  if(1) testMesonFieldNormGridStd<A2Apolicies_std, A2Apolicies_grid>(a2a_arg, tol);

  if(1) testMesonFieldComputeReference<A2Apolicies_std>(a2a_arg, tol);

  if(1) testMesonFieldComputePackedReference<A2Apolicies_std>(a2a_arg, tol);
    
  if(1) testMesonFieldComputeSingleReference<A2Apolicies_std>(a2a_arg, tol);
  
  if(1) testMesonFieldComputeSingleMulti<A2Apolicies_std>(a2a_arg, tol);

  if(1) testGridMesonFieldCompute<A2Apolicies_std, A2Apolicies_grid>(a2a_arg, nthreads, tol);
  if(1) testGridMultiSourceMesonFieldCompute<A2Apolicies_grid>(a2a_arg, nthreads, tol);
  if(1) testGridShiftMultiSourceMesonFieldCompute<A2Apolicies_grid>(a2a_arg, nthreads, tol);

  if(1) testFFTopt<A2Apolicies_std>();
#ifdef USE_GRID
  if(1) testFFTopt<A2Apolicies_grid>();
#endif

#ifdef USE_GRID
  if(1) testGridGetTwistedFFT<A2Apolicies_grid>(a2a_arg, nthreads, tol);
#endif
  
#ifdef USE_GRID
  if(1) testGridMesonFieldComputeManySimple<A2Apolicies_grid>(V_grid,W_grid,a2a_arg,lattice,simd_dims_3d,simd_dims,tol);
#endif
  
  if(1) testPionContractionGridStd<A2Apolicies_std, A2Apolicies_grid>(V_std, W_std,
							 V_grid, W_grid,
							 lattice, simd_dims_3d, tol);

  if(1) testKaonContractionGridStd<A2Apolicies_std, A2Apolicies_grid>(V_std, W_std,
							 V_grid, W_grid,
							 lattice, simd_dims_3d, tol);

  if(1) testPiPiContractionGridStd<A2Apolicies_std, A2Apolicies_grid>(V_std, W_std,
								V_grid, W_grid,
								lattice, simd_dims_3d, tol);

  if(1) testConvertComplexD();

  if(1) testBasicComplexArray<A2Apolicies_grid>();
  
  //Test the openmp Grid vs non-Grid implementation
  if(1) testKtoPiPiType1GridOmpStd<A2Apolicies_std, A2Apolicies_grid>(a2a_arg,
								      W_grid, V_grid, Wh_grid, Vh_grid,
								      W_std, V_std, Wh_std, Vh_std,
  								      tol);
  if(1) testhostDeviceMirroredContainer();

  if(1) testvMvGridOrigGparity<A2Apolicies_std, A2Apolicies_grid>(a2a_arg, nthreads, tol);
  if(1) testvMvGridOrigGparityTblock<A2Apolicies_std, A2Apolicies_grid>(a2a_arg, nthreads, tol);
  if(1) testvMvFieldTimesliceRange<A2Apolicies_grid>(a2a_arg, tol);
  if(1) testvMvFieldArbitraryNtblock<A2Apolicies_grid>(a2a_arg, do_arg, tol);


  if(1) testVVgridOrigGparity<A2Apolicies_std, A2Apolicies_grid>(a2a_arg, nthreads, tol);
  if(1) testVVgridOrigGparityTblock<A2Apolicies_std, A2Apolicies_grid>(a2a_arg, nthreads, tol);

  
  if(1) testCPSmatrixField<A2Apolicies_grid>(tol);

  if(1) testKtoPiPiType4FieldContraction<A2Apolicies_grid>(tol);
  
  if(1) testKtoPiPiType1FieldFull<A2Apolicies_grid>(a2a_arg,tol);
  if(1) testKtoPiPiType2FieldFull<A2Apolicies_grid>(a2a_arg,tol);

  if(1) testKtoPiPiType3FieldFull<A2Apolicies_grid>(a2a_arg,tol);
  if(1) testKtoPiPiType4FieldFull<A2Apolicies_grid>(a2a_arg,tol);

  if(1) testKtoSigmaType12FieldFull<A2Apolicies_grid>(a2a_arg,tol);
  if(1) testKtoSigmaType3FieldFull<A2Apolicies_grid>(a2a_arg,tol);
  if(1) testKtoSigmaType4FieldFull<A2Apolicies_grid>(a2a_arg,tol);

  if(1) testKtoPiPiContractionGridStd<A2Apolicies_std, A2Apolicies_grid>(V_std, W_std,
  									 V_grid, W_grid,
  									 lattice, simd_dims_3d, tol);

  if(1) testModeMappingTranspose(a2a_arg);

#ifdef USE_GRID
  if(1) testComputeLowModeMADWF<A2Apolicies_grid>(a2a_arg, lanc_arg, lattice, simd_dims, tol);
#endif

#ifdef USE_GRID
  if(1) testMADWFprecon<A2Apolicies_grid>(a2a_arg, lanc_arg, lattice, simd_dims, tol);
#endif
  
  if(1) testCyclicPermute();
  if(1) demonstrateFFTreln<A2Apolicies_std>(a2a_arg);
  if(1) testA2AvectorFFTrelnGparity<A2Apolicies_grid>(a2a_arg, lattice);
  if(1) testMultiSource<A2Apolicies_grid>(a2a_arg, lattice);

  if(1) testSumSource<A2Apolicies_grid>(a2a_arg, lattice);

  if(1) testMfFFTreln<A2Apolicies_grid>(a2a_arg, lattice);
  if(1) testA2AFFTinv<A2Apolicies_grid>(a2a_arg, lattice);
  if(1) testGridg5Contract<grid_Complex>();
  if(1) testGaugeFixInvertible<A2Apolicies_grid>(lattice);
  if(1) testDestructiveFFT<A2ApoliciesSIMDdoubleManualAllocGparity>(a2a_arg, lattice);
  if(1) testMesonFieldReadWrite<A2Apolicies_std>(a2a_arg);
  if(1) testMesonFieldTraceSingle<A2Apolicies_grid>(a2a_arg,tol);
  if(1) testMesonFieldTraceSingleTblock<A2Apolicies_grid>(a2a_arg,tol);
  if(1) testMesonFieldTraceProduct<A2Apolicies_grid>(a2a_arg,tol);
  if(1) testMesonFieldTraceProductTblock<A2Apolicies_grid>(a2a_arg,tol);
  if(1) testMesonFieldTraceProductAllTimes<A2Apolicies_grid>(a2a_arg,tol);
  if(1) testCPSfieldImpex();
  if(1) testGridFieldImpex<A2Apolicies_grid>(lattice);
  if(1) testCPSfieldIO();
  if(1) testA2AvectorIO<A2Apolicies_grid>(a2a_arg);
  if(1) testLanczosIO<A2Apolicies_grid>(lattice);
  if(1) testSCFmat();
  if(1) testMesonFieldUnpackPack<A2Apolicies_grid>(a2a_arg,tol);
  if(1) testMesonFieldUnpackPackTblock<A2Apolicies_grid>(a2a_arg,tol);
  
#ifdef USE_GRID
  if(1) testGaugeFixOrigNew<A2Apolicies_std, A2Apolicies_grid>(simd_dims,lattice);
#endif

  if(1) testMesonFieldNodeDistributeUnique(a2a_arg);
  if(1) testMesonFieldNodeDistributeOneSided(a2a_arg);

  if(1) testA2AvectorTimesliceExtraction<A2Apolicies_grid>(a2a_arg);

  //if(1) testCompressedEvecInterface<A2Apolicies_grid>(lattice,tol);  Current compilation issues on Intel

  if(1) testA2AvectorWnorm<A2Apolicies_grid>(a2a_arg);

  if(1) testXconjAction<A2Apolicies_grid>(lattice, tol);  

  if(1) test_gamma_CPS_vs_Grid();

    /*
  if(1) testXconjWsrc<A2Apolicies_grid>(lattice);
  if(1) testXconjWsrcPostOp<A2Apolicies_grid>(lattice);
  if(1) testXconjWsrcInverse<A2Apolicies_grid>(lattice);
  if(1) testXconjWsrcFull<A2Apolicies_grid>(lattice);
  if(1) testXconjWsrcCConjRelnV<A2Apolicies_grid>(lattice);
  if(1) testXconjWsrcCConjReln<A2Apolicies_grid>(lattice);


  */

}




void testPeriodic(CommonArg &common_arg, A2AArg &a2a_arg, FixGaugeArg &fix_gauge_arg, LancArg &lanc_arg, int ntests, int nthreads, double tol, const Options &opt){
  //Setup types
  typedef A2ApoliciesSIMDdoubleAutoAlloc A2Apolicies_grid;
  typedef A2ApoliciesDoubleAutoAlloc A2Apolicies_std; 

  typedef A2Apolicies_std::ComplexType mf_Complex;
  typedef A2Apolicies_grid::ComplexType grid_Complex;
  typedef A2Apolicies_grid::FgridGFclass LatticeType;  
  typedef A2Apolicies_grid::GridFermionField GridFermionField;

  //Setup SIMD info
  int nsimd = grid_Complex::Nsimd();
  typename SIMDpolicyBase<4>::ParamType simd_dims;
  SIMDpolicyBase<4>::SIMDdefaultLayout(simd_dims,nsimd,2); //only divide over spatial directions
  
  printf("Nsimd = %d, SIMD dimensions:\n", nsimd);
  for(int i=0;i<4;i++)
    printf("%d ", simd_dims[i]);
  printf("\n");

  typename SIMDpolicyBase<3>::ParamType simd_dims_3d;
  SIMDpolicyBase<3>::SIMDdefaultLayout(simd_dims_3d,nsimd);
  
  //Setup lattice
  FgridParams grid_params; 
  grid_params.mobius_scale = 2.0;
  LatticeType lattice(grid_params);
  lattice.ImportGauge(); //lattice -> Grid  

  doGaugeFix(lattice,!opt.do_init_gauge_fix,fix_gauge_arg);
  
  A2AvectorV<A2Apolicies_grid> V_grid(a2a_arg,simd_dims), Vh_grid(a2a_arg,simd_dims);
  A2AvectorW<A2Apolicies_grid> W_grid(a2a_arg,simd_dims), Wh_grid(a2a_arg,simd_dims);

  LatRanGen lrg_bak = LRG;
  randomizeVW<A2Apolicies_grid>(V_grid,W_grid);
  randomizeVW<A2Apolicies_grid>(Vh_grid,Wh_grid);

  A2AvectorV<A2Apolicies_std> V_std(a2a_arg), Vh_std(a2a_arg);
  A2AvectorW<A2Apolicies_std> W_std(a2a_arg), Wh_std(a2a_arg);

  LRG = lrg_bak;
  randomizeVW<A2Apolicies_std>(V_std,W_std);
  randomizeVW<A2Apolicies_std>(Vh_std,Wh_std);


  //Sanity check to ensure Grid and non-Grid V,W are the same
  for(int i=0;i<V_grid.getNmodes();i++){
    double v = fabs(V_grid.getMode(i).norm2() - V_std.getMode(i).norm2());
    if(v > tol){
      std::cout << "Error in V setup, field " << i << " norm2 diff " << v << std::endl;
      exit(1);
    }
  }

  for(int i=0;i<W_grid.getNl();i++){
    double v = fabs(W_grid.getWl(i).norm2() - W_std.getWl(i).norm2());
    if(v > tol){
      std::cout << "Error in Wl setup, field " << i << " norm2 diff " << v << std::endl;
      exit(1);
    }
  }

  for(int i=0;i<W_grid.getNhits();i++){
    double v = fabs(W_grid.getWh(i).norm2() - W_std.getWh(i).norm2());
    if(v > tol){
      std::cout << "Error in Wh setup, field " << i << " norm2 diff " << v << std::endl;
      exit(1);
    }
  }

  
  std::cout << "OPENMP threads is " << omp_get_max_threads() << std::endl;
  std::cout << "Starting tests" << std::endl;


  if(1) testCshiftCconjBc();
  if(1) testCshiftCconjBcMatrix(simd_dims);
  if(1) testGfixCPSmatrixField(lattice, simd_dims);

  if(1) testGridGaugeFix(lattice, opt.gfix_alpha, simd_dims);

  ///   if(1) testCPSspinColorMatrix();
// #ifdef USE_GRID
//   if(1) testvMvGridOrigPeriodic<A2Apolicies_std, A2Apolicies_grid>(a2a_arg, nthreads, tol);
//   if(1) testVVgridOrigPeriodic<A2Apolicies_std, A2Apolicies_grid>(a2a_arg, 1, nthreads, tol);
// #endif

}



int main(int argc,char *argv[])
{
  if(argc < 3)
    ERR.General("","main","Need #GPdirs and working directory\n");
  
  Start(&argc, &argv);
  int ngp;
  { std::stringstream ss; ss << argv[1]; ss >> ngp; }

  if(!UniqueID()){
    if(ngp > 0) printf("Doing G-parity in %d directions\n",ngp);
    else printf("Doing periodic BCs\n",ngp);
  }
  
  std::string workdir = argv[2];
  if(chdir(workdir.c_str())!=0) ERR.General("","main","Unable to switch to directory '%s'\n",workdir.c_str());
  
  bool save_config(false);
  bool load_config(false);
  bool load_lrg(false);
  bool save_lrg(false);
  char *load_config_file;
  char *save_config_file;
  char *save_lrg_file;
  char *load_lrg_file;
  bool verbose(false);
  bool unit_gauge(false);
  bool load_lanc_arg(false);  
  std::string lanc_arg_file;

  Options opt;

  int size[] = {4,4,4,4,4};
  int nthreads = 1;
  int ntests = 10;
  
  double tol = 1e-5;

  int nl = 10;

  const int ngrid_arg = 16;
  const std::string grid_args[ngrid_arg] = { "--debug-signals", "--dslash-generic", "--dslash-unroll",
					     "--dslash-asm", "--shm", "--lebesgue",
					     "--cacheblocking", "--comms-concurrent", "--comms-sequential",
					     "--comms-overlap", "--log", "--comms-threads",
					     "--shm-hugepages", "--accelerator-threads",
					     "--device-mem", "--shm-mpi"};
  const int grid_args_skip[ngrid_arg] =    { 1  , 1 , 1,
					     1  , 2 , 1,
					     2  , 1 , 1,
					     1  , 2 , 2,
					     1  , 2,
					     2  , 2};
  
  printf("Argc is %d\n",argc);
  int i=3;
  while(i<argc){
    std::string cmd = argv[i];  
    if( cmd == "-save_config"){
      if(i==argc-1){ printf("-save_config requires an argument\n"); exit(-1); }
      save_config=true;
      save_config_file = argv[i+1];
      i+=2;
    }else if( cmd == "-load_config"){
      if(i==argc-1){ printf("-save_config requires an argument\n"); exit(-1); }
      load_config=true;
      load_config_file = argv[i+1];
      i+=2;
    }else if( cmd == "-old_gparity_cfg"){
      LatticeHeader::GparityMultPlaqByTwo() = true;
      i+=1;
    }else if( cmd == "-latt" ){
      if(i>argc-6){
	printf("Did not specify enough arguments for 'latt' (require 5 dimensions)\n"); exit(-1);
      }
      for(int d=0;d<5;d++)
	size[d] = toInt(argv[i+1+d]);
      i+=6;
    }else if( cmd == "-load_lrg"){
      if(i==argc-1){ printf("-load_lrg requires an argument\n"); exit(-1); }
      load_lrg=true;
      load_lrg_file = argv[i+1];
      i+=2;
    }else if( cmd == "-save_lrg"){
      if(i==argc-1){ printf("-save_lrg requires an argument\n"); exit(-1); }
      save_lrg=true;
      save_lrg_file = argv[i+1];
      i+=2;  
    }else if( cmd == "-verbose"){
      verbose=true;
      i++;
    }else if( cmd == "-nthread"){
      nthreads = toInt(argv[i+1]);
      printf("Set nthreads to %d\n", nthreads);
      i+=2;
    }else if( cmd == "-ntest" ){
      ntests = toInt(argv[i+1]);
      printf("Set ntests to %d\n", ntests);
      i+=2;
    }else if( cmd == "-unit_gauge"){
      unit_gauge=true;
      i++;
    }else if( cmd == "-tolerance"){
      std::stringstream ss; ss << argv[i+1];
      ss >> tol;
      if(!UniqueID()) printf("Set tolerance to %g\n",tol);
      i+=2;
    }else if( cmd == "-load_lanc_arg"){
      load_lanc_arg = true;
      lanc_arg_file = argv[i+1];
      i+=2;
    }else if( cmd == "-vMv_offload_blocksize" ){
      std::stringstream ss; ss  << argv[i+1]; ss >> BlockedvMvOffloadArgs::b;
      if(!UniqueID()) printf("Set vMv offload blocksize to %d\n", BlockedvMvOffloadArgs::b);
      i+=2;
    }else if( cmd == "-vMv_offload_inner_blocksize" ){
      std::stringstream ss; ss  << argv[i+1]; ss >> BlockedvMvOffloadArgs::bb;
      if(!UniqueID()) printf("Set vMv offload inner blocksize to %d\n", BlockedvMvOffloadArgs::bb);
      i+=2;
    }else if( cmd == "-vMv_split_blocksize" ){
      std::stringstream ss; ss  << argv[i+1]; ss >> BlockedSplitvMvArgs::b;
      if(!UniqueID()) printf("Set vMv split blocksize to %d\n", BlockedSplitvMvArgs::b);
      i+=2;
    }else if( cmd == "-nl" ){
      std::stringstream ss; ss  << argv[i+1]; ss >> nl;
      if(!UniqueID()) printf("Set nl to %d\n", nl);
      i+=2;
    }else if( cmd == "-gfix_alpha" ){
      std::stringstream ss; ss  << argv[i+1]; ss >> opt.gfix_alpha;
      if(!UniqueID()) printf("Set alpha to %f\n", opt.gfix_alpha);
      i+=2;
    }else if( cmd == "-mempool_verbose"){
      HolisticMemoryPoolManager::globalPool().setVerbose(true);
      DeviceMemoryPoolManager::globalPool().setVerbose(true);
      i++;
    }else if( cmd == "-do_init_gauge_fix"){
      opt.do_init_gauge_fix = true;
      std::cout << "Doing initial gauge fixing" << std::endl;
      i++;
    }else{
      bool is_grid_arg = false;
      for(int ii=0;ii<ngrid_arg;ii++){
	if( cmd == grid_args[ii] ){
	  if(!UniqueID()){ printf("main.C: Ignoring Grid argument %s\n",cmd.c_str()); fflush(stdout); }
	  i += grid_args_skip[ii];
	  is_grid_arg = true;
	  break;
	}
      }
      if(!is_grid_arg){
	if(UniqueID()==0) printf("Unrecognised argument: %s\n",cmd.c_str());
	exit(-1);
      }
    }
  }

  LancArg lanc_arg;
#define L(A,B) lanc_arg.A = B
  L(mass, 0.01);
  L(qr_rsd, 1e-10);
  L(EigenOper, DDAG);
  L(precon, 1);
  L(N_get, nl);
  L(N_use, nl+4);
  L(N_true_get, nl);
  L(ch_ord, 10);
  L(ch_alpha, 1e-3);
  L(ch_beta, 1.5);
  L(ch_sh, false);
  L(ch_mu, 0);
  L(lock, false);
  L(maxits, 100);
  L(fname, "");
    
  if(load_lanc_arg){
    if(!lanc_arg.Decode((char*)lanc_arg_file.c_str(),"lanc_arg")){
      lanc_arg.Encode("lanc_arg.templ","lanc_arg");
      VRB.Result("","main","Can't open lanc_arg.vml!\n");exit(1);
    }
  }

  A2AArg a2a_arg;
  a2a_arg.nl = lanc_arg.N_true_get;
  a2a_arg.nhits = 1;
  a2a_arg.rand_type = UONE;
  a2a_arg.src_width = 1;

  JobParams jp; //nothing actually needed from here
  
  printf("Lattice size is %d %d %d %d\n",size[0],size[1],size[2],size[3],size[4]);

  CommonArg common_arg;
  DoArg do_arg;  setupDoArg(do_arg,size,ngp,verbose);

  GJP.Initialize(do_arg);
  GJP.SetNthreads(nthreads);
  assert(omp_get_max_threads() == nthreads);
  std::cout << "Using " << nthreads << " threads" << std::endl;

#if TARGET == BGQ
  LRG.setSerial();
#endif
  LRG.Initialize(); //usually initialised when lattice generated, but I pre-init here so I can load the state from file

  FixGaugeArg fix_gauge_arg;  
  fix_gauge_arg.fix_gauge_kind = FIX_GAUGE_COULOMB_T; //if initial gauge fixing is disabled, CPS will think it has Coulomb gauge fixing matrices, but they will all be unit matrices
  fix_gauge_arg.hyperplane_start = 0;
  fix_gauge_arg.hyperplane_step = 1;
  fix_gauge_arg.hyperplane_num = GJP.Tnodes()*GJP.TnodeSites();
  fix_gauge_arg.stop_cond = 1e-8;
  fix_gauge_arg.max_iter_num = 10000;
  
  {
    GnoneFnone lattice_tmp; //is destroyed at end of scope but the underlying gauge field remains in memory

    if(load_lrg){
      if(UniqueID()==0) printf("Loading RNG state from %s\n",load_lrg_file);
      LRG.Read(load_lrg_file,32);
    }
    if(save_lrg){
      if(UniqueID()==0) printf("Writing RNG state to %s\n",save_lrg_file);
      LRG.Write(save_lrg_file,32);
    }					       
    if(!load_config){
      printf("Creating gauge field\n");
      if(!unit_gauge){
	std::cout << "Creating disordered gauge field" << std::endl;
	lattice_tmp.SetGfieldDisOrd();
      }
      else{
	std::cout << "Creating unit gauge field" << std::endl;
	lattice_tmp.SetGfieldOrd();
      }
    }else{
      ReadLatticeParallel readLat;
      if(UniqueID()==0) printf("Reading: %s (NERSC-format)\n",load_config_file);
      readLat.read(lattice_tmp,load_config_file);
      if(UniqueID()==0) printf("Config read.\n");
    }
    if(save_config){
      if(UniqueID()==0) printf("Saving config to %s\n",save_config_file);

      QioArg wt_arg(save_config_file,0.001);
    
      wt_arg.ConcurIONumber=32;
      WriteLatticeParallel wl;
      wl.setHeader("disord_id","disord_label",0);
      wl.write(lattice_tmp,wt_arg);
    
      if(!wl.good()) ERR.General("main","()","Failed write lattice %s",save_config_file);

      if(UniqueID()==0) printf("Config written.\n");
    }
  }

  if(ngp > 0) testGparity(common_arg, a2a_arg, fix_gauge_arg, lanc_arg, do_arg, ntests, nthreads, tol, opt);
  else testPeriodic(common_arg, a2a_arg, fix_gauge_arg, lanc_arg, ntests, nthreads, tol, opt);

  std::cout << "Done" << std::endl;

  End();
  return 0;
}
