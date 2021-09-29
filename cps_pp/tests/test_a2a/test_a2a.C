#include <alg/a2a/ktopipi_gparity.h>

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

void testGparity(CommonArg &common_arg, A2AArg &a2a_arg, FixGaugeArg &fix_gauge_arg, LancArg &lanc_arg, int ntests, int nthreads, double tol){
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

  AlgFixGauge fix_gauge(lattice,&common_arg,&fix_gauge_arg);
  fix_gauge.run();

  
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

  if(0) testCPSfieldDeviceCopy<A2Apolicies_grid>();
  if(0) testAutoView();
  if(0) testViewArray();
  if(0) testCPSfieldArray<A2Apolicies_grid>();
  
  if(0) testA2AfieldAccess<A2Apolicies_grid>();
  if(0) testCPSfieldDeviceCopy<A2Apolicies_grid>();
  if(0) testMultiSourceDeviceCopy<A2Apolicies_grid>();
  
  if(0) testCPSsquareMatrix();
  if(0) testCPSspinColorMatrix();

  if(0) checkCPSfieldGridImpex5Dcb<A2Apolicies_grid>(lattice);
  
#ifdef USE_GRID
#ifdef GRID_SYCL
  if(0) testOneMKLwrapper();
#endif
#endif

  if(0) testFlavorProjectedSourceView<A2Apolicies_grid>();
  
  if(0) testMFmult<A2Apolicies_std>(a2a_arg,tol);
#ifdef USE_GRID
  if(0) testMFmult<A2Apolicies_grid>(a2a_arg,tol);
#endif
  
  if(0) testGaugeFixAndPhasingGridStd<A2Apolicies_std, A2Apolicies_grid>(simd_dims,lattice);

  
  if(0) compareVgridstd<A2Apolicies_std, A2Apolicies_grid>(V_std, V_grid, tol);
  
  if(0) testFlavorMatrixSCcontractStd<A2Apolicies_std>(tol);
  if(0) testGparityInnerProduct<A2Apolicies_std>(tol);

  if(0) testA2AfieldGetFlavorDilutedVect<A2Apolicies_std>(a2a_arg, tol);

  if(0) testMesonFieldNormGridStd<A2Apolicies_std, A2Apolicies_grid>(a2a_arg, tol);
  
  if(0) testMesonFieldComputeReference<A2Apolicies_std>(a2a_arg, tol);

  if(0) testMesonFieldComputePackedReference<A2Apolicies_std>(a2a_arg, tol);
    
  if(0) testMesonFieldComputeSingleReference<A2Apolicies_std>(a2a_arg, tol);
  
  if(0) testMesonFieldComputeSingleMulti<A2Apolicies_std>(a2a_arg, tol);

  if(1) testGridMesonFieldCompute<A2Apolicies_std, A2Apolicies_grid>(a2a_arg, nthreads, tol);

  if(1) testGridMultiSourceMesonFieldCompute<A2Apolicies_grid>(a2a_arg, nthreads, tol);

  if(1) testGridShiftMultiSourceMesonFieldCompute<A2Apolicies_grid>(a2a_arg, nthreads, tol);

  if(0) testFFTopt<A2Apolicies_std>();
#ifdef USE_GRID
  if(0) testFFTopt<A2Apolicies_grid>();
#endif

#ifdef USE_GRID
  if(0) testGridGetTwistedFFT<A2Apolicies_grid>(a2a_arg, nthreads, tol);
#endif
  
#ifdef USE_GRID
  if(1) testGridMesonFieldComputeManySimple<A2Apolicies_grid>(V_grid,W_grid,a2a_arg,lattice,simd_dims_3d,simd_dims,tol);
#endif
  
  if(0) testPionContractionGridStd<A2Apolicies_std, A2Apolicies_grid>(V_std, W_std,
							 V_grid, W_grid,
							 lattice, simd_dims_3d, tol);

  if(0) testKaonContractionGridStd<A2Apolicies_std, A2Apolicies_grid>(V_std, W_std,
							 V_grid, W_grid,
							 lattice, simd_dims_3d, tol);

  if(0) testPiPiContractionGridStd<A2Apolicies_std, A2Apolicies_grid>(V_std, W_std,
								V_grid, W_grid,
								lattice, simd_dims_3d, tol);
 
#ifdef USE_GRID
  if(0) testConvertComplexD();
  
  //Test the openmp Grid vs non-Grid implementation
  if(0) testKtoPiPiType1GridOmpStd<A2Apolicies_std, A2Apolicies_grid>(a2a_arg,
								      W_grid, V_grid, Wh_grid, Vh_grid,
								      W_std, V_std, Wh_std, Vh_std,
								      tol);
  
  if(0) testvMvGridOrigGparity<A2Apolicies_std, A2Apolicies_grid>(a2a_arg, nthreads, tol);
  if(0) testVVgridOrigGparity<A2Apolicies_std, A2Apolicies_grid>(a2a_arg, nthreads, tol);
  if(0) testCPSmatrixField<A2Apolicies_grid>(tol);

  if(0) testKtoPiPiType4FieldContraction<A2Apolicies_grid>(tol);
  if(0) testKtoPiPiType4FieldFull<A2Apolicies_grid>(a2a_arg,tol);
  if(0) testKtoPiPiType1FieldFull<A2Apolicies_grid>(a2a_arg,tol);
  if(0) testKtoPiPiType2FieldFull<A2Apolicies_grid>(a2a_arg,tol);
  if(0) testKtoPiPiType3FieldFull<A2Apolicies_grid>(a2a_arg,tol);

  if(0) testKtoSigmaType12FieldFull<A2Apolicies_grid>(a2a_arg,tol);
  if(0) testKtoSigmaType3FieldFull<A2Apolicies_grid>(a2a_arg,tol);
  if(0) testKtoSigmaType4FieldFull<A2Apolicies_grid>(a2a_arg,tol);

  if(0) testKtoPiPiContractionGridStd<A2Apolicies_std, A2Apolicies_grid>(V_std, W_std,
									 V_grid, W_grid,
									 lattice, simd_dims_3d, tol);

#endif

  if(0) testModeMappingTranspose(a2a_arg);

#ifdef USE_GRID
  if(0) testComputeLowModeMADWF<A2Apolicies_grid>(a2a_arg, lanc_arg, lattice, simd_dims, tol);
#endif

#ifdef USE_GRID
  if(0) testMADWFprecon<A2Apolicies_grid>(a2a_arg, lanc_arg, lattice, simd_dims, tol);
#endif
  
}




void testPeriodic(CommonArg &common_arg, A2AArg &a2a_arg, FixGaugeArg &fix_gauge_arg, LancArg &lanc_arg, int ntests, int nthreads, double tol){
#if 0
  
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

  AlgFixGauge fix_gauge(lattice,&common_arg,&fix_gauge_arg);
  fix_gauge.run();

  //#define COMPUTE_VW

#ifdef COMPUTE_VW  
  LatticeSolvers solvers(jp,nthreads);
  Lanczos<LanczosPolicies> eig;
  eig.compute(lanc_arg, solvers, lattice);
#endif
  
  A2AvectorV<A2Apolicies_grid> V_grid(a2a_arg,simd_dims);
  A2AvectorW<A2Apolicies_grid> W_grid(a2a_arg,simd_dims);

  LatRanGen lrg_bak = LRG;
#ifdef COMPUTE_VW  
  computeA2Avectors<A2Apolicies_grid,LanczosPolicies>::compute(V_grid,W_grid,false,false, lattice, eig, solvers);
#else 
  randomizeVW<A2Apolicies_grid>(V_grid,W_grid);
#endif

  A2AvectorV<A2Apolicies_std> V_std(a2a_arg);
  A2AvectorW<A2Apolicies_std> W_std(a2a_arg);

  LRG = lrg_bak;
#ifdef COMPUTE_VW 
  computeA2Avectors<A2Apolicies_std,LanczosPolicies>::compute(V_std,W_std,false,false, lattice, eig, solvers);
#else 
  randomizeVW<A2Apolicies_std>(V_std,W_std);
#endif


  std::cout << "OPENMP threads is " << omp_get_max_threads() << std::endl;
  std::cout << "Starting tests" << std::endl;

  if(0) testCPSspinColorMatrix();
#ifdef USE_GRID
  if(0) testvMvGridOrigPeriodic<A2Apolicies_std, A2Apolicies_grid>(a2a_arg, nthreads, tol);
  if(0) testVVgridOrigPeriodic<A2Apolicies_std, A2Apolicies_grid>(a2a_arg, 1, nthreads, tol);
#endif

#endif
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

  int size[] = {4,4,4,4,4};
  int nthreads = 1;
  int ntests = 10;
  
  double tol = 1e-5;

  int nl = 10;

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
    }else if(cmd == "--shm"){
      i+=2;
    }else if(cmd == "--accelerator-threads"){
      i+=2;
    }else{
      if(UniqueID()==0) printf("Unrecognised argument: %s\n",cmd.c_str());
      exit(-1);
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
  fix_gauge_arg.fix_gauge_kind = FIX_GAUGE_COULOMB_T;
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
      if(!unit_gauge) lattice_tmp.SetGfieldDisOrd();
      else lattice_tmp.SetGfieldOrd();
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

  if(ngp > 0) testGparity(common_arg, a2a_arg, fix_gauge_arg, lanc_arg, ntests, nthreads, tol);
  else testPeriodic(common_arg, a2a_arg, fix_gauge_arg, lanc_arg, ntests, nthreads, tol);

  std::cout << "Done" << std::endl;

  End();
  return 0;
}
