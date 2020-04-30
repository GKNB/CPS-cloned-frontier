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



int main(int argc,char *argv[])
{
  if(argc < 3)
    ERR.General("","main","Need #GPdirs and working directory\n");
  
  Start(&argc, &argv);
  int ngp;
  { std::stringstream ss; ss << argv[1]; ss >> ngp; }

  if(!UniqueID()) printf("Doing G-parity in %d directions\n",ngp);
  assert(ngp > 0); //we use Gparity classes 
  
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
  L(N_get, 10);
  L(N_use, 14);
  L(N_true_get, 10);
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
  
#if TARGET == BGQ
  LRG.setSerial();
#endif
  LRG.Initialize(); //usually initialised when lattice generated, but I pre-init here so I can load the state from file


  if(1) testSpinFlavorMatrices();

  
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

  typedef A2ApoliciesSIMDdoubleAutoAllocGparity A2Apolicies_grid;
  typedef A2ApoliciesDoubleAutoAllocGparity A2Apolicies_std; 

  typedef A2Apolicies_std::ComplexType mf_Complex;
  typedef A2Apolicies_grid::ComplexType grid_Complex;

  // typedef double mf_Float;
  // typedef std::complex<mf_Float> mf_Complex;

  // typedef Grid::vComplexD grid_Complex;
  
  // typedef deduceA2Apolicies<grid_Complex> A2ApoliciesBase_grid;
  // typedef GridA2APoliciesBase LanczosPolicies;
  typedef A2Apolicies_grid::FgridGFclass LatticeType;  
  // typedef GridA2APolicies<A2ApoliciesBase_grid> A2Apolicies_grid; //combines A2ApoliciesBase and GridPoliciesBase
  typedef A2Apolicies_grid::GridFermionField GridFermionField;
  
  // typedef deduceA2Apolicies<mf_Complex> A2ApoliciesBase_std;
  // typedef GridA2APolicies<A2ApoliciesBase_std> A2Apolicies_std; //combines A2ApoliciesBase and GridPoliciesBase

  FgridParams grid_params; 
  grid_params.mobius_scale = 2.0;
  LatticeType lattice(grid_params);
  lattice.ImportGauge(); //lattice -> Grid  

  std::cout << "OPENMP threads is " << omp_get_max_threads() << std::endl;

  //Some checkerboard checking stuff
  if(0){
    Grid::GridCartesian* grid5d_full = lattice.getFGrid();
    Grid::GridCartesian* grid4d_full = lattice.getUGrid();
    Grid::GridRedBlackCartesian* grid5d_cb = lattice.getFrbGrid();
    Grid::GridRedBlackCartesian* grid4d_cb = lattice.getUrbGrid();

    
    std::vector<int> seeds4({1,2,3,4});
    std::vector<int> seeds5({5,6,7,8});
    Grid::GridParallelRNG          RNG5(grid5d_full);  RNG5.SeedFixedIntegers(seeds5);
    Grid::GridParallelRNG          RNG4(grid4d_full);  RNG4.SeedFixedIntegers(seeds4);

    {
      GridFermionField fivedin(grid5d_full); random(RNG5,fivedin);

      CPSfermion5D<cps::ComplexD> cpscp1;
      cpscp1.importGridField(fivedin);

      CPSfermion5D<cps::ComplexD> cpscp2;
      lattice.ImportFermion((Vector*)cpscp2.ptr(), fivedin);

      assert(cpscp1.equals(cpscp2));

      double nrm_cps = cpscp1.norm2();
      double nrm_grid = Grid::norm2(fivedin);
      
      std::cout << "5D import pass norms " << nrm_cps << " " << nrm_grid << std::endl;

      GridFermionField fivedout(grid5d_full);
      cpscp1.exportGridField(fivedout);
      std::cout << "Export to grid: " << Grid::norm2(fivedout) << std::endl;
      
    }
    {
      GridFermionField fivedin(grid5d_full); random(RNG5,fivedin);
      GridFermionField fivedcb(grid5d_cb);
      Grid::pickCheckerboard(Grid::Odd, fivedcb, fivedin);

      Grid::Coordinate test_site(5,0);
      test_site[1] = 3;

      typedef typename Grid::GridTypeMapper<GridFermionField::vector_object>::scalar_object sobj;
      sobj v1, v2;
      Grid::peekLocalSite(v1,fivedin,test_site);

      Grid::peekLocalSite(v2,fivedcb,test_site);
      
      std::cout << "v1:\n" << v1 << std::endl;
      std::cout << "v2:\n" << v2 << std::endl;
      

      CPSfermion5Dcb4Dodd<cps::ComplexD> cpscp1;
      std::cout << "From Grid CB\n";
      cpscp1.importGridField(fivedcb);

      double nrm_cps = cpscp1.norm2();
      double nrm_grid = Grid::norm2(fivedcb);

      GridFermionField tmp(grid5d_full);
      zeroit(tmp);
      Grid::setCheckerboard(tmp, fivedcb);

      double nrm2_grid = Grid::norm2(tmp);

      
      CPSfermion5Dcb4Dodd<cps::ComplexD> cpscp3;
      std::cout << "From Grid full\n";
      cpscp3.importGridField(fivedin);
      double nrm_cps2 = cpscp3.norm2();
      
      std::cout << "5D CB odd import norms CPS " << nrm_cps << " CPS direct " << nrm_cps2 << " Grid "  << nrm_grid << " Grid putback " << nrm2_grid << std::endl;
    }
    std::cout << "Done\n";
  }


  //#define COMPUTE_VW

#ifdef COMPUTE_VW  
  LatticeSolvers solvers(jp,nthreads);
  Lanczos<LanczosPolicies> eig;
  eig.compute(lanc_arg, solvers, lattice);
#endif
  
  //Setup SIMD info
  int nsimd = grid_Complex::Nsimd();
  typename SIMDpolicyBase<4>::ParamType simd_dims;
  SIMDpolicyBase<4>::SIMDdefaultLayout(simd_dims,nsimd,2); //only divide over spatial directions
  
  printf("Nsimd = %d, SIMD dimensions:\n", nsimd);
  for(int i=0;i<4;i++)
    printf("%d ", simd_dims[i]);
  printf("\n");


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
  
  int nl = V_std.getNl();
  int nh = V_std.getNh();
  
  for(int i=0;i<nl;i++){
    double nrm_grid = V_grid.getVl(i).norm2();
    double nrm_std = V_std.getVl(i).norm2();
    double diff = nrm_grid - nrm_std;
    std::cout << "vl " << i << " grid " << nrm_grid << " std " << nrm_std << " diff " << diff << std::endl;
    if(fabs(diff) > 1e-10){
      assert(false);
    }
  }
  for(int i=0;i<nh;i++){
    double nrm_grid = V_grid.getVh(i).norm2();
    double nrm_std = V_std.getVh(i).norm2();
    double diff = nrm_grid - nrm_std;
    std::cout << "vh " << i << " grid " << nrm_grid << " std " << nrm_std << " diff " << diff << std::endl;
    if(fabs(diff) > 1e-10){
      assert(false);
    }
  }

  
  FixGaugeArg fix_gauge_arg;  
  fix_gauge_arg.fix_gauge_kind = FIX_GAUGE_COULOMB_T;
  fix_gauge_arg.hyperplane_start = 0;
  fix_gauge_arg.hyperplane_step = 1;
  fix_gauge_arg.hyperplane_num = GJP.Tnodes()*GJP.TnodeSites();
  fix_gauge_arg.stop_cond = 1e-8;
  fix_gauge_arg.max_iter_num = 10000;
  AlgFixGauge fix_gauge(lattice,&common_arg,&fix_gauge_arg);
  fix_gauge.run();
  
  //Test gauge fixing and phase application
  ThreeMomentum p_plus( GJP.Bc(0)==BND_CND_GPARITY? 1 : 0,
			GJP.Bc(1)==BND_CND_GPARITY? 1 : 0,
			GJP.Bc(2)==BND_CND_GPARITY? 1 : 0 );
  ThreeMomentum p_minus = -p_plus;
  
  {
    typename A2Apolicies_std::FermionFieldType field_std;
    field_std.testRandom();
    typename A2Apolicies_grid::FermionFieldType field_grid(simd_dims);
    field_grid.importField(field_std);

    std::cout << "Import CPS->CPS/Grid " << field_std.norm2() << " " << field_grid.norm2() << std::endl;

    field_std.gaugeFix(lattice,true);
    field_grid.gaugeFix(lattice,true);

    std::cout << "After gauge fix CPS->CPS/Grid " << field_std.norm2() << " " << field_grid.norm2() << std::endl;

    typename A2Apolicies_std::FermionFieldType field_std_tmp;
    field_std_tmp.importField(field_grid);

    compareField(field_std, field_std_tmp, "Gauge fix test", 1e-10);
    
    std::cout << "Phasing with " << p_plus.str() << std::endl;
    field_std.applyPhase(p_plus.ptr(),true);
    field_grid.applyPhase(p_plus.ptr(),true);

    field_std_tmp.importField(field_grid);
    compareField(field_std, field_std_tmp, "Phase test", 1e-10);

    CPSfermion4DglobalInOneDir<typename A2Apolicies_grid::ScalarComplexType> dbl_grid(0);
    CPSfermion4DglobalInOneDir<typename A2Apolicies_std::ComplexType> dbl_std(0);

    dbl_std.gather(field_std);
    dbl_std.fft();
    
    dbl_grid.gather(field_grid);
    dbl_grid.fft();
    
    compareField(dbl_std, dbl_grid, "Gather test", 1e-10);

    dbl_grid.scatter(field_grid);
    dbl_std.scatter(field_std);

    field_std_tmp.importField(field_grid);
    compareField(field_std, field_std_tmp, "FFT/scatter test", 1e-10);
    
  }
  typename SIMDpolicyBase<3>::ParamType simd_dims_3d;
  SIMDpolicyBase<3>::SIMDdefaultLayout(simd_dims_3d,nsimd);

  
  StandardPionMomentaPolicy momenta;
  MesonFieldMomentumContainer<A2Apolicies_std> mf_ll_con_std;
  MesonFieldMomentumContainer<A2Apolicies_grid> mf_ll_con_grid;
  
  computeGparityLLmesonFields1s<A2Apolicies_std,StandardPionMomentaPolicy,15,sigma3>::computeMesonFields(mf_ll_con_std,momenta,W_std,V_std,2.0,lattice);
  computeGparityLLmesonFields1s<A2Apolicies_grid,StandardPionMomentaPolicy,15,sigma3>::computeMesonFields(mf_ll_con_grid,momenta,W_grid,V_grid,2.0,lattice,simd_dims_3d);

  fMatrix<typename A2Apolicies_std::ScalarComplexType> fmat_std;
  ComputePion<A2Apolicies_std>::compute<StandardPionMomentaPolicy>(fmat_std, mf_ll_con_std, momenta, 0);

  fMatrix<typename A2Apolicies_grid::ScalarComplexType> fmat_grid;

  ComputePion<A2Apolicies_grid>::compute<StandardPionMomentaPolicy>(fmat_grid, mf_ll_con_grid, momenta, 0);

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
  printf("Pion pass\n");

  StationaryKaonMomentaPolicy kaon_mom;

  std::vector<A2AmesonField<A2Apolicies_std,A2AvectorWfftw,A2AvectorVfftw> > mf_ls_std;
  std::vector<A2AmesonField<A2Apolicies_std,A2AvectorWfftw,A2AvectorVfftw> > mf_sl_std;
  ComputeKaon<A2Apolicies_std>::computeMesonFields(mf_ls_std, mf_sl_std,
						   W_std, V_std,
						   W_std, V_std,
						   kaon_mom,
						   2.0, lattice);

  std::vector<A2AmesonField<A2Apolicies_grid,A2AvectorWfftw,A2AvectorVfftw> > mf_ls_grid;
  std::vector<A2AmesonField<A2Apolicies_grid,A2AvectorWfftw,A2AvectorVfftw> > mf_sl_grid;
  ComputeKaon<A2Apolicies_grid>::computeMesonFields(mf_ls_grid, mf_sl_grid,
						    W_grid, V_grid,
						    W_grid, V_grid,
						    kaon_mom,
						    2.0, lattice, simd_dims_3d);

  ComputeKaon<A2Apolicies_std>::compute(fmat_std, mf_ls_std, mf_sl_std);
  ComputeKaon<A2Apolicies_grid>::compute(fmat_grid, mf_ls_grid, mf_sl_grid);
  
  fail = false;
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
  printf("Kaon pass\n");

  
  ThreeMomentum p_pi_plus = p_plus * 2;
  
  char diags[] = {'C','D','R'};
  for(int d=0;d<3;d++){
    MesonFieldProductStore<A2Apolicies_std> products_std;
    ComputePiPiGparity<A2Apolicies_std>::compute(fmat_std, diags[d], p_pi_plus, p_pi_plus, 2, 1, mf_ll_con_std, products_std);

    MesonFieldProductStore<A2Apolicies_grid> products_grid;
    ComputePiPiGparity<A2Apolicies_grid>::compute(fmat_grid, diags[d], p_pi_plus, p_pi_plus, 2, 1, mf_ll_con_grid, products_grid);

    fail = false;
    for(int r=0;r<fmat_std.nRows();r++){
      for(int c=0;c<fmat_std.nCols();c++){
	double rdiff = fmat_std(r,c).real() - fmat_grid(r,c).real();
	double idiff = fmat_std(r,c).imag() - fmat_grid(r,c).imag();
	if(rdiff > tol|| idiff > tol){
	  printf("Fail Pipi fig %c elem %d %d : (%f,%f) (%f,%f) diff (%g,%g)\n",diags[d],r,c, fmat_std(r,c).real(),  fmat_std(r,c).imag(), fmat_grid(r,c).real(), fmat_grid(r,c).imag(), rdiff, idiff);
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
    
    ComputePiPiGparity<A2Apolicies_std>::computeFigureVdis(pipi_figV_std, p_pi_plus, 1, mf_ll_con_std);
    ComputePiPiGparity<A2Apolicies_grid>::computeFigureVdis(pipi_figV_grid, p_pi_plus, 1, mf_ll_con_grid);

    fail = false;
    for(int r=0;r<pipi_figV_std.size();r++){
      double rdiff = pipi_figV_std(r).real() - pipi_figV_grid(r).real();
      double idiff = pipi_figV_std(r).imag() - pipi_figV_grid(r).imag();
      if(rdiff > tol|| idiff > tol){
	printf("Fail Pipi fig V elem %d : (%f,%f) (%f,%f) diff (%g,%g)\n",r, pipi_figV_std(r).real(),  pipi_figV_std(r).imag(), pipi_figV_grid(r).real(), pipi_figV_grid(r).imag(), rdiff, idiff);
	fail = true;
      }      
    }
    if(fail)ERR.General("","","Standard vs Grid implementation pipi fig V test failed\n");
    printf("Pipi fig V pass\n");    
  }

  StandardLSWWmomentaPolicy ww_mom;  
  std::vector<A2AmesonField<A2Apolicies_grid,A2AvectorWfftw,A2AvectorWfftw> > mf_ls_ww_grid;
  ComputeKtoPiPiGparity<A2Apolicies_grid>::generatelsWWmesonfields(mf_ls_ww_grid,W_grid,W_grid, ww_mom, 2.0,lattice, simd_dims_3d);

  std::vector<A2AmesonField<A2Apolicies_std,A2AvectorWfftw,A2AvectorWfftw> > mf_ls_ww_std;
  ComputeKtoPiPiGparity<A2Apolicies_std>::generatelsWWmesonfields(mf_ls_ww_std,W_std,W_std, ww_mom, 2.0,lattice);

  mf_ll_con_grid.printMomenta(std::cout);

  if(0){
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::ResultsContainerType type1_grid;
    ComputeKtoPiPiGparity<A2Apolicies_grid>::type1(type1_grid, 4, 2, 1, 1, p_pi_plus, mf_ls_ww_grid, mf_ll_con_grid, V_grid, V_grid, W_grid, W_grid);

    typename ComputeKtoPiPiGparity<A2Apolicies_std>::ResultsContainerType type1_std;
    ComputeKtoPiPiGparity<A2Apolicies_std>::type1(type1_std, 4, 2, 1, 1, p_pi_plus, mf_ls_ww_std, mf_ll_con_std, V_std, V_std, W_std, W_std);

  
    bool fail = false;
    for(int i=0;i<type1_std.nElementsTotal();i++){
      std::complex<double> val_std = convertComplexD(type1_std[i]);
      std::complex<double> val_grid = convertComplexD(type1_grid[i]);
    
      double rdiff = fabs(val_grid.real()-val_std.real());
      double idiff = fabs(val_grid.imag()-val_std.imag());
      if(rdiff > tol|| idiff > tol){
	printf("!!!Fail: Type1 elem %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",i, val_grid.real(),val_grid.imag(), val_std.real(),val_std.imag(), val_std.real()-val_grid.real(), val_std.imag()-val_grid.imag());
	fail = true;
      }//else printf("Pass: Type1 elem %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",i, val_grid.real(),val_grid.imag(), val_std.real(),val_std.imag(), val_std.real()-val_grid.real(), val_std.imag()-val_grid.imag());
    }
    if(fail) ERR.General("","","Standard vs Grid implementation type1 test failed\n");
    printf("Type 1 pass\n");
  }
  if(1){
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::ResultsContainerType type2_grid;
    ComputeKtoPiPiGparity<A2Apolicies_grid>::type2(type2_grid, 4, 2, 1, p_pi_plus, mf_ls_ww_grid, mf_ll_con_grid, V_grid, V_grid, W_grid, W_grid);

    typename ComputeKtoPiPiGparity<A2Apolicies_std>::ResultsContainerType type2_std;
    ComputeKtoPiPiGparity<A2Apolicies_std>::type2(type2_std, 4, 2, 1, p_pi_plus, mf_ls_ww_std, mf_ll_con_std, V_std, V_std, W_std, W_std);

  
    bool fail = false;
    for(int i=0;i<type2_std.nElementsTotal();i++){
      std::complex<double> val_std = convertComplexD(type2_std[i]);
      std::complex<double> val_grid = convertComplexD(type2_grid[i]);
    
      double rdiff = fabs(val_grid.real()-val_std.real());
      double idiff = fabs(val_grid.imag()-val_std.imag());
      if(rdiff > tol|| idiff > tol){
	printf("!!!Fail: Type2 elem %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",i, val_grid.real(),val_grid.imag(), val_std.real(),val_std.imag(), val_std.real()-val_grid.real(), val_std.imag()-val_grid.imag());
	fail = true;
      }//else printf("Pass: Type2 elem %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",i, val_grid.real(),val_grid.imag(), val_std.real(),val_std.imag(), val_std.real()-val_grid.real(), val_std.imag()-val_grid.imag());
    }
    if(fail) ERR.General("","","Standard vs Grid implementation type2 test failed\n");
    printf("Type 2 pass\n");
  }
  if(0){
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::ResultsContainerType type3_grid;
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::MixDiagResultsContainerType type3_mix_grid;
    ComputeKtoPiPiGparity<A2Apolicies_grid>::type3(type3_grid, type3_mix_grid, 4, 2, 1, p_pi_plus, mf_ls_ww_grid, mf_ll_con_grid, V_grid, V_grid, W_grid, W_grid);

    typename ComputeKtoPiPiGparity<A2Apolicies_std>::ResultsContainerType type3_std;
    typename ComputeKtoPiPiGparity<A2Apolicies_std>::MixDiagResultsContainerType type3_mix_std;
    ComputeKtoPiPiGparity<A2Apolicies_std>::type3(type3_std, type3_mix_std, 4, 2, 1, p_pi_plus, mf_ls_ww_std, mf_ll_con_std, V_std, V_std, W_std, W_std);

  
    bool fail = false;
    for(int i=0;i<type3_std.nElementsTotal();i++){
      std::complex<double> val_std = convertComplexD(type3_std[i]);
      std::complex<double> val_grid = convertComplexD(type3_grid[i]);
    
      double rdiff = fabs(val_grid.real()-val_std.real());
      double idiff = fabs(val_grid.imag()-val_std.imag());
      if(rdiff > tol|| idiff > tol){
	printf("!!!Fail: Type3 elem %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",i, val_grid.real(),val_grid.imag(), val_std.real(),val_std.imag(), val_std.real()-val_grid.real(), val_std.imag()-val_grid.imag());
	fail = true;
      }//else printf("Pass: Type3 elem %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",i, val_grid.real(),val_grid.imag(), val_std.real(),val_std.imag(), val_std.real()-val_grid.real(), val_std.imag()-val_grid.imag());
    }
    if(fail) ERR.General("","","Standard vs Grid implementation type3 test failed\n");
    printf("Type 3 pass\n");
  }
  if(0){
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::ResultsContainerType type4_grid;
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::MixDiagResultsContainerType type4_mix_grid;
    ComputeKtoPiPiGparity<A2Apolicies_grid>::type4(type4_grid, type4_mix_grid, 1, mf_ls_ww_grid, V_grid, V_grid, W_grid, W_grid);

    typename ComputeKtoPiPiGparity<A2Apolicies_std>::ResultsContainerType type4_std;
    typename ComputeKtoPiPiGparity<A2Apolicies_std>::MixDiagResultsContainerType type4_mix_std;
    ComputeKtoPiPiGparity<A2Apolicies_std>::type4(type4_std, type4_mix_std, 1, mf_ls_ww_std, V_std, V_std, W_std, W_std);
  
    bool fail = false;
    for(int i=0;i<type4_std.nElementsTotal();i++){
      std::complex<double> val_std = convertComplexD(type4_std[i]);
      std::complex<double> val_grid = convertComplexD(type4_grid[i]);
    
      double rdiff = fabs(val_grid.real()-val_std.real());
      double idiff = fabs(val_grid.imag()-val_std.imag());
      if(rdiff > tol|| idiff > tol){
	printf("!!!Fail: Type4 elem %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",i, val_grid.real(),val_grid.imag(), val_std.real(),val_std.imag(), val_std.real()-val_grid.real(), val_std.imag()-val_grid.imag());
	fail = true;
      }//else printf("Pass: Type4 elem %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",i, val_grid.real(),val_grid.imag(), val_std.real(),val_std.imag(), val_std.real()-val_grid.real(), val_std.imag()-val_grid.imag());
    }
    if(fail) ERR.General("","","Standard vs Grid implementation type4 test failed\n");
    printf("Type 4 pass\n");
  }
  
  std::cout << "Done" << std::endl;

  return 0;
}
