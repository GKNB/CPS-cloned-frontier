#include <alg/alg_fix_gauge.h>
#include <alg/a2a/a2a.h>
#include <alg/a2a/mesonfield.h>
#include <alg/a2a/utils_main.h>
#include <alg/a2a/grid_wrappers.h>
#include <alg/a2a/bfm_wrappers.h>
#include <alg/a2a/compute_kaon.h>
#include <alg/a2a/compute_pion.h>
#include <alg/a2a/compute_sigma.h>
#include <alg/a2a/compute_pipi.h>
#include <alg/a2a/compute_ktopipi.h>
#include "main.h" //softlink the main.h from production_code

#ifndef USE_GRID_A2A
#error "Must be using USE_GRID_A2A"
#endif
#ifndef USE_GRID_LANCZOS
#error "Must be using USE_GRID_LANCZOS"
#endif

//Command line argument store/parse
struct TuneCommandLineArgs{
  int nthreads;
  double inner_prec_start;

  bool load_evecs;
  std::string load_evecs_stub; //will be appended with ".<CONF>" where <CONF> is replaced by the configuration index
  
  TuneCommandLineArgs(int argc, char **argv, int begin){
    nthreads = 1;
#if TARGET == BGQ
    nthreads = 64;
#endif

    load_evecs = false;
    inner_prec_start = 1e-08;
    
    parse(argc,argv,begin);
  }

  void parse(int argc, char **argv, int begin){
    if(!UniqueID()){ printf("Arguments:\n"); fflush(stdout); }
    for(int i=0;i<argc;i++){
      if(!UniqueID()){ printf("%d \"%s\"\n",i,argv[i]); fflush(stdout); }
    }
    
    const int ngrid_arg = 10;
    const std::string grid_args[ngrid_arg] = { "--debug-signals", "--dslash-generic", "--dslash-unroll", "--dslash-asm", "--shm", "--lebesgue", "--cacheblocking", "--comms-concurrent", "--comms-sequential", "--comms-overlap" };
    const int grid_args_skip[ngrid_arg] =    { 1                , 1                 , 1                , 1             , 2      , 1           , 2                , 1              , 1                 , 1 };

    int arg = begin;
    while(arg < argc){
      char* cmd = argv[arg];
      if( strncmp(cmd,"-nthread",8) == 0){
	if(arg == argc-1){ if(!UniqueID()){ printf("-nthread must be followed by a number!\n"); fflush(stdout); } exit(-1); }
	nthreads = strToAny<int>(argv[arg+1]);
	if(!UniqueID()){ printf("Setting number of threads to %d\n",nthreads); }
	arg+=2;
      }else if( std::string(cmd) == "-load_evecs" ){
	load_evecs = true;
	load_evecs_stub = argv[arg+1];
	if(!UniqueID()){ printf("Loading evecs with stub %s\n",load_evecs_stub); }
	arg+=2;	
      }else if( std::string(cmd) == "-inner_prec_start" ){
        inner_prec_start = strToAny<double>(argv[arg+1]);
        if(!UniqueID()){ printf("Using initial inner CG residual %g\n",inner_prec_start); }
        arg+=2;
      }else if( strncmp(cmd,"--comms-isend",30) == 0){
	ERR.General("","main","Grid option --comms-isend is deprecated: use --comms-concurrent instead");
      }else if( strncmp(cmd,"--comms-sendrecv",30) == 0){
	ERR.General("","main","Grid option --comms-sendrecv is deprecated: use --comms-sequential instead");
      }else{
	bool is_grid_arg = false;
	for(int i=0;i<ngrid_arg;i++){
	  if( std::string(cmd) == grid_args[i] ){
	    if(!UniqueID()){ printf("main.C: Ignoring Grid argument %s\n",cmd); fflush(stdout); }
	    arg += grid_args_skip[i];
	    is_grid_arg = true;
	    break;
	  }
	}
	if(!is_grid_arg){
	  if(UniqueID()==0) printf("Unrecognised argument: %s\n",cmd);
	  exit(-1);
	}
      }
    }
  }

};


struct TuneParameters{
  CommonArg common_arg;
  DoArg do_arg;
  A2AArg a2a_arg;
  LancArg lanc_arg;
  JobParams jp;
  
  TuneParameters(const char* directory): common_arg("",""){
    if(chdir(directory)!=0) ERR.General("Parameters","Parameters","Unable to switch to directory '%s'\n",directory);

    if(!do_arg.Decode("do_arg.vml","do_arg")){
      do_arg.Encode("do_arg.templ","do_arg");
      VRB.Result("Parameters","Parameters","Can't open do_arg.vml!\n");exit(1);
    }
    if(!a2a_arg.Decode("a2a_arg.vml","a2a_arg")){
      a2a_arg.Encode("a2a_arg.templ","a2a_arg");
      VRB.Result("Parameters","Parameters","Can't open a2a_arg.vml!\n");exit(1);
    }
    if(!lanc_arg.Decode("lanc_arg.vml","lanc_arg")){
      lanc_arg.Encode("lanc_arg.templ","lanc_arg");
      VRB.Result("Parameters","Parameters","Can't open lanc_arg.vml!\n");exit(1);
    }
    if(!jp.Decode("job_params.vml","job_params")){
      jp.Encode("job_params.templ","job_params");
      VRB.Result("Parameters","Parameters","Can't open job_params.vml!\n");exit(1);
    }
    common_arg.set_filename(".");
  }

};



template< typename mf_Policies>
void doInvert(Lattice &lat, const std::vector<typename mf_Policies::GridFermionFieldF> &evec, const std::vector<Grid::RealD> &eval,
	      const double mass, const Float residual, const int max_iter, const int nl, double inner_prec_start){
  EvecInterfaceGridSinglePrec<mf_Policies> evecs(evec,eval,lat,mass);
  
  typedef typename mf_Policies::GridFermionField GridFermionField;
  typedef typename mf_Policies::FgridFclass FgridFclass;
  typedef typename mf_Policies::GridDirac GridDirac;
  
  int ngp = 0;
  for(int i=0;i<3;i++) if(GJP.Bc(i) == BND_CND_GPARITY) ++ngp;

#ifdef USE_GRID_GPARITY
  if(ngp == 0) ERR.General("A2AvectorW","computeVWlow","Fgrid is currently compiled for G-parity\n");
#else
  if(ngp != 0) ERR.General("A2AvectorW","computeVWlow","Fgrid is not currently compiled for G-parity\n");
#endif

  assert(lat.Fclass() == mf_Policies::FGRID_CLASS_NAME);
  FgridFclass &latg = dynamic_cast<FgridFclass&>(lat);

  //Grids and gauge field
  Grid::GridCartesian *UGrid = latg.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = latg.getUrbGrid();
  Grid::GridCartesian *FGrid = latg.getFGrid();
  Grid::GridRedBlackCartesian *FrbGrid = latg.getFrbGrid();
  Grid::QCD::LatticeGaugeFieldD *Umu = latg.getUmu();
  
  //Mobius parameters
  const double mob_b = latg.get_mob_b();
  const double mob_c = mob_b - 1.;   //b-c = 1
  const double M5 = GJP.DwfHeight();
  printf("Grid b=%g c=%g b+c=%g\n",mob_b,mob_c,mob_b+mob_c);

  const int gparity = GJP.Gparity();

  //Setup Grid Dirac operator
  typename GridDirac::ImplParams params;
  latg.SetParams(params);

  GridDirac Ddwf(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_c, params);
  Grid::SchurDiagMooeeOperator<GridDirac, GridFermionField> linop(Ddwf);

  //Allocate temp *double precision* storage for fermions
  //typedef typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType Field3DparamType;
  //typedef typename A2Apolicies::FermionFieldType::InputParamType Field4DparamType;
  //Field4DparamType field4dparams; setupFieldParams<typename A2Apolicies::FermionFieldType>(field4dparams);
  
  //CPSfermion4D<typename mf_Policies::ComplexTypeD,typename mf_Policies::FermionFieldType::FieldMappingPolicy, typename mf_Policies::FermionFieldType::FieldAllocPolicy> v4dfield(field4dparams);
  CPSfermion4D<cps::ComplexD> v4dfield;
  v4dfield.setGaussianRandom();

  //Random source on timeslice, spin, color and flavor index 0
  //Flavor 1 to zero
  if(GJP.Gparity()){
#pragma omp parallel for
    for(int i=0;i<GJP.VolNodeSites();i++){
      cps::ComplexD* site = v4dfield.site_ptr(i,1);
      for(int sc=0;sc<12;sc++) site[sc] = cps::ComplexD(0.,0.);
    }
  }
  //sc>0 to zero and t!=0 to zero
#pragma omp parallel for
  for(int i=0;i<GJP.VolNodeSites();i++){
    int coord[4]; v4dfield.siteUnmap(i,coord);    
    const int t_glb = coord[3] + GJP.TnodeCoor()*GJP.TnodeSites();
    const int start = t_glb == 0 ? 1 : 0;
    
    cps::ComplexD* site = v4dfield.site_ptr(i);
    for(int sc=start;sc<12;sc++) site[sc] = cps::ComplexD(0.,0.);
  }
  
  const int glb_ls = GJP.SnodeSites() * GJP.Snodes();

  GridFermionField gsrc(FGrid);
  GridFermionField gtmp_full(FGrid);
  GridFermionField tmp_full_4d(UGrid);

  v4dfield.exportGridField(tmp_full_4d);
  DomainWallFourToFive(gsrc, tmp_full_4d, 0, glb_ls-1);

  //Left-multiply by D-.  D- = (1-c*DW)
  Ddwf.DW(gsrc, gtmp_full, Grid::QCD::DaggerNo);
  axpy(gsrc, -mob_c, gtmp_full, gsrc); 

  Float inner_resid = inner_prec_start;
  
  for(int i=0;i<8;i++){
    evecs.overrideInitialInnerResid(inner_resid);    
    gtmp_full = Grid::zero;
    if(!UniqueID()){ printf("Starting CGNE_M_high with inner resid %g\n", inner_resid); fflush(stdout); }
    double dtime = -dclock();    
    Grid_CGNE_M_high<mf_Policies>(gtmp_full, gsrc, residual, max_iter, evecs, nl, latg, Ddwf, FGrid, FrbGrid);
    std::string msg;
    {
      std::ostringstream os; os << "CGNE_M_high with inner resid " << inner_resid;
      msg=os.str();
    }
    print_time("tune",msg.c_str(),dclock()+dtime);
    inner_resid *= 10.;
  }

  if(!UniqueID()){ printf("Double precision Dirac operator timings:\n"); fflush(stdout); }
  Ddwf.Report();
  if(!UniqueID()){ printf("Single precision Dirac operator timings:\n"); fflush(stdout); }
  evecs.Report();
}




int main(int argc,char *argv[])
{
  Start(&argc, &argv);
  assert(argc > 2);

  const char* vml_dir = argv[1];
  TuneParameters params(vml_dir);

  TuneCommandLineArgs cmdline(argc,argv,2);
  
  GJP.Initialize(params.do_arg);
  LRG.Initialize();

  LanczosWrapper eig;
  if(cmdline.load_evecs){
    if(!UniqueID()) printf("Reading light Lanczos from %s\n",cmdline.load_evecs_stub);
    double time = -dclock();
    LanczosLattice* lanczos_lat = createLattice<LanczosLattice,LANCZOS_LATMARK>::doit(LANCZOS_LATARGS);
    eig.readParallel(cmdline.load_evecs_stub,*lanczos_lat);
    delete lanczos_lat;
    time+=dclock();
    print_time("main","Light Lanczos read",time);
    if(eig.evec_f.size() == 0) //mixed CG requires single prec eigenvectors
      eig.toSingle();
  }else{
    computeEvecs(eig, params.lanc_arg, params.jp, "Lanczos", true, false, COMPUTE_EVECS_EXTRA_ARG_PASS);
  }

  A2ALattice &a2a_lat = *createLattice<A2ALattice,A2A_LATMARK>::doit(A2A_LATARGS); 
  
  doInvert<A2Apolicies>(a2a_lat, eig.evec_f, eig.eval, eig.mass, eig.resid, 10000, params.a2a_arg.nl, cmdline.inner_prec_start);

  
  return 0;
}


