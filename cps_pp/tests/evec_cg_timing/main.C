//Do Lanczos and then repeat CG with different numbers of deflating evecs

#define USE_GRID_A2A
#define USE_GRID_LANCZOS

#include<alg/a2a/main.h>

using namespace cps;

typedef A2ApoliciesSIMDdoubleAutoAlloc A2Apolicies;

inline int toInt(const char* a){
  std::stringstream ss; ss << a; int o; ss >> o;
  return o;
}


int main(int argc,char *argv[])
{  
  Start(&argc, &argv);

  const char* cname = "";
  const char* fname = "main";
  
  if(argc < 2) ERR.General("","main","Require working directory\n");
      
  int nthreads = 1;
  int step = 100;
  int min = 100;

  const int ngrid_arg = 7;
  const std::string grid_args[ngrid_arg] = { "--debug-signals", "--dslash-generic", "--dslash-unroll", "--dslash-asm", "--shm", "--lebesgue", "--cacheblocking" };
  const int grid_args_skip[ngrid_arg] = { 1, 1, 1, 1, 2, 1, 2 };
  
  int i=2;
  while(i<argc){
    char* cmd = argv[i];  
    if( strncmp(cmd,"-nthread",15) == 0){
      nthreads = toInt(argv[i+1]);
      printf("Set nthreads to %d\n", nthreads);
      i+=2;
    }else if( strncmp(cmd,"-step",15) == 0){
      step = toInt(argv[i+1]);
      printf("Set evec step to %d\n", nthreads);
      i+=2;
    }else if( strncmp(cmd,"-min",15) == 0){
      min = toInt(argv[i+1]);
      printf("Set evec min to %d\n", min);
      i+=2;
    }else{
      bool is_grid_arg = false;
      for(int a=0;a<ngrid_arg;a++){
	if( std::string(cmd) == grid_args[a] ){
	  if(!UniqueID()){ printf("main.C: Ignoring Grid argument %s\n",cmd); fflush(stdout); }
	  i += grid_args_skip[a];
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

  if(chdir(argv[1])!=0) ERR.General("",fname,"Unable to switch to directory '%s'\n",argv[1]);
  CommonArg common_arg("","");
  DoArg do_arg;
  JobParams jp;
  LancArg lanc_arg;

  if(!do_arg.Decode("do_arg.vml","do_arg")){
    do_arg.Encode("do_arg.templ","do_arg");
    VRB.Result(cname,fname,"Can't open do_arg.vml!\n");exit(1);
  }
  if(!jp.Decode("job_params.vml","job_params")){
    jp.Encode("job_params.templ","job_params");
    VRB.Result(cname,fname,"Can't open job_params.vml!\n");exit(1);
  }
  if(!lanc_arg.Decode("lanc_arg.vml","lanc_arg")){
    lanc_arg.Encode("lanc_arg.templ","lanc_arg");
    VRB.Result(cname,fname,"Can't open lanc_arg.vml!\n");exit(1);
  }

  GJP.Initialize(do_arg);

  int ngp = 0;
  for(int i=0;i<3;i++) if(GJP.Bc(i) == BND_CND_GPARITY) ++ngp;

#ifdef USE_GRID_GPARITY
  if(ngp == 0) ERR.General(cname,fname,"Fgrid is currently compiled for G-parity\n");
#else
  if(ngp != 0) ERR.General(cname,fname,"Fgrid is not currently compiled for G-parity\n");
#endif
  
  typedef typename A2Apolicies::LatticeType LatticeType;

  LatticeSolvers solvers(jp,nthreads);
  LatticeSetup<LatticeType> lattice_setup(jp,solvers);
  LatticeType &lat = lattice_setup.getLattice();

  Lanczos<A2Apolicies> eig;
  eig.compute(lanc_arg, solvers, lat);

  eig.toSingle();
  
  //Grids and gauge field
  typedef typename A2Apolicies::GridFermionField GridFermionField;
  typedef typename A2Apolicies::FgridFclass FgridFclass;
  typedef typename A2Apolicies::GridDirac GridDirac;
  
  assert(lat.Fclass() == A2Apolicies::FGRID_CLASS_NAME);
  FgridFclass &latg = dynamic_cast<FgridFclass&>(lat);

  Grid::GridCartesian *UGrid = latg.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = latg.getUrbGrid();
  Grid::GridCartesian *FGrid = latg.getFGrid();
  Grid::GridRedBlackCartesian *FrbGrid = latg.getFrbGrid();
  Grid::QCD::LatticeGaugeFieldD *Umu = latg.getUmu();
  
  const int glb_ls = GJP.SnodeSites() * GJP.Snodes();

  //Mobius parameters
  const double mob_b = latg.get_mob_b();
  const double mob_c = mob_b - 1.;   //b-c = 1
  const double M5 = GJP.DwfHeight();
  printf("Grid b=%g c=%g b+c=%g\n",mob_b,mob_c,mob_b+mob_c);

  //Setup Grid Dirac operator
  typename GridDirac::ImplParams params;
  latg.SetParams(params);

  GridDirac Ddwf(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid, lanc_arg.mass, M5, mob_b, mob_c, params);
  Grid::SchurDiagMooeeOperator<GridDirac, GridFermionField> linop(Ddwf);

  
  //Generate random 4D source and import into Grid
  CPSfermion4D<cps::ComplexD> tmp_rand;
  tmp_rand.setGaussianRandom();
  GridFermionField tmp_full_4d(UGrid);
  tmp_rand.exportGridField(tmp_full_4d);

  GridFermionField gsrc(FGrid);
  GridFermionField gtmp_full(FGrid);
  DomainWallFourToFive(gsrc, tmp_full_4d, 0, glb_ls-1);

  //Left-multiply by D-.  D- = (1-c*DW)
  Ddwf.DW(gsrc, gtmp_full, Grid::QCD::DaggerNo);
  axpy(gsrc, -mob_c, gtmp_full, gsrc); 


  
  EvecInterfaceGridSinglePrec<A2Apolicies> ev(eig.evec_f, eig.eval, lat, lanc_arg.mass);
  
  const int max_iter = 30000;
  const double residual = 1e-08;
  
  
  for(int nev = lanc_arg.N_true_get; nev >= min; nev -= step){
    if(!UniqueID()) printf("MAIN: Number of evecs %d\n",nev);
    eig.eval.resize(nev);

    const int sz = eig.evec_f.size();
    for(int d=0;d< sz-nev; d++) eig.evec_f.pop_back();
    assert( int(eig.evec_f.size()) == nev );
    
    gtmp_full = Grid::zero;

    Grid_CGNE_M_high<A2Apolicies>(gtmp_full, gsrc, residual, max_iter, ev, nev, latg, Ddwf, FGrid, FrbGrid);
  }
  
  if(!UniqueID()) printf("Done\n");
  End();
  return 0;
}



