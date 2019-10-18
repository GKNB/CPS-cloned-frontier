//In your makefile choose between the following
//#define USE_BFM_A2A
//#define USE_BFM_LANCZOS
//#define USE_GRID_A2A
//#define USE_GRID_LANCZOS

#include "benchmark_mesonfield.h"

using namespace cps;

#ifdef USE_GRID
#ifdef USE_DESTRUCTIVE_FFT
typedef A2ApoliciesSIMDdoubleManualAlloc GridA2Apolicies;
#else
typedef A2ApoliciesSIMDdoubleAutoAlloc GridA2Apolicies;
#endif
typedef typename GridA2Apolicies::ComplexType grid_Complex;
#endif

#ifdef USE_DESTRUCTIVE_FFT
typedef A2ApoliciesDoubleManualAlloc ScalarA2Apolicies;
#else
typedef A2ApoliciesDoubleAutoAlloc ScalarA2Apolicies;
#endif
typedef typename ScalarA2Apolicies::ComplexType mf_Complex;
typedef typename mf_Complex::value_type mf_Float;


int main(int argc,char *argv[])
{
  std::cout << "Begin: omp_get_max_threads()=" << omp_get_max_threads() << std::endl;
  Start(&argc, &argv);

  std::cout << "0: omp_get_max_threads()=" << omp_get_max_threads() << std::endl;
  
  int ngp;
  { std::stringstream ss; ss << argv[1]; ss >> ngp; }

  if(!UniqueID()) printf("Doing G-parity in %d directions\n",ngp);

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

  int size[] = {2,2,2,2,2};
  int nthreads = 1;
  int ntests = 10;
  
  double tol = 1e-8;
  int nlowmodes = 100;
  printf("Argc is %d\n",argc);
  int i=2;
  while(i<argc){
    char* cmd = argv[i];  
    if( strncmp(cmd,"-save_config",15) == 0){
      if(i==argc-1){ printf("-save_config requires an argument\n"); exit(-1); }
      save_config=true;
      save_config_file = argv[i+1];
      i+=2;
    }else if( strncmp(cmd,"-load_config",15) == 0){
      if(i==argc-1){ printf("-save_config requires an argument\n"); exit(-1); }
      load_config=true;
      load_config_file = argv[i+1];
      i+=2;
    }else if( strncmp(cmd,"-latt",10) == 0){
      if(i>argc-6){
	printf("Did not specify enough arguments for 'latt' (require 5 dimensions)\n"); exit(-1);
      }
      for(int d=0;d<5;d++)
	size[d] = toInt(argv[i+1+d]);
      i+=6;
    }else if( strncmp(cmd,"-load_lrg",15) == 0){
      if(i==argc-1){ printf("-load_lrg requires an argument\n"); exit(-1); }
      load_lrg=true;
      load_lrg_file = argv[i+1];
      i+=2;
    }else if( strncmp(cmd,"-save_lrg",15) == 0){
      if(i==argc-1){ printf("-save_lrg requires an argument\n"); exit(-1); }
      save_lrg=true;
      save_lrg_file = argv[i+1];
      i+=2;  
    }else if( strncmp(cmd,"-verbose",15) == 0){
      verbose=true;
      i++;
    }else if( strncmp(cmd,"-nthread",15) == 0){
      nthreads = toInt(argv[i+1]);
      printf("Set nthreads to %d\n", nthreads);
      i+=2;
    }else if( strncmp(cmd,"-ntest",15) == 0){
      ntests = toInt(argv[i+1]);
      printf("Set ntests to %d\n", ntests);
      i+=2;
    }else if( strncmp(cmd,"-unit_gauge",15) == 0){
      unit_gauge=true;
      i++;
    }else if( strncmp(cmd,"-tolerance",15) == 0){
      std::stringstream ss; ss << argv[i+1];
      ss >> tol;
      if(!UniqueID()) printf("Set tolerance to %g\n",tol);
      i+=2;
    }else if( strncmp(cmd,"-nl",15) == 0){
      std::stringstream ss; ss << argv[i+1];
      ss >> nlowmodes;
      if(!UniqueID()) printf("Set nl to %d\n",nlowmodes);
      i+=2;
    }else if( strncmp(cmd,"-mf_outerblocking",15) == 0){
      int* b[3] = { &BlockedMesonFieldArgs::bi, &BlockedMesonFieldArgs::bj, &BlockedMesonFieldArgs::bp };
      for(int a=0;a<3;a++){
	std::stringstream ss; ss << argv[i+1+a];
	ss >> *b[a];
      }
      i+=4;
    }else if( strncmp(cmd,"-mf_innerblocking",15) == 0){
      int* b[3] = { &BlockedMesonFieldArgs::bii, &BlockedMesonFieldArgs::bjj, &BlockedMesonFieldArgs::bpp };
      for(int a=0;a<3;a++){
	std::stringstream ss; ss << argv[i+1+a];
	ss >> *b[a];
      }
      i+=4;
    }else{
      i++;
      //if(UniqueID()==0) printf("Unrecognised argument: %s\n",cmd);
      //exit(-1);
    }
  }

  printf("Lattice size is %d %d %d %d\n",size[0],size[1],size[2],size[3],size[4]);

  CommonArg common_arg;
  DoArg do_arg;  setupDoArg(do_arg,size,ngp,verbose);

  std::cout << "1: omp_get_max_threads()=" << omp_get_max_threads() << std::endl;
  
  GJP.Initialize(do_arg);


  std::cout << "2: omp_get_max_threads()=" << omp_get_max_threads() << std::endl;
  
  GJP.SetNthreads(nthreads);

  std::cout << "3: omp_get_max_threads()=" << omp_get_max_threads() << std::endl;
#if TARGET == BGQ
  LRG.setSerial();
#endif
  LRG.Initialize(); //usually initialised when lattice generated, but I pre-init here so I can load the state from file

#if !defined(USE_GRID) || defined(ARCH_BGQ) 
  GnoneFnone lattice;
#else

   
  
#ifdef USE_GRID_GPARITY
  assert(ngp != 0);
  std::cout << "Using Gparity BCs\n";
#else
  assert(ngp == 0);
  std::cout << "Using standard BCs\n";
#endif

#ifndef ARCH_BGQ //on BGQ we just use Grid for its SIMD wrappers
  FgridParams fgp; fgp.epsilon = 0.; fgp.mobius_scale = 32./12.;
  typename GridA2Apolicies::FgridGFclass lattice(fgp);
#endif
#endif
  
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
    if(!unit_gauge) lattice.SetGfieldDisOrd();
    else lattice.SetGfieldOrd();
  }else{
    ReadLatticeParallel readLat;
    if(UniqueID()==0) printf("Reading: %s (NERSC-format)\n",load_config_file);
    readLat.read(lattice,load_config_file);
    if(UniqueID()==0) printf("Config read.\n");
  }
  if(save_config){
    if(UniqueID()==0) printf("Saving config to %s\n",save_config_file);

    QioArg wt_arg(save_config_file,0.001);
    
    wt_arg.ConcurIONumber=32;
    WriteLatticeParallel wl;
    wl.setHeader("disord_id","disord_label",0);
    wl.write(lattice,wt_arg);
    
    if(!wl.good()) ERR.General("main","()","Failed write lattice %s",save_config_file);

    if(UniqueID()==0) printf("Config written.\n");
  }

  A2AArg a2a_args;
  a2a_args.nl = nlowmodes;
  a2a_args.nhits = 1;
  a2a_args.rand_type = UONE;
  a2a_args.src_width = 1;

  if(0) testCyclicPermute();
  
  if(0) demonstrateFFTreln<ScalarA2Apolicies>(a2a_args);


  if(0) testA2AvectorFFTrelnGparity<ScalarA2Apolicies>(a2a_args,lattice);
#ifdef USE_GRID
  if(0) testA2AvectorFFTrelnGparity<GridA2Apolicies>(a2a_args,lattice);
#endif
  
  if(0) testSumSource<ScalarA2Apolicies>(a2a_args,lattice);
#ifdef USE_GRID
  if(0) testSumSource<GridA2Apolicies>(a2a_args,lattice);
#endif


  if(0) testMfFFTreln<ScalarA2Apolicies>(a2a_args,lattice);
#ifdef USE_GRID
  if(0) testMfFFTreln<GridA2Apolicies>(a2a_args,lattice);
#endif
  
  if(0) testFFTopt<ScalarA2Apolicies>();
#ifdef USE_GRID
  if(0) testFFTopt<GridA2Apolicies>();
#endif
  
  if(0) testA2AFFTinv<ScalarA2Apolicies>(a2a_args,lattice);
  
  if(0) testVVdag<ScalarA2Apolicies>(lattice);
#ifdef USE_GRID
  if(0) testVVdag<GridA2Apolicies>(lattice);

  if(0) testvMvGridOrig<ScalarA2Apolicies,GridA2Apolicies>(a2a_args, 1, nthreads, tol);
#endif
  
  if(0) testDestructiveFFT<A2ApoliciesDoubleManualAlloc>(a2a_args,lattice);
  
  if(0) testA2AallocFree(a2a_args,lattice);

#ifdef USE_GRID
  if(0) benchmarkMFcontractKernel<GridA2Apolicies>(ntests,nthreads);
#endif

#ifdef USE_GRID
  if(0) testGridg5Contract<Grid::vComplexD>();
#endif
  
  if(0) benchmarkTrace(ntests,tol);
  if(0) benchmarkSpinFlavorTrace(ntests,tol);
  if(0) benchmarkTraceProd(ntests,tol);
  if(0) benchmarkColorTranspose(ntests,tol);
  if(0) benchmarkmultGammaLeft(ntests, tol);

#ifdef USE_GRID
  if(0) testMFcontract<ScalarA2Apolicies,GridA2Apolicies>(a2a_args, nthreads,tol);
#endif

  if(0) testMultiSource<ScalarA2Apolicies>(a2a_args,lattice);
#ifdef USE_GRID
  if(0) testMultiSource<GridA2Apolicies>(a2a_args,lattice);
#endif

  
#ifdef USE_GRID
  if(1) benchmarkMFcontract<ScalarA2Apolicies,GridA2Apolicies>(a2a_args, ntests, nthreads);
  if(0) benchmarkMultiSrcMFcontract<ScalarA2Apolicies,GridA2Apolicies>(a2a_args, ntests, nthreads);
#endif

  if(0) testTraceSingle<ScalarA2Apolicies>(a2a_args,tol);

  if(0) testMFmult<ScalarA2Apolicies>(a2a_args,tol);

  if(0) testCPSfieldImpex();
#if defined(USE_GRID) && !defined(ARCH_BGQ)
  if(0) testGridFieldImpex<GridA2Apolicies>(lattice);
  if(0) testLanczosIO<GridA2Apolicies>(lattice);
#endif
  
  if(0) testCPSfieldIO();
  if(0) testA2AvectorIO<ScalarA2Apolicies>(a2a_args);
  if(0) testA2AvectorIO<GridA2Apolicies>(a2a_args);

  if(0) benchmarkCPSfieldIO();

  if(0) testPointSource();

#if defined(USE_GRID) && !defined(ARCH_BGQ)
  if(0) testLMAprop<GridA2Apolicies>(lattice,argc,argv);
#endif

  if(0) testSCFmat();

#ifdef USE_GRID
  if(0) testKtoPiPiType3<GridA2Apolicies>(a2a_args,lattice);
#endif

#ifdef USE_GRID
  if(0) testMFmult<GridA2Apolicies>(a2a_args, tol);
  if(0) benchmarkMFmult<GridA2Apolicies>(a2a_args, ntests);
#endif

  if(0) timeAllReduce(false);
  if(0) timeAllReduce(true);


  Grid::vComplexD a, b;

  Grid::vsplat(a, std::complex<double>(1,2));
  Grid::vsplat(b, std::complex<double>(2,1));
  assert(!equals(a,b));
  assert(!equals(b,a));
  assert(equals(a,a));
  assert(equals(b,b));
    
  if(!UniqueID()){ printf("Finished\n"); fflush(stdout); }
  End();
  
  return 0;
}

