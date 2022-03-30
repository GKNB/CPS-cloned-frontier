#include "benchmark_mesonfield.h"
#include <sstream>

using namespace cps;

#ifndef USE_GRID
typedef void A2ApoliciesSIMDdoubleManualAllocGparity;
typedef void A2ApoliciesSIMDdoubleManualAlloc;
typedef void A2ApoliciesSIMDdoubleAutoAllocGparity;
typedef void A2ApoliciesSIMDdoubleAutoAlloc;
#endif

struct Options{
  bool load_lrg;
  std::string load_lrg_file;
  
  bool save_lrg;
  std::string save_lrg_file;

  bool load_config;
  std::string load_config_file;

  bool save_config;
  std::string save_config_file;

  bool unit_gauge;

  int ntests;
  int nthreads;
  double tol;
  int nlowmodes;

  int nshift; //for shifted src benchmark

  std::vector<int> tsep_k_pi;
  
  Options(){
    load_lrg=false;
    save_lrg=false;
    load_config=false;
    save_config=false;
    unit_gauge = false;

    nthreads = 1;
    ntests = 10;    
    tol = 1e-8;
    nlowmodes = 100;

    nshift = 4;

    tsep_k_pi = {4};
  }


};


template<typename ScalarA2ApoliciesType, typename ScalarA2ApoliciesManualAllocType, typename GridA2ApoliciesType>
void runBenchmarks(int argc,char *argv[], const Options &opt){
  A2AArg a2a_args;
  a2a_args.nl = opt.nlowmodes;
  a2a_args.nhits = 1;
  a2a_args.rand_type = UONE;
  a2a_args.src_width = 1;
  
  int ntests = opt.ntests;
  int nthreads = opt.nthreads;
  double tol = opt.tol;

		   
#if !defined(USE_GRID) || defined(ARCH_BGQ) 
  GnoneFnone lattice;
#else
#ifndef ARCH_BGQ //on BGQ we just use Grid for its SIMD wrappers
  FgridParams fgp; fgp.epsilon = 0.; fgp.mobius_scale = 32./12.;
  typename GridA2ApoliciesType::FgridGFclass lattice(fgp);

  std::cout << "Lattice created with b+c = " << lattice.get_mob_b() + lattice.get_mob_c() << std::endl;
#endif
#endif

  if(opt.load_lrg){
    if(UniqueID()==0) printf("Loading RNG state from %s\n",opt.load_lrg_file.c_str());
    LRG.Read(opt.load_lrg_file.c_str(),32);
  }
  if(opt.save_lrg){
    if(UniqueID()==0) printf("Writing RNG state to %s\n",opt.save_lrg_file.c_str());
    LRG.Write(opt.save_lrg_file.c_str(),32);
  }					       
  if(!opt.load_config){
    if(!opt.unit_gauge){
      std::cout << "Creating random gauge field" << std::endl;
      lattice.SetGfieldDisOrd();
    }else{
      std::cout << "Creating unit gauge field" << std::endl;      
      lattice.SetGfieldOrd();
    }
  }else{
    ReadLatticeParallel readLat;
    if(UniqueID()==0) printf("Reading: %s (NERSC-format)\n",opt.load_config_file.c_str());
    readLat.read(lattice,opt.load_config_file.c_str());
    if(UniqueID()==0) printf("Config read.\n");
  }
  if(opt.save_config){
    if(UniqueID()==0) printf("Saving config to %s\n",opt.save_config_file.c_str());

    QioArg wt_arg(opt.save_config_file.c_str(),0.001);
    
    wt_arg.ConcurIONumber=32;
    WriteLatticeParallel wl;
    wl.setHeader("disord_id","disord_label",0);
    wl.write(lattice,wt_arg);
    
    if(!wl.good()) ERR.General("main","()","Failed write lattice %s",opt.save_config_file.c_str());

    if(UniqueID()==0) printf("Config written.\n");
  }

  if(0) benchmarkFFT<ScalarA2ApoliciesType>(ntests);

#ifdef USE_GRID
  if(0) benchmarkMFcontractKernel<GridA2ApoliciesType>(ntests,nthreads);
#endif
 
  if(0) benchmarkTrace(ntests,tol);
  if(0) benchmarkSpinFlavorTrace(ntests,tol);
  if(0) benchmarkTraceProd(ntests,tol);
  if(0) benchmarkColorTranspose(ntests,tol);
  if(0) benchmarkmultGammaLeft(ntests, tol);
 
#ifdef USE_GRID
  if(0) benchmarkMFcontract<ScalarA2ApoliciesType,GridA2ApoliciesType>(a2a_args, ntests, nthreads);
  if(0) benchmarkMultiSrcMFcontract<ScalarA2ApoliciesType,GridA2ApoliciesType>(a2a_args, ntests, nthreads);
  if(0) benchmarkMultiShiftMFcontract<GridA2ApoliciesType>(a2a_args, opt.nshift);

#endif

  if(0) benchmarkCPSfieldIO();

#ifdef USE_GRID
  if(0) benchmarkMFmult<GridA2ApoliciesType>(a2a_args, ntests);
#endif

  if(0) timeAllReduce(false);
  if(0) timeAllReduce(true);

#ifdef USE_GRID
  if(0) benchmarkvMvGridOrig<ScalarA2ApoliciesType,GridA2ApoliciesType>(a2a_args, ntests, nthreads);

  if(1) benchmarkvMvGridOffload<GridA2ApoliciesType>(a2a_args, ntests, nthreads);
  if(0) benchmarkVVgridOffload<GridA2ApoliciesType>(a2a_args, ntests, nthreads);
  if(0) benchmarkCPSmatrixField<GridA2ApoliciesType>(ntests);
  if(0) benchmarkKtoPiPiType1offload<GridA2ApoliciesType>(a2a_args, lattice, opt.tsep_k_pi);
  if(0) benchmarkKtoPiPiType4offload<GridA2ApoliciesType>(a2a_args, lattice, opt.tsep_k_pi);

  if(0) benchmarkDeflation<GridA2ApoliciesType>(lattice ,opt.nlowmodes, argc, argv);

  if(0) benchmarkMfTraceProd<GridA2ApoliciesType>(a2a_args, ntests);
  if(0) benchmarkMfTraceProdGPU<GridA2ApoliciesType>(a2a_args, ntests);
  if(0) benchmarkMfVectorTraceProd<GridA2ApoliciesType>(a2a_args, ntests);
  if(0) benchmarkPiPiContractions<GridA2ApoliciesType>(a2a_args);

  if(0) benchmarkMesonFieldUnpack<GridA2ApoliciesType>(a2a_args, ntests);
  if(0) benchmarkMesonFieldPack<GridA2ApoliciesType>(a2a_args, ntests);
  if(0) benchmarkMesonFieldUnpackDevice<GridA2ApoliciesType>(a2a_args, ntests);
  if(0) benchmarkMesonFieldPackDevice<GridA2ApoliciesType>(a2a_args, ntests);
#endif

  if(0) benchmarkMesonFieldGather(a2a_args, ntests);
}



int main(int argc,char *argv[])
{
  Start(&argc, &argv);
 
  int ngp;
  { std::stringstream ss; ss << argv[1]; ss >> ngp; }

  if(!UniqueID()) printf("Doing G-parity in %d directions\n",ngp);

  bool verbose(false);
  int size[] = {2,2,2,2,2};
  int nlowmodes = 100;
  bool use_destructive_FFT = false;
  
  Options opt;

  printf("Argc is %d\n",argc);
  int i=2;
  while(i<argc){
    std::string cmd(argv[i]);  
    if( cmd == "-save_config" ){
      if(i==argc-1){ printf("-save_config requires an argument\n"); exit(-1); }
      opt.save_config=true;
      opt.save_config_file = argv[i+1];
      i+=2;
    }else if( cmd == "-load_config" ){
      if(i==argc-1){ printf("-save_config requires an argument\n"); exit(-1); }
      opt.load_config=true;
      opt.load_config_file = argv[i+1];
      i+=2;
    }else if( cmd == "-latt"){
      if(i>argc-6){
	printf("Did not specify enough arguments for 'latt' (require 5 dimensions)\n"); exit(-1);
      }
      for(int d=0;d<5;d++)
	size[d] = toInt(argv[i+1+d]);
      i+=6;
    }else if( cmd == "-load_lrg"){
      if(i==argc-1){ printf("-load_lrg requires an argument\n"); exit(-1); }
      opt.load_lrg=true;
      opt.load_lrg_file = argv[i+1];
      i+=2;
    }else if( cmd == "-save_lrg" ){
      if(i==argc-1){ printf("-save_lrg requires an argument\n"); exit(-1); }
      opt.save_lrg=true;
      opt.save_lrg_file = argv[i+1];
      i+=2;  
    }else if( cmd == "-verbose" ){
      verbose=true;
      i++;
    }else if( cmd == "-nthread" ){
      opt.nthreads = toInt(argv[i+1]);
      printf("Set nthreads to %d\n", opt.nthreads);
      i+=2;
    }else if( cmd == "-ntest"){
      opt.ntests = toInt(argv[i+1]);
      printf("Set ntests to %d\n", opt.ntests);
      i+=2;
    }else if( cmd == "-unit_gauge"){
      opt.unit_gauge=true;
      i++;
    }else if( cmd == "-tolerance" ){
      std::stringstream ss; ss << argv[i+1];
      ss >> opt.tol;
      if(!UniqueID()) printf("Set tolerance to %g\n",opt.tol);
      i+=2;
    }else if( cmd == "-nl" ){
      std::stringstream ss; ss << argv[i+1];
      ss >> opt.nlowmodes;
      if(!UniqueID()) printf("Set nl to %d\n",opt.nlowmodes);
      i+=2;
    }else if( cmd == "-mf_outerblocking" ){
      int* b[3] = { &BlockedMesonFieldArgs::bi, &BlockedMesonFieldArgs::bj, &BlockedMesonFieldArgs::bp };
      for(int a=0;a<3;a++){
	std::stringstream ss; ss << argv[i+1+a];
	ss >> *b[a];
      }
      i+=4;
    }else if( cmd == "-mf_innerblocking" ){
      int* b[3] = { &BlockedMesonFieldArgs::bii, &BlockedMesonFieldArgs::bjj, &BlockedMesonFieldArgs::bpp };
      for(int a=0;a<3;a++){
	std::stringstream ss; ss << argv[i+1+a];
	ss >> *b[a];
      }
      i+=4;
    }else if( cmd == "-use_destructive_FFT" ){
      use_destructive_FFT = true;
      i++;
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
    }else if( cmd == "-nshift" ){
      std::stringstream ss; ss << argv[i+1];
      ss >> opt.nshift;
      if(!UniqueID()) printf("Set nshift to %d\n",opt.nshift);
      i+=2;
    }else if( cmd == "-mf_burst_store_stub"){ //set the file stub for the meson field disk storage 
      BurstBufferMemoryStorage::filestub() = argv[i+1];
      if(!UniqueID()) printf("Set mesonfield burst buffer file stub to %s\n", argv[i+1]);
      i+=2;
    }else if( cmd == "-tsep_k_pi"){ //provide a list in Grid's  a.b.c.d  format
      std::istringstream iss(argv[i+1]);
      std::string token;
      while (std::getline(iss, token, '.')) {
	if (!token.empty())
	  opt.tsep_k_pi.push_back(toInt(token.c_str()));
      }
      std::cout << "Set tsep_k_pi to {";
      for(int v: opt.tsep_k_pi){ std::cout << v << " "; }
      std::cout <<std::endl;      
      i+=2;      
    }else{
      i++;
    }
  }

  printf("Lattice size is %d %d %d %d\n",size[0],size[1],size[2],size[3],size[4]);

  CommonArg common_arg;
  DoArg do_arg;  setupDoArg(do_arg,size,ngp,verbose);

  GJP.Initialize(do_arg);
  GJP.SetNthreads(opt.nthreads);

#if TARGET == BGQ
  LRG.setSerial();
#endif
  LRG.Initialize(); //usually initialised when lattice generated, but I pre-init here so I can load the state from file

  if(GJP.Gparity()){
    // if(use_destructive_FFT) runBenchmarks<A2ApoliciesDoubleManualAllocGparity, 
    // 					  A2ApoliciesDoubleManualAllocGparity,
    // 					  A2ApoliciesSIMDdoubleManualAllocGparity>(argc, argv, opt);
    // else 
      runBenchmarks<A2ApoliciesDoubleAutoAllocGparity, 
		    A2ApoliciesDoubleManualAllocGparity,
		    A2ApoliciesSIMDdoubleAutoAllocGparity>(argc, argv, opt);
  }else{    
    // if(use_destructive_FFT) runBenchmarks<A2ApoliciesDoubleManualAlloc, 
    // 					  A2ApoliciesDoubleManualAlloc,
    // 					  A2ApoliciesSIMDdoubleManualAlloc>(argc, argv, opt);
    // else 
    //   runBenchmarks<A2ApoliciesDoubleAutoAlloc, 
    // 		    A2ApoliciesDoubleManualAlloc,
    // 		    A2ApoliciesSIMDdoubleAutoAlloc>(argc, argv, opt);
  }

  if(!UniqueID()){ printf("Finished\n"); fflush(stdout); }
  End();
  
  return 0;
}


