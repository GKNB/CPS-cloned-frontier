#include <alg/a2a/base/utils_main.h>

CPS_START_NAMESPACE

//Gauge and RNG read
void ReadGaugeField(const MeasArg &meas_arg, bool double_latt){
  double time = -dclock();
  const char *cname = "main";
  const char *fname = "ReadGaugeField";

  GwilsonFdwf lat;
  std::ostringstream os;
  os << meas_arg.GaugeStem << '.' << meas_arg.TrajCur;
  std::string lat_file = os.str();

#ifdef MEMTEST_MODE
  lat.SetGfieldOrd();
#else
  ReadLatticeParallel rl;
  if(double_latt) rl.disableGparityReconstructUstarField();

  rl.read(lat,lat_file.c_str());
  if(!rl.good())ERR.General(cname,fname,"Failed read lattice %s",lat_file.c_str());
#endif

  time += dclock();
  print_time(cname,fname,time);
}

void ReadRngFile(const MeasArg &meas_arg, bool double_latt){
  double time = -dclock();
  const char *cname = "main";
  const char *fname = "ReadRngFile";

  std::ostringstream os;
  os << meas_arg.RNGStem << '.' << meas_arg.TrajCur;
  std::string rng_file = os.str();
#ifndef MEMTEST_MODE
  if(!LRG.Read(rng_file.c_str())) ERR.General(cname,fname,"Failed read rng file %s",rng_file.c_str());
#endif
  time += dclock();
  print_time(cname,fname,time);
}

//Read both gauge and RNG field contingent on DoArgs start_conf_kind and start_seed_kind, allowing easy use of random lattices for testing
void readGaugeRNG(const DoArg &do_arg, const MeasArg &meas_arg, const bool double_latt){
  if(do_arg.start_conf_kind == START_CONF_FILE){
    if(!UniqueID()) printf("Reading gauge field from file\n");
    ReadGaugeField(meas_arg,double_latt);
  }else if(do_arg.start_conf_kind == START_CONF_ORD){
    if(!UniqueID()) printf("Using ordered gauge field\n");
    GwilsonFdwf lat;
    lat.SetGfieldOrd();
  }else if(do_arg.start_conf_kind == START_CONF_DISORD){
    //Generate new random config for every traj in outer loop
    if(!UniqueID()) printf("Using disordered gauge field\n");
    GwilsonFdwf lat;
    lat.SetGfieldDisOrd();
  }else ERR.General("","readGaugeRNG","Unsupported do_arg.start_conf_kind\n");

  if(do_arg.start_seed_kind == START_SEED_FILE){
    if(!UniqueID()) printf("Reading RNG state from file\n");
    ReadRngFile(meas_arg,double_latt); 
  }else{
    if(!UniqueID()) printf("Using existing RNG state\n");
  }
}

//Rotate the temporal boundary links by a phase exp(i degrees/180 *pi)
void TboundaryTwist(const double degrees){
  std::complex<double> twist = std::polar(1., degrees/180. * M_PI);
  std::complex<double> ctwist = std::conj(twist);

  GwilsonFdwf lat;
  if(GJP.NodeCoor(3) == GJP.Nodes(3)-1){ //nodes on outer temporal boundary
    int x[4];
    x[3] = GJP.NodeSites(3)-1;
    for(x[0]=0;x[0]<GJP.NodeSites(0);x[0]++){
      for(x[1]=0;x[1]<GJP.NodeSites(1);x[1]++){
	for(x[2]=0;x[2]<GJP.NodeSites(2);x[2]++){
	  for(int f=0;f<GJP.Gparity()-1;f++){
	    std::complex<double>* m = (std::complex<double>*)(lat.GaugeField() + 3 + lat.GsiteOffset(x) + 4*f*GJP.VolNodeSites());
	    for(int i=0;i<9;i++) m[i] *= ( f == 0 ? twist : ctwist );
	  }
	}
      }
    }
  }
}


//Initialize OpenMP, GJP and QDP (if using BFM)
void initCPS(int argc, char **argv, const DoArg &do_arg, const int nthreads){
  if(!UniqueID()) printf("Initializing CPS\n");
  
  //Stop GJP from loading the gauge and RNG field on initialization; we handle these ourselves under a config loop
  DoArg do_arg_tmp(do_arg);
  do_arg_tmp.start_conf_kind = START_CONF_ORD;
  do_arg_tmp.start_seed_kind = START_SEED_FIXED;
  
  GJP.Initialize(do_arg_tmp);
  LRG.Initialize();
 
#ifdef USE_BFM
  cps_qdp_init(&argc,&argv);
  //Chroma::initialize(&argc,&argv);
#endif
  GJP.SetNthreads(nthreads);
#ifdef USE_GRID
  Grid::GridThread::SetThreads(nthreads);
  if(!UniqueID()) printf("Set number of Grid threads to %d (check returns %d)\n", nthreads, Grid::GridThread::GetThreads()); 
#endif
}


//Do the gauge fixing if !skip_gauge_fix
void doGaugeFix(Lattice &lat, const bool skip_gauge_fix, const FixGaugeArg &fix_gauge_arg){
  CommonArg carg;
  AlgFixGauge fix_gauge(lat, &carg, const_cast<FixGaugeArg *>(&fix_gauge_arg) );
  if( (lat.FixGaugeKind() != FIX_GAUGE_NONE) || (lat.FixGaugePtr() != NULL) )
    lat.FixGaugeFree(); //in case it has previously been allocated
  if(skip_gauge_fix){
    if(!UniqueID()) printf("Skipping gauge fix -> Setting all GF matrices to unity\n");
    gaugeFixUnity(lat,fix_gauge_arg);      
  }else{
    if(!UniqueID()){ printf("Gauge fixing\n"); fflush(stdout); }
    double time = -dclock();
#ifndef MEMTEST_MODE
    fix_gauge.run();
#else
    gaugeFixUnity(lat,fix_gauge_arg);
#endif      
    time += dclock();
    print_time("main","Gauge fix",time);
  }

  if(!UniqueID()) printf("Memory after gauge fix:\n");
  printMem();
}



CPS_END_NAMESPACE
