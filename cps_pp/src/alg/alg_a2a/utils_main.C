#include <alg/a2a/base/utils_main.h>
#include <alg/a2a/lattice/CPSfield.h>
#ifdef USE_OMP
#include<omp.h>
#endif

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
    LOGA2A << "Reading gauge field from file" << std::endl;
    ReadGaugeField(meas_arg,double_latt);
  }else if(do_arg.start_conf_kind == START_CONF_ORD){
    LOGA2A << "Using ordered gauge field" << std::endl;
    GwilsonFdwf lat;
    lat.SetGfieldOrd();
  }else if(do_arg.start_conf_kind == START_CONF_DISORD){
    //Generate new random config for every traj in outer loop
    LOGA2A << "Using disordered gauge field" << std::endl;
    GwilsonFdwf lat;
    lat.SetGfieldDisOrd();
  }else ERR.General("","readGaugeRNG","Unsupported do_arg.start_conf_kind\n");

  if(do_arg.start_seed_kind == START_SEED_FILE){
    LOGA2A << "Reading RNG state from file" << std::endl;
    ReadRngFile(meas_arg,double_latt); 
  }else if(do_arg.start_seed_kind == START_SEED_INPUT){
    //Here the input seed in the do_arg is used as a base seed to which 23*meas_arg.TrajCur is added
    //This ensures different random seeds for each configuration in a consistent way
    GJP.SetSeedKind(START_SEED_INPUT); // was overridden in initCPS
    GJP.SettSeedValue(do_arg.start_seed_value + 23*meas_arg.TrajCur);
    LOGA2A << "Seeding RNG with base seed " << GJP.StartSeedValue() << std::endl;
    LRG.Reinitialize();
  }else{
    LOGA2A << "Using existing RNG state" << std::endl;
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


//Initialize OpenMP, GJP and QDP
void initCPS(int argc, char **argv, const DoArg &do_arg, const int nthreads){
  LOGA2A << "Initializing CPS" << std::endl;
  
  //Stop GJP from loading the gauge and RNG field on initialization; we handle these ourselves under a config loop
  DoArg do_arg_tmp(do_arg);
  do_arg_tmp.start_conf_kind = START_CONF_ORD;
  do_arg_tmp.start_seed_kind = START_SEED_FIXED;
  
  GJP.Initialize(do_arg_tmp);
  LRG.Initialize();
 
#if defined(USE_OMP) && !defined(_OPENMP)

  //This error can trigger when compiling for accelerators in the device compilation stage
  //We can avoid if we have Grid by using Grid's macro that recognizes when device compilation is occurring
  #ifdef USE_GRID
    #ifndef GRID_SIMT
      #error "CPS was configured to use openmp but preprocessor flag _OPENMP not set!  We have tested using GRID_SIMT to ensure this is not occuring at device compilation stage"
    #endif
  #else
      #error "CPS was configured to use openmp but preprocessor flag _OPENMP not set!  This may be occuring at device compilation stage"  
  #endif
  
#endif

#ifndef _OPENMP
  LOGA2A << "WARNING: CPS was not compiled with openmp!" << std::endl;
#endif

  a2a_printf("omp_get_max_threads()=%d (pre GJP.SetNthreads())\n",omp_get_max_threads());
  GJP.SetNthreads(nthreads);
  a2a_printf("omp_get_max_threads()=%d (post GJP.SetNthreads())\n",omp_get_max_threads());
#ifdef USE_GRID
  Grid::GridThread::SetThreads(nthreads);
  a2a_printf("Set number of Grid threads to %d (check returns %d,   omp_get_max_threads()=%d)\n", nthreads, Grid::GridThread::GetThreads(), omp_get_max_threads()); 
#endif
}

//Do the gauge fixing if !skip_gauge_fix
void doGaugeFix(Lattice &lat, const bool skip_gauge_fix, const FixGaugeArg &fix_gauge_arg){
  CommonArg carg;
  if( (lat.FixGaugeKind() != FIX_GAUGE_NONE) || (lat.FixGaugePtr() != NULL) )
    lat.FixGaugeFree(); //in case it has previously been allocated
  if(fix_gauge_arg.fix_gauge_kind == FIX_GAUGE_NONE || skip_gauge_fix){
    LOGA2A << "Skipping gauge fix -> Setting all GF matrices to unity" << std::endl;
    gaugeFixUnity(lat,fix_gauge_arg);      
  }else{
    LOGA2A << "Gauge fixing" << std::endl;
    double time = -dclock();
#ifndef MEMTEST_MODE
    AlgFixGauge fix_gauge(lat, &carg, const_cast<FixGaugeArg*>(&fix_gauge_arg) );
    fix_gauge.run();
#else
    gaugeFixUnity(lat,fix_gauge_arg);
#endif      
    time += dclock();
    a2a_print_time("main","Gauge fix",time);
  }

  LOGA2A << "Memory after gauge fix:" << std::endl;
  printMem();
}



//Do the gauge fixing if !skip_gauge_fix
void doGaugeFix(Lattice &lat, const bool skip_gauge_fix, const FixGaugeArgGrid &fix_gauge_arg){
  FixGaugeArg farg;
  farg.fix_gauge_kind = fix_gauge_arg.fix_gauge_kind;
  farg.stop_cond = fix_gauge_arg.stop_cond;
  farg.max_iter_num = fix_gauge_arg.max_iter_num;
  farg.hyperplane_start = 0;
  farg.hyperplane_step = 1;
  switch(farg.fix_gauge_kind){
  case FIX_GAUGE_COULOMB_X:
    farg.hyperplane_num = GJP.Xnodes()*GJP.XnodeSites(); break;
  case FIX_GAUGE_COULOMB_Y:
    farg.hyperplane_num = GJP.Ynodes()*GJP.YnodeSites(); break;
  case FIX_GAUGE_COULOMB_Z:
    farg.hyperplane_num = GJP.Znodes()*GJP.ZnodeSites(); break;
  case FIX_GAUGE_COULOMB_T:
    farg.hyperplane_num = GJP.Tnodes()*GJP.TnodeSites(); break;
  case FIX_GAUGE_LANDAU:
    farg.hyperplane_num = 1; break;
  default:
    ERR.General("","doGaugeFix","Unknown gauge fix type");
  }

  //Fallback to non-Grid implementation for unsupported types
  if( 
#ifdef USE_GRID
     (fix_gauge_arg.fix_gauge_kind != FIX_GAUGE_COULOMB_T &&
      fix_gauge_arg.fix_gauge_kind != FIX_GAUGE_LANDAU) 
     || skip_gauge_fix
#else
     1
#endif
      ){
    return doGaugeFix(lat,skip_gauge_fix,farg);
  }

  //Copy gauge field to Grid
#ifdef USE_GRID
  LOGA2A << "Gauge fixing with Grid's Fourier-accelerated algorithm" << std::endl;
  double time = -dclock();
  FgridBase &lat_fgrid = dynamic_cast<FgridBase &>(lat);
  
  NullObject null_obj;
  CPSfield<cps::ComplexD,4*9,FourDpolicy<OneFlavorPolicy> > cps_gauge((cps::ComplexD*)lat.GaugeField(),null_obj);
  Grid::LatticeGaugeFieldD U_grid(lat_fgrid.getUGrid());
  cps_gauge.exportGridField(U_grid);
  typename Grid::ConjugateGimplD::GaugeLinkField V_grid(lat_fgrid.getUGrid());
  CPSfield<cps::ComplexD,9,FourDpolicy<OneFlavorPolicy> > cps_gfmat(null_obj);
 
  //Set up the gauge boundary conditions (periodic in directions with dirs4[i]=0)
  std::vector<int> dirs4(4,0);
  for(int i=0;i<4;i++) 
    if(GJP.Bc(i) == BND_CND_GPARITY) dirs4[i] = 1;
  Grid::ConjugateGimplD::setDirections(dirs4); 
  
  bool Fourier = true; //use Fourier acceleration
  double stop_Omega = fix_gauge_arg.stop_cond/20; //stop on  1 - \sum_x tr( g(x) )/Nc/V    where g(x) = exp(-i\alpha \partial_\mu A_\mu(x) )
                                                  //factor of 20 is a rule of thumb that appears to reproduce CPS stopping condition
  double stop_Phi = 1; //stop on relative difference in  \sum_x\sum_\mu tr( U_\mu(x) )

  //Do the gauge fixing
  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  std::vector<int> h_planes;  
  if(fix_gauge_arg.fix_gauge_kind == FIX_GAUGE_COULOMB_T){
    Grid::FourierAcceleratedGaugeFixer<Grid::ConjugateGimplD>::SteepestDescentGaugeFix(U_grid,V_grid,fix_gauge_arg.alpha,fix_gauge_arg.max_iter_num,stop_Omega,stop_Phi,Fourier,3);
    h_planes.resize(Lt); for(int i=0;i<Lt;i++) h_planes[i] = i;
  }else if(fix_gauge_arg.fix_gauge_kind == FIX_GAUGE_LANDAU){
    Grid::FourierAcceleratedGaugeFixer<Grid::ConjugateGimplD>::SteepestDescentGaugeFix(U_grid,V_grid,fix_gauge_arg.alpha,fix_gauge_arg.max_iter_num,stop_Omega,stop_Phi,Fourier);
    h_planes.resize(1,0);
  }

  //Copy resulting V matrix into Lattice object
  cps_gfmat.importGridField(V_grid);
  if(lat.FixGaugePtr() != nullptr) lat.FixGaugeFree();  
  lat.FixGaugeAllocate(fix_gauge_arg.fix_gauge_kind, h_planes.size(), h_planes.data()); 
  
  {
    CPSautoView(cps_gfmat_v,cps_gfmat,HostRead);
#pragma omp parallel for
    for(int s=0;s<GJP.VolNodeSites();s++){
      cps::ComplexD* to = (cps::ComplexD*)lat.FixGaugeMatrix(s,0);
      cps::ComplexD const* from = cps_gfmat_v.site_ptr(s);
      memcpy(to, from, 9*sizeof(cps::ComplexD));
      if(GJP.Gparity()){
	to = (cps::ComplexD*)lat.FixGaugeMatrix(s,1);
	for(int i=0;i<9;i++) to[i] = std::conj(from[i]);
      }
    }
  }

  LOGA2A << "Testing gauge fixing result against CPS' convergence metric: " << 
    gaugeFixTest::delta(lat, (fix_gauge_arg.fix_gauge_kind == FIX_GAUGE_COULOMB_T ? 3 : -1) ) << " expect " << fix_gauge_arg.stop_cond << std::endl;

  time += dclock();
  a2a_print_time("main","Gauge fix (Grid Fourier accelerated)",time);

  LOGA2A << "Memory after gauge fix:" << std::endl;
  printMem();
#endif
}

void a2a_printf(const char* format, ...){
  if(UniqueID()) return;
  int n; //not counting null character
  {
    char buf[1024];
    va_list argptr;
    va_start(argptr, format);    
    n = vsnprintf(buf, 1024, format, argptr);
    va_end(argptr);
    if(n < 1024){
      LOGA2A << buf;
      return;
    }
  }
  
  char buf[n+1];
  va_list argptr;
  va_start(argptr, format);    
  int n2 = vsnprintf(buf, 1024, format, argptr);
  va_end(argptr);
  assert(n2 <= n);
  LOGA2A << buf;
}

void a2a_printfnt(const char* format, ...){
  if(UniqueID()) return;
  int n; //not counting null character
  {
    char buf[1024];
    va_list argptr;
    va_start(argptr, format);    
    n = vsnprintf(buf, 1024, format, argptr);
    va_end(argptr);
    if(n < 1024){
      LOGA2ANT << buf;
      return;
    }
  }
  
  char buf[n+1];
  va_list argptr;
  va_start(argptr, format);    
  int n2 = vsnprintf(buf, 1024, format, argptr);
  va_end(argptr);
  assert(n2 <= n);
  LOGA2ANT << buf;
}


CPS_END_NAMESPACE
