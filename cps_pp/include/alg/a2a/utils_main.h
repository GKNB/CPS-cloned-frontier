#ifndef _UTILS_MAIN_H_
#define _UTILS_MAIN_H_

#include <util/time_cps.h>
#include <util/lattice/fgrid.h>
#include <alg/meas_arg.h>
#include <alg/ktopipi_jobparams.h>
#include <alg/a2a/bfm_wrappers.h>
#include <alg/a2a/CPSfield.h>
#include <alg/alg_fix_gauge.h>

//Useful functions for main programs
CPS_START_NAMESPACE


//Gauge and RNG read
void ReadGaugeField(const MeasArg &meas_arg, bool double_latt = false){
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

void ReadRngFile(const MeasArg &meas_arg, bool double_latt = false){
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


//template-factory for CPS lattice class
struct isGridtype{};
struct isBFMtype{};

template<typename LatticeType, typename BFMorGrid>
struct createLattice{};

#ifdef USE_BFM
template<typename LatticeType>
struct createLattice<LatticeType, isBFMtype>{
  static LatticeType* doit(BFMsolvers &bfm_solvers){
    LatticeType* lat = new LatticeType;
    bfm_solvers.importLattice(lat);
    return lat;
  }
};
#endif

#ifdef USE_GRID
template<typename LatticeType>
struct createLattice<LatticeType, isGridtype>{
  static LatticeType* doit(const JobParams &jp){
    assert(jp.solver == BFM_HmCayleyTanh);
    FgridParams grid_params; 
    grid_params.mobius_scale = jp.mobius_scale;
    LatticeType* lat = new LatticeType(grid_params);
        
    NullObject null_obj;
    lat->BondCond();
    CPSfield<cps::ComplexD,4*9,FourDpolicy<OneFlavorPolicy> > cps_gauge((cps::ComplexD*)lat->GaugeField(),null_obj);
    cps_gauge.exportGridField(*lat->getUmu());
    lat->BondCond();
    
    return lat;
  }
};
#endif


//Initialize OpenMP, GJP and QDP (if using BFM)
void initCPS(int argc, char **argv, const DoArg &do_arg, const int nthreads){
  if(!UniqueID()) printf("Initializing CPS\n");
  
  //Stop GJP from loading the gauge and RNG field on initialization; we handle these ourselves under a config loop
  DoArg do_arg_tmp(do_arg);
  do_arg_tmp.start_conf_kind = START_CONF_ORD;
  do_arg_tmp.start_seed_kind = START_SEED_FIXED;
  
  GJP.Initialize(do_arg_tmp);
  LRG.Initialize();

#if defined(USE_GRID_A2A) || defined(USE_GRID_LANCZOS)
  if(GJP.Gparity()){
#ifndef USE_GRID_GPARITY
    ERR.General("","","Must compile main program with flag USE_GRID_GPARITY to enable G-parity\n");
#endif
  }else{
#ifdef USE_GRID_GPARITY
    ERR.General("","","Must compile main program with flag USE_GRID_GPARITY off to disable G-parity\n");
#endif
  }      
#endif
  
#ifdef USE_BFM
  cps_qdp_init(&argc,&argv);
  //Chroma::initialize(&argc,&argv);
#endif
  GJP.SetNthreads(nthreads);
#ifdef USE_GRID
  Grid::GridThread::SetThreads(nthreads);
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


#endif
