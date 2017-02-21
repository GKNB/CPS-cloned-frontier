#ifndef _UTILS_MAIN_H_
#define _UTILS_MAIN_H_

#include <util/time_cps.h>
#include <util/lattice/fgrid.h>
#include <alg/meas_arg.h>
#include <alg/ktopipi_jobparams.h>
#include <alg/a2a/bfm_wrappers.h>
#include <alg/a2a/CPSfield.h>

//Useful functions for main programs
CPS_START_NAMESPACE



void ReadGaugeField(const MeasArg &meas_arg, bool double_latt = false){
  double time = -dclock();
  const char *cname = "main";
  const char *fname = "ReadGaugeField";

  GwilsonFdwf lat;
  std::ostringstream os;
  os << meas_arg.GaugeStem << '.' << meas_arg.TrajCur;
  std::string lat_file = os.str();

  ReadLatticeParallel rl;
  if(double_latt) rl.disableGparityReconstructUstarField();

  rl.read(lat,lat_file.c_str());
  if(!rl.good())ERR.General(cname,fname,"Failed read lattice %s",lat_file.c_str());

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

  if(!LRG.Read(rng_file.c_str())) ERR.General(cname,fname,"Failed read rng file %s",rng_file.c_str());
  time += dclock();
  print_time(cname,fname,time);
}

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
    CPSfield<cps::ComplexD,4*9,FourDpolicy,OneFlavorPolicy> cps_gauge((cps::ComplexD*)lat->GaugeField(),null_obj);
    cps_gauge.exportGridField(*lat->getUmu());
    lat->BondCond();
    
    return lat;
  }
};
#endif


CPS_END_NAMESPACE


#endif
