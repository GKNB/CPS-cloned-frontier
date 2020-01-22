#ifndef _A2A_BFM_GRID_COMBINED_WRAPPERS_H_
#define _A2A_BFM_GRID_COMBINED_WRAPPERS_H_

#include <alg/a2a/lattice/fermion/grid_wrappers.h>
#include <alg/a2a/lattice/fermion/bfm_wrappers.h>
#include <alg/a2a/a2a_fields.h>
#include "utils_main.h"

CPS_START_NAMESPACE

//A struct to pass around the BFM solvers if they are needed
struct BFMGridSolverWrapper{  
#if defined(USE_BFM_A2A) || defined(USE_BFM_LANCZOS)
  BFMsolvers bfm_solvers;
  
  BFMGridSolverWrapper(const int nthreads, const double mass, const double residual, const int max_iters, const BfmSolverType bfm_solver, const double mobius_scale):
    bfm_solvers(nthreads, mass, residual, max_iters, bfm_solver, mobius_scale){
  }
#endif
};

//Unified container for the BFM and Grid Lanczos wrappers
template<typename A2Apolicies>
struct BFMGridLanczosWrapper{
  BFMGridSolverWrapper &solvers;
  const JobParams &jp;

  BFMGridLanczosWrapper(BFMGridSolverWrapper &_solvers, const JobParams &_jp): solvers(_solvers), jp(_jp){}
  
#ifdef USE_GRID_LANCZOS
  GridLanczosWrapper<A2Apolicies> wrapper;
  typedef typename A2Apolicies::FgridGFclass LanczosLattice;
  
  void compute(const LancArg &lanc_arg){
    LanczosLattice* lanczos_lat = createLattice<LanczosLattice,isGridtype>::doit(jp);
    wrapper.compute(lanc_arg, *lanczos_lat);
    wrapper.moveDPevecsToIndependentGrid(*lanczos_lat); //make sure the underlying Grids don't get deleted when we delete the lattice instance
    delete lanczos_lat;
  }
  void randomizeEvecs(const LancArg &lanc_arg){
    LanczosLattice* lanczos_lat = createLattice<LanczosLattice,isGridtype>::doit(jp);
    wrapper.randomizeEvecs(lanc_arg, *lanczos_lat);
    wrapper.moveDPevecsToIndependentGrid(*lanczos_lat);
    delete lanczos_lat;
  }
    
#elif defined USE_BFM_LANCZOS
  BFMLanczosWrapper wrapper;
  typedef GwilsonFdwf LanczosLattice;

  void compute(const LancArg &lanc_arg){
    LanczosLattice* lanczos_lat = createLattice<LanczosLattice,isBFMtype>::doit(solvers.bfm_solvers);
    wrapper.compute(lanc_arg, solvers.bfm_solvers);
    delete lanczos_lat;
  }
  void randomizeEvecs(const LancArg &lanc_arg){
    LanczosLattice* lanczos_lat = createLattice<LanczosLattice,isBFMtype>::doit(solvers.bfm_solvers);
    wrapper.randomizeEvecs(lanc_arg, solvers.bfm_solvers);
    delete lanczos_lat;
  }
#endif

  void toSingle(){
    wrapper.toSingle();
  }
  void freeEvecs(){
    wrapper.freeEvecs();
  }
  void writeParallel(const std::string &file_stub, FP_FORMAT fileformat = FP_AUTOMATIC) const{
    wrapper.writeParallel(file_stub, fileformat);
  }  
  void readParallel(const std::string &file_stub){
#ifdef USE_GRID_LANCZOS
    LanczosLattice* lanczos_lat = createLattice<LanczosLattice,isGridtype>::doit(jp);
    wrapper.readParallel(file_stub, *lanczos_lat);
    delete lanczos_lat;
#else
    wrapper.readParallel(file_stub);
#endif
  }
  
};


//Wrap the creation of the CPS lattice instance and the computation of V and W vectors
template<typename A2Apolicies>
struct BFMGridA2ALatticeWrapper{
#ifdef USE_GRID_A2A
  typedef typename A2Apolicies::FgridGFclass A2ALattice;
#elif defined(USE_BFM_A2A)
  typedef GwilsonFdwf A2ALattice;
#else
# error "BFMGridA2ALatticeWrapper: Must compile with either USE_GRID_A2A or USE_BFM_A2A"
#endif

  A2ALattice *a2a_lat;

  BFMGridSolverWrapper &solvers;
  const JobParams &jp;

  BFMGridA2ALatticeWrapper(BFMGridSolverWrapper &_solvers, const JobParams &_jp): solvers(_solvers), jp(_jp){
#ifdef USE_GRID_A2A
    a2a_lat = createLattice<A2ALattice,isGridtype>::doit(jp);
#else
    a2a_lat = createLattice<A2ALattice,isBFMtype>::doit(solvers.bfm_solvers);
#endif
  }

  void computeVW(A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W, const BFMGridLanczosWrapper<A2Apolicies> &eig, const CGcontrols &cg_controls, bool randomize_vw) const{
#ifdef USE_DESTRUCTIVE_FFT
    V.allocModes(); W.allocModes();
#endif  
    if(!randomize_vw){
#ifdef USE_BFM_LANCZOS
      W.computeVW(V, *a2a_lat, *eig.wrapper.eig, eig.wrapper.singleprec_evecs, cg_controls, solvers.bfm_solvers.dwf_d, &solvers.bfm_solvers.dwf_f);
#else
      if(eig.wrapper.singleprec_evecs){
	W.computeVW(V, *a2a_lat, eig.wrapper.evec_f, eig.wrapper.eval, eig.wrapper.mass, cg_controls);
      }else{
	W.computeVW(V, *a2a_lat, eig.wrapper.evec, eig.wrapper.eval, eig.wrapper.mass, cg_controls);
      }
#endif     
    }else randomizeVW<A2Apolicies>(V,W);  
  }    

  ~BFMGridA2ALatticeWrapper(){
    delete a2a_lat;
  }
};



//Compute the eigenvectors and convert to single precision
template<typename A2Apolicies>
void computeEvecs(BFMGridLanczosWrapper<A2Apolicies> &eig, const LancArg &lanc_arg, const JobParams &jp, const char* name, const bool randomize_evecs){
  if(!UniqueID()) printf("Running %s quark Lanczos\n",name);
  double time = -dclock();
  if(randomize_evecs) eig.randomizeEvecs(lanc_arg);
  else eig.compute(lanc_arg);
  time += dclock();

  std::ostringstream os; os << name << " quark Lanczos";
      
  print_time("main",os.str().c_str(),time);

  if(!UniqueID()) printf("Memory after %s quark Lanczos:\n",name);
  printMem();      

#ifndef A2A_LANCZOS_SINGLE
  if(jp.convert_evecs_to_single_precision){ 
    eig.toSingle();
    if(!UniqueID()) printf("Memory after single-prec conversion of %s quark evecs:\n",name);
    printMem();
  }
#endif
#ifdef USE_BFM_LANCZOS
  eig.wrapper.checkEvecMemGuards();
#endif
}



CPS_END_NAMESPACE

#endif
