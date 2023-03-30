#ifndef _GRID_WRAPPERS_H_
#define _GRID_WRAPPERS_H_
#ifdef USE_GRID

#include "grid_lanczos.h"
#include "evec_io.h"
#include<alg/a2a/base/utils_main.h>

CPS_START_NAMESPACE

//Return a pointer to a CPS FGrid derived lattice type. Handles the BCs correctly
template<typename LatticeType>
LatticeType* createFgridLattice(const JobParams &jp){
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

//Generates and stores evecs and evals
template<typename GridPolicies>
struct GridLanczosWrapper{
  std::vector<Grid::RealD> eval; 
  std::vector<typename GridPolicies::GridFermionField> evec;
  std::vector<typename GridPolicies::GridFermionFieldF> evec_f;
  double mass;
  double resid;
  bool singleprec_evecs;
  
  GridLanczosWrapper(): singleprec_evecs(false){
  }
  
  void compute(const LancArg &lanc_arg, typename GridPolicies::FgridGFclass &lat, A2Apreconditioning precon_type = SchurOriginal){
    mass = lanc_arg.mass;
    resid = lanc_arg.stop_rsd;
    assert(lanc_arg.precon);

# ifdef A2A_LANCZOS_SINGLE
    gridSinglePrecLanczos<GridPolicies>(eval,evec_f,lanc_arg,lat, lat.getUGridF(), lat.getUrbGridF(), lat.getFGridF(), lat.getFrbGridF(), precon_type);
    singleprec_evecs = true;

    evec_f.resize(lanc_arg.N_true_get, lat.getFrbGridF()); //in case the Lanczos implementation does not explicitly remove the extra evecs used for the restart
    eval.resize(lanc_arg.N_true_get);
    
# else    
    gridLanczos<GridPolicies>(eval,evec,lanc_arg,lat, precon_type);
    singleprec_evecs = false;

    evec.resize(lanc_arg.N_true_get, lat.getFrbGrid());
    eval.resize(lanc_arg.N_true_get);
   
#  ifndef MEMTEST_MODE
    test_eigenvectors(evec,eval,lanc_arg.mass,lat,precon_type);
#  endif

# endif    
  }

  //This version creates a FGrid lattice using CPS factory and uses it in the above
  void compute(const JobParams &jp, const LancArg &lanc_arg){
    auto lanczos_lat = createFgridLattice<typename GridPolicies::FgridGFclass>(jp);
    A2Apreconditioning precond = SchurOriginal;
    if(lanc_arg.precon && jp.cg_controls.CGalgorithm == AlgorithmMixedPrecisionMADWF) precond = jp.cg_controls.madwf_params.precond; //SchurDiagTwo often used for ZMADWF
    compute(lanc_arg, *lanczos_lat, precond);
    delete lanczos_lat;
  }

  //Test double prec eigenvectors (TODO: generalize to test single prec)
  static void test_eigenvectors(const std::vector<typename GridPolicies::GridFermionField> &evec, const std::vector<Grid::RealD> &eval, 
				const double mass, typename GridPolicies::FgridGFclass &lattice, A2Apreconditioning precon_type = SchurOriginal){
    if(!UniqueID()) printf("Starting eigenvector residual test\n");
    typedef typename GridPolicies::GridFermionField GridFermionField;
    typedef typename GridPolicies::FgridFclass FgridFclass;
    typedef typename GridPolicies::GridDirac GridDirac;
  
    Grid::GridCartesian *UGrid = lattice.getUGrid();
    Grid::GridRedBlackCartesian *UrbGrid = lattice.getUrbGrid();
    Grid::GridCartesian *FGrid = lattice.getFGrid();
    Grid::GridRedBlackCartesian *FrbGrid = lattice.getFrbGrid();
    Grid::LatticeGaugeFieldD *Umu = lattice.getUmu();
    double mob_b = lattice.get_mob_b();
    double mob_c = mob_b - 1.;   //b-c = 1
    double M5 = GJP.DwfHeight();

    typename GridDirac::ImplParams params;
    lattice.SetParams(params);

    GridDirac Ddwf(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_c, params);
    Grid::SchurOperatorBase<GridFermionField>  *HermOp;
    if(precon_type == SchurOriginal) HermOp = new Grid::SchurDiagMooeeOperator<GridDirac, GridFermionField>(Ddwf);
    else if(precon_type == SchurDiagTwo) HermOp = new Grid::SchurDiagTwoOperator<GridDirac, GridFermionField>(Ddwf);
    else assert(0);
  
    GridFermionField tmp1(FrbGrid);
    GridFermionField tmp2(FrbGrid);
    GridFermionField tmp3(FrbGrid);
  
    for(int i=0;i<evec.size();i++){
      HermOp->Mpc(evec[i], tmp1);
      HermOp->MpcDag(tmp1, tmp2); //tmp2 = M^dag M v

      tmp3 = eval[i] * evec[i]; //tmp3 = lambda v

      double nrm = sqrt(axpy_norm(tmp1, -1., tmp2, tmp3)); //tmp1 = tmp3 - tmp2
    
      if(!UniqueID()) printf("Idx %d Eval %g Resid %g #Evecs %d #Evals %d\n",i,eval[i],nrm,evec.size(),eval.size());
    }
    delete HermOp;
  }

  //Randomize double precision eigenvectors
  void randomizeEvecs(const LancArg &lanc_arg,typename GridPolicies::FgridGFclass &lat){
    typedef typename GridPolicies::GridFermionField GridFermionField;

    mass = lanc_arg.mass;
    resid = lanc_arg.stop_rsd;
    
    evec.resize(lanc_arg.N_true_get, lat.getFrbGrid());
    eval.resize(lanc_arg.N_true_get);
    
    CPSfermion5D<cps::ComplexD> tmp;

    IncludeCBsite<5> oddsites(1, 0);
    IncludeCBsite<5> oddsitesf0(1, 0, 0);
    IncludeCBsite<5> oddsitesf1(1, 0, 1);
    IncludeCBsite<5> evensites(0, 0);
    IncludeCBsite<5> evensitesf0(0, 0, 0);
    IncludeCBsite<5> evensitesf1(0, 0, 1);

#ifdef USE_C11_RNG
    CPS_RNG rng(1234);
    std::uniform_real_distribution<double> dist(0.1,10);
#endif

    for(int i=0;i<lanc_arg.N_true_get;i++){
      evec[i].Checkerboard() = Grid::Odd;
      tmp.setGaussianRandom();
      double nrmcps = tmp.norm2();
      double nrmoddcps = tmp.norm2(oddsites);
      double nrmoddf0cps = tmp.norm2(oddsitesf0);
      double nrmoddf1cps = tmp.norm2(oddsitesf1);

      double nrmevencps = tmp.norm2(evensites);
      double nrmevenf0cps = tmp.norm2(evensitesf0);
      double nrmevenf1cps = tmp.norm2(evensitesf1);

      tmp.exportGridField(evec[i]);
      
      double nrm = Grid::norm2(evec[i]);
#ifdef USE_C11_RNG
      eval[i] = dist(rng);
#else 
      eval[i] = LRG.Lrand(10,0.1); //same on all nodes
#endif

      if(!UniqueID()){
	printf("random evec %d Grid norm %g CPS norm %g (odd %g : odd f0 %g, odd f1 %g) (even %g : even f0 %g, even f1 %g) and eval %g\n",i,nrm,nrmcps,nrmoddcps,nrmoddf0cps,nrmoddf1cps,nrmevencps,nrmevenf0cps,nrmevenf1cps,eval[i]);
      }

    }
    singleprec_evecs = false;
  }
  
  //This version creates a FGrid lattice using CPS factory and uses it in the above
  void randomizeEvecs(const JobParams &jp, const LancArg &lanc_arg){
    auto lanczos_lat = createFgridLattice<typename GridPolicies::FgridGFclass>(jp);
    randomizeEvecs(lanc_arg, *lanczos_lat);
    delete lanczos_lat;
  }

  //Convert double to single precision eigenvectors
  void toSingle(){
    if(singleprec_evecs) return;
    
    typedef typename GridPolicies::GridFermionField GridFermionField;
    typedef typename GridPolicies::GridFermionFieldF GridFermionFieldF;

    int nev = evec.size();
    
    if(!UniqueID() && nev > 0){
      std::cout << "Double-precision 5D even-odd Grid info:\n";
      evec.back().Grid()->show_decomposition();

      std::cout << "Single-precision 5D even-odd Grid info:\n";
      FgridBase::getFrbGridF()->show_decomposition();
    }

    if(!UniqueID()) printf("Lanczos container holds %d eigenvectors\n", nev);

    for(int i=0;i<nev;i++){      
      GridFermionFieldF tmp_f(FgridBase::getFrbGridF());
# ifndef MEMTEST_MODE
      precisionChange(tmp_f, evec.back());
# endif
      evec.pop_back();
      evec_f.push_back(std::move(tmp_f));
    }
    //These are in reverse order!
    std::reverse(evec_f.begin(), evec_f.end());
    singleprec_evecs = true;
  }

  static void IOfilenames(std::string &info_file, std::string &eval_file, std::string &evec_file, const std::string &file_stub){
    info_file = file_stub + "_info.xml";
    eval_file = file_stub + "_evals.hdf5";
    evec_file = file_stub + "_evecs.scidac";
  }

  void writeParallel(const std::string &file_stub) const{
    if(evec.size() == 0 && evec_f.size() == 0) ERR.General("GridLanczosWrapper","writeParallel","No eigenvectors to write!\n");
    std::string info_file,eval_file,evec_file;
    IOfilenames(info_file,eval_file,evec_file,file_stub);
   
    if(!UniqueID()){
      Grid::XmlWriter WRx(info_file);
      int prec = singleprec_evecs ? 1 : 2;
      write(WRx,"precision", prec);
    }

    if(singleprec_evecs) writeEvecsEvals(evec_f,eval,evec_file,eval_file);
    else writeEvecsEvals(evec,eval,evec_file,eval_file);
  }

  void readParallel(const std::string &file_stub){
    { //clear all memory associated with existing evecs
      std::vector<typename GridPolicies::GridFermionField>().swap(evec);
      std::vector<typename GridPolicies::GridFermionFieldF>().swap(evec_f);
    }
    std::string info_file,eval_file,evec_file;
    IOfilenames(info_file,eval_file,evec_file,file_stub);

    Grid::XmlReader RDx(info_file);
    int prec = -1;
    read(RDx,"precision",prec);

    singleprec_evecs = prec == 1 ? true : false;

    if(singleprec_evecs){
      readEvecsEvals(evec_f,eval,evec_file,eval_file,FgridBase::getFrbGridF());
    }else{
      readEvecsEvals(evec,eval,evec_file,eval_file,FgridBase::getFrbGrid());
    }
  }

  

  
  void freeEvecs(){
    std::vector<typename GridPolicies::GridFermionField>().swap(evec); //evec.clear();
    std::vector<typename GridPolicies::GridFermionFieldF>().swap(evec_f);
  }

  ~GridLanczosWrapper(){
    freeEvecs();
  }
};

CPS_END_NAMESPACE


#endif
#endif
