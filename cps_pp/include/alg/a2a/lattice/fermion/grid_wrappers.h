#ifndef _GRID_WRAPPERS_H_
#define _GRID_WRAPPERS_H_
#ifdef USE_GRID

#include "grid_lanczos.h"
#include "evec_io.h"
#include "evec_interface.h"
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

template<typename GridDirac, typename GridFermionFieldD, typename FgridGFclass>
void testEigenvectors(const EvecInterface<GridFermionFieldD> &evecs, const double mass, FgridGFclass &lattice, A2Apreconditioning precon_type = SchurOriginal){ 
  a2a_printf("Testing eigenvectors with mass %f\n",mass); 
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
  std::unique_ptr<Grid::SchurOperatorBase<GridFermionFieldD> > HermOp;
  if(precon_type == SchurOriginal) HermOp.reset(new Grid::SchurDiagMooeeOperator<GridDirac, GridFermionFieldD>(Ddwf));
  else if(precon_type == SchurDiagTwo) HermOp.reset(new Grid::SchurDiagTwoOperator<GridDirac, GridFermionFieldD>(Ddwf));
  else assert(0);
  
  conformable(FrbGrid, evecs.getEvecGridD());
  GridFermionFieldD tmp1(FrbGrid), tmp2(FrbGrid), tmp3(FrbGrid), evec(FrbGrid);

  int N = evecs.nEvecs();
  
  for(int i=0;i<N;i++){
    double eval = evecs.getEvecD(evec, i);

    HermOp->Mpc(evec, tmp1);
    HermOp->MpcDag(tmp1, tmp2); //tmp2 = M^dag M v

    tmp3 = eval * evec; //tmp3 = lambda v

    double nrm = sqrt(axpy_norm(tmp1, -1., tmp2, tmp3)); //tmp1 = tmp3 - tmp2
    
    a2a_printf("Idx %d Eval %g Resid %g #Evecs %d #Evals %d\n",i,eval,nrm,N,N);
  }
}

template<typename GridFermionField>
void randomizeEvecs(std::vector<GridFermionField> &evecs, std::vector<double> &evals){
  assert(evecs.size() == evals.size());
  int N = evecs.size();
  if(N==0) return;
  
  typedef typename GridCPSfieldFermionFlavorPolicyMap<GridFermionField>::value FlavorPolicy;

  CPSfermion5D<cps::ComplexD, FiveDpolicy<FlavorPolicy> > tmp;
  IncludeCBsite<5> oddsites(1, 0), evensites(0, 0);

#ifdef USE_C11_RNG
  CPS_RNG rng(1234);
  std::uniform_real_distribution<double> dist(0.1,10);
#endif

  for(int i=0;i<N;i++){
    evecs[i].Checkerboard() = Grid::Odd;
    tmp.setGaussianRandom();
    double nrmcps = tmp.norm2();
    double nrmoddcps = tmp.norm2(oddsites);  
    double nrmevencps = tmp.norm2(evensites);

    tmp.exportGridField(evecs[i]);
      
    double nrm = Grid::norm2(evecs[i]);
#ifdef USE_C11_RNG
    evals[i] = dist(rng);
#else 
    evals[i] = LRG.Lrand(10,0.1); //same on all nodes
#endif

    a2a_printf("random evec %d Grid norm %g CPS norm %g (odd %g) (even %g) and eval %g\n",i,nrm,nrmcps,nrmoddcps,nrmevencps,evals[i]);    
  }
}

template<typename _GridFermionFieldD, typename _GridFermionFieldF>
class EvecManager{ 
public:
  typedef _GridFermionFieldD GridFermionFieldD;
  typedef _GridFermionFieldF GridFermionFieldF;

  static void IOfilenames(std::string &info_file, std::string &eval_file, std::string &evec_file, const std::string &file_stub){
    info_file = file_stub + "_info.xml";
    eval_file = file_stub + "_evals.hdf5";
    evec_file = file_stub + "_evecs.scidac";
  }

  virtual void writeParallel(const std::string &file_stub) const = 0;
  virtual void readParallel(const std::string &file_stub) = 0;
    
  virtual void compute(const LancArg &lanc_arg, Lattice &lat, A2Apreconditioning precon_type = SchurOriginal) = 0;
  virtual void randomizeEvecs(const LancArg &lanc_arg, Lattice &lat) = 0;
  
  virtual std::unique_ptr<EvecInterfaceMixedPrec<GridFermionFieldD,GridFermionFieldF>> createInterface() const = 0;

  virtual void freeEvecs() = 0;
  
  virtual ~EvecManager(){}
};

//Create the evecs using a double-precision solver, but convert to single precision afterwards to save memory
template<typename GridPolicies>
class GridLanczosDoubleConvSingle: public EvecManager<typename GridPolicies::GridFermionField,typename GridPolicies::GridFermionFieldF>{
public:
  typedef typename GridPolicies::GridFermionField GridFermionFieldD;
  typedef typename GridPolicies::GridFermionFieldF GridFermionFieldF;
private:
  std::vector<Grid::RealD> eval; 
  std::vector<GridFermionFieldF> evec_f;
public:

  std::unique_ptr<EvecInterfaceMixedPrec<GridFermionFieldD,GridFermionFieldF>> createInterface() const override{
    return std::unique_ptr<EvecInterfaceMixedPrec<GridFermionFieldD,GridFermionFieldF>>(
			new EvecInterfaceSinglePrec<GridFermionFieldD,GridFermionFieldF>(evec_f,eval, FgridBase::getFrbGrid(), FgridBase::getFrbGridF()) );
  }
  
  void compute(const LancArg &lanc_arg, Lattice &latb, A2Apreconditioning precon_type = SchurOriginal) override{
    assert(lanc_arg.precon);
    typename GridPolicies::FgridGFclass &lat = dynamic_cast<typename GridPolicies::FgridGFclass &>(latb);
    {
      std::vector<GridFermionFieldF>().swap(evec_f);
    }
    std::vector<GridFermionFieldD> evec;
    LOGA2A << "GridLanczosDoubleConvSingle: computing double precision eigenvectors" << std::endl;
    gridLanczos<GridPolicies>(eval,evec,lanc_arg,lat, precon_type);

    int nev = evec.size();    
    //Test the evecs
    if(nev > 0){
      EvecInterfaceDoublePrec<GridFermionFieldD> ei(evec,eval,evec[0].Grid());
      testEigenvectors<typename GridPolicies::GridDirac>(ei,lanc_arg.mass,lat,precon_type);
    }

    //Convert to single precision
    LOGA2A << "GridLanczosDoubleConvSingle: converting evecs to single precision" << std::endl;
    Grid::precisionChangeWorkspace wk(lat.getFrbGridF(), lat.getFrbGrid());

    for(int i=0;i<nev;i++){      
      GridFermionFieldF tmp_f(lat.getFrbGridF());
      precisionChange(tmp_f, evec.back(),wk);
      evec.pop_back();
      evec_f.push_back(std::move(tmp_f));
    }
    //These are in reverse order!
    std::reverse(evec_f.begin(), evec_f.end());
    LOGA2A << "GridLanczosDoubleConvSingle: completed eigenvector calculation" << std::endl;
  }
  
  void randomizeEvecs(const LancArg &lanc_arg, Lattice &latb) override{
    typename GridPolicies::FgridGFclass &lat = dynamic_cast<typename GridPolicies::FgridGFclass &>(latb);
    evec_f.clear();
    evec_f.resize(lanc_arg.N_true_get, lat.getFrbGridF());
    eval.resize(lanc_arg.N_true_get);
    cps::randomizeEvecs(evec_f,eval);
  }
    
  void writeParallel(const std::string &file_stub) const override{
    if(eval.size() == 0 && evec_f.size() == 0) return;
    std::string info_file,eval_file,evec_file;
    this->IOfilenames(info_file,eval_file,evec_file,file_stub);
   
    if(!UniqueID()){
      Grid::XmlWriter WRx(info_file);
      write(WRx,"precision", 1);
    }
    writeEvecsEvals(evec_f,eval,evec_file,eval_file);
  }

  void readParallel(const std::string &file_stub) override{
    { //clear all memory associated with existing evecs
      std::vector<GridFermionFieldF>().swap(evec_f);
    }
    std::string info_file,eval_file,evec_file;
    this->IOfilenames(info_file,eval_file,evec_file,file_stub);

    Grid::XmlReader RDx(info_file);
    int prec = -1;
    read(RDx,"precision",prec);
    
    if(prec != 1) ERR.General("GridLanczosDoubleConvSingle","readParallel","Expect single precision eigenvectors");
    readEvecsEvals(evec_f,eval,evec_file,eval_file,FgridBase::getFrbGridF());
  }
 
  void freeEvecs() override{
    std::vector<GridFermionFieldF>().swap(evec_f);
  }
  
};





//Create the evecs using a double-precision solver, but convert to single precision afterwards to save memory
template<typename GridPolicies>
class GridXconjLanczosDoubleConvSingle: public EvecManager<typename GridPolicies::GridFermionField,typename GridPolicies::GridFermionFieldF>{
public:
  typedef typename GridPolicies::GridFermionField GridFermionFieldD;
  typedef typename GridPolicies::GridFermionFieldF GridFermionFieldF;
  typedef typename GridPolicies::GridXconjFermionField GridXconjFermionFieldD;
  typedef typename GridPolicies::GridXconjFermionFieldF GridXconjFermionFieldF;

private:
  std::vector<Grid::RealD> eval; 
  std::vector<GridXconjFermionFieldF> evec_f;
public:

  std::unique_ptr<EvecInterfaceMixedPrec<GridFermionFieldD,GridFermionFieldF>> createInterface() const override{
    return std::unique_ptr<EvecInterfaceMixedPrec<GridFermionFieldD,GridFermionFieldF>>(
											new EvecInterfaceXconjSinglePrec<GridFermionFieldD,GridXconjFermionFieldD,
											GridFermionFieldF,GridXconjFermionFieldF>(evec_f,eval, FgridBase::getFrbGrid(), FgridBase::getFrbGridF()) );
  }
  
  void compute(const LancArg &lanc_arg, Lattice &latb, A2Apreconditioning precon_type = SchurOriginal) override{
    assert(lanc_arg.precon);
    typename GridPolicies::FgridGFclass &lat = dynamic_cast<typename GridPolicies::FgridGFclass &>(latb);
    {
      std::vector<GridXconjFermionFieldF>().swap(evec_f);
    }
    std::vector<GridXconjFermionFieldD> evec;
    LOGA2A << "GridXconjLanczosDoubleConvSingle: computing double precision eigenvectors" << std::endl;
    gridLanczosXconj<GridPolicies>(eval,evec,lanc_arg,lat, precon_type);

    int nev = evec.size();    
    //Test the evecs
    if(nev > 0){
      EvecInterfaceXconjDoublePrec<GridFermionFieldD,GridXconjFermionFieldD> ei(evec,eval,evec[0].Grid());
      testEigenvectors<typename GridPolicies::GridDirac>(ei,lanc_arg.mass,lat,precon_type); //tests with 2-flavor Dirac operator :)
    }

    //Convert to single precision
    LOGA2A << "GridLanczosDoubleConvSingle: converting evecs to single precision" << std::endl;
    Grid::precisionChangeWorkspace wk(lat.getFrbGridF(), lat.getFrbGrid());

    for(int i=0;i<nev;i++){      
      GridXconjFermionFieldF tmp_f(lat.getFrbGridF());
      precisionChange(tmp_f, evec.back(),wk);
      evec.pop_back();
      evec_f.push_back(std::move(tmp_f));
    }
    //These are in reverse order!
    std::reverse(evec_f.begin(), evec_f.end());
    LOGA2A << "GridXconjLanczosDoubleConvSingle: completed eigenvector calculation" << std::endl;
  }
  
  void randomizeEvecs(const LancArg &lanc_arg, Lattice &latb) override{
    typename GridPolicies::FgridGFclass &lat = dynamic_cast<typename GridPolicies::FgridGFclass &>(latb);
    evec_f.clear();
    evec_f.resize(lanc_arg.N_true_get, lat.getFrbGridF());
    eval.resize(lanc_arg.N_true_get);
    cps::randomizeEvecs(evec_f,eval);
  }
    
  void writeParallel(const std::string &file_stub) const override{
    if(eval.size() == 0 && evec_f.size() == 0) return;
    std::string info_file,eval_file,evec_file;
    this->IOfilenames(info_file,eval_file,evec_file,file_stub);
   
    if(!UniqueID()){
      Grid::XmlWriter WRx(info_file);
      write(WRx,"precision", 1);
    }
    writeEvecsEvals(evec_f,eval,evec_file,eval_file);
  }

  void readParallel(const std::string &file_stub) override{
    { //clear all memory associated with existing evecs
      std::vector<GridXconjFermionFieldF>().swap(evec_f);
    }
    std::string info_file,eval_file,evec_file;
    this->IOfilenames(info_file,eval_file,evec_file,file_stub);

    Grid::XmlReader RDx(info_file);
    int prec = -1;
    read(RDx,"precision",prec);
    
    if(prec != 1) ERR.General("GridXconjLanczosDoubleConvSingle","readParallel","Expect single precision eigenvectors");
    readEvecsEvals(evec_f,eval,evec_file,eval_file,FgridBase::getFrbGridF());
  }
 
  void freeEvecs() override{
    std::vector<GridXconjFermionFieldF>().swap(evec_f);
  }

};



//Create the evecs using a double-precision solver, but convert to single precision afterwards to save memory
template<typename GridPolicies>
class GridXconjBlockLanczosDoubleConvSingle: public EvecManager<typename GridPolicies::GridFermionField,typename GridPolicies::GridFermionFieldF>{
public:
  typedef typename GridPolicies::GridFermionField GridFermionFieldD;
  typedef typename GridPolicies::GridFermionFieldF GridFermionFieldF;
  typedef typename GridPolicies::GridXconjFermionField GridXconjFermionFieldD;
  typedef typename GridPolicies::GridXconjFermionFieldF GridXconjFermionFieldF;

private:
  std::vector<Grid::RealD> eval; 
  std::vector<GridXconjFermionFieldF> evec_f;
  std::vector<int> split_grid_geom;
public:

  GridXconjBlockLanczosDoubleConvSingle(const std::vector<int> &split_grid_geom): split_grid_geom(split_grid_geom){
    if(split_grid_geom.size() != 4) ERR.General("GridXconjBlockLanczosDoubleConvSingle","constructor","Split grid geometry has wrong dimension!");
    for(int i=0;i<4;i++) if( GJP.Nodes(i) % split_grid_geom[i] != 0 ) ERR.General("GridXconjBlockLanczosDoubleConvSingle","constructor","Split grid geometry must exactly subdivide the lattice!");
  }

  std::unique_ptr<EvecInterfaceMixedPrec<GridFermionFieldD,GridFermionFieldF>> createInterface() const override{
    return std::unique_ptr<EvecInterfaceMixedPrec<GridFermionFieldD,GridFermionFieldF>>(
											new EvecInterfaceXconjSinglePrec<GridFermionFieldD,GridXconjFermionFieldD,
											GridFermionFieldF,GridXconjFermionFieldF>(evec_f,eval, FgridBase::getFrbGrid(), FgridBase::getFrbGridF()) );
  }
  
  void compute(const LancArg &lanc_arg, Lattice &latb, A2Apreconditioning precon_type = SchurOriginal) override{
    assert(lanc_arg.precon);
    typename GridPolicies::FgridGFclass &lat = dynamic_cast<typename GridPolicies::FgridGFclass &>(latb);
    {
      std::vector<GridXconjFermionFieldF>().swap(evec_f);
    }
    std::vector<GridXconjFermionFieldD> evec;
    LOGA2A << "GridXconjBlockLanczosDoubleConvSingle: computing double precision eigenvectors" << std::endl;
    gridBlockLanczosXconj<GridPolicies>(eval,evec,lanc_arg,lat,split_grid_geom,precon_type);

    int nev = evec.size();    
    //Test the evecs
    if(nev > 0){
      EvecInterfaceXconjDoublePrec<GridFermionFieldD,GridXconjFermionFieldD> ei(evec,eval,evec[0].Grid());
      testEigenvectors<typename GridPolicies::GridDirac>(ei,lanc_arg.mass,lat,precon_type); //tests with 2-flavor Dirac operator :)
    }

    //Convert to single precision
    LOGA2A << "GridXconjBlockLanczosDoubleConvSingle: converting evecs to single precision" << std::endl;
    Grid::precisionChangeWorkspace wk(lat.getFrbGridF(), lat.getFrbGrid());

    for(int i=0;i<nev;i++){      
      GridXconjFermionFieldF tmp_f(lat.getFrbGridF());
      precisionChange(tmp_f, evec.back(),wk);
      evec.pop_back();
      evec_f.push_back(std::move(tmp_f));
    }
    //These are in reverse order!
    std::reverse(evec_f.begin(), evec_f.end());
    LOGA2A << "GridXconjBlockLanczosDoubleConvSingle: completed eigenvector calculation" << std::endl;
  }
  
  void randomizeEvecs(const LancArg &lanc_arg, Lattice &latb) override{
    typename GridPolicies::FgridGFclass &lat = dynamic_cast<typename GridPolicies::FgridGFclass &>(latb);
    evec_f.clear();
    evec_f.resize(lanc_arg.N_true_get, lat.getFrbGridF());
    eval.resize(lanc_arg.N_true_get);
    cps::randomizeEvecs(evec_f,eval);
  }
    
  void writeParallel(const std::string &file_stub) const override{
    if(eval.size() == 0 && evec_f.size() == 0) return;
    std::string info_file,eval_file,evec_file;
    this->IOfilenames(info_file,eval_file,evec_file,file_stub);
   
    if(!UniqueID()){
      Grid::XmlWriter WRx(info_file);
      write(WRx,"precision", 1);
    }
    writeEvecsEvals(evec_f,eval,evec_file,eval_file);
  }

  void readParallel(const std::string &file_stub) override{
    { //clear all memory associated with existing evecs
      std::vector<GridXconjFermionFieldF>().swap(evec_f);
    }
    std::string info_file,eval_file,evec_file;
    this->IOfilenames(info_file,eval_file,evec_file,file_stub);

    Grid::XmlReader RDx(info_file);
    int prec = -1;
    read(RDx,"precision",prec);
    
    if(prec != 1) ERR.General("GridXconjBlockLanczosDoubleConvSingle","readParallel","Expect single precision eigenvectors");
    readEvecsEvals(evec_f,eval,evec_file,eval_file,FgridBase::getFrbGridF());
  }
 
  void freeEvecs() override{
    std::vector<GridXconjFermionFieldF>().swap(evec_f);
  }

};




//Create the evecs using a single-precision block solver
template<typename GridPolicies>
class GridXconjBlockLanczosSingle: public EvecManager<typename GridPolicies::GridFermionField,typename GridPolicies::GridFermionFieldF>{
public:
  typedef typename GridPolicies::GridFermionField GridFermionFieldD;
  typedef typename GridPolicies::GridFermionFieldF GridFermionFieldF;
  typedef typename GridPolicies::GridXconjFermionField GridXconjFermionFieldD;
  typedef typename GridPolicies::GridXconjFermionFieldF GridXconjFermionFieldF;

private:
  std::vector<Grid::RealD> eval; 
  std::vector<GridXconjFermionFieldF> evec_f;
  std::vector<int> split_grid_geom;
public:

  GridXconjBlockLanczosSingle(const std::vector<int> &split_grid_geom): split_grid_geom(split_grid_geom){
    if(split_grid_geom.size() != 4) ERR.General("GridXconjBlockLanczosSingle","constructor","Split grid geometry has wrong dimension!");
    for(int i=0;i<4;i++) if( GJP.Nodes(i) % split_grid_geom[i] != 0 ) ERR.General("GridXconjBlockLanczosSingle","constructor","Split grid geometry must exactly subdivide the lattice!");
  }

  std::unique_ptr<EvecInterfaceMixedPrec<GridFermionFieldD,GridFermionFieldF>> createInterface() const override{
    return std::unique_ptr<EvecInterfaceMixedPrec<GridFermionFieldD,GridFermionFieldF>>(
											new EvecInterfaceXconjSinglePrec<GridFermionFieldD,GridXconjFermionFieldD,
											GridFermionFieldF,GridXconjFermionFieldF>(evec_f,eval, FgridBase::getFrbGrid(), FgridBase::getFrbGridF()) );
  }
  
  void compute(const LancArg &lanc_arg, Lattice &latb, A2Apreconditioning precon_type = SchurOriginal) override{
    assert(lanc_arg.precon);
    typename GridPolicies::FgridGFclass &lat = dynamic_cast<typename GridPolicies::FgridGFclass &>(latb);
    {
      std::vector<GridXconjFermionFieldF>().swap(evec_f);
    }
    LOGA2A << "GridXconjBlockLanczosSingle: computing single precision eigenvectors" << std::endl;
    gridBlockLanczosXconjSingle<GridPolicies>(eval,evec_f,lanc_arg,lat,split_grid_geom,precon_type);

    int nev = evec_f.size();
    //Test the evecs
    if(nev > 0){
      EvecInterfaceXconjSinglePrec<GridFermionFieldD,GridXconjFermionFieldD,
				   GridFermionFieldF,GridXconjFermionFieldF> ei(evec_f,eval,lat.getFrbGrid(),lat.getFrbGridF()); //enable retrieval of double prec fields
      testEigenvectors<typename GridPolicies::GridDirac>(ei,lanc_arg.mass,lat,precon_type); //tests with 2-flavor Dirac operator, double precision :)
    }

    LOGA2A << "GridXconjBlockLanczosSingle: completed eigenvector calculation" << std::endl;
  }
  
  void randomizeEvecs(const LancArg &lanc_arg, Lattice &latb) override{
    typename GridPolicies::FgridGFclass &lat = dynamic_cast<typename GridPolicies::FgridGFclass &>(latb);
    evec_f.clear();
    evec_f.resize(lanc_arg.N_true_get, lat.getFrbGridF());
    eval.resize(lanc_arg.N_true_get);
    cps::randomizeEvecs(evec_f,eval);
  }
    
  void writeParallel(const std::string &file_stub) const override{
    if(eval.size() == 0 && evec_f.size() == 0) return;
    std::string info_file,eval_file,evec_file;
    this->IOfilenames(info_file,eval_file,evec_file,file_stub);
   
    if(!UniqueID()){
      Grid::XmlWriter WRx(info_file);
      write(WRx,"precision", 1);
    }
    writeEvecsEvals(evec_f,eval,evec_file,eval_file);
  }

  void readParallel(const std::string &file_stub) override{
    { //clear all memory associated with existing evecs
      std::vector<GridXconjFermionFieldF>().swap(evec_f);
    }
    std::string info_file,eval_file,evec_file;
    this->IOfilenames(info_file,eval_file,evec_file,file_stub);

    Grid::XmlReader RDx(info_file);
    int prec = -1;
    read(RDx,"precision",prec);
    
    if(prec != 1) ERR.General("GridXconjBlockLanczosDoubleConvSingle","readParallel","Expect single precision eigenvectors");
    readEvecsEvals(evec_f,eval,evec_file,eval_file,FgridBase::getFrbGridF());
  }
 
  void freeEvecs() override{
    std::vector<GridXconjFermionFieldF>().swap(evec_f);
  }

};




template<typename GridPolicies>
std::unique_ptr<EvecManager<typename GridPolicies::GridFermionField,typename GridPolicies::GridFermionFieldF> >
A2ALanczosFactory(const LanczosControls &args){
  std::vector<int> block_lanc_geom(args.block_lanczos_split_grid_geometry.block_lanczos_split_grid_geometry_len);
  for(int i=0;i<args.block_lanczos_split_grid_geometry.block_lanczos_split_grid_geometry_len;i++)
    block_lanc_geom[i] = args.block_lanczos_split_grid_geometry.block_lanczos_split_grid_geometry_val[i];

  std::unique_ptr<EvecManager<typename GridPolicies::GridFermionField,typename GridPolicies::GridFermionFieldF> > out;
  switch(args.lanczos_type){
  case A2AlanczosTypeDoubleConvSingle:
    out.reset(new GridLanczosDoubleConvSingle<GridPolicies>()); break;
  case A2AlanczosTypeXconjDoubleConvSingle: 
    out.reset(new GridXconjLanczosDoubleConvSingle<GridPolicies>()); break;
  case A2AlanczosTypeBlockXconjDoubleConvSingle:
    out.reset(new GridXconjBlockLanczosDoubleConvSingle<GridPolicies>(block_lanc_geom)); break;
  case A2AlanczosTypeBlockXconjSingle:
    out.reset(new GridXconjBlockLanczosSingle<GridPolicies>(block_lanc_geom)); break;
  default:
    assert(0);
  }
  return out;
}




CPS_END_NAMESPACE


#endif
#endif
