#ifndef _GRID_WRAPPERS_H_
#define _GRID_WRAPPERS_H_
#ifdef USE_GRID

#include <alg/a2a/grid_lanczos.h>

CPS_START_NAMESPACE

//Generates and stores evecs and evals
template<typename GridPolicies>
struct GridLanczosWrapper{
  std::vector<Grid::RealD> eval; 
  std::vector<typename GridPolicies::GridFermionField> evec;
  std::vector<typename GridPolicies::GridFermionFieldF> evec_f;
  double mass;
  double resid;
  bool singleprec_evecs;
  
  //Keep a copy of the double-precision Grid because we delete the Lattice instance after compute
  Grid::GridRedBlackCartesian *FrbGrid_d; 

  //For precision change
  Grid::GridCartesian *UGrid_f;
  Grid::GridRedBlackCartesian *UrbGrid_f;
  Grid::GridCartesian *FGrid_f;
  Grid::GridRedBlackCartesian *FrbGrid_f;
  
  GridLanczosWrapper(): UGrid_f(NULL), UrbGrid_f(NULL), FGrid_f(NULL), FrbGrid_f(NULL), FrbGrid_d(NULL), singleprec_evecs(false){
  }

  void setupSPgrids(){
    //Make single precision Grids
    int Ls = GJP.Snodes()*GJP.SnodeSites();
    std::vector<int> nodes(4);
    std::vector<int> vol(4);
    for(int i=0;i<4;i++){
      vol[i]= GJP.NodeSites(i)*GJP.Nodes(i);;
      nodes[i]= GJP.Nodes(i);
    }
    Grid::Coordinate simd_layout = Grid::GridDefaultSimd(Grid::Nd,Grid::vComplexF::Nsimd());
    if(!UniqueID()) printf("Created single-prec Grids: nodes (%d,%d,%d,%d) vol (%d,%d,%d,%d) and SIMD layout (%d,%d,%d,%d)\n",nodes[0],nodes[1],nodes[2],nodes[3],vol[0],vol[1],vol[2],vol[3],simd_layout[0],simd_layout[1],simd_layout[2],simd_layout[3]);
    
    UGrid_f = Grid::SpaceTimeGrid::makeFourDimGrid(vol,simd_layout,nodes);
    UrbGrid_f = Grid::SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid_f);
    FGrid_f = Grid::SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid_f);
    FrbGrid_f = Grid::SpaceTimeGrid::makeFiveDimRedBlackGrid(GJP.SnodeSites()*GJP.Snodes(),UGrid_f);     
  }

  //Copy from one lattice to another under the programmer's assurance that the Grid instances are equivalent (not Grid::conformable which checks pointer equality)
  void copyVecNoGridCheck(typename GridPolicies::GridFermionField &out, const typename GridPolicies::GridFermionField &in){
    out.Checkerboard() = in.Checkerboard();
    auto iview = in.View();
    auto oview = out.View();
    typedef typename GridPolicies::GridFermionField::vector_object vobj;
    accelerator_for(ss,iview.size(),vobj::Nsimd(),{
      coalescedWrite(oview[ss],iview(ss));
    });
  }

  //If we delete the lattice class we need to move the evecs to a different Grid instance that is not deleted
  void moveDPevecsToIndependentGrid(typename GridPolicies::FgridGFclass &lat){
    if(FrbGrid_d == NULL) FrbGrid_d = new Grid::GridRedBlackCartesian(lat.getFrbGrid());
    typename GridPolicies::GridFermionField tmp(FrbGrid_d);

    for(int i=0;i<evec.size();i++){
      //Painful double-copy. Could be avoided if it were possible to change the Grid pointer under the hood. Used to be possible but is no longer!
      copyVecNoGridCheck(tmp, evec[i]);
      assert(tmp.Grid() == FrbGrid_d);
      evec[i].reset(FrbGrid_d);
      evec[i] = tmp;
      assert(evec[i].Grid() == FrbGrid_d);
    }
  }
  
  void compute(const LancArg &lanc_arg, typename GridPolicies::FgridGFclass &lat){
    mass = lanc_arg.mass;
    resid = lanc_arg.stop_rsd;
    
# ifdef A2A_LANCZOS_SINGLE
    setupSPgrids();
    gridSinglePrecLanczos<GridPolicies>(eval,evec_f,lanc_arg,lat,UGrid_f,UrbGrid_f,FGrid_f,FrbGrid_f);
    singleprec_evecs = true;

    evec_f.resize(lanc_arg.N_true_get, FrbGrid_f); //in case the Lanczos implementation does not explicitly remove the extra evecs used for the restart
    eval.resize(lanc_arg.N_true_get);
    
# else    
    gridLanczos<GridPolicies>(eval,evec,lanc_arg,lat);
    singleprec_evecs = false;

    evec.resize(lanc_arg.N_true_get, lat.getFrbGrid());
    eval.resize(lanc_arg.N_true_get);
   
#  ifndef MEMTEST_MODE
    test_eigenvectors(evec,eval,lanc_arg.mass,lat);
#  endif

# endif    
  }

  static void test_eigenvectors(const std::vector<typename GridPolicies::GridFermionField> &evec, const std::vector<Grid::RealD> &eval, const double mass, typename GridPolicies::FgridGFclass &lattice){
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
    Grid::SchurDiagMooeeOperator<GridDirac, GridFermionField> HermOp(Ddwf);
  
    GridFermionField tmp1(FrbGrid);
    GridFermionField tmp2(FrbGrid);
    GridFermionField tmp3(FrbGrid);
  
    for(int i=0;i<evec.size();i++){
      HermOp.Mpc(evec[i], tmp1);
      HermOp.MpcDag(tmp1, tmp2); //tmp2 = M^dag M v

      tmp3 = eval[i] * evec[i]; //tmp3 = lambda v

      double nrm = sqrt(axpy_norm(tmp1, -1., tmp2, tmp3)); //tmp1 = tmp3 - tmp2
    
      if(!UniqueID()) printf("Idx %d Eval %g Resid %g #Evecs %d #Evals %d\n",i,eval[i],nrm,evec.size(),eval.size());
    }

  }

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
      eval[i] = LRG.Lrand(10,0.1); //same on all nodes
      if(!UniqueID()){
	printf("random evec %d Grid norm %g CPS norm %g (odd %g : odd f0 %g, odd f1 %g) (even %g : even f0 %g, even f1 %g) and eval %g\n",i,nrm,nrmcps,nrmoddcps,nrmoddf0cps,nrmoddf1cps,nrmevencps,nrmevenf0cps,nrmevenf1cps,eval[i]);
      }

    }
    singleprec_evecs = false;
  }
  
  void toSingle(){
    if(singleprec_evecs) return;
    
    typedef typename GridPolicies::GridFermionField GridFermionField;
    typedef typename GridPolicies::GridFermionFieldF GridFermionFieldF;

    if(FrbGrid_f == NULL) setupSPgrids();

    int nev = evec.size();
    
    if(!UniqueID() && nev > 0){
      std::cout << "Double-precision 5D even-odd Grid info:\n";
      evec.back().Grid()->show_decomposition();

      std::cout << "Single-precision 5D even-odd Grid info:\n";
      FrbGrid_f->show_decomposition();
    }

    if(!UniqueID()) printf("Lanczos container holds %d eigenvectors\n", nev);

    for(int i=0;i<nev;i++){      
      GridFermionFieldF tmp_f(FrbGrid_f);
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

  void writeParallel(const std::string &file_stub, FP_FORMAT fileformat = FP_AUTOMATIC) const{
    if(evec.size() == 0 && evec_f.size() == 0) ERR.General("GridLanczosWrapper","writeParallel","No eigenvectors to write!\n");
    
    bool single_prec = singleprec_evecs;
    int n_evec = single_prec ? evec_f.size() : evec.size();

    Grid::GridBase* grd = single_prec ? evec_f[0].Grid() : evec[0].Grid();
    bool is_rb(false); for(int i=0;i<5;i++) if(grd->CheckerBoarded(i)){ is_rb = true; break; }
    assert(is_rb);
    Grid::GridRedBlackCartesian* grd_rb = dynamic_cast<Grid::GridRedBlackCartesian*>(grd);
    assert(grd_rb->_checker_dim_mask[0] == 0); //4d checkerboarding

    std::ostringstream filename; filename << file_stub << "." << UniqueID();
    std::ofstream file(filename.str().c_str());
    assert(!file.fail());
    file.exceptions ( std::ofstream::failbit | std::ofstream::badbit );

    arrayIO<Grid::RealD> evalio(fileformat);
    
    file << "BEGIN_HEADER\n";
    file << "HDR_VERSION = 1\n";
    file << "N_EVECS = " << n_evec << "\n";
    file << "PRECISION = " << (single_prec ? 1 : 2) << "\n";
    file << "MASS = " << mass << "\n";
    file << "RESID = " << resid << "\n";
    file << "END_HEADER\n";
    file << "BEGIN_EVALS\n";

    file << "DATA_FORMAT = " << evalio.getFileFormatString() << '\n';
    file << "CHECKSUM = " << evalio.checksum(eval.data(),eval.size()) << '\n';
    evalio.write(file,eval.data(),eval.size());
    
    file << "END_EVALS\n";
    file << "BEGIN_EVECS\n";

    CPSfermion5Dcb4Dodd<cps::ComplexF> c_odd_f;
    CPSfermion5Dcb4Dodd<cps::ComplexD> c_odd_d;
    
    for(int i=0;i<n_evec;i++){
      if(single_prec){
	c_odd_f.importGridField(evec_f[i]);
	c_odd_f.writeParallel(file,fileformat);
      }else{
	c_odd_d.importGridField(evec[i]);
	c_odd_d.writeParallel(file,fileformat);
      }
    }    

    file << "END_EVECS\n";
    file.close();
  }

  void readParallel(const std::string &file_stub, typename GridPolicies::FgridGFclass &lat){
    { //clear all memory associated with existing evecs
      std::vector<typename GridPolicies::GridFermionField>().swap(evec);
      std::vector<typename GridPolicies::GridFermionFieldF>().swap(evec_f);
    }

    std::ostringstream os; os << file_stub << "." << UniqueID();
    std::ifstream file(os.str().c_str(),std::ifstream::in);
    file.exceptions ( std::ofstream::failbit | std::ofstream::badbit );
    assert(!file.fail());

    std::string str;
    getline(file,str); assert(str == "BEGIN_HEADER");
    getline(file,str); assert(str == "HDR_VERSION = 1");
  
    int read_nvecs;
    getline(file,str); assert( sscanf(str.c_str(),"N_EVECS = %d",&read_nvecs) == 1 );

    int read_precision;
    getline(file,str); assert( sscanf(str.c_str(),"PRECISION = %d",&read_precision) == 1 );

    getline(file,str); assert( sscanf(str.c_str(),"MASS = %lf",&mass) == 1 );
    getline(file,str); assert( sscanf(str.c_str(),"RESID = %lf",&resid) == 1 );
    
    bool single_prec = (read_precision == 1);
    getline(file,str); assert(str == "END_HEADER");
    getline(file,str); assert(str == "BEGIN_EVALS");

    char dformatbuf[256];
    getline(file,str); assert( sscanf(str.c_str(),"DATA_FORMAT = %s",dformatbuf) == 1 );
    
    unsigned int read_checksum;
    getline(file,str); assert( sscanf(str.c_str(),"CHECKSUM = %u",&read_checksum) == 1 );

    eval.resize(read_nvecs);
    arrayIO<Grid::RealD> evalio(dformatbuf);
    evalio.read(file,eval.data(),eval.size());
    
    //assert( evalio.checksum(eval.data(),eval.size()) == read_checksum );
    unsigned int actual_checksum = evalio.checksum(eval.data(),eval.size());
    if(actual_checksum != read_checksum) ERR.General("GridLanczosWrapper","readParallel","Eval array checksum error, expected %u got %u\n",read_checksum, actual_checksum);    

    getline(file,str); assert(str == "END_EVALS");
    getline(file,str); assert(str == "BEGIN_EVECS");

    if(single_prec){
      CPSfermion5Dcb4Dodd<cps::ComplexF> c_odd_f;
      if(FrbGrid_f == NULL) setupSPgrids();
      evec_f.resize(read_nvecs, FrbGrid_f);
      for(int i=0;i<read_nvecs;i++){
	c_odd_f.readParallel(file);
	evec_f[i].Checkerboard() = Grid::Odd;
	c_odd_f.exportGridField(evec_f[i]);
      }
      singleprec_evecs = true;
    }else{
      CPSfermion5Dcb4Dodd<cps::ComplexD> c_odd_d;
      Grid::GridRedBlackCartesian *FrbGrid = lat.getFrbGrid();
      evec.resize(read_nvecs, FrbGrid);
      for(int i=0;i<read_nvecs;i++){
	c_odd_d.readParallel(file);
	evec[i].Checkerboard() = Grid::Odd;
	c_odd_d.exportGridField(evec[i]);
      }
      singleprec_evecs = false;
    }

    getline(file,str); assert(str == "END_EVECS");

    file.close();
  }

  

  
  void freeEvecs(){
    std::vector<typename GridPolicies::GridFermionField>().swap(evec); //evec.clear();
    std::vector<typename GridPolicies::GridFermionFieldF>().swap(evec_f);
#define CKDEL(A) if(A != NULL){ delete A; A=NULL; }
    CKDEL(UGrid_f);
    CKDEL(UrbGrid_f);
    CKDEL(FGrid_f);
    CKDEL(FrbGrid_f);
    CKDEL(FrbGrid_d);
#undef CKDEL
  }

  ~GridLanczosWrapper(){
    freeEvecs();
  }
};

CPS_END_NAMESPACE


#endif
#endif
