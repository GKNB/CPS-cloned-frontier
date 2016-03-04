#include<config.h>
#include<util/lattice.h>

#ifndef INCLUDED_FGRID_H
#define INCLUDED_FGRID_H

#ifdef USE_GRID
#include<Grid.h>
// using namespace Grid; //Not ideal in a header
// using namespace Grid::QCD;


//typedef GparityDomainWallFermionD::FermionField GparityLatticeFermionD;
//typedef GparityDomainWallFermionF::FermionField GparityLatticeFermionF;
//typedef Grid::QCD::DomainWallFermionD::FermionField LatticeFermionD;
//typedef Grid::QCD::DomainWallFermionF::FermionField LatticeFermionF;

#define GRID_GPARITY

#ifdef GRID_GPARITY

typedef Grid::QCD::MobiusFermion<Grid::QCD::GparityWilsonImplD> GparityMobiusFermionD;

#define DIRAC GparityMobiusFermionD
#define MOB	,mob_b,mob_b-1.
//#define DIRAC GparityDomainWallFermionD
//#define MOB	
#define IMPL Grid::QCD::GparityWilsonImplD	
#define LATTICE_FERMION DIRAC ::FermionField
#define SITE_FERMION Grid::QCD::iGparitySpinColourVector<Grid::ComplexD>
#define PARAMS	,params
#define GP gp

#else

#define DIRAC Grid::QCD::MobiusFermionD
#define MOB	,mob_b,mob_b-1.
#define LATTICE_FERMION DIRAC ::FermionField
#define SITE_FERMION Grid::QCD::iSpinColourVector<Grid::ComplexD>
#define IMPL Grid::QCD::WilsonImplD	
#define PARAMS	
#define GP 

#endif

//typedef iGparitySpinColourVector<std::complex<double> > siteGrid;
//typedef GparityDomainWallFermionR::FermionField GparityLatticeFermion;

CPS_START_NAMESPACE

class FgridParams {
public:
  Float mobius_scale;
  FgridParams():mobius_scale(1.) {}
  ~FgridParams(){}
};

class Fgrid: public virtual Lattice, public FgridParams,
	     public virtual FwilsonTypes{

  using RealD = Grid::RealD;
  using RealF = Grid::RealF;
public:
  typedef enum EvenOdd {Even,Odd,All} EvenOdd;

private:
  const char *cname;
  const int  Nc = Grid::QCD::Nc;
  const int  Nd = Grid::QCD::Nd;
  const int  Ns = Grid::QCD::Ns;
  int n_gp;
  //	GparityDomainWallFermionD *gp_d;
  //	GparityDomainWallFermionF *gp_f;
  Grid::GridCartesian *UGrid;
  Grid::GridRedBlackCartesian *UrbGrid;
  Grid::GridCartesian *FGrid;
  Grid::GridRedBlackCartesian *FrbGrid;
  Grid::QCD::LatticeGaugeFieldD *Umu;
  int threads;
  std::vector< int > vol; // global volume
  std::vector< int > nodes;
  RealD mass; 
  int Ls;
  double mob_b;

  template <typename vobj,typename sobj>
  void ImpexFermion(Vector *cps_f, vobj &grid_f, int cps2grid, EvenOdd eo ){
    using namespace Grid;
    using namespace Grid::QCD;

    const char *fname="ImpexFermion()";
    unsigned long vol;
    GridBase *grid = grid_f._grid;
    unsigned long fourvol = GJP.VolNodeSites();
    int ncb =2 ;
    if (eo!=All) {fourvol /=2; ncb=1;}
    //		if ((eo==All) && ((grid_f._grid)->lSites())!= (vol = GJP.VolNodeSites()*GJP.SnodeSites()))
    if (((grid_f._grid)->lSites())!= (vol = fourvol*(2/ncb)*GJP.SnodeSites()))
      ERR.General(cname,fname,"numbers of grid(%d) and GJP(%d) does not match\n",grid->lSites(),vol);
    std::vector<int> grid_coor;
    //		 iGparitySpinColourVector<Grid::ComplexD> siteGrid;
    sobj siteGrid;
    for(int site=0;site<vol;site++){
      grid->CoorFromIndex(grid_coor,site,grid->_ldimensions);
      int pos[4];
      for(int i=0;i<4;i++) pos[i] =grid_coor[i+1];
      if ((ncb>1)||(pos[0]+pos[1]+pos[2]+pos[3])%2==eo)
	if(cps2grid){
	  //			VRB.Debug(cname,fname,"site=%d grid_coor=%d %d %d %d %d 4dindex = %d \n",site,grid_coor[0],grid_coor[1],grid_coor[2],grid_coor[3],grid_coor[4], (FsiteOffset(pos)/(2/ncb)));
	  //			printf("%d: %s:%s: site=%d grid_coor=%d %d %d %d %d 4dindex = %d \n", UniqueID(),cname,fname, site,grid_coor[0],grid_coor[1],grid_coor[2],grid_coor[3],grid_coor[4], (FsiteOffset(pos)/(2/ncb)) );
	  for(int gp=0;gp<n_gp;gp++)
	    for(int s=0;s<Ns;s++)
	      for(int i=0;i<Nc;i++){
		int index = (FsiteOffset(pos)/(2/ncb)) + fourvol*(gp+n_gp*grid_coor[0]);
		Float *cps = (Float *)cps_f;
		cps += 2*(i+Nc*(s+Ns*(index)));
		std::complex<double> elem(*cps,*(cps+1));
		//				if (norm(elem)>0.01) printf("%d %d %d %d: cps[%d][%d][%d][%d] = %g %g\n", GJP.NodeCoor(0),GJP.NodeCoor(1),GJP.NodeCoor(2),GJP.NodeCoor(3), site,gp,s,i,elem.real(),elem.imag());
		siteGrid(GP)(s)(i) = elem;
	      }
	  pokeLocalSite(siteGrid,grid_f,grid_coor);
	} else {
	  grid->CoorFromIndex(grid_coor,site,grid->_ldimensions);
	  peekLocalSite(siteGrid,grid_f,grid_coor);
	  for(int gp=0;gp<n_gp;gp++)
	    for(int s=0;s<Ns;s++)
	      for(int i=0;i<Nc;i++){
		std::complex<double> elem;
		elem = siteGrid(GP)(s)(i);
		//				if (norm(elem)>0.01) printf("%d %d %d %d %d: grid[%d][%d][%d][%d] = %g %g\n",
		//grid_coor[0], grid_coor[1], grid_coor[2], grid_coor[3], grid_coor[4],
		//site,gp,s,i,elem.real(),elem.imag());
		int index = (FsiteOffset(pos)/(2/ncb)) + fourvol*(gp+n_gp*grid_coor[0]);
		Float *cps = (Float *)cps_f;
		cps += 2*(i+Nc*(s+Ns*(index)));
		*cps=elem.real();
		*(cps+1)=elem.imag();
	      }
	}
    } 
  }

public:

  inline Grid::GridCartesian * getUGrid(){ return UGrid; }
  inline Grid::GridRedBlackCartesian * getUrbGrid(){ return UrbGrid; }

  inline Grid::GridCartesian * getFGrid(){ return FGrid; }
  inline Grid::GridRedBlackCartesian * getFrbGrid(){ return FrbGrid; }

  inline Grid::QCD::LatticeGaugeFieldD * getUmu(){ return Umu; }
 
  inline double get_mob_b(){ return mob_b; }

  Fgrid(FgridParams &params): cname("Fgrid"),vol(4,0),nodes(4,0){
    using namespace Grid;
    using namespace Grid::QCD;

    const char *fname("Fgrid()");
    if(!grid_initted) Grid_init(GJP.argc_p(),GJP.argv_p());
    *((FgridParams *) this) = params;
    VRB.Result(cname,fname,"mobius_scale=%g\n",mobius_scale);
    mob_b = 0.5*(mobius_scale+1.);
    VRB.Func(cname,fname);
    if(!GJP.Gparity()){
      //		ERR.General(cname,fname,"Only implemented for Grid with Gparity at the moment\n");
      n_gp=1;
    } else n_gp = 2;
    VRB.Result(cname,fname,"Grid initted\n");
    threads = GridThread::GetThreads();
    for(int i=0;i<4;i++) vol[i]= GJP.NodeSites(i)*GJP.Nodes(i);;
    for(int i=0;i<4;i++) nodes[i]= GJP.Nodes(i);
    VRB.Result(cname,fname,"vol nodes Nd=%d\n",Nd);
    for(int i=0;i<4;i++) 
      VRB.Result(cname,fname,"%d %d \n",vol[i],nodes[i]);
    UGrid = SpaceTimeGrid::makeFourDimGrid(vol,GridDefaultSimd(Nd,vComplex::Nsimd()),nodes);
    VRB.Result(cname,fname,"UGrid.lSites()=%d\n",UGrid->lSites());
    SetLs(GJP.SnodeSites());
    UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
    FGrid = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
    VRB.Result(cname,fname,"FGrid.lSites()=%d\n",FGrid->lSites());
    FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);
    Umu = new LatticeGaugeFieldD(UGrid);
    grid_initted=true;
    VRB.FuncEnd(cname,fname);
    BondCond();
  }
  virtual ~Fgrid(void){
    if(Umu) delete Umu;
    BondCond();
    //		Grid_finalize();
  }
  Float SetMass(Float _mass){mass = _mass;return mass;}
  int SetLs(int _Ls){Ls = _Ls;return Ls;}
  static bool grid_initted;
  void ImportFermion(Vector *cps_f, LATTICE_FERMION &grid_f,EvenOdd eo = All){
    ImpexFermion<LATTICE_FERMION, SITE_FERMION>(cps_f,grid_f,0,eo);
  }
  void ImportFermion(LATTICE_FERMION &grid_f,Vector *cps_f,EvenOdd eo = All){
    ImpexFermion<LATTICE_FERMION,SITE_FERMION>(cps_f,grid_f,1,eo);
  }

#if 0
  void ImportGauge( Matrix *mom){ ImportGauge( NULL, mom,1); }

  void ImportGauge( Grid::QCD::LatticeGaugeFieldD *grid_lat)
  { ImportGauge( grid_lat,NULL,1); }

#endif
  void ImportGauge( Grid::QCD::LatticeGaugeFieldD *grid_lat, Matrix *mom){
    ImpexGauge( grid_lat, mom, 1);
  }

  void ImportGauge( Matrix *mom, Grid::QCD::LatticeGaugeFieldD *grid_lat){
    ImpexGauge( grid_lat, mom, 0);
  }

  void ImportGauge() { ImpexGauge(NULL,NULL,1); }
		
  void ImpexGauge( Grid::QCD::LatticeGaugeFieldD *grid_lat, Matrix *mom, int cps2grid){

    Float *gauge = (Float *) mom;
    if(!mom)  gauge = (Float *)GaugeField();
    if (!grid_lat)  grid_lat = Umu;
    unsigned long vol;
    const char *fname="ImpexGauge()";
    Grid::GridBase *grid = grid_lat->_grid;
    if (grid_lat->_grid->lSites()!= (vol = GJP.VolNodeSites()))
      ERR.General(cname,fname,"numbers of grid(%d) and GJP(%d) does not match\n",grid_lat->_grid->lSites(),vol);
    std::vector<int> grid_coor;
    Grid::QCD::LorentzColourMatrixD siteGrid;
    for(int site=0;site<vol;site++)
      for(int mu=0;mu<4;mu++){
	if(cps2grid){
	  for(int i=0;i<Nc;i++)
	    for(int j=0;j<Nc;j++){
	      Float *cps = gauge +18*(site*4+mu)+6*j+2*i;
	      std::complex<double> elem(*cps,*(cps+1));
	      siteGrid(mu)()(j,i) = elem;
	      //				if (norm(elem)>0.01) printf("gauge[%d][%d][%d][%d] = %g %g\n",site,mu,i,j,elem.real(),elem.imag());
	    }
	  grid->CoorFromIndex(grid_coor,site,grid->_ldimensions);
	  pokeLocalSite(siteGrid,*grid_lat,grid_coor);
	} else {
	  grid->CoorFromIndex(grid_coor,site,grid->_ldimensions);
	  peekLocalSite(siteGrid,*grid_lat,grid_coor);
	  for(int i=0;i<Nc;i++)
	    for(int j=0;j<Nc;j++){
	      std::complex<double> elem;
	      elem  = siteGrid(mu)()(j,i);
	      Float *cps = gauge +18*(site*4+mu)+6*j+2*i;
	      *cps = elem.real();*(cps+1)=elem.imag();
	      //				if (norm(elem)>0.01) printf("gauge[%d][%d][%d][%d] = %g %g\n",site,mu,i,j,elem.real(),elem.imag());
	    }
	}
      } 
  }
  std::vector<int> SetTwist(){
    std::vector<int> twists(Nd,0);
    for(int i=0;i<3;i++){
      twists[i] = (GJP.Bc(i) == BND_CND_GPARITY) ? 1 : 0 ;
      if(twists[i]) VRB.Result(cname,"SetTwist()","gparity[%d]=1\n",i);
    }
    return twists;
  }

  void  SetParams( DIRAC ::ImplParams &params){
#ifdef GRID_GPARITY
    std::vector<int> twists = SetTwist();
    params.twists = twists;
#endif

  }



  void Fdslash(Vector *f_out, Vector *f_in, CgArg *cg_arg,
	       CnvFrmType cnv_frm, int dir_flag){
    const char *fname("Fdslash()");
#if 1
    mass = cg_arg->mass;
    VRB.Result(cname,fname,"mass=%0.14g\n",mass);
    RealD M5 = GJP.DwfHeight();
    ImportGauge();
    std::vector<int> twists = SetTwist();
	

    LATTICE_FERMION grid_in(FGrid),grid_out(FGrid);
    ImportFermion(grid_in, f_in);
    DIRAC ::ImplParams params;
    SetParams( params );
    //	GparityDomainWallFermionD GPDdwf(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,params);
    DIRAC Ddwf(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5 MOB PARAMS );
    Ddwf.M(grid_in,grid_out);
    ImportFermion(f_out, grid_out);
#endif
  }
  FclassType Fclass()const {
    return F_CLASS_GRID;
  }
  // It returns the type of fermion class
  
  //! Multiplication of a lattice spin-colour vector by gamma_5.
  //  void Gamma5(Vector *v_out, Vector *v_in, int num_sites);

  int FsiteOffsetChkb(const int *x) const
  {ERR.NotImplemented(cname,"FsiteOffsetChkb");}
  // Sets the offsets for the fermion fields on a 
  // checkerboard. The fermion field storage order
  // is not the canonical one but it is particular
  // to the Dwf fermion type. x[i] is the 
  // ith coordinate where i = {0,1,2,3} = {x,y,z,t}.
  
#if 0
  int FsiteOffset(const int *x) const
  {ERR.NotImplemented(cname,"FsiteOffset");}
  // Sets the offsets for the fermion fields on a 
  // checkerboard. The fermion field storage order
  // is the canonical one. X[I] is the
  // ith coordinate where i = {0,1,2,3} = {x,y,z,t}.
#endif
  
  int FsiteSize() const {
    int size= 24 * Ls;
    return size;
  }
  // Returns the number of fermion field 
  // components (including real/imaginary) on a
  // site of the 4-D lattice.
  
  int FchkbEvl() const {
    return 1;
  }
  // Returns 0 => If no checkerboard is used for the evolution
  //      or the CG that inverts the evolution matrix.
  
  int FmatEvlInv(Vector *f_out, Vector *f_in, 
		 CgArg *cg_arg, 
		 Float *true_res,
		 CnvFrmType cnv_frm = CNV_FRM_YES)
  {ERR.NotImplemented(cname,"FmatEvlInv");}
  // It calculates f_out where A * f_out = f_in and
  // A is the preconditioned fermion matrix that appears
  // in the HMC evolution (even/odd preconditioning 
  // of [Dirac^dag Dirac]). The inversion is done
  // with the conjugate gradient. cg_arg is the structure
  // that contains all the control parameters, f_in is the
  // fermion field source vector, f_out should be set to be
  // the initial guess and on return is the solution.
  // f_in and f_out are defined on a checkerboard.
  // If true_res !=0 the value of the true residual is returned
  // in true_res.
  // *true_res = |src - MatPcDagMatPc * sol| / |src|
  // The function returns the total number of CG iterations.
  int FmatEvlInv(Vector *f_out, Vector *f_in, 
		 CgArg *cg_arg, 
		 CnvFrmType cnv_frm = CNV_FRM_YES)
  {ERR.NotImplemented(cname,"FmatEvlInv");}
  
  int FmatEvlMInv(Vector **f_out, Vector *f_in, Float *shift, 
		  int Nshift, int isz, CgArg **cg_arg, 
		  CnvFrmType cnv_frm, MultiShiftSolveType type, Float *alpha,
		  Vector **f_out_d)
  {
    using namespace Grid;
    using namespace Grid::QCD;

    const char *fname("FmatEvlMInv()");
#if 1
    mass = cg_arg[0]->mass;
    VRB.Result(cname,fname,"mass=%0.14g\n",mass);
    RealD M5 = GJP.DwfHeight();
    ImportGauge();
    std::vector<int> twists = SetTwist();

    DIRAC ::ImplParams params;
    SetParams( params );
    DIRAC Ddwf(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5 MOB PARAMS );

    LATTICE_FERMION grid_in(FGrid),psi(FGrid);
    std::vector<LATTICE_FERMION> grid_rb_out(Nshift,FrbGrid);
    LATTICE_FERMION grid_rb_in(FrbGrid),grid_out(FGrid);

    //	Float stp_cnd = cg_arg->stop_rsd;
    //	stp_cnd = stp_cnd*stp_cnd*norm2(grid_in);

#if 0
    ConjugateGradient<LATTICE_FERMION> CG(stp_cnd,cg_arg->max_num_iter);
    SchurRedBlackDiagMooeeSolve<LATTICE_FERMION> SchurSolver(CG);
    SchurSolver(Ddwf,grid_in,grid_out);
#else 
    MultiShiftFunction shifts;
    shifts.order=Nshift;
    shifts.poles.resize(Nshift);
    shifts.tolerances.resize(Nshift);
    shifts.residues.resize(Nshift);
    for(int i=0;i<Nshift;i++){
      shifts.poles[i]  = shift[i];
      shifts.residues[i]  =alpha[i];
      shifts.tolerances[i]  =cg_arg[i]->stop_rsd;
    }
    ImportFermion(grid_in, f_in,Odd);

    pickCheckerboard(Odd,grid_rb_in,grid_in);
    SchurDifferentiableOperator< IMPL > MdagM(Ddwf);
    ConjugateGradientMultiShift<LATTICE_FERMION> 
      MSCG(cg_arg[0]->max_num_iter,shifts);
    MSCG(MdagM,grid_rb_in,grid_rb_out); 
    for(int i=0;i<Nshift;i++){
      setCheckerboard(grid_out,grid_rb_out[i]);
      ImportFermion(f_out[i], grid_out,Odd);
    }
	
#endif
#if 0
    if(1){
      LATTICE_FERMION temp(FGrid);
      Ddwf.M(grid_out,temp);
      temp = temp - grid_in;
      true_res = std::sqrt(norm2(temp)/norm2(grid_in));
      VRB.Result(cname,fname,"true_res=%g\n",true_res);
    }
    ImportFermion(f_out, grid_out,Odd);
#endif
#endif
  }
  
  void FminResExt(Vector *sol, Vector *source, Vector **sol_old, 
		  Vector **vm, int degree, CgArg *cg_arg, CnvFrmType cnv_frm)
  {ERR.NotImplemented(cname,"FminResExt");}
  
  int FmatInv(Vector *f_out, Vector *f_in, 
	      CgArg *cg_arg, 
	      Float *true_res,
	      CnvFrmType cnv_frm = CNV_FRM_YES,
	      PreserveType prs_f_in = PRESERVE_YES){
    using namespace Grid;
    using namespace Grid::QCD;

    const char *fname("FmatInv()");
#if 1
    mass = cg_arg->mass;
    VRB.Result(cname,fname,"mass=%0.14g\n",mass);
    RealD M5 = GJP.DwfHeight();
    ImportGauge();
    std::vector<int> twists = SetTwist();

    LATTICE_FERMION grid_in(FGrid),grid_out(FGrid);
    ImportFermion(grid_in, f_in);
    ImportFermion(grid_out, f_out);
    DIRAC ::ImplParams params;
    SetParams( params );
    DIRAC Ddwf(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5 MOB PARAMS );

    Float stp_cnd = cg_arg->stop_rsd;
    //	stp_cnd = stp_cnd*stp_cnd*norm2(grid_in);
#if 1

    ConjugateGradient<LATTICE_FERMION> CG(stp_cnd,cg_arg->max_num_iter);
    SchurRedBlackDiagMooeeSolve<LATTICE_FERMION> SchurSolver(CG);
    SchurSolver(Ddwf,grid_in,grid_out);
#else
    MdagMLinearOperator<DIRAC,LATTICE_FERMION> HermOp(Ddwf);
    ConjugateGradient<LATTICE_FERMION> CG(stp_cnd,cg_arg->max_num_iter);
    CG(HermOp,grid_in,grid_out);
#endif
    if(true_res){
      LATTICE_FERMION temp(FGrid);
      Ddwf.M(grid_out,temp);
      temp = temp - grid_in;
      *true_res = std::sqrt(norm2(temp)/norm2(grid_in));
      VRB.Result(cname,fname,"true_res=%g\n",*true_res);
    }
    ImportFermion(f_out, grid_out);
#endif
  }
  int FmatInv(Vector *f_out, Vector *f_in, 
	      CgArg *cg_arg, 
	      CnvFrmType cnv_frm = CNV_FRM_YES,
	      PreserveType prs_f_in = PRESERVE_YES)
  {ERR.NotImplemented(cname,"FmatInv");}
  
  void Ffour2five(Vector *five, Vector *four, int s_u, int s_l, int Ncb=2);
  //!< Transforms a 4-dimensional fermion field into a 5-dimensional field.
  /* The 5d field is zero */
  // The 5d field is zero
  // except for the upper two components (right chirality)
  // at s = s_u which are equal to the ones of the 4d field
  // and the lower two components (left chirality) 
  // at s_l, which are equal to the ones of the 4d field
  // For spread-out DWF s_u, s_l refer to the global
  // s coordinate i.e. their range is from 
  // 0 to [GJP.Snodes() * GJP.SnodeSites() - 1]
  
  void Ffive2four(Vector *four, Vector *five, int s_u, int s_l, int Ncb=2);
  //!< Transforms a 5-dimensional fermion field into a 4-dimensional field.
  //The 4d field has
  // the upper two components (right chirality) equal to the
  // ones of the 5d field at s = s_u and the lower two 
  // components (left chirality) equal to the
  // ones of the 5d field at s = s_l, where s is the 
  // coordinate in the 5th direction.
  // For spread-out DWF s_u, s_l refer to the global
  // s coordinate i.e. their range is from 
  // 0 to [GJP.Snodes() * GJP.SnodeSites() - 1]
  // The same 4D field is generarted in all s node slices.
  
  int FeigSolv(Vector **f_eigenv, Float *lambda,
	       Float *chirality, int *valid_eig,
	       Float **hsum,
	       EigArg *eig_arg, 
	       CnvFrmType cnv_frm = CNV_FRM_YES)
  {ERR.NotImplemented(cname,"FeigSolv");}
  
  Float SetPhi(Vector *phi, Vector *frm1, Vector *frm2,
	       Float mass, Float epsilon, DagType dag)
  {ERR.NotImplemented(cname,"SetPhi");}
  // It sets the pseudofermion field phi from frm1, frm2.

  Float SetPhi(Vector *phi, Vector *frm1, Vector *frm2,
	       Float mass, DagType dag)
  {ERR.NotImplemented(cname,"SetPhi");}
    
  void MatPc(Vector *out, Vector *in, Float mass, Float epsilon, DagType dag)
  {ERR.NotImplemented(cname,"MatPc");}

  void MatPc(Vector *out, Vector *in, Float mass, DagType dag)
  {ERR.NotImplemented(cname,"MatPc");}

  ForceArg EvolveMomFforce(Matrix *mom, Vector *frm,
                           Float mass, Float epsilon, Float step_size)
  {ERR.NotImplemented(cname,"EvolveMomFforce");}

  ForceArg EvolveMomFforce(Matrix *mom, Vector *frm,
                           Float mass, Float step_size){
    return EvolveMomFforce(mom,frm,mass,-12345,step_size);
  }
  // It evolves the canonical momentum mom by step_size
  // using the fermion force.

  ForceArg EvolveMomFforce(Matrix *mom, Vector *phi, Vector *eta,
                           Float mass, Float epsilon, Float step_size) {
    return EvolveMomFforceBase(mom, phi, eta, mass, epsilon, -step_size);
  }
  ForceArg EvolveMomFforce(Matrix *mom, Vector *phi, Vector *eta,
                           Float mass, Float step_size) {
    return EvolveMomFforceBase(mom, phi, eta, mass, -12345, -step_size);
  }

  // It evolves the canonical Momemtum mom:
  // mom += coef * (phi1^\dag e_i(M) \phi2 + \phi2^\dag e_i(M^\dag) \phi1)
  // note: this function does not exist in the base Lattice class.
  //CK: This function is not used, so I have not modified it for WilsonTM

  ForceArg EvolveMomFforceBase(Matrix *mom,
                               Vector *phi1,
                               Vector *phi2,
                               Float mass_, Float epsilon,
                               Float coef){
    using namespace Grid;
    using namespace Grid::QCD;

    const char *fname("EvolveMomFforceBase()");
    mass = mass_;
    VRB.Result(cname,fname,"mass=%0.14g\n",mass);
    RealD M5 = GJP.DwfHeight();

    ImportGauge();

    LatticeGaugeFieldD grid_mom(UGrid), dSdU(UGrid),force(UGrid);
    ImportGauge(&grid_mom, mom);

    //	std::vector<int> twists = SetTwist();
    DIRAC ::ImplParams params;
    SetParams( params );

    LATTICE_FERMION grid_phi(FGrid),grid_Y(FGrid);
    LATTICE_FERMION X(FrbGrid),Y(FrbGrid);
    ImportFermion(grid_phi, phi1,Odd);
    ImportFermion(grid_Y, phi2,Odd);
    pickCheckerboard(Odd,X,grid_phi);
    pickCheckerboard(Odd,Y,grid_Y);

    DIRAC DenOp(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5 MOB PARAMS );
    DIRAC NumOp(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,1.,M5 MOB PARAMS );

    // not really used. Just to fill invoke TwoFlavourEvenOddRatioPseudoFermionAction
    ConjugateGradient<LATTICE_FERMION> CG(1e-8,10000);
	
    TwoFlavourEvenOddRatioPseudoFermionAction< IMPL > quo(NumOp,DenOp,CG,CG);
	
#if 0
    quo.cps_deriv(*Umu,X,Y,dSdU);
#else
    {
      NumOp.ImportGauge(*Umu);
      DenOp.ImportGauge(*Umu);
	
      SchurDifferentiableOperator< IMPL > Mpc(DenOp);
      SchurDifferentiableOperator< IMPL > Vpc(NumOp);

      force=0.;
      Mpc.MpcDeriv(force,X,Y);  	dSdU=force;
#if 1
      Mpc.MpcDagDeriv(force,Y,X); dSdU=dSdU+force;
#endif

      //		LATTICE_FERMION X(FGrid),Y(FGrid);

    }
#endif
#if 1
    grid_mom +=2.*Ta(dSdU);
    ImportGauge(mom, &grid_mom);
#endif
  }
  // It evolves the canonical Momemtum mom:
  // mom += coef * (phi1^\dag e_i(M) \phi2 + \phi2^\dag e_i(M^\dag) \phi1)
  // note: this function does not exist in the base Lattice class.
  ForceArg EvolveMomFforceBase(Matrix *mom,
                               Vector *phi1,
                               Vector *phi2,
                               Float mass,
                               Float coef){
    return EvolveMomFforceBase(mom,phi1,phi2,mass,-12345,coef);
  }

  // It evolve the canonical momentum mom  by step_size
  // using the bosonic quotient force.
  
  ForceArg RHMC_EvolveMomFforce(Matrix *mom, Vector **sol, int degree,
				int isz, Float *alpha, Float mass, Float epsilon, Float dt,
				Vector **sol_d, ForceMeasure measure)
  {
    const char *fname("RHMC_EvolveMomFforce()");
#if 0
    //	mass = cg_arg[0]->mass;
    VRB.Result(cname,fname,"mass=%0.14g\n",mass);
    RealD M5 = GJP.DwfHeight();
    ImportGauge();
    std::vector<int> twists = SetTwist();
    LatticeGaugeFieldD mom_grid(UGrid);
	

    DIRAC ::ImplParams params;
    params.twists = twists;
    DIRAC Ddwf(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5 MOB PARAMS );

    LATTICE_FERMION grid_in(FGrid),psi(FGrid);
    std::vector<LATTICE_FERMION> grid_rb_out(Nshift,FrbGrid);
    LATTICE_FERMION grid_rb_in(FrbGrid),grid_out(FGrid);

    //	Float stp_cnd = cg_arg->stop_rsd;
    //	stp_cnd = stp_cnd*stp_cnd*norm2(grid_in);

#if 0
    ConjugateGradient<LATTICE_FERMION> CG(stp_cnd,cg_arg->max_num_iter);
    SchurRedBlackDiagMooeeSolve<LATTICE_FERMION> SchurSolver(CG);
    SchurSolver(Ddwf,grid_in,grid_out);
#else 
    MultiShiftFunction shifts;
    shifts.order=Nshift;
    shifts.poles.resize(Nshift);
    shifts.tolerances.resize(Nshift);
    shifts.residues.resize(Nshift);
    for(int i=0;i<Nshift;i++){
      shifts.poles[i]  = shift[i];
      shifts.residues[i]  =alpha[i];
      shifts.tolerances[i]  =cg_arg[i]->stop_rsd;
    }
    ImportFermion(grid_in, f_in,Odd);

    pickCheckerboard(Odd,grid_rb_in,grid_in);
    SchurDifferentiableOperator< IMPL > MdagM(Ddwf);
    ConjugateGradientMultiShift<LATTICE_FERMION> 
      MSCG(cg_arg[0]->max_num_iter,shifts);
    MSCG(MdagM,grid_rb_in,grid_rb_out); 
    for(int i=0;i<Nshift;i++){
      setCheckerboard(grid_out,grid_rb_out[i]);
      ImportFermion(f_out[i], grid_out,Odd);
    }
	
#endif
#if 0
    if(1){
      LATTICE_FERMION temp(FGrid);
      Ddwf.M(grid_out,temp);
      temp = temp - grid_in;
      true_res = std::sqrt(norm2(temp)/norm2(grid_in));
      VRB.Result(cname,fname,"true_res=%g\n",true_res);
    }
    ImportFermion(f_out, grid_out,Odd);
#endif
#endif
  }
  ForceArg RHMC_EvolveMomFforce(Matrix *mom, Vector **sol, int degree,
				int isz, Float *alpha, Float mass, Float dt,
				Vector **sol_d, ForceMeasure measure) {
    return RHMC_EvolveMomFforce(mom,sol,degree,isz,alpha,mass,-12345,dt,sol_d,measure);
  }
  
  Float FhamiltonNode( Vector *phi,  Vector *chi)
  {ERR.NotImplemented(cname,"FhamiltonNode");}
  // The fermion Hamiltonian of the node sublattice.
  // chi must be the solution of Cg with source phi.	       
  
#if 1
  // doing nothing for now
  // Convert fermion field f_field from -> to
  void Fconvert(Vector *f_field,
		StrOrdType to,
		StrOrdType from)
  {}
#endif
  
  Float BhamiltonNode(Vector *boson, Float mass, Float epsilon)
  {ERR.NotImplemented(cname,"BhamiltonNode");}
  Float BhamiltonNode(Vector *boson, Float mass)
  {ERR.NotImplemented(cname,"BhamiltonNode");}
  // The boson Hamiltonian of the node sublattice
  int SpinComponents() const {
    return 4;
  }

  int ExactFlavors() const {
    return 2;
  }
    
  //!< Method to ensure bosonic force works (does nothing for Wilson
  //!< theories.
  void BforceVector(Vector *in, CgArg *cg_arg)
  {ERR.NotImplemented(cname,"BforceVec");}

};
#if 1

//------------------------------------------------------------------
class GnoneFgrid 
  : public virtual Lattice,
    public virtual Gnone,
    public virtual Fgrid
{
private:
  const char *cname;    // Class name.

public:
  GnoneFgrid(FgridParams &params):cname("GnoneFgrid"),Fgrid(params) {
    const char *fname = "GnoneFgrid()";
    VRB.Func(cname,fname);
  }

  ~GnoneFgrid() {
    const char *fname = "~GnoneFgrid()";
    VRB.Func(cname,fname);
  }
    
};
#endif
CPS_END_NAMESPACE
#else  

#warning "Compiling without Grid"

//#error Does not compile without Grid for now. Needs fake implementations 
#endif //#ifdef USE_GRID


#endif
