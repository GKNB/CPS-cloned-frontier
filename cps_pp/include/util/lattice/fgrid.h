#include<config.h>
#include<util/lattice.h>
#ifndef INCLUDED_FGRID_H
#define INCLUDED_FGRID_H

#ifdef USE_GRID
#include<Grid/Grid.h>
//using namespace Grid;
//using namespace Grid::QCD;





CPS_START_NAMESPACE

class FgridParams {
	public:
	Float mobius_scale;
	Float epsilon;//WilsonTM
	FgridParams():mobius_scale(1.){}
	~FgridParams(){}
};

class FgridBase: public virtual Lattice, public virtual FgridParams,
	public virtual FwilsonTypes{

   using RealD = Grid::RealD;
   using RealF = Grid::RealF;
   public:
	typedef enum EvenOdd {Even,Odd,All} EvenOdd;
	const char *cname;

   protected:
   	const int  Nc = Grid::QCD::Nc;
   	const int  Nd = Grid::QCD::Nd;
   	const int  Ns = Grid::QCD::Ns;
	int n_gp;
	Grid::GridCartesian *UGrid;
	Grid::GridRedBlackCartesian *UrbGrid;
	Grid::GridCartesian *FGrid;
	Grid::GridRedBlackCartesian *FrbGrid;
	Grid::QCD::LatticeGaugeFieldD *Umu;
	int threads;
	std::vector< int > vol; // global volume
	std::vector< int > nodes;
	RealD mass; 
	RealD eps; // WilsonTM 
	int Ls;
	double mob_b; //Mobius
	static bool grid_initted;

public: 
        double get_mob_b(){return mob_b;};

	Grid::GridCartesian *getFGrid(){return FGrid;}
	Grid::GridRedBlackCartesian *getFrbGrid(){return FrbGrid;}
	Grid::GridCartesian *getUGrid(){return UGrid;}
	Grid::GridRedBlackCartesian *getUrbGrid(){return UrbGrid;}
	Grid::QCD::LatticeGaugeFieldD *getUmu(){return Umu;}
	FgridBase(FgridParams &params): cname("FgridBase"),vol(4,0),nodes(4,0),mass(1.), Ls(1){
//,epsilon(0.),
		const char *fname("FgridBase()");
		if(!grid_initted) Grid::Grid_init(GJP.argc_p(),GJP.argv_p());
		*((FgridParams *) this) = params;
		eps = params.epsilon;
		
//		VRB.Debug(cname,fname,"mobius_scale=%g\n",mobius_scale);
		mob_b = 0.5*(mobius_scale+1.);
		VRB.Func(cname,fname);
		if(!GJP.Gparity()){
//		ERR.General(cname,fname,"Only implemented for Grid with Gparity at the moment\n");
			n_gp=1;
		} else n_gp = 2;
		VRB.Result(cname,fname,"Grid initted\n");
		threads = Grid::GridThread::GetThreads();
		for(int i=0;i<4;i++) vol[i]= GJP.NodeSites(i)*GJP.Nodes(i);;
		for(int i=0;i<4;i++) nodes[i]= GJP.Nodes(i);
		VRB.Result(cname,fname,"vol nodes Nd=%d Grid::vComplexD::Nsimd()=%d\n",Nd,Grid::vComplexD::Nsimd());
		for(int i=0;i<4;i++) 
		VRB.Result(cname,fname,"%d %d \n",vol[i],nodes[i]);
		UGrid = Grid::QCD::SpaceTimeGrid::makeFourDimGrid(vol,Grid::GridDefaultSimd(Nd,Grid::vComplexD::Nsimd()),nodes);
		VRB.Result(cname,fname,"UGrid.lSites()=%d\n",UGrid->lSites());
		SetLs(GJP.SnodeSites());
		UrbGrid = Grid::QCD::SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
		FGrid = Grid::QCD::SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
		VRB.Result(cname,fname,"FGrid.lSites()=%d\n",FGrid->lSites());
		FrbGrid = Grid::QCD::SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);
		for(int i=0;i<5;i++)
		  VRB.Result(cname,fname,"FGrid.gdimensions[%d]=%d\n",i,FGrid->_gdimensions[i]);
		Umu = new Grid::QCD::LatticeGaugeFieldD(UGrid);
		grid_initted=true;
		VRB.FuncEnd(cname,fname);
//		BondCond();

		bool fail = false;
		for(int t=0;t<GJP.Tnodes();t++)
		  for(int z=0;z<GJP.Znodes();z++)
		    for(int y=0;y<GJP.Ynodes();y++)
		      for(int x=0;x<GJP.Xnodes();x++){
			std::vector<int> node {x,y,z,t};
			int cps_rank = QMP_get_node_number_from(&node[0]); //is a MPI_COMM_WORLD rank
			int grid_rank = UGrid->RankFromProcessorCoor(node); //is an MPI_Cart rank. However this MPI_Cart is drawn from MPI_COMM_WORLD and so the rank mapping to physical processors should be the same. However check below
			int fail = 0;
			if(UGrid->_processor == grid_rank){
			  int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
			  if(world_rank != UGrid->_processor) fail = 1;
			}
			QMP_status_t ierr = QMP_sum_int(&fail);
			if(ierr != QMP_SUCCESS)
			  ERR.General("FgridBase","FgridBase","Rank check sum failed\n");
			if(fail != 0)
			  ERR.General("FgridBase","FgridBase","Grid MPI_Cart rank does not align with MPI_COMM_WORLD rank\n");
						
			if(cps_rank != grid_rank){
			  if(!UniqueID()){ std::cout << "Error in FgridBase constructor: node (" << node[0] << "," << node[1] << "," << node[2] << "," << node[3] << ") maps to different MPI ranks for Grid " << grid_rank << " and CPS " << cps_rank << std::endl;
			    std::cout.flush();
			  }
			  fail = true;
			}
		      }
		if(fail)
		  exit(0);	
	}
	virtual ~FgridBase(void){
		if(Umu) delete Umu;
//		BondCond();
//		Grid_finalize();
	}
	int SetLs(int _Ls){Ls = _Ls;return Ls;}
	Float SetMass(Float _mass){mass = _mass;return mass;}
	Float SetEpsilon(Float _epsilon){eps = _epsilon;return eps;}


	void ImportGauge( Grid::QCD::LatticeGaugeFieldD *grid_lat, Matrix *mom){
		ImpexGauge( grid_lat, mom, 1);
	}

	void ImportGauge( Matrix *mom, Grid::QCD::LatticeGaugeFieldD *grid_lat){
		ImpexGauge( grid_lat, mom, 0);
	}

	void ImportGauge() { ImpexGauge(NULL,NULL,1); }
		
	void ImpexGauge( Grid::QCD::LatticeGaugeFieldD *grid_lat, Matrix *mom, int cps2grid){

	  BondCond(); //Apply - sign to boundary for APRD directions. Does nothing for GPBC dirs.
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
			Grid::Lexicographic::CoorFromIndex(grid_coor,site,grid->_ldimensions);
			pokeLocalSite(siteGrid,*grid_lat,grid_coor);
			} else {
			Grid::Lexicographic::CoorFromIndex(grid_coor,site,grid->_ldimensions);
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
		BondCond(); //Unapply - sign for APRD directions.
		if(cps2grid) std::cout << "Imported gauge field:\n" << *grid_lat << std::endl;
			    
	}
	std::vector<int> SetTwist(){
	std::vector<int> twists(Nd,0);
	for(int i=0;i<3;i++){
		 twists[i] = (GJP.Bc(i) == BND_CND_GPARITY) ? 1 : 0 ;
		if(twists[i]) VRB.Result(cname,"SetTwist()","gparity[%d]=1\n",i);
	}
	return twists;
}



  FclassType Fclass()const {
    return F_CLASS_GRID;
  }
  // It returns the type of fermion class
  
  //! Multiplication of a lattice spin-colour vector by gamma_5.
//  void Gamma5(Vector *v_out, Vector *v_in, int num_sites);

#if 0
  int FsiteOffsetChkb(const int *x) const
  {ERR.NotImplemented(cname,"FsiteOffsetChkb");}
  // Sets the offsets for the fermion fields on a 
  // checkerboard. The fermion field storage order
  // is not the canonical one but it is particular
  // to the Dwf fermion type. x[i] is the 
  // ith coordinate where i = {0,1,2,3} = {x,y,z,t}.
//  int FsiteOffset(const int *x) const;
#endif
  
  
#if 1
  // doing nothing for now
  // Convert fermion field f_field from -> to
  void Fconvert(Vector *f_field,
		StrOrdType to,
		StrOrdType from)
  {}
#endif
  
  // The boson Hamiltonian of the node sublattice
  int SpinComponents() const {
    return 4;
  }

  int ExactFlavors() const {
    return 2;
  }
  virtual int FsiteSize() const = 0;

#if 1
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
#endif
    
};
CPS_END_NAMESPACE

#define PASTER(x,y) x ## y
#define EVALUATOR(x,y) PASTER(x,y)
#define GFCLASS(class) EVALUATOR(class, FGRID )
#define XSTR(s) STR(s)
#define STR(s) #s

//#ifdef GRID_GPARITY
#define GRID_GPARITY
#define IF_FIVE_D 
#define FGRID FgridGparityMobius
#define CLASS_NAME F_CLASS_GRID_GPARITY_MOBIUS
#define DIRAC Grid::QCD::GparityMobiusFermionD
#define MOB	,M5,mob_b,mob_b-1.
#define IMPL Grid::QCD::GparityWilsonImplD	
#define LATTICE_FERMION DIRAC ::FermionField
#define SITE_FERMION Grid::QCD::iGparitySpinColourVector<Grid::ComplexD>
#define PARAMS	,params
#define GP gp
#include "fgrid.h.inc"

#define GRID_GPARITY
#undef IF_FIVE_D 
#define FGRID FgridGparityWilsonTM
#define CLASS_NAME F_CLASS_GRID_GPARITY_WILSON_TM
#define DIRAC Grid::QCD::GparityWilsonTMFermionD
#define MOB  ,eps
#define IMPL Grid::QCD::GparityWilsonImplD	
#define LATTICE_FERMION DIRAC ::FermionField
#define SITE_FERMION Grid::QCD::iGparitySpinColourVector<Grid::ComplexD>
#define PARAMS	,params
#define GP gp
#include "fgrid.h.inc"

//#else
#undef GRID_GPARITY
#define IF_FIVE_D 
#define FGRID FgridMobius
#define CLASS_NAME F_CLASS_GRID_MOBIUS
#define DIRAC Grid::QCD::MobiusFermionD
#define MOB	,M5,mob_b,mob_b-1.
#define LATTICE_FERMION DIRAC ::FermionField
#define SITE_FERMION Grid::QCD::iSpinColourVector<Grid::ComplexD>
#define IMPL Grid::QCD::WilsonImplD	
#define PARAMS	
#define GP 
#include "fgrid.h.inc"

#undef GRID_GPARITY
#undef IF_FIVE_D
#define FGRID FgridWilsonTM
#define CLASS_NAME F_CLASS_GRID_WILSON_TM
#define DIRAC Grid::QCD::WilsonTMFermionD
#define MOB  ,eps
#define IMPL Grid::QCD::WilsonImplD	
#define LATTICE_FERMION DIRAC ::FermionField
#define SITE_FERMION Grid::QCD::iSpinColourVector<Grid::ComplexD>
#define PARAMS	
#define GP 
#include "fgrid.h.inc"

#undef FGRID
#undef CLASS_NAME
#undef DIRAC
#undef MOB
#undef LATTICE_FERMION
#undef SITE_FERMION
#undef IMPL
#undef PARAMS	
#undef GP 

#else  
//#error Does not compile without Grid for now. Needs fake implementations 
#warning "FGrid without Grid!"
#endif //#ifdef USE_GRID


#endif
