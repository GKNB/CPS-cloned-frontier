#ifndef INCLUDED_FGRID_H
#define INCLUDED_FGRID_H
#include<stdlib.h>
#include<config.h>
#include<assert.h>
#ifdef USE_GRID
#include<Grid/Grid.h>
#include<util/lattice.h>
#include<util/time_cps.h>
#include<util/lattice/SchurRedBlackMixed.h>
#ifdef USE_BFM
#include<util/lattice/bfm_mixed_solver.h>
#endif
#include<util/multi_cg_controller.h>
#include<util/eigen_container.h>
#include<util/eigen_grid.h>
//#include<Grid/algorithms/iterative/SimpleLanczos.h>
//using namespace Grid;
//using namespace Grid::QCD;
#undef HAVE_HANDOPT

namespace Grid
{

  template < typename Float, class Field >
    class Guesser:public LinearFunction < Field >
  {
  public:
    int neig;
      std::vector < Float > &eval;
      std::vector < Field > &evec;
      Guesser (int n, std::vector < Float > &_eval,
	       std::vector < Field > &_evec)
    : neig (n), eval (_eval), evec (_evec)
    {
       std::cout << "Guesser::neig= "<<neig <<std::endl;
      if (neig>0){
      assert (eval.size () >= neig);
      assert (evec.size () >= neig);
    }
    }
    void operator  () (const Field & in, Field & out)
    {
      std::cout << GridLogMessage << "guesser() called " << std::endl;
      out = 0.;
      for (int i = 0; i < neig; i++) {
	Grid::ComplexD coef = innerProduct (evec[i], in);
	coef = coef / eval[i];
	if (cps::VRB.IsActivated(cps::VERBOSE_DEBUG_LEVEL))
          std::cout<<GridLogMessage <<"eval coef norm(evec) "<<i<<" : "<<eval[i]<<" "<<coef<<" "<<norm2(evec[i])<< std::endl;
	out += coef * evec[i];
      }
      std::
	cout << GridLogMessage << "norm(out)  : " << norm2 (out) << std::endl;
    }
  };

}

CPS_START_NAMESPACE class FgridParams
{
public:
  Float mobius_scale;
  Float mobius_bmc;
  std::vector <Grid::ComplexD> omega;	//ZMobius
  Float epsilon;		//WilsonTM
    FgridParams ():mobius_scale (1.), mobius_bmc (1.)
  {
  }
   ~FgridParams ()
  {
  }
  void setZmobius (cps::Complex * bs, int ls)
  {
    omega.clear ();
    for (int i = 0; i < ls; i++) {
      std::complex < double >temp = 1. / (2. * bs[i] - 1.);
      VRB.Result ("FgridParams", "setZmobius", "bs[%d]=%g %g, omega=%g %g\n",
		  i, bs[i].real (), bs[i].imag (), i, temp.real (),
		  temp.imag ());
      omega.push_back(Grid::ComplexD(temp.real(),temp.imag()));
    }
  }
};

class FgridBase:public virtual Lattice, public virtual FgridParams,
  public virtual FwilsonTypes
{

  using RealD = Grid::RealD;
  using RealF = Grid::RealF;
public:
  //  typedef enum EvenOdd
  //  { Even, Odd, All } EvenOdd;
  const char *cname;
  static bool grid_initted;
  static bool grid_layouts_initted;
protected:
  static const int Nc = Grid::Nc;
  static const int Nd = Grid::Nd;
  static const int Ns = Grid::Ns;
  int n_gp;
  static Grid::GridCartesian * UGridD;
  static Grid::GridCartesian * UGridF;
  static Grid::GridRedBlackCartesian * UrbGridD;
  static Grid::GridRedBlackCartesian * UrbGridF;
  static Grid::GridCartesian * FGridD;
  static Grid::GridCartesian * FGridF;
  static Grid::GridRedBlackCartesian * FrbGridF;
  static Grid::GridRedBlackCartesian * FrbGridD;
  Grid::LatticeGaugeFieldD * Umu;
  //      Grid::LatticeGaugeFieldF *Umu_f;
  int threads;
  static std::vector < int >vol;	// global volume
  static std::vector < int >nodes;
  RealD mass;
  RealD mob_b;			//Mobius
  RealD mob_c;			//Mobius
  RealD eps;			// WilsonTM 
  std::vector <Grid::ComplexD>omegas;	//ZMobius
  static int Ls;
  
  static void setGridInitted(const bool value = true){ grid_initted = value; }
  static void setGridLayoutsInitted(const bool value = true){ grid_layouts_initted = value; }
public:
  //Initialize Grid
  //Can be called manually a priori, if not it is called in FGrid constructor
  //Will use the argc,argv pointers provided. If these are null it will use those stored in GJP
  static void initializeGrid(int *argc = NULL, char ***argv = NULL);

  //Initialize the static Grid objects
  //Can be called manually a priori, if not it is called in FGrid constructor
  //We separate this from initializeGrid because initializeGrid can be called before GJP is initialized (eg in src/comms/qmp/sysfunc.C)
  static void initializeGridLayouts();

  //Check if Grid and the Grids have been initialized
  static bool getGridInitted(){ return grid_initted; }

  inline double get_mob_b () const{
    return mob_b;
  };
  inline double get_mob_c () const{
    return mob_c;
  };
  inline double get_mass () const{
    return mass;
  }  

  static Grid::GridCartesian * getFGrid () {
    return FGridD;
  }
  static Grid::GridCartesian * getFGridF () {
    return FGridF;
  }
  static Grid::GridRedBlackCartesian * getFrbGrid () {
    return FrbGridD;
  }
  static Grid::GridRedBlackCartesian * getFrbGridF () {
    return FrbGridF;
  }
  static Grid::GridCartesian * getUGrid () {
    return UGridD;
  }
  static Grid::GridCartesian * getUGridF () {
    return UGridF;
  }
  static Grid::GridRedBlackCartesian * getUrbGrid () {
    return UrbGridD;
  }
  static Grid::GridRedBlackCartesian * getUrbGridF () {
    return UrbGridF;
  }
  Grid::LatticeGaugeFieldD * getUmu () {
    return Umu;
  }
  //      Grid::LatticeGaugeFieldF *getUmu_f(){return Umu_f;}

  FgridBase (FgridParams & params);

  void ResetParams (FgridParams & params)
  {
    this->mobius_scale = params.mobius_scale;
    this->mobius_bmc = params.mobius_bmc;
    this->eps = params.epsilon;
    this->omegas = params.omega;
  }
  virtual ~ FgridBase (void)
  {
    if (Umu)
      delete Umu;
  }

  //Set Ls to new value (defaults to that of GJP)
  //This function regenerates the 5D Grids. However it does not delete the old ones as that would make
  //any Grid objects currently using those Grids no longer function
  static int SetLs (int _Ls);

  Float SetMass (Float _mass)
  {
    mass = _mass;
    return mass;
  }
  Float SetEpsilon (Float _epsilon)
  {
    eps = _epsilon;
    return eps;
  }


  void ImportGauge (Grid::LatticeGaugeFieldD * grid_lat, Matrix * mom)
  {
    ImpexGauge (grid_lat, NULL, mom, 1);
  }

  void ImportGauge (Matrix * mom, Grid::LatticeGaugeFieldD * grid_lat)
  {
    ImpexGauge (grid_lat, NULL, mom, 0);
  }

  void ImportGauge ()
  {
    ImpexGauge (Umu, NULL, NULL, 1);
  }

  void ImpexGauge (Grid::LatticeGaugeFieldD * grid_lat,
		   Grid::LatticeGaugeFieldF * grid_lat_f, Matrix * mom,
		   int cps2grid)
  {

    BondCond ();
    Float *gauge = (Float *) mom;
    if (!mom)
      gauge = (Float *) GaugeField ();
//              if (!grid_lat)  grid_lat = Umu;
//              if (!grid_lat_f && cps2grid )  grid_lat_f = Umu_f;
    unsigned long vol;
    const char *fname = "ImpexGauge()";
    Grid::GridBase * grid = grid_lat->Grid();
    if (grid_lat->Grid()->lSites () != (vol = GJP.VolNodeSites ()))
      ERR.General (cname, fname,
		   "numbers of grid(%d) and GJP(%d) does not match\n",
		   grid_lat->Grid()->lSites (), vol);
    Grid::Coordinate grid_coor;
    Grid::LorentzColourMatrixD siteGrid;
    Grid::LorentzColourMatrixF siteGrid_f;
    for (int site = 0; site < vol; site++)
      for (int mu = 0; mu < 4; mu++) {
	if (cps2grid) {
	  for (int i = 0; i < Nc; i++)
	    for (int j = 0; j < Nc; j++) {
	      Float *cps = gauge + 18 * (site * 4 + mu) + 6 * j + 2 * i;
	      std::complex < double >elem (*cps, *(cps + 1));
	      siteGrid (mu) ()(j, i) = elem;
	      siteGrid_f (mu) ()(j, i) = elem;
//                              if (norm(elem)>0.01) printf("gauge[%d][%d][%d][%d] = %g %g\n",site,mu,i,j,elem.real(),elem.imag());
	    }
	  Grid::Lexicographic::CoorFromIndex (grid_coor, site,
					      grid->_ldimensions);
	  pokeLocalSite (siteGrid, *grid_lat, grid_coor);
	  if (grid_lat_f)
	    pokeLocalSite (siteGrid_f, *grid_lat_f, grid_coor);
	} else {
	  Grid::Lexicographic::CoorFromIndex (grid_coor, site,
					      grid->_ldimensions);
	  peekLocalSite (siteGrid, *grid_lat, grid_coor);
	  for (int i = 0; i < Nc; i++)
	    for (int j = 0; j < Nc; j++) {
	      std::complex < double >elem;
	      elem = siteGrid (mu) ()(j, i);
	      Float *cps = gauge + 18 * (site * 4 + mu) + 6 * j + 2 * i;
	      *cps = elem.real ();
	      *(cps + 1) = elem.imag ();
//i	if (norm(elem)>0.01) printf("gauge[%d][%d][%d][%d] = %g %g\n",site,mu,i,j,elem.real(),elem.imag());
	    }
	}
      }
    Float *f_tmp = (Float *) gauge;
    VRB.Debug (cname, fname, "mom=(%g %g)(%g %g)(%g %g)\n",
	       *f_tmp, *(f_tmp + 1), *(f_tmp + 2),
	       *(f_tmp + 3), *(f_tmp + 4), *(f_tmp + 5));
    BondCond ();
  }
  std::vector < int >SetTwist ()
  {
    std::vector < int >twists (Nd, 0);
    for (int i = 0; i < 3; i++) {
      twists[i] = (GJP.Bc (i) == BND_CND_GPARITY) ? 1 : 0;
      if (twists[i])
	VRB.Debug (cname, "SetTwist()", "gparity[%d]=1\n", i);
    }
    return twists;
  }



  FclassType Fclass () const
  {
    return F_CLASS_GRID;
  }
  // It returns the type of fermion class

  //! Multiplication of a lattice spin-colour vector by gamma_5.
//  void Gamma5(Vector *v_out, Vector *v_in, int num_sites);


#if 1
  // doing nothing for now
  // Convert fermion field f_field from -> to
  void Fconvert (Vector * f_field, StrOrdType to, StrOrdType from)
  {
  }
#endif

  // The boson Hamiltonian of the node sublattice
  int SpinComponents () const
  {
    return 4;
  }

  int ExactFlavors () const
  {
    return 2;
  }
  virtual int FsiteSize () const = 0;

#if 1
  void Ffour2five (Vector * five, Vector * four, int s_u, int s_l, int Ncb = 2);
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

  void Ffive2four (Vector * four, Vector * five, int s_u, int s_l, int Ncb = 2);
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
// Match to Fbfm twisted wilson instead of FwilsonTM
#undef USE_F_CLASS_WILSON_TM
#undef TWOKAPPA
#define GRID_GPARITY
#define IF_FIVE_D
#undef IF_TM
#define FGRID FgridGparityMobius
#define CLASS_NAME F_CLASS_GRID_GPARITY_MOBIUS
#define DIRAC Grid::GparityMobiusFermionD
#define DIRAC_F Grid::GparityMobiusFermionF
#define MOB	,M5,mob_b,mob_b-1.
#define IMPL Grid::GparityWilsonImplD
#define IMPL_F Grid::GparityWilsonImplF
#define SITE_FERMION Grid::iGparitySpinColourVector<Grid::ComplexD>
#define SITE_FERMION_F Grid::iGparitySpinColourVector<Grid::ComplexF>
#define PARAMS	,params
#define GP gp
#include "fgrid.h.inc"
#undef GRID_GPARITY
#undef IF_FIVE_D
#undef IF_TM
#undef FGRID
#undef CLASS_NAME
#undef DIRAC
#undef DIRAC_F
#undef MOB
#undef SITE_FERMION
#undef SITE_FERMION_F
#undef IMPL
#undef PARAMS
#undef GP
#define GRID_GPARITY
#undef IF_FIVE_D
#define IF_TM
#define FGRID FgridGparityWilsonTM
#define CLASS_NAME F_CLASS_GRID_GPARITY_WILSON_TM
#define DIRAC Grid::GparityWilsonTMFermionD
#define DIRAC_F Grid::GparityWilsonTMFermionF
#define MOB  ,eps
#define IMPL Grid::GparityWilsonImplD
#define IMPL_F Grid::GparityWilsonImplF
#define SITE_FERMION Grid::iGparitySpinColourVector<Grid::ComplexD>
#define SITE_FERMION_F Grid::iGparitySpinColourVector<Grid::ComplexF>
#define PARAMS	,params
#define GP gp
#include "fgrid.h.inc"
#undef GRID_GPARITY
#undef IF_FIVE_D
#undef IF_TM
#undef FGRID
#undef CLASS_NAME
#undef DIRAC
#undef DIRAC_F
#undef MOB
#undef SITE_FERMION
#undef SITE_FERMION_F
#undef IMPL
#undef PARAMS
#undef GP
#undef GRID_GPARITY
#define IF_FIVE_D
#undef IF_TM
#define FGRID FgridMobius
#define CLASS_NAME F_CLASS_GRID_MOBIUS
#define DIRAC Grid::MobiusFermionD
#define DIRAC_F Grid::MobiusFermionF
#define MOB	,M5,mob_b,mob_b-1.
#define SITE_FERMION Grid::iSpinColourVector<Grid::ComplexD>
#define SITE_FERMION_F Grid::iSpinColourVector<Grid::ComplexF>
#define IMPL Grid::WilsonImplD
#define IMPL_F Grid::WilsonImplF
#define MOB_ASYM
#undef MOB_SYM2
#define PARAMS
#define GP
#undef TWOKAPPA
#include "fgrid.h.inc"
#undef GRID_GPARITY
#undef IF_FIVE_D
#undef IF_TM
#undef FGRID
#undef CLASS_NAME
#undef DIRAC
#undef DIRAC_F
#undef MOB
#undef SITE_FERMION
#undef SITE_FERMION_F
#undef IMPL
#undef PARAMS
#undef GP
#undef GRID_GPARITY
#undef MOB_ASYM
#define IF_FIVE_D
#undef IF_TM
#define FGRID FgridMobiusSYM2
#define CLASS_NAME F_CLASS_GRID_MOBIUS_SYM2
#define DIRAC Grid::MobiusFermionD
#define DIRAC_F Grid::MobiusFermionF
#define MOB	,M5,mob_b,mob_b-1.
#define SITE_FERMION Grid::iSpinColourVector<Grid::ComplexD>
#define SITE_FERMION_F Grid::iSpinColourVector<Grid::ComplexF>
#define IMPL Grid::WilsonImplD
#define IMPL_F Grid::WilsonImplF
#undef MOB_ASYM
#define MOB_SYM2
#define PARAMS
#define GP
#undef TWOKAPPA
#include "fgrid.h.inc"
#undef GRID_GPARITY
#undef IF_FIVE_D
#undef FGRID
#undef CLASS_NAME
#undef DIRAC
#undef DIRAC_F
#undef MOB
#undef SITE_FERMION
#undef SITE_FERMION_F
#undef IMPL
#undef PARAMS
#undef GP
#undef MOB_SYM2
#define IF_FIVE_D
#define GRID_ZMOB
#define FGRID FgridZmobius
#define CLASS_NAME F_CLASS_GRID_ZMOBIUS
#define DIRAC Grid::ZMobiusFermionD
#define DIRAC_F Grid::ZMobiusFermionF
#define MOB	,M5,omegas,1.,0.
#define SITE_FERMION Grid::iSpinColourVector<Grid::ComplexD>
#define SITE_FERMION_F Grid::iSpinColourVector<Grid::ComplexF>
#define IMPL Grid::ZWilsonImplD
#define IMPL_F Grid::ZWilsonImplF
#define PARAMS
#define GP
// Using TwoKappa only for zMobius for now, really SYM2
//#define TWOKAPPA
//#define GRID_MADWF
#include "fgrid.h.inc"
#undef GRID_GPARITY
#undef IF_FIVE_D
#undef GRID_ZMOB
#undef IF_TM
#undef FGRID
#undef CLASS_NAME
#undef DIRAC
#undef DIRAC_F
#undef MOB
#undef SITE_FERMION
#undef SITE_FERMION_F
#undef IMPL
#undef IMPL_F
#undef PARAMS
#undef GP
#undef TWOKAPPA
#undef GRID_MADWF
#undef IF_FIVE_D
#define IF_TM
#define FGRID FgridWilsonTM
#define CLASS_NAME F_CLASS_GRID_WILSON_TM
#define DIRAC Grid::WilsonTMFermionD
#define DIRAC_F Grid::WilsonTMFermionF
#define MOB  ,eps
#define IMPL Grid::WilsonImplD
#define IMPL_F Grid::WilsonImplF
#define SITE_FERMION Grid::iSpinColourVector<Grid::ComplexD>
#define SITE_FERMION_F Grid::iSpinColourVector<Grid::ComplexF>
#define PARAMS
#define GP
#include "fgrid.h.inc"
#undef GRID_GPARITY
#undef IF_FIVE_D
#undef IF_TM
#undef FGRID
#undef CLASS_NAME
#undef DIRAC
#undef DIRAC_F
#undef MOB
#undef SITE_FERMION
#undef SITE_FERMION_F
#undef IMPL
#undef PARAMS
#undef GP
#define FGRID FgridWilson
#define CLASS_NAME F_CLASS_GRID_WILSON
#define DIRAC Grid::WilsonFermionD
#define DIRAC_F Grid::WilsonFermionF
#define MOB
#define IMPL Grid::WilsonImplD
#define IMPL_F Grid::WilsonImplF
#define SITE_FERMION Grid::iSpinColourVector<Grid::ComplexD>
#define SITE_FERMION_F Grid::iSpinColourVector<Grid::ComplexF>
#define PARAMS
#define GP
#undef NONHERMSOLVE
#include "fgrid.h.inc"
#undef GRID_GPARITY
#undef IF_FIVE_D
#undef IF_TM
#undef FGRID
#undef CLASS_NAME
#undef DIRAC
#undef DIRAC_F
#undef MOB
#undef SITE_FERMION
#undef SITE_FERMION_F
#undef IMPL
#undef PARAMS
#undef GP
#undef NONHERMSOLVE
#define FGRID FgridWilsonClover
#define CLASS_NAME F_CLASS_GRID_WILSON_CLOVER
#define DIRAC Grid::WilsonCloverFermionD
#define DIRAC_F Grid::WilsonCloverFermionF
#define MOB  , csw, csw
#define IMPL Grid::WilsonImplD
#define IMPL_F Grid::WilsonImplF
#define SITE_FERMION Grid::iSpinColourVector<Grid::ComplexD>
#define SITE_FERMION_F Grid::iSpinColourVector<Grid::ComplexF>
#define PARAMS
#define GP
#undef NONHERMSOLVE
#include "fgrid.h.inc"
#undef GRID_GPARITY
#undef IF_FIVE_D
#undef IF_TM
#undef FGRID
#undef CLASS_NAME
#undef DIRAC
#undef DIRAC_F
#undef MOB
#undef SITE_FERMION
#undef SITE_FERMION_F
#undef IMPL
#undef PARAMS
#undef GP
#undef NONHERMSOLVE
#endif //#ifdef USE_GRID
#endif
