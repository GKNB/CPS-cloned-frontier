#pragma once

//Functionality for X-conjugate fields

#include<config.h>

#if defined(USE_GRID) && defined(USE_GRID_A2A)
#include<Grid/Grid.h>

CPS_START_NAMESPACE

inline Grid::Gamma Xmatrix(){
  using namespace Grid;
  static Gamma C = Gamma(Gamma::Algebra::MinusGammaY) * Gamma(Gamma::Algebra::GammaT);
  static Gamma g5 = Gamma(Gamma::Algebra::Gamma5);
  static Gamma X = C*g5;
  return X;
}

//From the matrices extract the column with the given flavor, spin and color index
template<typename GridFermionField2f, typename GridPropagatorField2f>
void extractColumn(GridFermionField2f &into, const GridPropagatorField2f &from, const int fs, const int ss, const int cs){
  using namespace Grid;
  {
    size_t Nsimd = GridFermionField2f::vector_type::Nsimd();
    autoView( from_v, from, AcceleratorRead);
    autoView( into_v, into, AcceleratorWrite);
    accelerator_for( i, from_v.size(), Nsimd, {
	auto site_matrix = from_v(i);
	auto site_into = into_v(i);
	for(int fr=0;fr<Ngp;fr++)
	  for(int sr=0;sr<Ns;sr++)
	    for(int cr=0;cr<Nc;cr++)
	      site_into(fr)(sr)(cr) = site_matrix(fr,fs)(sr,ss)(cr,cs);
	coalescedWrite(into_v[i], site_into);
      });
  }
}
template<typename GridFermionField2f, typename GridPropagatorField2f>
void insertColumn(GridPropagatorField2f &into, const GridFermionField2f &from, const int fs, const int ss, const int cs){
  using namespace Grid;
  size_t Nsimd = GridFermionField2f::vector_type::Nsimd();
  autoView( from_v, from, AcceleratorRead);
  autoView( into_v, into, AcceleratorWrite);
  accelerator_for( i, from_v.size(), Nsimd, {
      auto site_spinor = from_v(i);
      auto site_into = into_v(i);
      for(int fr=0;fr<Ngp;fr++)
	for(int sr=0;sr<Ns;sr++)
	  for(int cr=0;cr<Nc;cr++)
	    site_into(fr,fs)(sr,ss)(cr,cs) = site_spinor(fr)(sr)(cr);
      coalescedWrite(into_v[i], site_into);
    });
}


template<typename GridFermionField2f>
void multU(GridFermionField2f &into, const GridFermionField2f &from){
  using namespace Grid;
  GridFermionField2f Xfrom = Xmatrix() * from;
  autoView(from_v,from,AcceleratorRead);  
  autoView(Xfrom_v,Xfrom,AcceleratorRead);
  autoView(into_v,into,AcceleratorWrite);

  accelerator_for(x, from.Grid()->oSites(), GridFermionField2f::vector_object::Nsimd(), {
      auto fs = from_v(x);
      auto Xfs = Xfrom_v(x);
      decltype(fs) out;
      RealD nrm = 1./sqrt(2.);
      ComplexD I(0,1);

      //U = 1/sqrt(2) |  -X    i  |
      //              |  -1    iX |
      for(int s=0;s<Ns;s++)
	for(int c=0;c<Nc;c++){
	  out(0)(s)(c) = -nrm * Xfs(0)(s)(c) + nrm * I * fs(1)(s)(c);
	  out(1)(s)(c) = -nrm * fs(0)(s)(c) + nrm * I * Xfs(1)(s)(c);
	}
      coalescedWrite(into_v[x],out);
    });
}

template<typename GridFermionField2f>
void multUdag(GridFermionField2f &into, const GridFermionField2f &from){
  using namespace Grid;
  GridFermionField2f Xfrom = Xmatrix() * from;
  autoView(from_v,from,AcceleratorRead);  
  autoView(Xfrom_v,Xfrom,AcceleratorRead);
  autoView(into_v,into,AcceleratorWrite);

  accelerator_for(x, from.Grid()->oSites(), GridFermionField2f::vector_object::Nsimd(), {
      auto fs = from_v(x);
      auto Xfs = Xfrom_v(x);
      decltype(fs) out;
      RealD nrm = 1./sqrt(2.);
      ComplexD I(0,1);

      //U^dag = 1/sqrt(2) |  X    -1  |
      //                  |  -i    iX |
      for(int s=0;s<Ns;s++)
	for(int c=0;c<Nc;c++){
	  out(0)(s)(c) = nrm * Xfs(0)(s)(c) - nrm * fs(1)(s)(c);
	  out(1)(s)(c) = -nrm * I * fs(0)(s)(c) + nrm * I * Xfs(1)(s)(c);
	}
      coalescedWrite(into_v[x],out);
    });
}





template<typename GridPropagatorField2f>
void multUleft(GridPropagatorField2f &into, const GridPropagatorField2f &from){
  using namespace Grid;
  autoView(from_v,from,AcceleratorRead);  
  autoView(into_v,into,AcceleratorWrite);

  accelerator_for(x, from.Grid()->oSites(), GridPropagatorField2f::vector_object::Nsimd(), {
      auto fs = from_v(x);
      decltype(fs) out;
      RealD nrm = 1./sqrt(2.);
      ComplexD I(0,1);
      Gamma C = Gamma(Gamma::Algebra::MinusGammaY) * Gamma(Gamma::Algebra::GammaT);
      Gamma g5 = Gamma(Gamma::Algebra::Gamma5);
      Gamma X = C*g5;
      auto Xfs = X * fs;

      //U = 1/sqrt(2) |  -X    i  |
      //              |  -1    iX |
      for(int fc=0;fc<Ngp;fc++){
	for(int sc=0;sc<Ns;sc++){
	  for(int cc=0;cc<Nc;cc++){

	    for(int sr=0;sr<Ns;sr++)
	      for(int cr=0;cr<Nc;cr++){	  
		out(0,fc)(sr,sc)(cr,cc) = -nrm * Xfs(0,fc)(sr,sc)(cr,cc) + nrm * I * fs(1,fc)(sr,sc)(cr,cc);
		out(1,fc)(sr,sc)(cr,cc) = -nrm * fs(0,fc)(sr,sc)(cr,cc) + nrm * I * Xfs(1,fc)(sr,sc)(cr,cc);
	      }
	  }
	}
      }
      coalescedWrite(into_v[x],out);
    });
}

template<typename GridPropagatorField2f>
void multUdagLeft(GridPropagatorField2f &into, const GridPropagatorField2f &from){
  using namespace Grid;
  autoView(from_v,from,AcceleratorRead);  
  autoView(into_v,into,AcceleratorWrite);

  accelerator_for(x, from.Grid()->oSites(), GridPropagatorField2f::vector_object::Nsimd(), {
      auto fs = from_v(x);
      decltype(fs) out;
      RealD nrm = 1./sqrt(2.);
      ComplexD I(0,1);
      Gamma C = Gamma(Gamma::Algebra::MinusGammaY) * Gamma(Gamma::Algebra::GammaT);
      Gamma g5 = Gamma(Gamma::Algebra::Gamma5);
      Gamma X = C*g5;
      auto Xfs = X * fs;

      //U^dag = 1/sqrt(2) |  X    -1  |
      //                  |  -i    iX |
      for(int fc=0;fc<Ngp;fc++){
	for(int sc=0;sc<Ns;sc++){
	  for(int cc=0;cc<Nc;cc++){

	    for(int sr=0;sr<Ns;sr++)
	      for(int cr=0;cr<Nc;cr++){	  
		out(0,fc)(sr,sc)(cr,cc) = nrm * Xfs(0,fc)(sr,sc)(cr,cc) - nrm * fs(1,fc)(sr,sc)(cr,cc);
		out(1,fc)(sr,sc)(cr,cc) = -nrm * I * fs(0,fc)(sr,sc)(cr,cc) + nrm * I * Xfs(1,fc)(sr,sc)(cr,cc);
	      }
	  }
	}
      }
      coalescedWrite(into_v[x],out);
    });
}


template<typename GridPropagatorField2f>
void multUright(GridPropagatorField2f &into, const GridPropagatorField2f &from){
  using namespace Grid;
  autoView(from_v,from,AcceleratorRead);  
  autoView(into_v,into,AcceleratorWrite);

  accelerator_for(x, from.Grid()->oSites(), GridPropagatorField2f::vector_object::Nsimd(), {
      auto fs = from_v(x);
      decltype(fs) out;
      RealD nrm = 1./sqrt(2.);
      ComplexD I(0,1);
      Gamma C = Gamma(Gamma::Algebra::MinusGammaY) * Gamma(Gamma::Algebra::GammaT);
      Gamma g5 = Gamma(Gamma::Algebra::Gamma5);
      Gamma X = C*g5;
      auto fsX = fs * X;

      //U = 1/sqrt(2) |  -X    i  |
      //              |  -1    iX |
      for(int fr=0;fr<Ngp;fr++){
	for(int sr=0;sr<Ns;sr++){
	  for(int cr=0;cr<Nc;cr++){

	    for(int sc=0;sc<Ns;sc++)
	      for(int cc=0;cc<Nc;cc++){	  
		out(fr,0)(sr,sc)(cr,cc) = -nrm * fsX(fr,0)(sr,sc)(cr,cc) - nrm * fs(fr,1)(sr,sc)(cr,cc);
		out(fr,1)(sr,sc)(cr,cc) = nrm * I * fs(fr,0)(sr,sc)(cr,cc) + nrm * I * fsX(fr,1)(sr,sc)(cr,cc);
	      }
	  }
	}
      }
      coalescedWrite(into_v[x],out);
    });
}


template<typename GridPropagatorField2f>
void multUdagRight(GridPropagatorField2f &into, const GridPropagatorField2f &from){
  using namespace Grid;
  autoView(from_v,from,AcceleratorRead);  
  autoView(into_v,into,AcceleratorWrite);

  accelerator_for(x, from.Grid()->oSites(), GridPropagatorField2f::vector_object::Nsimd(), {
      auto fs = from_v(x);
      decltype(fs) out;
      RealD nrm = 1./sqrt(2.);
      ComplexD I(0,1);
      Gamma C = Gamma(Gamma::Algebra::MinusGammaY) * Gamma(Gamma::Algebra::GammaT);
      Gamma g5 = Gamma(Gamma::Algebra::Gamma5);
      Gamma X = C*g5;
      auto fsX = fs * X;

      //U^dag = 1/sqrt(2) |  X    -1  |
      //                  |  -i    iX |
      for(int fr=0;fr<Ngp;fr++){
	for(int sr=0;sr<Ns;sr++){
	  for(int cr=0;cr<Nc;cr++){

	    for(int sc=0;sc<Ns;sc++)
	      for(int cc=0;cc<Nc;cc++){	  
		out(fr,0)(sr,sc)(cr,cc) = nrm * fsX(fr,0)(sr,sc)(cr,cc) - nrm * I * fs(fr,1)(sr,sc)(cr,cc);
		out(fr,1)(sr,sc)(cr,cc) = -nrm * fs(fr,0)(sr,sc)(cr,cc) + nrm * I * fsX(fr,1)(sr,sc)(cr,cc);
	      }
	  }
	}
      }
      coalescedWrite(into_v[x],out);
    });
}


template<typename TwoFlavorField, typename OneFlavorField>
void XconjugateBoost(TwoFlavorField &into, const OneFlavorField &from){  
  using namespace Grid;
  conformable(into.Grid(),from.Grid());
  PokeIndex<GparityFlavourIndex>(into, from, 0);
  OneFlavorField tmp(from.Grid());
  tmp = -(Xmatrix()*conjugate(from));
  PokeIndex<GparityFlavourIndex>(into, tmp, 1);
  into.Checkerboard() = from.Checkerboard();
}

template<typename TwoFlavorField>
bool XconjugateCheck(const TwoFlavorField &v, const double tol = 1e-10, bool verbose=true){
  decltype( Grid::PeekIndex<GparityFlavourIndex>(v,0) ) tmp(v.Grid());
  tmp = -(Xmatrix()*conjugate(Grid::PeekIndex<GparityFlavourIndex>(v,0))) - Grid::PeekIndex<GparityFlavourIndex>(v,1);
  double n = norm2(tmp);
  if(n > tol){
    LOGA2A << "Failed Xconj check, got " << n << " (expect 0)" << std::endl;
    return false;
  }
  return true;
} 

//This test ensures the X-conjugate action is working
//TODO: find out why --dslash-unroll implementation is failing!
template<typename A2Apolicies>
void testXconjAction(typename A2Apolicies::FgridGFclass &lattice){
  using namespace Grid;

  typedef typename A2Apolicies::GridXconjFermionField FermionField1f;
  typedef typename A2Apolicies::GridFermionField FermionField2f;

  typedef typename A2Apolicies::FgridFclass FgridFclass;

  typedef typename A2Apolicies::GridDiracXconj GridDiracXconj;
  typedef typename A2Apolicies::GridDirac GridDiracGP;

  LOGA2A << "Testing X-conj action" << std::endl;
  if(!lattice.getGridFullyInitted()) ERR.General("","gridLanczosXconj","Grid/Grids are not initialized!");
  
  Grid::GridCartesian *UGrid = lattice.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = lattice.getUrbGrid();
  Grid::GridCartesian *FGrid = lattice.getFGrid();
  Grid::GridRedBlackCartesian *FrbGrid = lattice.getFrbGrid();

  {
    //Independent test (other than Grids)
    LOGA2A << "Doing independent test (other than Grids)" << std::endl;
    double bpc=2.0;
    double bmc=1.0;
    double b=0.5*(bpc+bmc);
    double c=bpc-b;
    double M5=1.8;
    double mass = 0.01;

    LatticeGaugeFieldD Umu(UGrid);
    GridParallelRNG RNG4(UGrid);  
    RNG4.SeedFixedIntegers({1,2,3,4});
    gaussian(RNG4,Umu);
    
    typename GridDiracGP::ImplParams params_gp;
    params_gp.twists = Coordinate({1,1,1,0});

    typename GridDiracXconj::ImplParams params_x;
    params_x.twists = params_gp.twists;
    params_x.boundary_phase = 1.0;
    
    GridDiracXconj actionX(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c, params_x);
    GridDiracGP actionGP(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c, params_gp);
    
    GridParallelRNG RNG(FGrid);  
    RNG.SeedFixedIntegers({1,2,3,4});
    FermionField1f rnd_1f(FGrid), tmp1f(FGrid), M1f(FGrid);
    gaussian(RNG, rnd_1f);

    FermionField2f rnd_2f(FGrid), M2f(FGrid);
    XconjugateBoost(rnd_2f, rnd_1f);
  
    actionX.M(rnd_1f, M1f);
    actionGP.M(rnd_2f, M2f);

    FermionField1f M2f_f0 = PeekIndex<GparityFlavourIndex>(M2f,0); 
  
    FermionField1f diff = M2f_f0 - M1f;
    LOGA2A << "GP vs Xconj action test on Xconj src : GP " << norm2(M2f_f0) << " Xconj: " << norm2(M1f) << " diff ( expect 0): " << norm2(diff) << std::endl;
    assert(norm2(diff)/norm2(rnd_1f) < 1e-4);

    //Check 2f result is X-conjugate
    FermionField1f M2f_f1 = PeekIndex<GparityFlavourIndex>(M2f,1); 
    tmp1f = -(Xmatrix()*conjugate(M2f_f0));

    diff = tmp1f - M2f_f1;
    LOGA2A << "GP action on Xconj src produces X-conj vector (expect 0): " << norm2(diff) << std::endl;
    assert(norm2(diff)/norm2(rnd_1f) < 1e-4);
  }
  {
    LOGA2A << "Doing test using gauge fields and parameters from CPS" << std::endl;

    Grid::LatticeGaugeFieldD *Umu = lattice.getUmu();

    double b = lattice.get_mob_b();
    double c = b - 1.;   //b-c = 1
    double M5 = GJP.DwfHeight();
    double mass = 0.01;

    typename GridDiracGP::ImplParams params_gp;
    lattice.SetParams(params_gp);

    typename GridDiracXconj::ImplParams params_x;
    params_x.twists = params_gp.twists;
    params_x.boundary_phase = 1.0;

    GridDiracXconj actionX(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c, params_x);
    GridDiracGP actionGP(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c, params_gp);
  
    GridParallelRNG RNG(FGrid);  
    RNG.SeedFixedIntegers({1,2,3,4});
    FermionField1f rnd_1f(FGrid), tmp1f(FGrid), M1f(FGrid);
    gaussian(RNG, rnd_1f);

    FermionField2f rnd_2f(FGrid), M2f(FGrid);
    XconjugateBoost(rnd_2f, rnd_1f);
  
    actionX.M(rnd_1f, M1f);
    actionGP.M(rnd_2f, M2f);

    FermionField1f M2f_f0 = PeekIndex<GparityFlavourIndex>(M2f,0); 
  
    FermionField1f diff = M2f_f0 - M1f;
    LOGA2A << "GP vs Xconj action test on Xconj src : GP " << norm2(M2f_f0) << " Xconj: " << norm2(M1f) << " diff ( expect 0): " << norm2(diff) << std::endl;
    assert(norm2(diff)/norm2(rnd_1f) < 1e-4);

    //Check 2f result is X-conjugate
    FermionField1f M2f_f1 = PeekIndex<GparityFlavourIndex>(M2f,1); 
    tmp1f = -(Xmatrix()*conjugate(M2f_f0));

    diff = tmp1f - M2f_f1;
    LOGA2A << "GP action on Xconj src produces X-conj vector (expect 0): " << norm2(diff) << std::endl;
    assert(norm2(diff)/norm2(rnd_1f) < 1e-4);
  }

  LOGA2A << "Finished testing X-conj action" << std::endl;
}

CPS_END_NAMESPACE
#endif

