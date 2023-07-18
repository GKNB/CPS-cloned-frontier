#pragma once

//Classes that enclose eigenvectors and makes them accessible as Grid fields

#include<config.h>

#if defined(USE_GRID) && defined(USE_GRID_A2A)
#include<Grid/Grid.h>
#include <Grid/algorithms/iterative/LocalCoherenceLanczos.h>
#include "grid_Xconj.h"

CPS_START_NAMESPACE

//Compute the guess for Nfield sources with Nevecs eigenvectors
//in, out are arrays of length Nfield
template<typename GridFermionField, typename EvecGetter>
void basicDeflatedGuess(GridFermionField *out, GridFermionField const *in, int Nfield, int Nevecs, const EvecGetter &get){
  double t_total = -dclock();
  double t_get = 0;
  double t_inner = 0;
  double t_linalg = 0;

  for(int s=0;s<Nfield;s++){
    out[s] = Grid::Zero();
    out[s].Checkerboard() = in[0].Checkerboard();
  }
  Grid::GridBase* grid = in[0].Grid();

  GridFermionField evec(grid);
  for(int i=0;i<Nevecs;i++){
    t_get -= dclock();
    double eval = get(evec, i);
    t_get += dclock();

    for(int s=0;s<Nfield;s++){
      assert(in[s].Checkerboard() == evec.Checkerboard());
      t_inner -= dclock();
      Grid::ComplexD dot = innerProduct(evec, in[s]);
      t_inner += dclock();

      t_linalg -= dclock();
      dot = dot / eval;
      out[s] = out[s] + dot * evec;
      t_linalg += dclock();
    }
  }
  t_total += dclock();
  LOGA2A << "Deflated " << Nfield << " fields with " << Nevecs << " evecs in " << t_total 
	    << "s:  evec_load:" << t_get << "s, inner_product:" << t_inner << "s, linalg:" << t_linalg << "s" << std::endl;

}

//Base class of interfaces, expose double precision evecs and deflation
template<typename _GridFermionFieldD>
class EvecInterface{
public:
  typedef _GridFermionFieldD GridFermionFieldD;

  //Get the double precision evec Grid
  virtual Grid::GridBase* getEvecGridD() const = 0;

  //Get the double precision evec and eval
  virtual double getEvecD(GridFermionFieldD &into, const int idx) const = 0;

  //Get the number of evecs
  virtual int nEvecs() const = 0;

  //Get the deflated guess for a set of 5D source vectors
  //if use_Nevecs is set the number of evecs used will be constrained to that number
  virtual void deflatedGuessDp(GridFermionFieldD *out, GridFermionFieldD const *in, int Nfield, int use_Nevecs = -1) const{
    if(use_Nevecs = -1) use_Nevecs = this->nEvecs();
    else assert(use_Nevecs <= this->nEvecs());    
    basicDeflatedGuess(out, in, Nfield, use_Nevecs, [&](GridFermionFieldD &into, const int i){ return this->getEvecD(into,i); });
  }

  void deflatedGuessD(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in, int use_Nevecs = -1) const{
    assert(out.size() == in.size());
    deflatedGuessDp(out.data(),in.data(),in.size(),use_Nevecs);
  }
  
  virtual ~EvecInterface(){}
};

//A Grid guesser, only valid for double precision fields
template<class FieldD>
class EvecInterfaceGuesser: public Grid::LinearFunction<FieldD> {
  const EvecInterface<FieldD> &interface;
public:
  EvecInterfaceGuesser(const EvecInterface<FieldD> &interface): interface(interface){}

  void operator()(const FieldD &src,FieldD &guess) {
    interface.deflatedGuessD(&guess, &src, 1);
  }
};

//Guesser for single prec fields that converts to double prec for the intermediate computation using double prec evecs
template<class FieldF, class FieldD>
class EvecInterfaceSinglePrecGuesser: public Grid::LinearFunction<FieldF> {
  const EvecInterface<FieldD> &interface;
  Grid::GridBase* doublePrecGrid;
  Grid::GridBase* singlePrecGrid;
  FieldD tmp1D, tmp2D;
  Grid::precisionChangeWorkspace pc_d_to_f;
  Grid::precisionChangeWorkspace pc_f_to_d;
public:
  EvecInterfaceSinglePrecGuesser(const EvecInterface<FieldD> &interface, Grid::GridBase* singlePrecGrid): interface(interface), doublePrecGrid(interface.getEvecGridD()), 
													  singlePrecGrid(singlePrecGrid), tmp1D(doublePrecGrid),tmp2D(doublePrecGrid),
													  pc_d_to_f(singlePrecGrid,doublePrecGrid),
													  pc_f_to_d(doublePrecGrid,singlePrecGrid)
  {}

  void operator()(const FieldF &src,FieldF &guess) {
    assert(src.Grid() == singlePrecGrid && guess.Grid() == singlePrecGrid);
    precisionChange(tmp1D,src,pc_f_to_d);
    interface.deflatedGuessDp(&tmp2D, &tmp1D, 1);
    precisionChange(guess,tmp2D,pc_d_to_f);
  }
};

//Derived abstract class that also exposes single precision operations
template<typename _GridFermionFieldD, typename _GridFermionFieldF>
class EvecInterfaceMixedPrec: public EvecInterface<_GridFermionFieldD>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
  typedef _GridFermionFieldF GridFermionFieldF;

  //Get the single precision evec Grid
  virtual Grid::GridBase* getEvecGridF() const = 0;

  //Get the single precision evec and eval
  virtual double getEvecF(GridFermionFieldF &into, const int idx) const = 0;

  virtual void deflatedGuessFp(GridFermionFieldF *out, GridFermionFieldF const *in, int Nfield, int use_Nevecs = -1) const{
    if(use_Nevecs = -1) use_Nevecs = this->nEvecs();
    else assert(use_Nevecs <= this->nEvecs());    
    basicDeflatedGuess(out, in, Nfield, use_Nevecs, [&](GridFermionFieldF &into, const int i){ return this->getEvecF(into,i); });
  }

  void deflatedGuessF(std::vector<GridFermionFieldF> &out, const std::vector<GridFermionFieldF> &in, int use_Nevecs = -1) const{
    assert(out.size() == in.size());
    deflatedGuessFp(out.data(),in.data(),in.size(),use_Nevecs);
  }
  
  virtual ~EvecInterfaceMixedPrec(){}
};

//An implementation of the interface for the case of unused evecs
template<typename _GridFermionFieldD, typename _GridFermionFieldF>
class EvecInterfaceMixedPrecNone: public EvecInterfaceMixedPrec<_GridFermionFieldD,_GridFermionFieldF>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
  typedef _GridFermionFieldF GridFermionFieldF;
private:
  Grid::GridBase* grid_d;
  Grid::GridBase* grid_f;
public:
  EvecInterfaceMixedPrecNone(Grid::GridBase* grid_d, Grid::GridBase* grid_f): grid_d(grid_d), grid_f(grid_f){}

  Grid::GridBase* getEvecGridD() const override{ return grid_d; }
  Grid::GridBase* getEvecGridF() const override{ return grid_f; }

  //Get the double precision evec and eval
  double getEvecD(GridFermionFieldD &into, const int idx) const override { assert(0); } //should not get here! 
  double getEvecF(GridFermionFieldF &into, const int idx) const override { assert(0); }

  //Get the number of evecs
  int nEvecs() const override{ return 0; }
};




template<typename Field, class FieldD, class FieldF>
struct _choose_deflate{};

template<class FieldD, class FieldF>
struct _choose_deflate<FieldD,FieldD,FieldF>{
  static void doit(const EvecInterfaceMixedPrec<FieldD,FieldF> &interface, const FieldD &src, FieldD &guess){
    interface.deflatedGuessDp(&guess, &src, 1);
  }
};

template<class FieldD, class FieldF>
struct _choose_deflate<FieldF,FieldD,FieldF>{
  static void doit(const EvecInterfaceMixedPrec<FieldD,FieldF> &interface, const FieldF &src, FieldF &guess){
    interface.deflatedGuessFp(&guess, &src, 1);
  }
};


//A Grid guesser, valid for both double and single precision fields
template<class Field, class FieldD, class FieldF>
class EvecInterfaceMixedPrecGuesser: public Grid::LinearFunction<Field> {
  const EvecInterfaceMixedPrec<FieldD,FieldF> &interface;
public:
  EvecInterfaceMixedPrecGuesser(const EvecInterfaceMixedPrec<FieldD,FieldF> &interface): interface(interface){}

  void operator()(const Field &src,Field &guess) {
    _choose_deflate<Field,FieldD,FieldF>::doit(interface, src, guess);
  }

};


//Implementation of container for array of double precision evecs
template<typename _GridFermionFieldD>
class EvecInterfaceDoublePrec : public EvecInterface<_GridFermionFieldD>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;

private:
  Grid::GridBase* m_evecGridD;
  const std::vector<GridFermionFieldD> &m_evecs;
  const std::vector<double> &m_evals;

public:
  EvecInterfaceDoublePrec(const std::vector<GridFermionFieldD> &evecs, const std::vector<double> &evals, Grid::GridBase* evecGridD): 
    m_evecs(evecs), m_evals(evals), m_evecGridD(evecGridD){
    assert(m_evecs.size() == m_evals.size());
  }
  
  Grid::GridBase* getEvecGridD() const override{ return m_evecGridD; }

  //Get an eigenvector and eigenvalue
  double getEvecD(GridFermionFieldD &into, const int idx) const override{
    into = m_evecs[idx];
    return m_evals[idx];
  }

  int nEvecs() const override{ return m_evals.size(); }
};


//Implementation of mixed prec container for array of single precision evecs
template<typename _GridFermionFieldD, typename _GridFermionFieldF>
class EvecInterfaceSinglePrec : public EvecInterfaceMixedPrec<_GridFermionFieldD, _GridFermionFieldF>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
  typedef _GridFermionFieldF GridFermionFieldF;

private:
  Grid::GridBase* m_evecGridD;
  Grid::GridBase* m_evecGridF;
  const std::vector<GridFermionFieldF> &m_evecsF;
  const std::vector<double> &m_evals;
  Grid::precisionChangeWorkspace pc_f_to_d;
public:
  EvecInterfaceSinglePrec(const std::vector<GridFermionFieldF> &evecsF, const std::vector<double> &evals, Grid::GridBase* evecGridD, Grid::GridBase* evecGridF): 
    m_evecsF(evecsF), m_evals(evals), m_evecGridD(evecGridD), m_evecGridF(evecGridF), pc_f_to_d(m_evecGridD,m_evecGridF){
    assert(m_evecsF.size() == m_evals.size());
  }
  
  Grid::GridBase* getEvecGridD() const override{ return m_evecGridD; }
  Grid::GridBase* getEvecGridF() const override{ return m_evecGridF; }

  //Get an eigenvector and eigenvalue
  double getEvecD(GridFermionFieldD &into, const int idx) const override{
    precisionChange(into, m_evecsF[idx], pc_f_to_d);
    return m_evals[idx];
  }

  double getEvecF(GridFermionFieldF &into, const int idx) const override{
    into = m_evecsF[idx];
    return m_evals[idx];
  }

  int nEvecs() const override{ return m_evals.size(); }
};


//Implementation of mixed prec container for array of double precision evecs
template<typename _GridFermionFieldD, typename _GridFermionFieldF>
class EvecInterfaceMixedDoublePrec : public EvecInterfaceMixedPrec<_GridFermionFieldD, _GridFermionFieldF>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
  typedef _GridFermionFieldF GridFermionFieldF;

private:
  Grid::GridBase* m_evecGridD;
  Grid::GridBase* m_evecGridF;
  const std::vector<GridFermionFieldD> &m_evecsD;
  const std::vector<double> &m_evals;
  Grid::precisionChangeWorkspace pc_d_to_f;
public:
  EvecInterfaceMixedDoublePrec(const std::vector<GridFermionFieldD> &evecsD, const std::vector<double> &evals, Grid::GridBase* evecGridD, Grid::GridBase* evecGridF): 
    m_evecsD(evecsD), m_evals(evals), m_evecGridD(evecGridD), m_evecGridF(evecGridF), pc_d_to_f(m_evecGridF,m_evecGridD){
    assert(m_evecsD.size() == m_evals.size());
  }
  
  Grid::GridBase* getEvecGridD() const override{ return m_evecGridD; }
  Grid::GridBase* getEvecGridF() const override{ return m_evecGridF; }

  //Get an eigenvector and eigenvalue
  double getEvecD(GridFermionFieldD &into, const int idx) const override{
    into = m_evecsD[idx];
    return m_evals[idx];
  }

  double getEvecF(GridFermionFieldF &into, const int idx) const override{
    precisionChange(into, m_evecsD[idx], pc_d_to_f);
    return m_evals[idx];
  }

  int nEvecs() const override{ return m_evals.size(); }
};


template<typename GridFermionField, typename CoarseField, typename EvecGetter, typename Projector, typename Promoter>
void compressedDeflatedGuess(GridFermionField *out, GridFermionField const *in, int Nfield,  int Nevecs, Grid::GridBase* coarseGrid,
			     const EvecGetter &get, const Projector &proj, const Promoter &promote){
  double t_total = -dclock();
  double t_get = 0;
  double t_project = 0;
  double t_inner = 0;
  double t_linalg = 0;
  double t_promote = 0;

  for(int i=0;i<Nfield;i++){
    out[i] = Grid::Zero();
    out[i].Checkerboard() = in[0].Checkerboard();
  }

  t_project -= dclock();
  std::vector<CoarseField> eta_proj(Nfield, coarseGrid);
  for(int i=0;i<Nfield;i++) proj(eta_proj[i], in[i]);
  t_project += dclock();

  std::vector<CoarseField> rho(Nfield, coarseGrid);
  for(int i=0;i<Nfield;i++) rho[i] = Grid::Zero();

  CoarseField cevec(coarseGrid);
  for(int a=0;a<Nevecs;a++){      
    t_get -= dclock();
    double eval = get(cevec, a);
    t_get += dclock();

    for(int i=0;i<Nfield;i++){
      t_inner -= dclock();
      Grid::ComplexD dot = Grid::innerProduct(cevec, eta_proj[i]);
      t_inner += dclock();
	
      t_linalg -= dclock();
      rho[i] = rho[i] + dot / eval * cevec;
      t_linalg += dclock();
    }
  }

  t_promote -= dclock();    
  for(int i=0;i<Nfield;i++) promote(out[i],rho[i]);
  t_promote += dclock();    

  t_total += dclock();
  LOGA2A << "Deflated " << Nfield << " fields with " << Nevecs << " compressed evecs in " << t_total 
	    << "s:  get: " << t_get << "s, projection:" << t_project << "s, inner_product:" << t_inner << "s, linalg:" << t_linalg << "s, promote:" << t_promote << "s" << std::endl;
}






template<typename _GridFermionFieldD, typename _GridFermionFieldF, int _basis_size>
class EvecInterfaceCompressedMixedDoublePrec : public EvecInterfaceMixedPrec<_GridFermionFieldD, _GridFermionFieldF>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
  typedef _GridFermionFieldF GridFermionFieldF;
  enum { basis_size = _basis_size };
private:
  Grid::GridBase* m_evecGridD;
  Grid::GridBase* m_evecGridF;
  typedef typename GridFermionFieldD::vector_object SiteSpinor;
  typedef Grid::LocalCoherenceLanczos<SiteSpinor,Grid::vTComplexD,basis_size> LCL;
  typedef typename LCL::CoarseSiteVector vCoarseSiteVector;
  typedef typename LCL::CoarseField CoarseField;
  typedef typename vCoarseSiteVector::scalar_object CoarseSiteVector;

  const std::vector<GridFermionFieldD> &m_basisD;
  const std::vector<CoarseField> &m_coarseEvecsD;
  const std::vector<double> &m_evals;

  mutable GridFermionFieldD tmpD;
  Grid::precisionChangeWorkspace pc_d_to_f;
  Grid::precisionChangeWorkspace pc_f_to_d;
public:
  EvecInterfaceCompressedMixedDoublePrec( const std::vector<CoarseField> &coarseEvecsD, const std::vector<GridFermionFieldD> &basisD, 
				const std::vector<double> &evals, Grid::GridBase* evecGridD, Grid::GridBase* evecGridF): 
    m_coarseEvecsD(coarseEvecsD), m_basisD(basisD), m_evals(evals), m_evecGridD(evecGridD), m_evecGridF(evecGridF), tmpD(evecGridD), 
    pc_d_to_f(evecGridF,evecGridD), pc_f_to_d(evecGridD,evecGridF){
    assert(m_basisD.size() == basis_size);
    assert(m_coarseEvecsD.size() == m_evals.size());
  }
  
  Grid::GridBase* getEvecGridD() const override{ return m_evecGridD; }
  Grid::GridBase* getEvecGridF() const override{ return m_evecGridF; }

  //Get an eigenvector and eigenvalue
  double getEvecD(GridFermionFieldD &into, const int idx) const override{
    Grid::blockPromote(m_coarseEvecsD[idx],into,m_basisD); 
    return m_evals[idx];
  }

  double getEvecF(GridFermionFieldF &into, const int idx) const override{
    Grid::blockPromote(m_coarseEvecsD[idx],tmpD,m_basisD); 
    precisionChange(into, tmpD, pc_d_to_f);
    return m_evals[idx];
  }

  int nEvecs() const override{ return m_evals.size(); }
  
  //Perform the deflation in the blocked space to avoid uncompression overheads
  void deflatedGuessDp(GridFermionFieldD *out, GridFermionFieldD const *in, int Nfield, int use_Nevecs = -1) const override{
    if(use_Nevecs = -1) use_Nevecs = this->nEvecs();
    else assert(use_Nevecs <= this->nEvecs());    

    assert(m_coarseEvecsD.size() > 0);
    Grid::GridBase* coarseGrid = m_coarseEvecsD[0].Grid();

    compressedDeflatedGuess<GridFermionFieldD,CoarseField>(out, in, Nfield, use_Nevecs, coarseGrid,
							   [&](CoarseField &into, const int idx){ into = m_coarseEvecsD[idx]; return m_evals[idx]; },
							   [&](CoarseField &out, const GridFermionFieldD &in){ Grid::blockProject(out,in, m_basisD); },
							   [&](GridFermionFieldD &out, const CoarseField &in){ Grid::blockPromote(in, out, m_basisD); }
							   );
  }

  void deflatedGuessFp(GridFermionFieldF *out, GridFermionFieldF const *in, int Nfield, int use_Nevecs = -1) const override{
    if(use_Nevecs = -1) use_Nevecs = this->nEvecs();
    else assert(use_Nevecs <= this->nEvecs());    

    assert(m_coarseEvecsD.size() > 0);
    Grid::GridBase* coarseGrid = m_coarseEvecsD[0].Grid();

    //use double precision fields internally
    compressedDeflatedGuess<GridFermionFieldF,CoarseField>(out, in, Nfield, use_Nevecs, coarseGrid,
							   [&](CoarseField &into, const int idx){ into = m_coarseEvecsD[idx]; return m_evals[idx]; },
							   [&](CoarseField &out, const GridFermionFieldF &in){ precisionChange(tmpD,in,pc_f_to_d); Grid::blockProject(out,tmpD, m_basisD); },
							   [&](GridFermionFieldF &out, const CoarseField &in){ Grid::blockPromote(in, tmpD, m_basisD); precisionChange(out,tmpD,pc_d_to_f); }
							   );
  }


};



template<typename _GridFermionFieldD, typename _GridFermionFieldF, int _basis_size>
class EvecInterfaceCompressedSinglePrec : public EvecInterfaceMixedPrec<_GridFermionFieldD, _GridFermionFieldF>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
  typedef _GridFermionFieldF GridFermionFieldF;
  enum { basis_size = _basis_size };
private:
  Grid::GridBase* m_evecGridD;
  Grid::GridBase* m_evecGridF;
  typedef typename GridFermionFieldF::vector_object SiteSpinor;
  typedef Grid::LocalCoherenceLanczos<SiteSpinor,Grid::vTComplexF,basis_size> LCL;
  typedef typename LCL::CoarseSiteVector vCoarseSiteVector;
  typedef typename LCL::CoarseField CoarseField;
  typedef typename vCoarseSiteVector::scalar_object CoarseSiteVector;

  const std::vector<GridFermionFieldF> &m_basisF;
  const std::vector<CoarseField> &m_coarseEvecsF;
  const std::vector<double> &m_evals;
  mutable GridFermionFieldF tmpF;

  Grid::precisionChangeWorkspace pc_d_to_f;
  Grid::precisionChangeWorkspace pc_f_to_d;
public:
  EvecInterfaceCompressedSinglePrec( const std::vector<CoarseField> &coarseEvecsF, const std::vector<GridFermionFieldF> &basisF, 
				     const std::vector<double> &evals, Grid::GridBase* evecGridD, Grid::GridBase* evecGridF): 
    m_coarseEvecsF(coarseEvecsF), m_basisF(basisF), m_evals(evals), m_evecGridD(evecGridD), m_evecGridF(evecGridF), tmpF(evecGridF),
    pc_d_to_f(evecGridF,evecGridD), pc_f_to_d(evecGridD,evecGridF){

    assert(m_basisF.size() == basis_size);
    assert(m_coarseEvecsF.size() == m_evals.size());
  }
  
  Grid::GridBase* getEvecGridD() const override{ return m_evecGridD; }
  Grid::GridBase* getEvecGridF() const override{ return m_evecGridF; }

  //Get an eigenvector and eigenvalue
  double getEvecD(GridFermionFieldD &into, const int idx) const override{
    Grid::blockPromote(m_coarseEvecsF[idx],tmpF,m_basisF); 
    precisionChange(into,tmpF,pc_f_to_d);
    return m_evals[idx];
  }

  double getEvecF(GridFermionFieldF &into, const int idx) const override{
    Grid::blockPromote(m_coarseEvecsF[idx],into,m_basisF); 
    return m_evals[idx];
  }

  int nEvecs() const override{ return m_evals.size(); }
  
  //Perform the deflation in the blocked space to avoid uncompression overheads
  void deflatedGuessDp(GridFermionFieldD *out, GridFermionFieldD const *in, int Nfield, int use_Nevecs = -1) const override{
    if(use_Nevecs = -1) use_Nevecs = this->nEvecs();
    else assert(use_Nevecs <= this->nEvecs());    

    assert(m_coarseEvecsF.size() > 0);
    Grid::GridBase* coarseGrid = m_coarseEvecsF[0].Grid();

    //use single precision fields internally
    compressedDeflatedGuess<GridFermionFieldD,CoarseField>(out, in, Nfield, use_Nevecs, coarseGrid,
							   [&](CoarseField &into, const int idx){ into = m_coarseEvecsF[idx]; return m_evals[idx]; },
							   [&](CoarseField &out, const GridFermionFieldD &in){ precisionChange(tmpF,in,pc_d_to_f); Grid::blockProject(out,tmpF, m_basisF); },
							   [&](GridFermionFieldD &out, const CoarseField &in){ Grid::blockPromote(in, tmpF, m_basisF); precisionChange(out, tmpF,pc_f_to_d); }
							   );
  }

  void deflatedGuessFp(GridFermionFieldF *out, GridFermionFieldF const *in, int Nfield, int use_Nevecs = -1) const override{
    if(use_Nevecs = -1) use_Nevecs = this->nEvecs();
    else assert(use_Nevecs <= this->nEvecs());    

    assert(m_coarseEvecsF.size() > 0);
    Grid::GridBase* coarseGrid = m_coarseEvecsF[0].Grid();

    compressedDeflatedGuess<GridFermionFieldF,CoarseField>(out, in, Nfield, use_Nevecs, coarseGrid,
							   [&](CoarseField &into, const int idx){ into = m_coarseEvecsF[idx]; return m_evals[idx]; },
							   [&](CoarseField &out, const GridFermionFieldF &in){ Grid::blockProject(out,in, m_basisF); },
							   [&](GridFermionFieldF &out, const CoarseField &in){ Grid::blockPromote(in, out, m_basisF); }
							   );
  }
};




//Implementation of container for array of double precision X-conjugate evecs
template<typename _GridFermionFieldD, typename _GridXconjFermionFieldD>
class EvecInterfaceXconjDoublePrec : public EvecInterface<_GridFermionFieldD>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
  typedef _GridXconjFermionFieldD GridXconjFermionFieldD;
private:
  Grid::GridBase* m_evecGridD;
  const std::vector<GridXconjFermionFieldD> &m_evecs;
  const std::vector<double> &m_evals;

public:
  EvecInterfaceXconjDoublePrec(const std::vector<GridXconjFermionFieldD> &evecs, const std::vector<double> &evals, Grid::GridBase* evecGridD): 
    m_evecs(evecs), m_evals(evals), m_evecGridD(evecGridD){
    assert(m_evecs.size() == m_evals.size());
  }
  
  Grid::GridBase* getEvecGridD() const override{ return m_evecGridD; }

  //Get an eigenvector and eigenvalue
  double getEvecD(GridFermionFieldD &into, const int idx) const override{
    XconjugateBoost(into, m_evecs[idx]);
    return m_evals[idx];
  }

  int nEvecs() const override{ return m_evals.size(); }
};


//Implementation of mixed prec container for array of single precision evecs
template<typename _GridFermionFieldD, typename _GridXconjFermionFieldD,
	 typename _GridFermionFieldF, typename _GridXconjFermionFieldF>
class EvecInterfaceXconjSinglePrec : public EvecInterfaceMixedPrec<_GridFermionFieldD, _GridFermionFieldF>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
  typedef _GridFermionFieldF GridFermionFieldF;
  typedef _GridXconjFermionFieldD GridXconjFermionFieldD;
  typedef _GridXconjFermionFieldF GridXconjFermionFieldF;
private:
  Grid::GridBase* m_evecGridD;
  Grid::GridBase* m_evecGridF;
  const std::vector<GridXconjFermionFieldF> &m_evecsF;
  const std::vector<double> &m_evals;
  Grid::precisionChangeWorkspace pc_f_to_d;
public:
  EvecInterfaceXconjSinglePrec(const std::vector<GridXconjFermionFieldF> &evecsF, const std::vector<double> &evals, Grid::GridBase* evecGridD, Grid::GridBase* evecGridF): 
    m_evecsF(evecsF), m_evals(evals), m_evecGridD(evecGridD), m_evecGridF(evecGridF), pc_f_to_d(m_evecGridD,m_evecGridF){
    assert(m_evecsF.size() == m_evals.size());
  }
  
  Grid::GridBase* getEvecGridD() const override{ return m_evecGridD; }
  Grid::GridBase* getEvecGridF() const override{ return m_evecGridF; }

  //Get an eigenvector and eigenvalue
  double getEvecD(GridFermionFieldD &into, const int idx) const override{
    GridXconjFermionFieldD tmp(m_evecGridD);
    precisionChange(tmp, m_evecsF[idx], pc_f_to_d);
    XconjugateBoost(into,tmp);
    return m_evals[idx];
  }

  double getEvecF(GridFermionFieldF &into, const int idx) const override{
    XconjugateBoost(into,m_evecsF[idx]);
    return m_evals[idx];
  }

  int nEvecs() const override{ return m_evals.size(); }
};


CPS_END_NAMESPACE

#endif
