#pragma once

//Classes that enclose eigenvectors and makes them accessible as Grid fields

#include<config.h>

#if defined(USE_GRID) && defined(USE_GRID_A2A)
#include<Grid/Grid.h>

CPS_START_NAMESPACE

//Compute the guess for Nfield sources with Nevecs eigenvectors
//in, out are arrays of length Nfield
template<typename GridFermionField, typename EvecGetter>
void basicDeflatedGuess(GridFermionField *out, GridFermionField const *in, int Nfield, int Nevecs, const EvecGetter &get){
  for(int s=0;s<Nfield;s++){
    out[s] = Grid::Zero();
    out[s].Checkerboard() = in[0].Checkerboard();
  }
  Grid::GridBase* grid = in[0].Grid();

  GridFermionField evec(grid);
  for(int i=0;i<Nevecs;i++){
    double eval = get(evec, i);

    for(int s=0;s<Nfield;s++){
      assert(in[s].Checkerboard() == evec.Checkerboard());
      Grid::ComplexD dot = innerProduct(evec, in[s]);
      dot = dot / eval;
      out[s] = out[s] + dot * evec;
    }
  }
}

//Base class of interfaces, expose double precision evecs and deflation
template<typename _GridFermionFieldD>
class EvecInterface{
public:
  typedef _GridFermionFieldD GridFermionFieldD;

  //Get the double precision evec Grid
  virtual Grid::GridBase* getEvecGridD() const = 0;

  //Get the double precision evec and eval
  virtual double getEvec(GridFermionFieldD &into, const int idx) const = 0;

  //Get the number of evecs
  virtual int nEvecs() const = 0;

  //Get the deflated guess for a set of 5D source vectors
  //if use_Nevecs is set the number of evecs used will be constrained to that number
  virtual void deflatedGuess(GridFermionFieldD *out, GridFermionFieldD const *in, int Nfield, int use_Nevecs = -1) const{
    if(use_Nevecs = -1) use_Nevecs = this->nEvecs();
    else assert(use_Nevecs <= this->nEvecs());    
    basicDeflatedGuess(out, in, Nfield, use_Nevecs, [&](GridFermionFieldD &into, const int i){ return this->getEvec(into,i); });
  }

  void deflatedGuess(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in, int use_Nevecs = -1) const{
    assert(out.size() == in.size());
    deflatedGuess(out.data(),in.data(),in.size(),use_Nevecs);
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
    interface.deflatedGuess(&guess, &src, 1);
  }
};

//Guesser for single prec fields that converts to double prec for the intermediate computation
template<class FieldF, class FieldD>
class EvecInterfaceSinglePrecGuesser: public Grid::LinearFunction<FieldF> {
  const EvecInterface<FieldD> &interface;
  Grid::GridBase* doublePrecGrid;
  FieldD tmp1D, tmp2D;
public:
  EvecInterfaceSinglePrecGuesser(const EvecInterface<FieldD> &interface): interface(interface), doublePrecGrid(interface.getEvecGridD()), 
									    tmp1D(doublePrecGrid),tmp2D(doublePrecGrid){}

  void operator()(const FieldF &src,FieldF &guess) {
    precisionChange(tmp1D,src);
    interface.deflatedGuess(&tmp2D, &tmp1D, 1);
    precisionChange(guess,tmp2D);
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
  virtual double getEvec(GridFermionFieldF &into, const int idx) const = 0;

  virtual void deflatedGuess(GridFermionFieldF *out, GridFermionFieldF const *in, int Nfield, int use_Nevecs = -1) const{
    if(use_Nevecs = -1) use_Nevecs = this->nEvecs();
    else assert(use_Nevecs <= this->nEvecs());    
    basicDeflatedGuess(out, in, Nfield, use_Nevecs, [&](GridFermionFieldF &into, const int i){ return this->getEvec(into,i); });
  }

  void deflatedGuess(std::vector<GridFermionFieldF> &out, const std::vector<GridFermionFieldF> &in, int use_Nevecs = -1) const{
    assert(out.size() == in.size());
    deflatedGuess(out.data(),in.data(),in.size(),use_Nevecs);
  }
  
  virtual ~EvecInterfaceMixedPrec(){}
};

//A Grid guesser, valid for both double and single precision fields
template<class Field, class FieldD, class FieldF>
class EvecInterfaceMixedPrecGuesser: public Grid::LinearFunction<Field> {
  const EvecInterfaceMixedPrec<FieldD,FieldF> &interface;
public:
  EvecInterfaceMixedPrecGuesser(const EvecInterfaceMixedPrec<FieldD,FieldF> &interface): interface(interface){}

  void operator()(const Field &src,Field &guess) {
    interface.deflatedGuess(&guess, &src, 1);
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
  double getEvec(GridFermionFieldD &into, const int idx) const override{
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

public:
  EvecInterfaceSinglePrec(const std::vector<GridFermionFieldF> &evecsF, const std::vector<double> &evals, Grid::GridBase* evecGridD, Grid::GridBase* evecGridF): 
    m_evecsF(evecsF), m_evals(evals), m_evecGridD(evecGridD), m_evecGridF(evecGridF){
    assert(m_evecsF.size() == m_evals.size());
  }
  
  Grid::GridBase* getEvecGridD() const override{ return m_evecGridD; }
  Grid::GridBase* getEvecGridF() const override{ return m_evecGridF; }

  //Get an eigenvector and eigenvalue
  double getEvec(GridFermionFieldD &into, const int idx) const override{
    precisionChange(into, m_evecsF[idx]);
    return m_evals[idx];
  }

  double getEvec(GridFermionFieldF &into, const int idx) const override{
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

public:
  EvecInterfaceMixedDoublePrec(const std::vector<GridFermionFieldD> &evecsD, const std::vector<double> &evals, Grid::GridBase* evecGridD, Grid::GridBase* evecGridF): 
    m_evecsD(evecsD), m_evals(evals), m_evecGridD(evecGridD), m_evecGridF(evecGridF){
    assert(m_evecsD.size() == m_evals.size());
  }
  
  Grid::GridBase* getEvecGridD() const override{ return m_evecGridD; }
  Grid::GridBase* getEvecGridF() const override{ return m_evecGridF; }

  //Get an eigenvector and eigenvalue
  double getEvec(GridFermionFieldD &into, const int idx) const override{
    into = m_evecsD[idx];
    return m_evals[idx];
  }

  double getEvec(GridFermionFieldF &into, const int idx) const override{
    precisionChange(into, m_evecsD[idx]);
    return m_evals[idx];
  }

  int nEvecs() const override{ return m_evals.size(); }
};



CPS_END_NAMESPACE

#endif
