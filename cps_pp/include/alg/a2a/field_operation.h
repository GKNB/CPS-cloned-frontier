#ifndef _FIELD_OPERATION_H
#define _FIELD_OPERATION_H

CPS_START_NAMESPACE

//Perform some operation on a field
template <typename mf_Complex>
class fieldOperation{
public:
  virtual void operator()(const CPSfermion4D<mf_Complex> &in, CPSfermion4D<mf_Complex> &out) = 0;
};

template <typename mf_Complex>
class gaugeFixAndTwist: public fieldOperation<mf_Complex>{
  int p[3];
  Lattice *lat;
public:
  gaugeFixAndTwist(const int _p[3], Lattice &_lat): lat(&_lat){ for(int i=0;i<3;i++) p[i] = _p[i]; }

  void operator()(const CPSfermion4D<mf_Complex> &in, CPSfermion4D<mf_Complex> &out){
    //Gauge fix and apply phase in parallel (i.e. don't parallelize over modes)
    out = in;
    out.gaugeFix(*lat,true);
    out.applyPhase(p,true);
  }
};

CPS_END_NAMESPACE

#endif
