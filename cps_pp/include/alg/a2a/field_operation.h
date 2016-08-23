#ifndef _FIELD_OPERATION_H
#define _FIELD_OPERATION_H

CPS_START_NAMESPACE

//Perform some operation on a field
template <typename FieldType>
class fieldOperation{
public:
  virtual void operator()(const FieldType &in, FieldType &out) = 0;
};

template <typename FieldType>
class gaugeFixAndTwist: public fieldOperation<FieldType>{
  int p[3];
  Lattice *lat;
public:
  gaugeFixAndTwist(const int _p[3], Lattice &_lat): lat(&_lat){ for(int i=0;i<3;i++) p[i] = _p[i]; }

  void operator()(const FieldType &in, FieldType &out){
    //Gauge fix and apply phase in parallel (i.e. don't parallelize over modes)
    out = in;
#ifndef MEMTEST_MODE
    out.gaugeFix(*lat,true);
    out.applyPhase(p,true);
#endif
  }
};

CPS_END_NAMESPACE

#endif
