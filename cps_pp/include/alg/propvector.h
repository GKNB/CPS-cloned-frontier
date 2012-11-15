CPS_START_NAMESPACE
#ifndef PROP_VECTOR_H
#define PROP_VECTOR_H
CPS_END_NAMESPACE

#include<config.h>
#include <alg/propagatorcontainer.h>

CPS_START_NAMESPACE

//container for multiple props with destructor that deletes all props
class PropVector{
public:
  const static int MAX_SIZE = 100;
  PropagatorContainer & operator[](const int &idx);
  PropagatorContainer & addProp(PropagatorArg &arg);
  void clear();
  const int &size() const;
  PropVector();
  ~PropVector();
private:
  PropagatorContainer *props[MAX_SIZE];
  int sz;
};

#endif
CPS_END_NAMESPACE
