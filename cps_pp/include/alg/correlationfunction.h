CPS_START_NAMESPACE
#ifndef CORRELATION_FUNCTION_H
#define CORRELATION_FUNCTION_H
CPS_END_NAMESPACE

#include<config.h>


CPS_START_NAMESPACE


class CorrelationFunction{
  char *label;
  int ncontract;
  int time_size;
  Rcomplex** wick; //the wick contractions, each a function of t

  void sumLattice(); //sum each element of wick[contraction][t] over all nodes
public:
  void setNcontractions(const int &n);
  void write(const char *file);
  void write(FILE *fp);
  Rcomplex & operator()(const int &contraction_idx, const int &t);
  CorrelationFunction(const char *_label);
  ~CorrelationFunction();
};

#endif
CPS_END_NAMESPACE
