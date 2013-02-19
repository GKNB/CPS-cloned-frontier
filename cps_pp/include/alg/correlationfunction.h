CPS_START_NAMESPACE
#ifndef CORRELATION_FUNCTION_H
#define CORRELATION_FUNCTION_H
CPS_END_NAMESPACE

#include<config.h>


CPS_START_NAMESPACE


class CorrelationFunction{
public:
  enum ThreadType { THREADED, UNTHREADED };
private:
  char *label;
  int ncontract;
  int time_size;
  Rcomplex** wick; //unthreaded wick contractions [contraction][t]
  Rcomplex*** wick_threaded; //threaded wick contractions [thread][contraction][t]

  ThreadType threadtype;
  int max_threads; //the max amount of threads when the array of Rcomplex was created
public:
  void setNcontractions(const int &n);
  void write(const char *file);
  void write(FILE *fp);
  Rcomplex & operator()(const int &contraction_idx, const int &t);
  Rcomplex & operator()(const int &thread_idx, const int &contraction_idx, const int &t);

  void sumLattice(); //sum each element of wick[contraction][t] over all nodes (and threads if threaded). Result is stored in wick and after sum accessible via operator() (non-threaded version)
  void clear();

  CorrelationFunction(const char *_label, const ThreadType &thread = UNTHREADED);
  ~CorrelationFunction();
};

#endif
CPS_END_NAMESPACE
