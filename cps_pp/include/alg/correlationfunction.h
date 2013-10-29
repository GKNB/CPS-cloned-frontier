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
  bool global_sum_on_write; //do a global summation before writing. Defaults to true.
public:
  void setNcontractions(const int &n);
  void write(const char *file);
  void write(FILE *fp);
  Rcomplex & operator()(const int &contraction_idx, const int &t);
  Rcomplex & operator()(const int &thread_idx, const int &contraction_idx, const int &t);

  void sumThreads(); //Form the sum of wick[contraction][t] over all threads on the local node. Result is stored in wick and after sum accessible via operator() (i.e. the non-threaded version)
  void sumLattice(); //Form the sum of wick[contraction][t] over all nodes (and threads if threaded). Result is stored in wick and after sum accessible via operator() (i.e. the non-threaded version)
  void clear();

  void setGlobalSumOnWrite(const bool &b){ global_sum_on_write = b; }

  const ThreadType & threadType() const{ return threadtype; }

  CorrelationFunction(const char *_label, const ThreadType &thread = UNTHREADED);
  CorrelationFunction(const char *_label, const int &n_contractions, const ThreadType &thread = UNTHREADED);

  ~CorrelationFunction();
};

#endif
CPS_END_NAMESPACE
