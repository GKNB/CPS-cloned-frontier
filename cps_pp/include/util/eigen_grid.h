#ifndef _EIGEN_GRID_H
#define _EIGEN_GRID_H
#include<util/eigen_container.h>
#ifdef USE_GRID
#include<Grid/Grid.h>

CPS_START_NAMESPACE

template < class Field > class EigenCacheGrid:public EigenCache {
private:
  char *cname;
//  const char *fname;
public:

  Grid::GridBase * grid;

  std::vector < Field > *evec_grid;
  EigenCacheGrid (const char *name):EigenCache (name), grid (NULL), evec_grid(NULL), cname ("EigenCacheGrid") {
  }
  ~EigenCacheGrid () { }

  void load( std::vector<Grid::RealD> &_evals, std::vector <Field> & evecs){
    alloc_flag=2;
    ordering=GRID;
    int size = _evals.size();
    evals.resize(size);
    for(int i=0;i<size;i++) evals[i] = _evals[i];
    assert(size == evecs.size());
    grid = evecs[0].Grid();
    evec_grid = & evecs;
  }

};

CPS_END_NAMESPACE
#endif

#endif
