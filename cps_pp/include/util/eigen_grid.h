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
    neig = _evals.size();
    evals.resize(neig);
    for(int i=0;i<neig;i++) evals[i] = _evals[i];
    assert(neig == evecs.size());
    grid = evecs[0].Grid();
    evec_grid = & evecs;
    VRB.Result(cname,"EigenCacheGrid()","grid=%p\n",grid);
  }

};

CPS_END_NAMESPACE
#endif

#endif
