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
 
#ifdef ECACHE_GRID
// assumes alloc
  int read_compressed (char *root_, const char *checksum_dir=NULL, Grid::GridBase *_grid=NULL, int interval=-1){
    const char *fname="read_compressed()";
    VRB.Result(cname,fname,"Entered\n");
    if(alloc_flag==0) ERR.General(cname,fname,"should be allocated first\n");
    if(!grid) { 
      if(!_grid) ERR.General(cname,fname,"GridBase pointer not set\n");
      grid = _grid;
    } else {
      if(_grid && (grid!=_grid)) 
      ERR.General(cname,fname,"GridBase pointer different!(%p != %p)\n",grid,_grid);
    }
      EigenCache::read_compressed(root,checksum_dir,interval);
    evec_grid = new std::vector<Field> 
    for(int i=0;i<neig;i++){
    }
    alloc_flag=2;
    ordering=GRID;
  }
#endif
};

CPS_END_NAMESPACE
#endif

#endif
