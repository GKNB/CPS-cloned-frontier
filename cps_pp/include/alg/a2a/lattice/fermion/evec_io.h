#pragma once
#ifdef USE_GRID

#include "grid_lanczos.h"
#include<alg/a2a/base/utils_main.h>

CPS_START_NAMESPACE

template<typename FermionField>
void writeEvecsEvals(const std::vector<FermionField> &evecs, const std::vector<Grid::RealD> &evals,
		     const std::string &evecs_file, const std::string &evals_file){
#ifndef HAVE_LIME
  ERR.General("","writeEvecsEvals","Requires LIME");
#else
  using namespace Grid;
  if(!UniqueID()){
    Hdf5Writer WRx(evals_file);
    write(WRx,"evals", evals);
  }

  if(evecs.size()){
    Grid::GridBase* grid = evecs[0].Grid();
    emptyUserRecord record;
    ScidacWriter WR(grid->IsBoss());
    WR.open(evecs_file);
    for(int k=0;k<evecs.size();k++) {
      if(evecs[k].Checkerboard() != Odd) ERR.General("","writeEvecsEvals","Only implemented for odd-checkerboard evecs");
      WR.writeScidacFieldRecord(const_cast<FermionField&>(evecs[k]),record);
    }
    WR.close();
  }
#endif
}

template<typename FermionField>
void readEvecsEvals(std::vector<FermionField> &evecs, std::vector<Grid::RealD> &evals,
		    const std::string &evecs_file, const std::string &evals_file, Grid::GridBase *grid){
#ifndef HAVE_LIME
  ERR.General("","readEvecsEvals","Requires LIME");
#else
  using namespace Grid;
  Hdf5Reader RDx(evals_file);
  read(RDx,"evals",evals);

  int N = evals.size();
  evecs.clear();
  evecs.resize(N,grid);
  
  emptyUserRecord record;
  ScidacReader RD ;
  RD.open(evecs_file);
  for(int k=0;k<N;k++) {
    evecs[k].Checkerboard()=Odd;
    RD.readScidacFieldRecord(evecs[k],record);
  }
  RD.close();
#endif
}

#endif
CPS_END_NAMESPACE
