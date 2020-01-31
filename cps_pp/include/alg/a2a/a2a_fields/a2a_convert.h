#ifndef _CPS_A2A_CONVERT_H_
#define _CPS_A2A_CONVERT_H_

#include "a2a_fields.h"

CPS_START_NAMESPACE

#ifdef USE_GRID

//Convert a W field to Grid format
template<typename GridFieldType, typename A2Apolicies>
void convertToGrid(std::vector<GridFieldType> &W_out, const A2AvectorW<A2Apolicies> &W_in, Grid::GridBase *grid){
  W_out.resize(W_in.getNv(), GridFieldType(grid));

  for(int i=0;i<W_in.getNl();i++)
    W_in.getWl(i).exportGridField(W_out[i]);

  typename A2Apolicies::FermionFieldType tmp_ferm(W_in.getWh(0).getDimPolParams());
  for(int i=W_in.getNl();i<W_in.getNv();i++){
    W_in.getDilutedSource(tmp_ferm, i-W_in.getNl());
    tmp_ferm.exportGridField(W_out[i]);
  }
}

//Convert a V field to Grid format
template<typename GridFieldType, typename A2Apolicies>
void convertToGrid(std::vector<GridFieldType> &V_out, const A2AvectorV<A2Apolicies> &V_in, Grid::GridBase *grid){
  V_out.resize(V_in.getNv(), GridFieldType(grid));

  for(int i=0;i<V_in.getNv();i++)
    V_in.getMode(i).exportGridField(V_out[i]);
}

#endif

CPS_END_NAMESPACE

#endif
