#ifndef _CPS_MESONFIELD_CONVERT_H
#define _CPS_MESONFIELD_CONVERT_H

#ifdef USE_GRID
#include<Hadrons/A2AMatrix.hpp>
#endif

#include "mesonfield.h"

CPS_START_NAMESPACE

#ifdef USE_GRID

template<typename A2Apolicies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void convertToGrid(Grid::Hadrons::A2AMatrix<Grid::ComplexD> &mf_out, const A2AmesonField<A2Apolicies, A2AfieldL, A2AfieldR> &mf_in){
  int nmodes_l = mf_in.getNrowsFull();
  int nmodes_r = mf_in.getNcolsFull();
  
  mf_out.resize(nmodes_l, nmodes_r);
  
#pragma omp parallel for
  for(size_t ij=0;ij<nmodes_l*nmodes_r;ij++){ //j + nmodes_r * i
    size_t i = ij / nmodes_r;
    size_t j = ij % nmodes_r;

    mf_out(i,j) = mf_in.elem(i,j);
  }
}

#endif


CPS_END_NAMESPACE

#endif
