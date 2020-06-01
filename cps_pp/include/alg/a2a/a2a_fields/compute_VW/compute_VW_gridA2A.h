#ifndef A2A_COMPUTE_VW_GRIDA2A_H_
#define A2A_COMPUTE_VW_GRIDA2A_H_

#include <alg/a2a/a2a_fields.h>

#ifndef USE_GRID
#error "Must be using Grid"
#endif

#ifdef USE_GRID_A2A

CPS_START_NAMESPACE

//Compute the high mode parts of V and W using either standard CG or a multi-RHS CG variant, respectively
template<typename Policies>
void computeVWhighSingle(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, EvecInterface<Policies> &evecs, const Float mass, const CGcontrols &cg_controls);
template<typename Policies>
void computeVWhighMulti(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, EvecInterface<Policies> &evecs, const Float mass, const CGcontrols &cg_controls);
template<typename Policies>
void computeVWhighSingleMADWF(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, EvecInterface<Policies> &evecs, const Float mass, const CGcontrols &cg_controls);

//Chooses the appropriate function from the previous ones based on the cg_control
template<typename Policies>
void computeVWhigh(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, EvecInterface<Policies> &evecs, const Float mass, const CGcontrols &cg_controls);

//Compute the low mode part of the W and V vectors.
template<typename Policies>
void computeVWlow(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, EvecInterface<Policies> &evecs, const Float mass, const CGcontrols &cg_controls);

template<typename Policies>
void computeVW(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, EvecInterface<Policies> &evecs, const Float mass, const CGcontrols &cg_controls){
  computeVWlow(V,W,lat,evecs,mass,cg_controls);
  computeVWhigh(V,W,lat,evecs,mass,cg_controls);
}


#if defined(USE_GRID_LANCZOS)
  //Pure Grid for both Lanczos and A2A
template<typename Policies>
void computeVWlow(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, const std::vector<typename Policies::GridFermionField> &evec, const std::vector<Grid::RealD> &eval, const double mass, const CGcontrols &cg_controls);

template<typename Policies>
void computeVWhigh(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, const std::vector<typename Policies::GridFermionField> &evec, const std::vector<Grid::RealD> &eval, const double mass, const CGcontrols &cg_controls);

template<typename Policies>
void computeVW(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, const std::vector<typename Policies::GridFermionField> &evec, const std::vector<Grid::RealD> &eval, const double mass, const CGcontrols &cg_controls){
  computeVWlow(V,W,lat,evec,eval,mass,cg_controls);
  computeVWhigh(V,W,lat,evec,eval,mass,cg_controls);
}

  //Single-precision variants (use mixed_CG internally)
template<typename Policies>
void computeVWlow(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, const std::vector<typename Policies::GridFermionFieldF> &evec, const std::vector<Grid::RealD> &eval, const double mass, const CGcontrols &cg_controls);

template<typename Policies>
void computeVWhigh(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, const std::vector<typename Policies::GridFermionFieldF> &evec, const std::vector<Grid::RealD> &eval, const double mass, const CGcontrols &cg_controls);

template<typename Policies>
void computeVW(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, const std::vector<typename Policies::GridFermionFieldF> &evec, const std::vector<Grid::RealD> &eval, const double mass, const CGcontrols &cg_controls){
  computeVWlow(V,W,lat,evec,eval,mass,cg_controls);
  computeVWhigh(V,W,lat,evec,eval,mass,cg_controls);
}
#endif

#ifdef USE_BFM_LANCZOS
//Compute the low mode part of the W and V vectors. In the Lanczos class you can choose to store the vectors in single precision (despite the overall precision, which is fixed to double here)
//Set 'singleprec_evecs' if this has been done
template< typename Policies>
void computeVWlow(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, BFM_Krylov::Lanczos_5d<double> &eig, bfm_evo<double> &dwf, bool singleprec_evecs, const CGcontrols &cg_controls);
#endif


#include "implementation/compute_VW_gridA2A_common.tcc"
#include "implementation/compute_VWlow_gridA2A.tcc"
#include "implementation/compute_VWhigh_gridA2A.tcc"

CPS_END_NAMESPACE

#endif

#endif
