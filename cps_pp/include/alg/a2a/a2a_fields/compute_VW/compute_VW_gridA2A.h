#ifndef A2A_COMPUTE_VW_GRIDA2A_H_
#define A2A_COMPUTE_VW_GRIDA2A_H_

#include <memory>
#include <alg/a2a/a2a_fields.h>

#ifndef USE_GRID
#error "Must be using Grid"
#endif

#ifdef USE_GRID_A2A

CPS_START_NAMESPACE

//Compute the low mode V and W vectors using the abstract implementation classes provided
template<typename GridFermionFieldD, typename Vtype, typename Wtype>
void computeVWlow(Vtype &V, Wtype &W, const EvecInterface<GridFermionFieldD> &evecs,  
		  const A2AlowModeCompute<GridFermionFieldD> &impl);

//block_size is the number of sources deflated simultaneously, and if the inverter supports it, inverted concurrently
template<typename GridFermionField, typename Vtype, typename Wtype>
void computeVWhigh(Vtype &V, Wtype &W, 
		   const A2AhighModeSource<typename Vtype::Policies> &Wsrc_impl,
		   const EvecInterface<GridFermionField> &evecs,  
		   const A2AhighModeCompute<GridFermionField> &impl,
		   size_t block_size = 1);

//Compute both V and W with the internal inverter, operators and parameters are created internally controlled by CGcontrols 
//thus matching the original interface
template<typename Vtype, typename Wtype>
void computeVW(Vtype &V, Wtype &W, Lattice &lat, 
	       const EvecInterface<typename Vtype::Policies::GridFermionField> &evecs,
	       const double mass, const CGcontrols &cg);



#if defined(USE_GRID_LANCZOS)

//Compute both V and W with the internal inverter, operators and parameters are created internally controlled by CGcontrols
//Pure Grid for both Lanczos and A2A

//Single precision eigenvectors
template<typename Vtype, typename Wtype>
void computeVW(Vtype &V, Wtype &W, Lattice &lat, 
	       const std::vector<typename Vtype::Policies::GridFermionFieldF> &evec, const std::vector<Grid::RealD> &eval, const double mass, const CGcontrols &cg_controls);

//Double precision eigenvectors
template<typename Vtype, typename Wtype>
void computeVW(Vtype &V, Wtype &W, Lattice &lat, 
	       const std::vector<typename Vtype::Policies::GridFermionField> &evec, const std::vector<Grid::RealD> &eval, const double mass, const CGcontrols &cg_controls);

#endif

#include "implementation/compute_VWlow_gridA2A.tcc"
#include "implementation/compute_VWhigh_gridA2A.tcc"
#include "implementation/compute_VW_gridA2A.tcc"

CPS_END_NAMESPACE

#endif

#endif
