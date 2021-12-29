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
template<typename A2Apolicies, typename FermionOperatorTypeD>
void computeVWlow(A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W, const EvecInterface<typename FermionOperatorTypeD::FermionField> &evecs,  const A2AlowModeCompute<FermionOperatorTypeD> &impl);

//Allow the operator used for the high mode inversions (1) to differ from that used for the low mode contribution, (2) eg for MADWF
//block_size is the number of sources deflated simultaneously, and if the inverter supports it, inverted concurrently
template<typename A2Apolicies, typename FermionOperatorTypeD1, typename FermionOperatorTypeD2>
void computeVWhigh(A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W, 
		   const EvecInterface<typename FermionOperatorTypeD2::FermionField> &evecs,  
		   const A2AlowModeCompute<FermionOperatorTypeD2> &impl,
		   const A2Ainverter4dBase<FermionOperatorTypeD1> &inverter,
		   size_t block_size = 1);

//Compute both V and W with the internal inverter, operators and parameters are created internally controlled by CGcontrols 
//thus matching the original interface
template<typename Policies>
void computeVW(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, 
	       const EvecInterface<typename Policies::GridFermionField> &evecs,
	       const double mass, const CGcontrols &cg);



#if defined(USE_GRID_LANCZOS)

//Compute both V and W with the internal inverter, operators and parameters are created internally controlled by CGcontrols
//Pure Grid for both Lanczos and A2A

//Single precision eigenvectors
template<typename Policies>
void computeVW(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, 
	       const std::vector<typename Policies::GridFermionFieldF> &evec, const std::vector<Grid::RealD> &eval, const double mass, const CGcontrols &cg_controls);

//Double precision eigenvectors
template<typename Policies>
void computeVW(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, 
	       const std::vector<typename Policies::GridFermionField> &evec, const std::vector<Grid::RealD> &eval, const double mass, const CGcontrols &cg_controls);

#endif

#ifdef USE_BFM_LANCZOS
//Compute the low mode part of the W and V vectors. In the Lanczos class you can choose to store the vectors in single precision (despite the overall precision, which is fixed to double here)
//Set 'singleprec_evecs' if this has been done
template< typename Policies>
void computeVWlow(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, BFM_Krylov::Lanczos_5d<double> &eig, bfm_evo<double> &dwf, bool singleprec_evecs, const CGcontrols &cg_controls);

#endif


#include "implementation/compute_VWlow_gridA2A.tcc"
#include "implementation/compute_VWhigh_gridA2A.tcc"
#include "implementation/compute_VW_gridA2A.tcc"

CPS_END_NAMESPACE

#endif

#endif
