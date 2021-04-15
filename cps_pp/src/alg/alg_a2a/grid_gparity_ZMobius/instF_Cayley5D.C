#include<alg/a2a/lattice/fermion/grid_gparity_ZMobius.h>

#ifdef USE_GRID

#include <Grid/qcd/action/fermion/FermionCore.h>
#include <Grid/qcd/action/fermion/implementation/CayleyFermion5DImplementation.h>
#include <Grid/qcd/action/fermion/implementation/CayleyFermion5Dcache.h>

NAMESPACE_BEGIN(Grid);

template class CayleyFermion5D<ZGparityWilsonImplF>;

NAMESPACE_END(Grid);

#endif
