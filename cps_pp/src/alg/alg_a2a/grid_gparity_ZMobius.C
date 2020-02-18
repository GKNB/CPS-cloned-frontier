//Grid currently does not implement ZMobius for G-parity
//Do do so we have to mirror Grid's fermion class instantiations
//(I wish this were easier)

#include<alg/a2a/lattice/fermion/grid_gparity_ZMobius.h>

#ifdef USE_GRID

#include <Grid/qcd/action/fermion/FermionCore.h>
#include <Grid/qcd/action/fermion/implementation/WilsonKernelsImplementation.h>
#include <Grid/qcd/action/fermion/implementation/WilsonKernelsAsmImplementation.h>
#include <Grid/qcd/action/fermion/implementation/WilsonKernelsHandImplementation.h>
#include <Grid/qcd/action/fermion/implementation/WilsonKernelsHandGparityImplementation.h>

#include <Grid/qcd/action/fermion/implementation/WilsonFermion5DImplementation.h>
#include <Grid/qcd/action/fermion/implementation/CayleyFermion5DImplementation.h>
#include <Grid/qcd/action/fermion/implementation/CayleyFermion5Dcache.h>

NAMESPACE_BEGIN(Grid);

// G-parity requires more specialised implementation.
#define DEFCON(IMPLEMENTATION)						\
  template <>								\
  void WilsonKernels<IMPLEMENTATION>::ContractConservedCurrentSiteFwd(const SitePropagator &q_in_1, \
								      const SitePropagator &q_in_2, \
								      SitePropagator &q_out, \
								      DoubledGaugeFieldView &U,	\
								      unsigned int sU, \
								      unsigned int mu, \
								      bool switch_sign)	\
  {									\
    assert(0);								\
  }									\
  template <>								\
  void WilsonKernels<IMPLEMENTATION>::ContractConservedCurrentSiteBwd( const SitePropagator &q_in_1, \
								       const SitePropagator &q_in_2, \
								       SitePropagator &q_out, \
								       DoubledGaugeFieldView &U, \
								       unsigned int mu,	\
								       unsigned int sU,	\
								       bool switch_sign) \
  {									\
    assert(0);								\
  }									\
									\
  HAND_SPECIALISE_GPARITY(IMPLEMENTATION);

DEFCON(ZGparityWilsonImplF);
DEFCON(ZGparityWilsonImplD);
DEFCON(ZGparityWilsonImplFH);
DEFCON(ZGparityWilsonImplDF);


template class WilsonKernels<ZGparityWilsonImplF>;
template class WilsonKernels<ZGparityWilsonImplD>;
template class WilsonKernels<ZGparityWilsonImplFH>;
template class WilsonKernels<ZGparityWilsonImplDF>;

template class WilsonFermion5D<ZGparityWilsonImplF>;
template class WilsonFermion5D<ZGparityWilsonImplD>;
template class WilsonFermion5D<ZGparityWilsonImplFH>;
template class WilsonFermion5D<ZGparityWilsonImplDF>;

template class CayleyFermion5D<ZGparityWilsonImplF>;
template class CayleyFermion5D<ZGparityWilsonImplD>;
template class CayleyFermion5D<ZGparityWilsonImplFH>;
template class CayleyFermion5D<ZGparityWilsonImplDF>;

NAMESPACE_END(Grid);

#endif
