#ifndef _A2A_POLICIES_H
#define _A2A_POLICIES_H

#include<alg/a2a/a2a_allocpolicies.h>
#include<alg/a2a/CPSfield.h>

CPS_START_NAMESPACE

//Type policies needed for sources

#ifdef USE_GRID
struct GridSIMDSourcePolicies{
  typedef Grid::vComplexD ComplexType;
  typedef ThreeDSIMDPolicy<OneFlavorPolicy> MappingPolicy;
  typedef Aligned128AllocPolicy AllocPolicy;
};
struct GridSIMDSourcePoliciesSingle{
  typedef Grid::vComplexF ComplexType;
  typedef ThreeDSIMDPolicy<OneFlavorPolicy> MappingPolicy;
  typedef Aligned128AllocPolicy AllocPolicy;
};
#endif

struct StandardSourcePolicies{
  typedef cps::ComplexD ComplexType;
  typedef SpatialPolicy<OneFlavorPolicy> MappingPolicy;
  typedef StandardAllocPolicy AllocPolicy;
};

//These typedefs are needed if Grid is being used at all even if the main program is not using SIMD vectorized data types
#ifdef USE_GRID

CPS_END_NAMESPACE
#include<util/lattice/fgrid.h>
CPS_START_NAMESPACE

struct BaseGridPolicies{
# ifdef USE_GRID_GPARITY
  typedef FgridGparityMobius FgridFclass;
  typedef GnoneFgridGparityMobius FgridGFclass;
  typedef Grid::QCD::GparityMobiusFermionD GridDirac;
  typedef Grid::QCD::GparityMobiusFermionF GridDiracF; //single prec
  typedef Grid::QCD::GparityMobiusFermionFH GridDiracFH; //half-precision comms
  enum { FGRID_CLASS_NAME=F_CLASS_GRID_GPARITY_MOBIUS };
# else
  typedef FgridMobius FgridFclass;
  typedef GnoneFgridMobius FgridGFclass;
  typedef Grid::QCD::MobiusFermionD GridDirac;
  typedef Grid::QCD::MobiusFermionF GridDiracF;
  typedef Grid::QCD::MobiusFermionFH GridDiracFH;
  enum { FGRID_CLASS_NAME=F_CLASS_GRID_MOBIUS };
# endif  
  
  typedef typename GridDirac::FermionField GridFermionField;
  typedef typename GridDiracF::FermionField GridFermionFieldF;
#ifdef GRID_INNER_CG_HALFPREC_COMMS
  typedef GridDiracFH GridDiracFMixedCGInner; //which single-precision fermion action to use for inner CG of Grid high mode calculation
#else
  typedef GridDiracF GridDiracFMixedCGInner;
#endif
};

#define INHERIT_BASE_GRID_TYPEDEFS \
  typedef typename BaseGridPolicies::FgridFclass FgridFclass;		\
  typedef typename BaseGridPolicies::FgridGFclass FgridGFclass;		\
  typedef typename BaseGridPolicies::GridDirac GridDirac;		\
  typedef typename BaseGridPolicies::GridDiracF GridDiracF;		\
  typedef typename BaseGridPolicies::GridDiracFH GridDiracFH;		\
  typedef typename BaseGridPolicies::GridDiracFMixedCGInner GridDiracFMixedCGInner; \
  typedef typename BaseGridPolicies::GridFermionField GridFermionField;	\
  typedef typename BaseGridPolicies::GridFermionFieldF GridFermionFieldF; \
  enum { FGRID_CLASS_NAME=BaseGridPolicies::FGRID_CLASS_NAME }

#endif


//Policy choices
struct A2ApoliciesDoubleAutoAlloc{
#ifdef USE_GRID
  INHERIT_BASE_GRID_TYPEDEFS;
#endif
  
  typedef cps::ComplexD ComplexType;
  typedef cps::ComplexD ComplexTypeD;
  typedef cps::ComplexF ComplexTypeF;
  typedef StandardAllocPolicy AllocPolicy;
  typedef cps::ComplexD ScalarComplexType;
  typedef CPSfermion4D<ComplexType, FourDpolicy<DynamicFlavorPolicy>, AllocPolicy> FermionFieldType;
  typedef CPScomplex4D<ComplexType, FourDpolicy<DynamicFlavorPolicy>, AllocPolicy> ComplexFieldType;
  typedef StandardSourcePolicies SourcePolicies;

  SET_A2AVECTOR_AUTOMATIC_ALLOC(A2ApoliciesDoubleAutoAlloc);
};

struct A2ApoliciesDoubleManualAlloc{
#ifdef USE_GRID
  INHERIT_BASE_GRID_TYPEDEFS;
#endif
  
  typedef cps::ComplexD ComplexType;
  typedef cps::ComplexD ComplexTypeD;
  typedef cps::ComplexF ComplexTypeF;
  typedef StandardAllocPolicy AllocPolicy;
  typedef cps::ComplexD ScalarComplexType;
  typedef CPSfermion4D<ComplexType, FourDpolicy<DynamicFlavorPolicy>, AllocPolicy> FermionFieldType;
  typedef CPScomplex4D<ComplexType, FourDpolicy<DynamicFlavorPolicy>, AllocPolicy> ComplexFieldType;
  typedef StandardSourcePolicies SourcePolicies;

  SET_A2AVECTOR_MANUAL_ALLOC(A2ApoliciesDoubleManualAlloc);
};


#ifdef USE_GRID

//Policy choices
struct A2ApoliciesSIMDdoubleAutoAlloc{
  INHERIT_BASE_GRID_TYPEDEFS;

  typedef Grid::vComplexD ComplexType;
  typedef Grid::vComplexD ComplexTypeD;
  typedef Grid::vComplexF ComplexTypeF;
  typedef Aligned128AllocPolicy AllocPolicy;
  typedef cps::ComplexD ScalarComplexType;
  typedef CPSfermion4D<ComplexType, FourDSIMDPolicy<DynamicFlavorPolicy>, AllocPolicy> FermionFieldType;
  typedef CPScomplex4D<ComplexType, FourDSIMDPolicy<DynamicFlavorPolicy>, AllocPolicy> ComplexFieldType;
  typedef GridSIMDSourcePolicies SourcePolicies;

  SET_A2AVECTOR_AUTOMATIC_ALLOC(A2ApoliciesSIMDdoubleAutoAlloc);
};

struct A2ApoliciesSIMDdoubleManualAlloc{
  INHERIT_BASE_GRID_TYPEDEFS;

  typedef Grid::vComplexD ComplexType;
  typedef Grid::vComplexD ComplexTypeD;
  typedef Grid::vComplexF ComplexTypeF;
  typedef Aligned128AllocPolicy AllocPolicy;
  typedef cps::ComplexD ScalarComplexType;
  typedef CPSfermion4D<ComplexType, FourDSIMDPolicy<DynamicFlavorPolicy>, AllocPolicy> FermionFieldType;
  typedef CPScomplex4D<ComplexType, FourDSIMDPolicy<DynamicFlavorPolicy>, AllocPolicy> ComplexFieldType;
  typedef GridSIMDSourcePolicies SourcePolicies;

  SET_A2AVECTOR_MANUAL_ALLOC(A2ApoliciesSIMDdoubleManualAlloc);
};

#endif

CPS_END_NAMESPACE

#endif
