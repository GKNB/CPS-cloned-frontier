#ifndef _A2A_POLICIES_H
#define _A2A_POLICIES_H

#include "a2a_allocpolicies.h"
#include<alg/a2a/lattice/CPSfield.h>

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
#ifdef USE_GRID
  typedef Grid::ComplexD ComplexType;
#else
  typedef cps::ComplexD ComplexType;
#endif
  typedef SpatialPolicy<OneFlavorPolicy> MappingPolicy;
  typedef StandardAllocPolicy AllocPolicy;
};

//These typedefs are needed if Grid is being used at all even if the main program is not using SIMD vectorized data types
#ifdef USE_GRID

CPS_END_NAMESPACE
#include<util/lattice/fgrid.h>
CPS_START_NAMESPACE

struct BaseGridPolicies{
  typedef FgridMobius FgridFclass;
  typedef GnoneFgridMobius FgridGFclass;
  typedef Grid::MobiusFermionD GridDirac;
  typedef Grid::MobiusFermionF GridDiracF;
  typedef Grid::MobiusFermionFH GridDiracFH;
  enum { FGRID_CLASS_NAME=F_CLASS_GRID_MOBIUS };
  
  typedef Grid::ZMobiusFermionD GridDiracZMobius;
  
  typedef typename GridDirac::FermionField GridFermionField;
  typedef typename GridDiracF::FermionField GridFermionFieldF;
#ifdef GRID_INNER_CG_HALFPREC_COMMS
  typedef GridDiracFH GridDiracFMixedCGInner; //which single-precision fermion action to use for inner CG of Grid high mode calculation
  typedef Grid::ZMobiusFermionFH GridDiracFZMobiusInner;
#else
  typedef GridDiracF GridDiracFMixedCGInner;
  typedef Grid::ZMobiusFermionF GridDiracFZMobiusInner;
#endif
};

struct BaseGridPoliciesGparity{
  typedef FgridGparityMobius FgridFclass;
  typedef GnoneFgridGparityMobius FgridGFclass;
  typedef Grid::GparityMobiusFermionD GridDirac;
  typedef Grid::GparityMobiusFermionF GridDiracF; //single prec
  typedef Grid::GparityMobiusFermionFH GridDiracFH; //half-precision comms
  enum { FGRID_CLASS_NAME=F_CLASS_GRID_GPARITY_MOBIUS };
  
  typedef Grid::ZGparityMobiusFermionD GridDiracZMobius;

  typedef typename GridDirac::FermionField GridFermionField;
  typedef typename GridDiracF::FermionField GridFermionFieldF;
#ifdef GRID_INNER_CG_HALFPREC_COMMS
  typedef GridDiracFH GridDiracFMixedCGInner; //which single-precision fermion action to use for inner CG of Grid high mode calculation
  typedef Grid::ZGparityMobiusFermionFH GridDiracFZMobiusInner;
#else
  typedef GridDiracF GridDiracFMixedCGInner;
  typedef Grid::ZGparityMobiusFermionF GridDiracFZMobiusInner;
#endif
};

#define INHERIT_BASE_GRID_TYPEDEFS(BGP)					\
  typedef typename BGP::FgridFclass FgridFclass;		\
  typedef typename BGP::FgridGFclass FgridGFclass;		\
  typedef typename BGP::GridDirac GridDirac;		\
  typedef typename BGP::GridDiracF GridDiracF;		\
  typedef typename BGP::GridDiracFH GridDiracFH;		\
  typedef typename BGP::GridDiracZMobius GridDiracZMobius;	\
  typedef typename BGP::GridFermionField GridFermionField;	\
  typedef typename BGP::GridFermionFieldF GridFermionFieldF; \
  typedef typename BGP::GridDiracFMixedCGInner GridDiracFMixedCGInner; \
  typedef typename BGP::GridDiracFZMobiusInner GridDiracFZMobiusInner; \
  enum { FGRID_CLASS_NAME=BGP::FGRID_CLASS_NAME }

#endif

//Policy choices for non-SIMD operations

//First setup the complex types and inherit extra Grid typedefs if using Grid
#ifdef USE_GRID

#define A2APOLICIES_SETUP(BASE_GRID_PARAMS)		\
 INHERIT_BASE_GRID_TYPEDEFS(BASE_GRID_PARAMS);				\
 typedef Grid::ComplexD ComplexType;					\
 typedef Grid::ComplexD ComplexTypeD;					\
 typedef Grid::ComplexF ComplexTypeF;					\
 									\
 typedef Grid::ComplexD ScalarComplexType;

#else

#define A2APOLICIES_SETUP(NULL_ARG)						\
 typedef cps::ComplexD ComplexType;					\
 typedef cps::ComplexD ComplexTypeD;					\
 typedef cps::ComplexF ComplexTypeF;					\
									\
 typedef cps::ComplexD ScalarComplexType;

#endif  

//This macro defines a template for all the non-SIMD A2A policies
#define A2APOLICIES_TEMPLATE(NAME, IS_GPARITY_POLICY, BASE_GRID_PARAMS, ALLOCATOR_MACRO) \
struct NAME{								\
 A2APOLICIES_SETUP(BASE_GRID_PARAMS)					\
 typedef StandardAllocPolicy AllocPolicy;				\
 typedef CPSfermion4D<ComplexType, FourDpolicy<DynamicFlavorPolicy>, AllocPolicy> FermionFieldType; \
 typedef CPScomplex4D<ComplexType, FourDpolicy<DynamicFlavorPolicy>, AllocPolicy> ComplexFieldType; \
 typedef FermionFieldType ScalarFermionFieldType; /*SIMD vectorized and scalar (non-vectorized) fields are the same*/ \
 typedef ComplexFieldType ScalarComplexFieldType;			\
 typedef StandardSourcePolicies SourcePolicies;				\
									\
 ALLOCATOR_MACRO(NAME);							\
 enum { GPARITY=IS_GPARITY_POLICY };					\
};

//Apply the template!
A2APOLICIES_TEMPLATE(A2ApoliciesDoubleAutoAlloc, 0, BaseGridPolicies, SET_A2AVECTOR_AUTOMATIC_ALLOC);
A2APOLICIES_TEMPLATE(A2ApoliciesDoubleManualAlloc, 0, BaseGridPolicies, SET_A2AVECTOR_MANUAL_ALLOC);
A2APOLICIES_TEMPLATE(A2ApoliciesDoubleAutoAllocGparity, 1, BaseGridPoliciesGparity, SET_A2AVECTOR_AUTOMATIC_ALLOC);
A2APOLICIES_TEMPLATE(A2ApoliciesDoubleManualAllocGparity, 1, BaseGridPoliciesGparity, SET_A2AVECTOR_MANUAL_ALLOC);





#ifdef USE_GRID

//This macro defines a template for all the SIMD A2A policies

#define A2APOLICIES_SIMD_TEMPLATE(NAME, IS_GPARITY_POLICY, BASE_GRID_PARAMS, ALLOCATOR_MACRO) \
struct NAME{					\
 INHERIT_BASE_GRID_TYPEDEFS(BASE_GRID_PARAMS);	\
						\
 typedef Grid::vComplexD ComplexType;		\
 typedef Grid::vComplexD ComplexTypeD;		\
 typedef Grid::vComplexF ComplexTypeF;		\
 typedef Aligned128AllocPolicy AllocPolicy;	\
 typedef Grid::ComplexD ScalarComplexType;	\
									\
 typedef CPSfermion4D<ComplexType, FourDSIMDPolicy<DynamicFlavorPolicy>, AllocPolicy> FermionFieldType;	\
 typedef CPScomplex4D<ComplexType, FourDSIMDPolicy<DynamicFlavorPolicy>, AllocPolicy> ComplexFieldType;	\
									\
 typedef CPSfermion4D<ScalarComplexType, FourDpolicy<DynamicFlavorPolicy>, AllocPolicy> ScalarFermionFieldType; \
 typedef CPScomplex4D<ScalarComplexType, FourDpolicy<DynamicFlavorPolicy>, AllocPolicy> ScalarComplexFieldType;	\
									\
 typedef GridSIMDSourcePolicies SourcePolicies;				\
									\
 ALLOCATOR_MACRO(NAME);							\
 enum { GPARITY=IS_GPARITY_POLICY };					\
};


//Apply the template!
A2APOLICIES_SIMD_TEMPLATE(A2ApoliciesSIMDdoubleAutoAlloc, 0, BaseGridPolicies, SET_A2AVECTOR_AUTOMATIC_ALLOC);
A2APOLICIES_SIMD_TEMPLATE(A2ApoliciesSIMDdoubleManualAlloc, 0, BaseGridPolicies, SET_A2AVECTOR_MANUAL_ALLOC);
A2APOLICIES_SIMD_TEMPLATE(A2ApoliciesSIMDdoubleAutoAllocGparity, 1, BaseGridPoliciesGparity, SET_A2AVECTOR_AUTOMATIC_ALLOC);
A2APOLICIES_SIMD_TEMPLATE(A2ApoliciesSIMDdoubleManualAllocGparity, 1, BaseGridPoliciesGparity, SET_A2AVECTOR_MANUAL_ALLOC);

#endif

CPS_END_NAMESPACE

#endif