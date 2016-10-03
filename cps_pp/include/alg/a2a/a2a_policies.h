#ifndef _A2A_POLICIES_H
#define _A2A_POLICIES_H

#include<alg/a2a/CPSfield.h>
CPS_START_NAMESPACE

//Type policies needed for sources

#ifdef USE_GRID
struct GridSIMDSourcePolicies{
  typedef Grid::vComplexD ComplexType;
  typedef ThreeDSIMDPolicy DimensionPolicy;
  typedef Aligned128AllocPolicy AllocPolicy;
};
struct GridSIMDSourcePoliciesSingle{
  typedef Grid::vComplexF ComplexType;
  typedef ThreeDSIMDPolicy DimensionPolicy;
  typedef Aligned128AllocPolicy AllocPolicy;
};
#endif

struct StandardSourcePolicies{
  typedef cps::ComplexD ComplexType;
  typedef SpatialPolicy DimensionPolicy;
  typedef StandardAllocPolicy AllocPolicy;
};

template<typename mf_Complex, typename mf_Complex_class>
struct _deduce_source_policies{};

template<typename T>
struct _deduce_source_policies<std::complex<T>, complex_double_or_float_mark>{
  typedef StandardSourcePolicies SourcePolicies;
};
#ifdef USE_GRID
template<>
struct _deduce_source_policies<Grid::vComplexD, grid_vector_complex_mark>{
  typedef GridSIMDSourcePolicies SourcePolicies;
};
template<>
struct _deduce_source_policies<Grid::vComplexF, grid_vector_complex_mark>{
  typedef GridSIMDSourcePoliciesSingle SourcePolicies;
};
#endif


//Deduction of fermion field properties given a complex type class. Don't have to use them but it can be useful
template<typename mf_Complex_class>
struct _deduce_a2a_dim_alloc_policies{};

template<>
struct _deduce_a2a_dim_alloc_policies<complex_double_or_float_mark>{
  typedef FourDpolicy DimensionPolicy;
  typedef StandardAllocPolicy AllocPolicy;
};

template<>
struct _deduce_a2a_dim_alloc_policies<grid_vector_complex_mark>{
  typedef FourDSIMDPolicy DimensionPolicy;
  typedef Aligned128AllocPolicy AllocPolicy;
};

template<typename mf_Complex, typename mf_Complex_class>
struct _deduce_scalar_complex_type{};

template<typename mf_Complex>
struct _deduce_scalar_complex_type<mf_Complex, complex_double_or_float_mark>{
  typedef mf_Complex ScalarComplexType;
};

template<typename mf_Complex>
struct _deduce_scalar_complex_type<mf_Complex, grid_vector_complex_mark>{
  typedef typename mf_Complex::scalar_type ScalarComplexType;
};

template<typename mf_Complex_class>
struct _deduce_double_single_variants{};

template<>
struct _deduce_double_single_variants<complex_double_or_float_mark>{
  typedef std::complex<double> ComplexTypeD;
  typedef std::complex<float> ComplexTypeF;
};
#ifdef USE_GRID
template<>
struct _deduce_double_single_variants<grid_vector_complex_mark>{
  typedef Grid::vComplexD ComplexTypeD;
  typedef Grid::vComplexF ComplexTypeF;
};
#endif


template<typename mf_Complex>
class _deduce_a2a_field_policies{
public:
  typedef mf_Complex ComplexType; //Can be SIMD-vectorized or scalar complex. Used internally.
private:
  typedef typename ComplexClassify<mf_Complex>::type ComplexClass;
  typedef typename _deduce_a2a_dim_alloc_policies<ComplexClass>::DimensionPolicy DimensionPolicy;
public:
  typedef typename _deduce_a2a_dim_alloc_policies<ComplexClass>::AllocPolicy AllocPolicy;
  typedef typename _deduce_double_single_variants<ComplexClass>::ComplexTypeD ComplexTypeD;
  typedef typename _deduce_double_single_variants<ComplexClass>::ComplexTypeF ComplexTypeF;
  typedef typename _deduce_scalar_complex_type<ComplexType, ComplexClass>::ScalarComplexType ScalarComplexType; //scalarized version of ComplexType if SIMD-vectorized, otherwise the same
  typedef CPSfermion4D<ComplexType, DimensionPolicy, DynamicFlavorPolicy, AllocPolicy> FermionFieldType;
  typedef CPScomplex4D<ComplexType, DimensionPolicy, DynamicFlavorPolicy, AllocPolicy> ComplexFieldType;
  typedef typename _deduce_source_policies<ComplexType,ComplexClass>::SourcePolicies SourcePolicies;
};

  
#ifdef USE_GRID
CPS_END_NAMESPACE
#include<util/lattice/fgrid.h>
CPS_START_NAMESPACE

struct GridA2APoliciesBase{
  //Extra policies needed internally by Grid implementations
#ifdef USE_GRID_GPARITY
  typedef FgridGparityMobius FgridFclass;
  typedef GnoneFgridGparityMobius FgridGFclass;
  typedef Grid::QCD::GparityMobiusFermionD GridDirac;
  typedef Grid::QCD::GparityMobiusFermionF GridDiracF; //single prec
  enum { FGRID_CLASS_NAME=F_CLASS_GRID_GPARITY_MOBIUS };
#else
  typedef FgridMobius FgridFclass;
  typedef GnoneFgridMobius FgridGFclass;
  typedef Grid::QCD::MobiusFermionD GridDirac;
  typedef Grid::QCD::MobiusFermionF GridDiracF;
  enum { FGRID_CLASS_NAME=F_CLASS_GRID_MOBIUS };
#endif

  typedef typename GridDirac::FermionField GridFermionField;
  typedef typename GridDiracF::FermionField GridFermionFieldF;
};

template<typename BaseA2Apolicies>
struct GridA2APolicies{
  //Inherit the base's generic A2A policies
  typedef typename BaseA2Apolicies::ComplexType ComplexType;
  typedef typename BaseA2Apolicies::ComplexTypeD ComplexTypeD;
  typedef typename BaseA2Apolicies::ComplexTypeF ComplexTypeF;
  typedef typename BaseA2Apolicies::ScalarComplexType ScalarComplexType;
  typedef typename BaseA2Apolicies::FermionFieldType FermionFieldType;
  typedef typename BaseA2Apolicies::ComplexFieldType ComplexFieldType;
  typedef typename BaseA2Apolicies::SourcePolicies SourcePolicies;
  typedef typename BaseA2Apolicies::AllocPolicy AllocPolicy;
  
  typedef typename GridA2APoliciesBase::FgridFclass FgridFclass;
  typedef typename GridA2APoliciesBase::FgridGFclass FgridGFclass;
  typedef typename GridA2APoliciesBase::GridDirac GridDirac;
  typedef typename GridA2APoliciesBase::GridFermionField GridFermionField;
  typedef typename GridA2APoliciesBase::GridDiracF GridDiracF;
  typedef typename GridA2APoliciesBase::GridFermionFieldF GridFermionFieldF;
  
  enum { FGRID_CLASS_NAME=GridA2APoliciesBase::FGRID_CLASS_NAME };
};

#endif

CPS_END_NAMESPACE

#endif
