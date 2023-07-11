#ifndef _CPS_MATRIX_FIELD_H__
#define _CPS_MATRIX_FIELD_H__

#include "CPSfield.h"
#include "CPSfield_utils.h"
#include <alg/a2a/lattice/spin_color_matrices.h>
#include <alg/a2a/utils/utils_complex.h>

//CPSfields of SIMD-vectorized matrices and associated functionality

CPS_START_NAMESPACE 

//Definition of CPSmatrixField
template<typename VectorMatrixType>
using CPSmatrixField = CPSfield<VectorMatrixType,1, FourDSIMDPolicy<OneFlavorPolicy>, CPSfieldDefaultAllocPolicy>;

template<typename VectorMatrixType>
double CPSmatrixFieldNorm2(const CPSmatrixField<VectorMatrixType> &f);

//For testRandom
template<typename T>
class _testRandom<T, typename std::enable_if<isCPSsquareMatrix<T>::value, void>::type>{
public:
  static void rand(T* f, size_t fsize, const Float hi, const Float lo);
};

#include "implementation/CPSmatrixField_meta.tcc"
#include "implementation/CPSmatrixField_functors.tcc"
#include "implementation/CPSmatrixField_func_templates.tcc"

template<typename VectorMatrixType>
inline auto Trace(const CPSmatrixField<VectorMatrixType> &a)->decltype( unop_v(a, _trV<VectorMatrixType>()) );

template<int Index, typename VectorMatrixType>
inline auto TraceIndex(const CPSmatrixField<VectorMatrixType> &a)->decltype( unop_v(a, _trIndexV<Index,VectorMatrixType>()) );

template<int Index, typename VectorMatrixType>
inline void TraceIndex(CPSmatrixField<typename _trIndexV<Index,VectorMatrixType>::OutputType > &out,  const CPSmatrixField<VectorMatrixType> &in);

template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<CPSflavorMatrix<ComplexType> > > ColorTrace(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in);

template<typename ComplexType>
inline void ColorTrace(CPSmatrixField<CPSspinMatrix<CPSflavorMatrix<ComplexType> > > &out, const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in);

//Trace over two indices of a nested matrix. Requires  Index1 < Index2
template<int Index1, int Index2, typename VectorMatrixType>
inline auto TraceTwoIndices(const CPSmatrixField<VectorMatrixType> &a)->decltype( unop_v(a, _trTwoIndicesV<Index1,Index2,VectorMatrixType>()) );

template<int Index1, int Index2, typename VectorMatrixType>
inline void TraceTwoIndices(CPSmatrixField<typename _trTwoIndicesV<Index1,Index2,VectorMatrixType>::OutputType > &out, const CPSmatrixField<VectorMatrixType> &in);

template<typename ComplexType>
inline CPSmatrixField<CPScolorMatrix<ComplexType> > SpinFlavorTrace(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in);

template<typename ComplexType>
inline void SpinFlavorTrace(CPSmatrixField<CPScolorMatrix<ComplexType> > &out, const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in);

template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> Transpose(const CPSmatrixField<VectorMatrixType> &a);

template<int Index, typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> TransposeOnIndex(const CPSmatrixField<VectorMatrixType> &in);

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > TransposeColor(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in);

template<int Index, typename VectorMatrixType>
inline void TransposeOnIndex(CPSmatrixField<VectorMatrixType> &out, const CPSmatrixField<VectorMatrixType> &in);

template<typename ComplexType>
inline void TransposeColor(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &out, const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in);

//Complex conjugate
template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> cconj(const CPSmatrixField<VectorMatrixType> &a);

//Hermitian conjugate
template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> Dagger(const CPSmatrixField<VectorMatrixType> &a);

//Left multiplication by gamma matrix
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > gl_r(const CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > gl_r(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline void gl_r(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &out, const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);

//Right multiplication by gamma matrix
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > gr_r(const CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > gr_r(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline void gr_r(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &out, const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);

//Left multiplication by gamma(dir)gamma(5)
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > glAx_r(const CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > glAx_r(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline void glAx_r(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &out, const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);

//Right multiplication by gamma(dir)gamma(5)
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > grAx_r(const CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > grAx_r(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline void grAx_r(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &out, const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);

template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> operator*(const CPSmatrixField<VectorMatrixType> &a, const CPSmatrixField<VectorMatrixType> &b);

//complex * matrixField
template<typename ScalarType, typename VectorMatrixType, typename std::enable_if< is_complex_double_or_float<ScalarType>::value, int>::type dummy = 0  >
inline CPSmatrixField<VectorMatrixType> operator*(const ScalarType &a, const CPSmatrixField<VectorMatrixType> &b);

//complex field * matrix field
template<typename VectorScalarType, typename VectorMatrixType, typename std::enable_if< is_grid_vector_complex<VectorScalarType>::value, int>::type dummy = 0>
inline CPSmatrixField<VectorMatrixType> operator*(const CPSmatrixField<VectorScalarType> &a, const CPSmatrixField<VectorMatrixType> &b);

template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> operator+(const CPSmatrixField<VectorMatrixType> &a, const CPSmatrixField<VectorMatrixType> &b);

template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> operator-(const CPSmatrixField<VectorMatrixType> &a, const CPSmatrixField<VectorMatrixType> &b);

//Trace(a*b) = \sum_{ij} a_{ij}b_{ji}
template<typename VectorMatrixType>
inline CPSmatrixField<typename _traceProdV<VectorMatrixType>::OutputType> Trace(const CPSmatrixField<VectorMatrixType> &a, const CPSmatrixField<VectorMatrixType> &b);

//in -> in
template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> & unit(CPSmatrixField<VectorMatrixType> &in);

//in -> i * in
template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> & timesI(CPSmatrixField<VectorMatrixType> &in);

//in -> -i * in
template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> & timesMinusI(CPSmatrixField<VectorMatrixType> &in);

//in -> -in
template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> & timesMinusOne(CPSmatrixField<VectorMatrixType> &in);

//in -> unit matrix
template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> & setUnit(CPSmatrixField<VectorMatrixType> &inout);

//Left multiplication by gamma matrix
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > & gl(CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & gl(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);

//Right multiplication by gamma matrix
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > & gr(CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & gr(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);

//Left multiplication by gamma^dir gamma^5
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > & glAx(CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & glAx(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);

//Right multiplication by gamma^dir gamma^5
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > & grAx(CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & grAx(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);


//Left multiplication by flavor matrix
template<typename ComplexType>
inline CPSmatrixField<CPSflavorMatrix<ComplexType> > & pl(CPSmatrixField<CPSflavorMatrix<ComplexType> > &in, const FlavorMatrixType type);

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & pl(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const FlavorMatrixType type);

//Right multiplication by flavor matrix
template<typename ComplexType>
inline CPSmatrixField<CPSflavorMatrix<ComplexType> > & pr(CPSmatrixField<CPSflavorMatrix<ComplexType> > &in, const FlavorMatrixType type);

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & pr(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const FlavorMatrixType type);

//Tr(a * b)
template<typename VectorMatrixType>			
CPSmatrixField<typename VectorMatrixType::scalar_type> Trace(const CPSmatrixField<VectorMatrixType> &a, const VectorMatrixType &b);

//Sum the matrix field over sides on this node
//Slow implementation
template<typename VectorMatrixType>			
VectorMatrixType localNodeSumSimple(const CPSmatrixField<VectorMatrixType> &a);
//Fast implementation
template<typename VectorMatrixType>			
VectorMatrixType localNodeSum(const CPSmatrixField<VectorMatrixType> &a);

//Simultaneous global and SIMD reduction (if applicable)
template<typename VectorMatrixType>			
inline auto globalSumReduce(const CPSmatrixField<VectorMatrixType> &a) ->decltype(Reduce(localNodeSum(a)));

//Perform the local-node 3d slice sum
//Output is an array of size GJP.TnodeSites()  (i.e. the local time coordinate)
//Slow implementation
template<typename VectorMatrixType>			
ManagedVector<VectorMatrixType>  localNodeSpatialSumSimple(const CPSmatrixField<VectorMatrixType> &a);

//Perform the local-node 3d slice sum
//Output is an array of size GJP.TnodeSites()  (i.e. the local time coordinate)
//Fast implementation
template<typename VectorMatrixType>			
ManagedVector<VectorMatrixType> localNodeSpatialSum(const CPSmatrixField<VectorMatrixType> &a);

//Unpack a CPSmatrixField into a linear array format
template<typename VectorMatrixType>
CPSfield<typename VectorMatrixType::scalar_type, VectorMatrixType::nScalarType(), FourDSIMDPolicy<OneFlavorPolicy>, CPSfieldDefaultAllocPolicy> linearUnpack(const CPSmatrixField<VectorMatrixType> &in);

//Pack a CPSmatrixField from a linear array format
template<typename VectorMatrixType>
CPSmatrixField<VectorMatrixType> linearRepack(const CPSfield<typename VectorMatrixType::scalar_type, VectorMatrixType::nScalarType(), FourDSIMDPolicy<OneFlavorPolicy>, CPSfieldDefaultAllocPolicy> &in);

//Cshift a field in the direction mu, for a field with complex-conjugate BCs in chosen directions
//Cshift(+mu) : f'(x) = f(x-\hat mu)
//Cshift(-mu) : f'(x) = f(x+\hat mu)
template<typename VectorMatrixType>
CPSmatrixField<VectorMatrixType> CshiftCconjBc(const CPSmatrixField<VectorMatrixType> &field, int mu, int pm, const std::vector<int> &conj_dirs = defaultConjDirs());

//Functionality for testing the gauge fixing matrices against the stopping condition used by CPS
namespace gaugeFixTest{
  inline CPSmatrixField<CPScolorMatrix<Grid::vComplexD> > getUmu(const CPSfield<Grid::vComplexD,4*9,FourDSIMDPolicy<OneFlavorPolicy> > &U,
							  const int mu){
    CPSmatrixField<CPScolorMatrix<Grid::vComplexD> > out(U.getDimPolParams());
    CPSautoView(out_v,out,HostWrite);
    CPSautoView(in_v,U,HostRead);
#pragma omp parallel for
    for(size_t i=0;i<U.nsites();i++){
      CPScolorMatrix<Grid::vComplexD> *out_p = out_v.site_ptr(i);
      CPScolorMatrix<Grid::vComplexD> const* in_p = (CPScolorMatrix<Grid::vComplexD> const*)( in_v.site_ptr(i) + 9*mu );
      *out_p = *in_p;    
    }
    return out;
  }
  //       Lj(n) = g(n) L0j(n) g^dag(n+j)
  inline CPSmatrixField<CPScolorMatrix<Grid::vComplexD> > computeLj(const CPSmatrixField<CPScolorMatrix<Grid::vComplexD> > &L0j,
							     const CPSmatrixField<CPScolorMatrix<Grid::vComplexD> > &g_orig,
							     const CPSmatrixField<CPScolorMatrix<Grid::vComplexD> > &g_plus_j,
							     int j){
    return g_orig * L0j * Dagger(g_plus_j);
  }

  //               3   +        
  //       A(n) = sum[Lj(n) + Lj(n-j)],
  //               j
  inline CPSmatrixField<CPScolorMatrix<Grid::vComplexD> > computeA(const std::vector< CPSmatrixField<CPScolorMatrix<Grid::vComplexD> > > &L0,
							    const CPSmatrixField<CPScolorMatrix<Grid::vComplexD> > &g_orig, int orthog_dir = -1){
    CPSmatrixField<CPScolorMatrix<Grid::vComplexD> > out(g_orig.getDimPolParams());
    out.zero();
    for(int j=0;j<4;j++){
      if(j != orthog_dir){
	CPSmatrixField<CPScolorMatrix<Grid::vComplexD> > g_plus_j = CshiftCconjBc(g_orig, j, -1);
	auto Lj = computeLj(L0[j],g_orig,g_plus_j,j);
	auto Ljm1 = CshiftCconjBc(Lj, j, +1); //sign is direction of data motion. Func is G-parity aware
	out = out + Dagger(Lj) + Ljm1;
      }
    }
    return out;
  }
  //
  //                       +     1         +                                 //
  //        B(n) = A(n) - A(n) - - Tr(A - A )                        (1)     //
  //                             3                                           //
  // This is equivalent to 2i \partial_\mu A_\mu(x)  
  // with A_\mu = \sum_\mu -i/2 [ U'_\mu(x) - U'_\mu^dag(x) - 1/Nc Tr( U'_\mu(x) - U'_\mu^dag(x) ) ]
  // U'_\mu(x) = g(x)U_\mu(x)g^dag(x+\mu)  
  inline CPSmatrixField<CPScolorMatrix<Grid::vComplexD> > computeB(const std::vector< CPSmatrixField<CPScolorMatrix<Grid::vComplexD> > > &L0,
							    const CPSmatrixField<CPScolorMatrix<Grid::vComplexD> > &g_orig, int orthog_dir = -1){
    auto A = computeA(L0,g_orig,orthog_dir);
    auto Adag = Dagger(A);
    auto AmAdag = A - Adag;
    CPSmatrixField<CPScolorMatrix<Grid::vComplexD> > one(g_orig.getDimPolParams()); setUnit(one);

    return AmAdag - Grid::ComplexD(1./3) * (Trace(AmAdag)*one);
  }

  //Evaluate the convergence metric used in CPS:   norm2(B)/(2 Nc^2 V)
  //orthog_dir:  Landau:-1  Coulomb_T:3
  inline double delta(Lattice &lat, const SIMDdims<4> &simd_dims, int orthog_dir=-1){
    NullObject null_obj;
    CPSfield<cps::ComplexD,4*9,FourDpolicy<OneFlavorPolicy> > cps_gauge_s((cps::ComplexD*)lat.GaugeField(),null_obj);
    CPSfield<Grid::vComplexD,4*9,FourDSIMDPolicy<OneFlavorPolicy> > cps_gauge_v(simd_dims); 
    cps_gauge_v.importField(cps_gauge_s);

    CPSfield<cps::ComplexD,9,FourDpolicy<OneFlavorPolicy> > cps_gfmat_s = getGaugeFixingMatrixFlavor0(lat);
    CPSfield<Grid::vComplexD,9,FourDSIMDPolicy<OneFlavorPolicy> > cps_gfmat_v(simd_dims); 
    cps_gfmat_v.importField(cps_gfmat_s);
  
    typedef CPSmatrixField<CPScolorMatrix<Grid::vComplexD> > CPSgfMatrixField;
    CPSgfMatrixField cps_gfmat = linearRepack<CPScolorMatrix<Grid::vComplexD> >(cps_gfmat_v);
  
    std::vector< CPSgfMatrixField > Umu(4, simd_dims);
    for(int i=0;i<4;i++) Umu[i] = getUmu(cps_gauge_v,i);

    CPSgfMatrixField B = computeB(Umu, cps_gfmat, orthog_dir);

    double V=1.;
    for(int i=0;i<4;i++) V*=GJP.Nodes(i)*GJP.NodeSites(i);

    //CPS stopping condition uses   \sum_x |B(x)|^2 < 2 Nc^2 V c   where c is in the input stopping condition
    //Four Coulomb gauge this is evaluated separately for every timeslice (with V the timeslice volume),
    //so what we compute here is the average over the Lt stopping conditions
    return CPSmatrixFieldNorm2(B)/18./V;
  }
  inline double delta(Lattice &lat, int orthog_dir=-1){
    int nsimd = Grid::vComplexD::Nsimd();
    typename SIMDpolicyBase<4>::ParamType simd_dims;
    SIMDpolicyBase<4>::SIMDdefaultLayout(simd_dims,nsimd,2);
    return delta(lat, simd_dims, orthog_dir);
  }
};

#include "implementation/CPSmatrixField_impl.tcc"

CPS_END_NAMESPACE

#endif
