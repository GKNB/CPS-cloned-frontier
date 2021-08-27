#ifndef _CPS_MATRIX_FIELD_H__
#define _CPS_MATRIX_FIELD_H__

#include "CPSfield.h"
#include <alg/a2a/lattice/spin_color_matrices.h>
#include <alg/a2a/lattice/spin_color_matrices_SIMT.h>

//CPSfields of SIMD-vectorized matrices and associated functionality

CPS_START_NAMESPACE 

template<typename VectorMatrixType>
using CPSmatrixField = CPSfield<VectorMatrixType,1, FourDSIMDPolicy<OneFlavorPolicy>, Aligned128AllocPolicy>;


//These structs allow the same interface for operators that are applicable for matrix and complex types
define_test_has_enum(isDerivedFromCPSsquareMatrix);

struct vector_matrix_mark{}; //Matrix of SIMD data
template<typename T>
struct MatrixTypeClassify{
  typedef typename TestElem< is_grid_vector_complex<T>::value, grid_vector_complex_mark,
			     TestElem< has_enum_isDerivedFromCPSsquareMatrix<T>::value && is_grid_vector_complex<typename T::scalar_type>::value, vector_matrix_mark, 
				       LastElem
				       >
			     >::type type;
};


template<typename T, typename Tclass>
struct getScalarType{};

template<typename T>
struct getScalarType<T, grid_vector_complex_mark>{
  typedef T type;
};
template<typename T>
struct getScalarType<T, vector_matrix_mark>{
  typedef typename T::scalar_type type;
};


//For testRandom
template<typename T>
class _testRandom<T, typename std::enable_if<isCPSsquareMatrix<T>::value, void>::type>{
public:
  static void rand(T* f, size_t fsize, const Float hi, const Float lo){
    static_assert(sizeof(T) == T::nScalarType()*sizeof(typename T::scalar_type));
    _testRandom<typename T::scalar_type>::rand( (typename T::scalar_type*)f, T::nScalarType()*fsize, hi, lo);
  }
};


/*
  Expect a functor that acts on SIMD data, pulling out the SIMD lane as appropriate
  Example:

  template<typename VectorMatrixType>
  struct _trV{
     typedef typename VectorMatrixType::scalar_type OutputType;  //must contain OutputType typedef
     accelerator_inline void operator()(OutputType &out, const VectorMatrixType &in, const int lane) const{     //must take SIMD lane as parameter
       SIMT<OutputType>::write(out,0,lane); //Should use SIMT accessors
       _LaneRecursiveTraceImpl<OutputType, VectorMatrixType, cps_square_matrix_mark>::doit(out, in, lane);
     }
  };
*/  

template<typename T, typename Functor>
auto unop_v(const CPSmatrixField<T> &in, const Functor &l)-> CPSmatrixField<typename Functor::OutputType>{
  using namespace Grid;
  constexpr int nsimd = T::scalar_type::Nsimd();
  CPSmatrixField<typename Functor::OutputType> out(in.getDimPolParams());
  auto ov = out.view();
  auto iv = in.view();
  accelerator_for(x4d, iv.size(), nsimd,
		    {
		      int lane = Grid::acceleratorSIMTlane(nsimd);
		      l(*ov.site_ptr(x4d), *iv.site_ptr(x4d), lane);
		    }
		    );
  return out;
}

template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _trV{
  typedef typename VectorMatrixType::scalar_type OutputType;
  accelerator_inline void operator()(OutputType &out, const VectorMatrixType &in, const int lane) const{ 
    Trace(out, in, lane);
  }
};

template<int Index, typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _trIndexV{
  typedef typename _PartialTraceFindReducedType<VectorMatrixType,Index>::type OutputType;
  accelerator_inline void operator()(OutputType &out, const VectorMatrixType &in, const int lane) const{ 
    TraceIndex<Index>(out, in, lane);
  }
};

template<int Index1, int Index2, typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _trTwoIndicesV{
  typedef typename _PartialDoubleTraceFindReducedType<VectorMatrixType,Index1,Index2>::type OutputType;
  accelerator_inline void operator()(OutputType &out, const VectorMatrixType &in, const int lane) const{ 
    TraceTwoIndices<Index1,Index2>(out, in, lane);
  }
};

template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _transposeV{
  typedef VectorMatrixType OutputType;
  accelerator_inline void operator()(VectorMatrixType &out, const VectorMatrixType &in, const int lane) const{ 
    Transpose(out, in, lane);
  }
};

template<int Index, typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _transIdx{
  typedef VectorMatrixType OutputType;
  accelerator_inline void operator()(VectorMatrixType &out, const VectorMatrixType &in, const int lane) const{ 
    TransposeOnIndex<Index>(out, in, lane);
  }
};


template<typename ComplexType>
struct _gl_r_V{
  typedef CPSspinMatrix<ComplexType> OutputType;
  int dir;
  _gl_r_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinMatrix<ComplexType> &out, const CPSspinMatrix<ComplexType> &in, const int lane) const{ 
    gl_r<gl_spinMatrixIterator<ComplexType> >(out, in, dir, lane);
  }
};
template<typename ComplexType>
struct _gl_r_scf_V{
  typedef CPSspinColorFlavorMatrix<ComplexType> OutputType;
  int dir;
  _gl_r_scf_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &out, const CPSspinColorFlavorMatrix<ComplexType> &in, const int lane) const{ 
    gl_r<gl_spinColorFlavorMatrixIterator<ComplexType> >(out, in, dir, lane);
  }
};

template<typename ComplexType>
struct _gr_r_V{
  typedef CPSspinMatrix<ComplexType> OutputType;
  int dir;
  _gr_r_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinMatrix<ComplexType> &out, const CPSspinMatrix<ComplexType> &in, const int lane) const{ 
    gr_r<gl_spinMatrixIterator<ComplexType> >(out, in, dir, lane);
  }
};
template<typename ComplexType>
struct _gr_r_scf_V{
  typedef CPSspinColorFlavorMatrix<ComplexType> OutputType;
  int dir;
  _gr_r_scf_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &out, const CPSspinColorFlavorMatrix<ComplexType> &in, const int lane) const{ 
    gr_r<gl_spinColorFlavorMatrixIterator<ComplexType> >(out, in, dir, lane);
  }
};



template<typename ComplexType>
struct _glAx_r_V{
  typedef CPSspinMatrix<ComplexType> OutputType;
  int dir;
  _glAx_r_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinMatrix<ComplexType> &out, const CPSspinMatrix<ComplexType> &in, const int lane) const{ 
    glAx_r<gl_spinMatrixIterator<ComplexType> >(out, in, dir, lane);
  }
};
template<typename ComplexType>
struct _glAx_r_scf_V{
  typedef CPSspinColorFlavorMatrix<ComplexType> OutputType;
  int dir;
  _glAx_r_scf_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &out, const CPSspinColorFlavorMatrix<ComplexType> &in, const int lane) const{ 
    glAx_r<gl_spinColorFlavorMatrixIterator<ComplexType> >(out, in, dir, lane);
  }
};


template<typename ComplexType>
struct _grAx_r_V{
  typedef CPSspinMatrix<ComplexType> OutputType;
  int dir;
  _grAx_r_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinMatrix<ComplexType> &out, const CPSspinMatrix<ComplexType> &in, const int lane) const{ 
    grAx_r<gl_spinMatrixIterator<ComplexType> >(out, in, dir, lane);
  }
};
template<typename ComplexType>
struct _grAx_r_scf_V{
  typedef CPSspinColorFlavorMatrix<ComplexType> OutputType;
  int dir;
  _grAx_r_scf_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &out, const CPSspinColorFlavorMatrix<ComplexType> &in, const int lane) const{ 
    grAx_r<gl_spinColorFlavorMatrixIterator<ComplexType> >(out, in, dir, lane);
  }
};



template<typename VectorMatrixType>
inline auto Trace(const CPSmatrixField<VectorMatrixType> &a)->decltype( unop_v(a, _trV<VectorMatrixType>()) ){
  return unop_v(a, _trV<VectorMatrixType>());
}

template<int Index, typename VectorMatrixType>
inline auto TraceIndex(const CPSmatrixField<VectorMatrixType> &a)->decltype( unop_v(a, _trIndexV<Index,VectorMatrixType>()) ){
  return unop_v(a, _trIndexV<Index,VectorMatrixType>());
}

template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<CPSflavorMatrix<ComplexType> > > ColorTrace(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in){
  return TraceIndex<1>(in);
}

//Trace over two indices of a nested matrix. Requires  Index1 < Index2
template<int Index1, int Index2, typename VectorMatrixType>
inline auto TraceTwoIndices(const CPSmatrixField<VectorMatrixType> &a)->decltype( unop_v(a, _trTwoIndicesV<Index1,Index2,VectorMatrixType>()) ){
  return unop_v(a, _trTwoIndicesV<Index1,Index2,VectorMatrixType>());
}

template<typename ComplexType>
inline CPSmatrixField<CPScolorMatrix<ComplexType> > SpinFlavorTrace(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in){
  return TraceTwoIndices<0,2>(in);
}

template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> Transpose(const CPSmatrixField<VectorMatrixType> &a){
  return unop_v(a, _transposeV<VectorMatrixType>());
}

template<int Index, typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> TransposeOnIndex(CPSmatrixField<VectorMatrixType> &in){
  return unop_v(in, _transIdx<Index,VectorMatrixType>());
}

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > TransposeColor(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in){
  return TransposeOnIndex<1>(in);
}

//Left multiplication by gamma matrix
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > gl_r(const CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir){
  return unop_v(in,_gl_r_V<ComplexType>(dir));
}
template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > gl_r(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir){
  return unop_v(in,_gl_r_scf_V<ComplexType>(dir));
}

//Right multiplication by gamma matrix
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > gr_r(const CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir){
  return unop_v(in,_gr_r_V<ComplexType>(dir));
}
template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > gr_r(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir){
  return unop_v(in,_gr_r_scf_V<ComplexType>(dir));
}


//Left multiplication by gamma(dir)gamma(5)
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > glAx_r(const CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir){
  return unop_v(in,_glAx_r_V<ComplexType>(dir));
}
template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > glAx_r(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir){
  return unop_v(in,_glAx_r_scf_V<ComplexType>(dir));
}

//Right multiplication by gamma(dir)gamma(5)
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > grAx_r(const CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir){
  return unop_v(in,_grAx_r_V<ComplexType>(dir));
}
template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > grAx_r(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir){
  return unop_v(in,_grAx_r_scf_V<ComplexType>(dir));
}




/*
  Expect functor of the form, e.g.

  template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
  struct _subV{
    typedef VectorMatrixType OutputType;
    accelerator_inline void operator()(VectorMatrixType &out, const VectorMatrixType &a, const VectorMatrixType &b, const int lane) const{ 
      sub(out, a, b, lane);
    }
  };
*/
template<typename T, typename Functor>
auto binop_v(const CPSmatrixField<T> &a, const CPSmatrixField<T> &b, const Functor &l)-> CPSmatrixField<typename Functor::OutputType>{
  using namespace Grid;
  assert(a.size() == b.size());
  constexpr int nsimd = getScalarType<T, typename MatrixTypeClassify<T>::type>::type::Nsimd();
  CPSmatrixField<typename Functor::OutputType> out(a.getDimPolParams());
  auto ov = out.view();
  auto av = a.view(), bv = b.view();
  
  accelerator_for(x4d, av.size(), nsimd,
		    {
		      int lane = Grid::acceleratorSIMTlane(nsimd);
		      l(*ov.site_ptr(x4d), *av.site_ptr(x4d), *bv.site_ptr(x4d), lane);
		    }
		    );
  return out;
}


template<typename VectorMatrixType>
struct _timesV{
  typedef VectorMatrixType OutputType;
  accelerator_inline void operator()(VectorMatrixType &out, const VectorMatrixType &a, const VectorMatrixType &b, const int lane) const{ 
    mult(out, a, b, lane);
  }
};

template<typename VectorMatrixType>
struct _addV{
  typedef VectorMatrixType OutputType;
  accelerator_inline void operator()(VectorMatrixType &out, const VectorMatrixType &a, const VectorMatrixType &b, const int lane) const{ 
    add(out, a, b, lane);
  }
};

template<typename VectorMatrixType>
struct _subV{
  typedef VectorMatrixType OutputType;
  accelerator_inline void operator()(VectorMatrixType &out, const VectorMatrixType &a, const VectorMatrixType &b, const int lane) const{ 
    sub(out, a, b, lane);
  }
};

template<typename VectorMatrixType>
struct _traceProdV{
  typedef typename VectorMatrixType::scalar_type OutputType;
  accelerator_inline void operator()(OutputType &out, const VectorMatrixType &a, const VectorMatrixType &b, const int lane) const{ 
    Trace(out, a, b, lane);
  }
};



template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> operator*(const CPSmatrixField<VectorMatrixType> &a, const CPSmatrixField<VectorMatrixType> &b){
  return binop_v(a,b,_timesV<VectorMatrixType>());
}
template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> operator+(const CPSmatrixField<VectorMatrixType> &a, const CPSmatrixField<VectorMatrixType> &b){
  return binop_v(a,b,_addV<VectorMatrixType>());
}
template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> operator-(const CPSmatrixField<VectorMatrixType> &a, const CPSmatrixField<VectorMatrixType> &b){
  return binop_v(a,b,_subV<VectorMatrixType>());
}

//Trace(a*b) = \sum_{ij} a_{ij}b_{ji}
template<typename VectorMatrixType>
inline CPSmatrixField<typename _traceProdV<VectorMatrixType>::OutputType> Trace(const CPSmatrixField<VectorMatrixType> &a, const CPSmatrixField<VectorMatrixType> &b){
  return binop_v(a,b,_traceProdV<VectorMatrixType>());
}




/*
  Expect Functor of the form, e.g.

  template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
  struct _unitV{
    accelerator_inline void operator()(VectorMatrixType &out, const int lane) const{ 
      unit(out, lane);
    }
  };
*/
//For convenience this function returns a reference to the modified input field
template<typename T, typename Functor>
CPSmatrixField<T> & unop_self_v(CPSmatrixField<T> &m, const Functor &l){
  using namespace Grid;
  constexpr int nsimd = T::scalar_type::Nsimd();
  auto mv = m.view();
  accelerator_for(x4d, m.size(), nsimd,
		    {
		      int lane = Grid::acceleratorSIMTlane(nsimd);
		      l(*mv.site_ptr(x4d), lane);
		    }
		    );
  return m;
}

template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _unitV{
  accelerator_inline void operator()(VectorMatrixType &in, const int lane) const{ 
    unit(in, lane);
  }
};

template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _timesIV{
  accelerator_inline void operator()(VectorMatrixType &in, const int lane) const{ 
    timesI(in, in, lane);
  }
};

template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _timesMinusIV{
  accelerator_inline void operator()(VectorMatrixType &in, const int lane) const{ 
    timesMinusI(in, in, lane);
  }
};

template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _timesMinusOneV{
  accelerator_inline void operator()(VectorMatrixType &in, const int lane) const{ 
    timesMinusOne(in, in, lane);
  }
};


template<typename ComplexType>
struct _gl_V{
  int dir;
  _gl_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinMatrix<ComplexType> &in, const int lane) const{ 
    gl<gl_spinMatrixIterator<ComplexType> >(in, dir, lane);
  }
};
template<typename ComplexType>
struct _gl_scf_V{
  int dir;
  _gl_scf_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &in, const int lane) const{ 
    gl<gl_spinColorFlavorMatrixIterator<ComplexType> >(in, dir, lane);
  }
};

template<typename ComplexType>
struct _gr_V{
  int dir;
  _gr_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinMatrix<ComplexType> &in, const int lane) const{ 
    gr<gl_spinMatrixIterator<ComplexType> >(in, dir, lane);
  }
};
template<typename ComplexType>
struct _gr_scf_V{
  int dir;
  _gr_scf_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &in, const int lane) const{ 
    gr<gl_spinColorFlavorMatrixIterator<ComplexType> >(in, dir, lane);
  }
};


template<typename ComplexType>
struct _glAx_V{
  int dir;
  _glAx_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinMatrix<ComplexType> &in, const int lane) const{ 
    glAx<gl_spinMatrixIterator<ComplexType> >(in, dir, lane);
  }
};
template<typename ComplexType>
struct _glAx_scf_V{
  int dir;
  _glAx_scf_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &in, const int lane) const{ 
    glAx<gl_spinColorFlavorMatrixIterator<ComplexType> >(in, dir, lane);
  }
};

template<typename ComplexType>
struct _grAx_V{
  int dir;
  _grAx_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinMatrix<ComplexType> &in, const int lane) const{ 
    grAx<gl_spinMatrixIterator<ComplexType> >(in, dir, lane);
  }
};
template<typename ComplexType>
struct _grAx_scf_V{
  int dir;
  _grAx_scf_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &in, const int lane) const{ 
    grAx<gl_spinColorFlavorMatrixIterator<ComplexType> >(in, dir, lane);
  }
};


template<typename ComplexType>
struct _pl_V{
  const FlavorMatrixType type;
  _pl_V(const FlavorMatrixType type): type(type){}

  accelerator_inline void operator()(CPSflavorMatrix<ComplexType> &in, const int lane) const{ 
    pl<pl_flavorMatrixIterator<ComplexType> >(in, type, lane);
  }
};
template<typename ComplexType>
struct _pl_scf_V{
  const FlavorMatrixType type;
  _pl_scf_V(const FlavorMatrixType type): type(type){}

  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &in, const int lane) const{ 
    pl<pl_spinColorFlavorMatrixIterator<ComplexType> >(in, type, lane);
  }
};

template<typename ComplexType>
struct _pr_V{
  const FlavorMatrixType type;
  _pr_V(const FlavorMatrixType type): type(type){}

  accelerator_inline void operator()(CPSflavorMatrix<ComplexType> &in, const int lane) const{ 
    pr<pl_flavorMatrixIterator<ComplexType> >(in, type, lane);
  }
};
template<typename ComplexType>
struct _pr_scf_V{
  const FlavorMatrixType type;
  _pr_scf_V(const FlavorMatrixType type): type(type){}

  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &in, const int lane) const{ 
    pr<pl_spinColorFlavorMatrixIterator<ComplexType> >(in, type, lane);
  }
};




template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> & unit(CPSmatrixField<VectorMatrixType> &in){
  return unop_self_v(in,_unitV<VectorMatrixType>());
}

//in -> i * in
template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> & timesI(CPSmatrixField<VectorMatrixType> &in){
  return unop_self_v(in,_timesIV<VectorMatrixType>());
}

//in -> -i * in
template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> & timesMinusI(CPSmatrixField<VectorMatrixType> &in){
  return unop_self_v(in,_timesMinusIV<VectorMatrixType>());
}

//in -> -in
template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> & timesMinusOne(CPSmatrixField<VectorMatrixType> &in){
  return unop_self_v(in,_timesMinusOneV<VectorMatrixType>());
}

//Left multiplication by gamma matrix
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > & gl(CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir){
  return unop_self_v(in,_gl_V<ComplexType>(dir));
}
template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & gl(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir){
  return unop_self_v(in,_gl_scf_V<ComplexType>(dir));
}

//Right multiplication by gamma matrix
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > & gr(CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir){
  return unop_self_v(in,_gr_V<ComplexType>(dir));
}
template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & gr(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir){
  return unop_self_v(in,_gr_scf_V<ComplexType>(dir));
}

//Left multiplication by gamma^dir gamma^5
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > & glAx(CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir){
  return unop_self_v(in,_glAx_V<ComplexType>(dir));
}
template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & glAx(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir){
  return unop_self_v(in,_glAx_scf_V<ComplexType>(dir));
}

//Right multiplication by gamma^dir gamma^5
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > & grAx(CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir){
  return unop_self_v(in,_grAx_V<ComplexType>(dir));
}
template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & grAx(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir){
  return unop_self_v(in,_grAx_scf_V<ComplexType>(dir));
}


//Left multiplication by flavor matrix
template<typename ComplexType>
inline CPSmatrixField<CPSflavorMatrix<ComplexType> > & pl(CPSmatrixField<CPSflavorMatrix<ComplexType> > &in, const FlavorMatrixType type){
  return unop_self_v(in,_pl_V<ComplexType>(type));
}
template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & pl(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const FlavorMatrixType type){
  return unop_self_v(in,_pl_scf_V<ComplexType>(type));
}

//Right multiplication by flavor matrix
template<typename ComplexType>
inline CPSmatrixField<CPSflavorMatrix<ComplexType> > & pr(CPSmatrixField<CPSflavorMatrix<ComplexType> > &in, const FlavorMatrixType type){
  return unop_self_v(in,_pr_V<ComplexType>(type));
}
template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & pr(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const FlavorMatrixType type){
  return unop_self_v(in,_pr_scf_V<ComplexType>(type));
}


template<typename VectorMatrixType>			
CPSmatrixField<typename VectorMatrixType::scalar_type> Trace(const CPSmatrixField<VectorMatrixType> &a, const VectorMatrixType &b){
  using namespace Grid;
  CPSmatrixField<typename VectorMatrixType::scalar_type> out(a.getDimPolParams());
  
  //On NVidia GPUs the compiler fails if b is passed by value into the kernel due to "formal parameter space overflow"
  //As we can't assume b is allocated in UVM memory we need to either create a UVM copy or explicitly copy b to the GPU
#ifdef GPU_VEC
  VectorMatrixType* bptr = (VectorMatrixType*)device_alloc_check(sizeof(VectorMatrixType));
  copy_host_to_device(bptr, &b, sizeof(VectorMatrixType));
#else
  VectorMatrixType const* bptr = &b;
#endif  

  static const int nsimd = VectorMatrixType::scalar_type::Nsimd();
  auto av = a.view();
  auto ov = out.view();
  accelerator_for(x4d, a.size(), nsimd,
  		  {
		    int lane = Grid::acceleratorSIMTlane(nsimd);
		    Trace(*ov.site_ptr(x4d), *av.site_ptr(x4d), *bptr, lane);
  		  }
  		  );

#ifdef GPU_VEC
  device_free(bptr);
#endif

  return out;
}



//Sum the matrix field over sides on this node
template<typename VectorMatrixType>			
VectorMatrixType localNodeSumSimple(const CPSmatrixField<VectorMatrixType> &a){
  VectorMatrixType out = *a.site_ptr(size_t(0));
  for(size_t i=1;i<a.size();i++) out = out + *a.site_ptr(i);
  return out;
}


template<typename VectorMatrixType>			
VectorMatrixType localNodeSum(const CPSmatrixField<VectorMatrixType> &a){
  using namespace Grid;
  VectorMatrixType out; memset(&out, 0, sizeof(VectorMatrixType));

  //Want to introduce some paralellism into this thing
  //Parallelize over fundamental complex type
  typedef typename getScalarType<VectorMatrixType, typename MatrixTypeClassify<VectorMatrixType>::type>::type ScalarType; 

  static const size_t nscalar = sizeof(VectorMatrixType)/sizeof(ScalarType);
  size_t field_size = a.size();
  //Have vol/2 threads sum   x[i] + x[i + vol/2]
  //Then vol/4 for the next round, and so on
  static const int nsimd = ScalarType::Nsimd();
  
  assert(field_size % 2 == 0);

  ScalarType *tmp = (ScalarType*)managed_alloc_check(nscalar*field_size/2*sizeof(ScalarType));
  ScalarType *tmp2 = (ScalarType*)managed_alloc_check(nscalar*field_size/2*sizeof(ScalarType));

  ScalarType *into = tmp;

  auto av = a.view();
  accelerator_for(offset, nscalar * field_size/2, nsimd,
		  {
		    typedef SIMT<ScalarType> ACC;
		    //Map offset as  s + nscalar*x
		    size_t s = offset % nscalar;
		    size_t x = offset / nscalar;

		    ScalarType const* aa1_ptr = (ScalarType const *)av.site_ptr(x); //pointer to complex type
		    auto aa1 = ACC::read(aa1_ptr[s]);

		    ScalarType const* aa2_ptr = (ScalarType const *)av.site_ptr(x + field_size/2); //pointer to complex type
		    auto aa2 = ACC::read(aa2_ptr[s]);
		    
		    ACC::write(into[offset], aa1+aa2);
		  }
		  );
  
  field_size/=2;

  size_t iter = 0;
  while(field_size % 2 == 0){
    size_t work = nscalar * field_size/2;
    size_t threads = Grid::acceleratorThreads();
    if(work % threads != 0){ //not enough work left even for one workgroup, little point in accelerating (it also causes errors in oneAPI, at least for iris GPus)
      break;
    }
      
    //swap back and forth between the two temp buffers,
    ScalarType const* from = (iter % 2 == 0 ? tmp : tmp2);
    into = (iter % 2 == 0 ? tmp2 : tmp);
    
    //std::cout << "Iteration " << iter << " work=" << work << " nthread=" << Grid::acceleratorThreads() << std::endl;
    
    accelerator_for(offset, work, nsimd,
    		    {
    		      typedef SIMT<ScalarType> ACC;
		      
    		      ScalarType const* aa1_ptr = from + offset;
    		      auto aa1 = ACC::read(*aa1_ptr);
		      
    		      ScalarType const* aa2_ptr = from + offset + nscalar*field_size/2;
    		      auto aa2 = ACC::read(*aa2_ptr);
		      
    		      ACC::write(into[offset], aa1+aa2);
    		    }
    		    );
    field_size/=2;
    ++iter;
  }

  ScalarType *out_s = (ScalarType*)&out;
  for(size_t s=0;s<nscalar;s++)
    for(size_t i=0; i<field_size; i++)
      out_s[s] = out_s[s] + into[s+nscalar*i];
  
  managed_free(tmp);
  managed_free(tmp2);
  return out;
}

//Simultaneous global and SIMD reduction (if applicable)
template<typename VectorMatrixType>			
inline auto globalSumReduce(const CPSmatrixField<VectorMatrixType> &a)
  ->decltype(Reduce(localNodeSum(a)))
{
  VectorMatrixType lsum = localNodeSum(a);
  auto slsum = Reduce(lsum);
  globalSum(&slsum);
  return slsum;
}


//Perform the local-node 3d slice sum
//Output is an array of size GJP.TnodeSites()  (i.e. the local time coordinate)
template<typename VectorMatrixType>			
ManagedVector<VectorMatrixType>  localNodeSpatialSumSimple(const CPSmatrixField<VectorMatrixType> &a){
  int Lt_loc = GJP.TnodeSites();

  size_t field_size = a.size();
  assert(a.nodeSites(3) == Lt_loc);
  size_t field_size_3d = field_size/Lt_loc; 

  ManagedVector<VectorMatrixType> out(Lt_loc); 

  for(int t=0;t<Lt_loc;t++){
    size_t x4d = a.threeToFour(0,t);
    out[t] = *a.site_ptr(x4d);
    for(size_t i=1;i<field_size_3d;i++){
      x4d = a.threeToFour(i,t);
      out[t] = out[t] + *a.site_ptr(x4d);
    }
  }
  return out;
}


//Perform the local-node 3d slice sum
//Output is an array of size GJP.TnodeSites()  (i.e. the local time coordinate)
template<typename VectorMatrixType>			
ManagedVector<VectorMatrixType> localNodeSpatialSum(const CPSmatrixField<VectorMatrixType> &a){
  using namespace Grid;
  int Lt_loc = GJP.TnodeSites();
  ManagedVector<VectorMatrixType> out(Lt_loc); 
  memset(out.data(), 0, Lt_loc*sizeof(VectorMatrixType));
  
  typedef typename getScalarType<VectorMatrixType, typename MatrixTypeClassify<VectorMatrixType>::type  >::type ScalarType; 
  static const size_t nscalar = sizeof(VectorMatrixType)/sizeof(ScalarType);
  size_t field_size = a.size();
  assert(a.nodeSites(3) == Lt_loc);
  size_t field_size_3d = field_size/Lt_loc; 
  //Have vol/2 threads sum   x[i] + x[i + vol/2]
  //Then vol/4 for the next round, and so on
  static const int nsimd = ScalarType::Nsimd();
  
  assert(field_size_3d % 2 == 0);

  ScalarType *tmp = (ScalarType*)managed_alloc_check(nscalar*field_size/2*sizeof(ScalarType));
  ScalarType *tmp2 = (ScalarType*)managed_alloc_check(nscalar*field_size/2*sizeof(ScalarType));

  ScalarType *into = tmp;

  auto av = a.view();
  accelerator_for(offset, nscalar * Lt_loc * field_size_3d/2, nsimd,
		  {
		    typedef SIMT<ScalarType> ACC;
		    //Map offset as  s + nscalar*(t + Lt_loc*x3d)
		    size_t rem = offset;
		    size_t s = rem % nscalar; rem /= nscalar;
		    size_t t = rem % Lt_loc; rem /= Lt_loc;
		    size_t x3d = rem;

		    size_t x4d = av.threeToFour(x3d,t);
		    size_t x4d_shift = av.threeToFour(x3d + field_size_3d/2, t);

		    ScalarType const* aa1_ptr = (ScalarType const *)av.site_ptr(x4d); //pointer to complex type
		    auto aa1 = ACC::read(aa1_ptr[s]);

		    ScalarType const* aa2_ptr = (ScalarType const *)av.site_ptr(x4d_shift); //pointer to complex type
		    auto aa2 = ACC::read(aa2_ptr[s]);
		    
		    ACC::write(into[offset], aa1+aa2);
		  }
		  );
  
  field_size_3d/=2;

  size_t iter = 0;
  while(field_size_3d % 2 == 0){
    //swap back and forth between the two temp buffers,
    ScalarType const* from = (iter % 2 == 0 ? tmp : tmp2);
    into = (iter % 2 == 0 ? tmp2 : tmp);
    
    accelerator_for(offset, nscalar * Lt_loc * field_size_3d/2, nsimd,
    		    {
    		      typedef SIMT<ScalarType> ACC;
		      
    		      ScalarType const* aa1_ptr = from + offset;
    		      auto aa1 = ACC::read(*aa1_ptr);
		      
    		      ScalarType const* aa2_ptr = from + offset + nscalar*Lt_loc*field_size_3d/2;
    		      auto aa2 = ACC::read(*aa2_ptr);
		      
    		      ACC::write(into[offset], aa1+aa2);
    		    }
    		    );
    field_size_3d/=2;
    ++iter;
  }

  for(int t=0;t<Lt_loc;t++){
    ScalarType *out_s = (ScalarType*)&out[t];
    for(size_t s=0;s<nscalar;s++)    
      for(size_t x3d=0; x3d<field_size_3d; x3d++)
	out_s[s] = out_s[s] + into[s+nscalar*(t+Lt_loc*x3d)];
  }

  managed_free(tmp);
  managed_free(tmp2);
  return out;
}



/*Generic unop using functor
  Sadly it does not work for lambdas as the matrix type will be evaluated with both the vector and scalar matrix arguments
  Instead use a templated functor e.g.

  struct _tr{
    template<typename MatrixType>
    accelerator_inline auto operator()(const MatrixType &matrix) const ->decltype(matrix.Trace()){ return matrix.Trace(); }  
  };

  then call
  
  unop(myfield, tr_());
*/
template<typename T, typename Lambda>
auto unop(const CPSmatrixField<T> &in, const Lambda &l)-> CPSmatrixField<typename std::decay<decltype( l(*in.site_ptr(size_t(0)) )  )>::type>{
  typedef typename std::decay<decltype(l(*in.site_ptr(size_t(0)) )  )>::type outMatrixType;
  using namespace Grid;
  constexpr int nsimd = T::scalar_type::Nsimd();
  CPSmatrixField<outMatrixType> out(in.getDimPolParams());
  auto ov = out.view();
  auto iv = in.view();
  accelerator_for(x4d, in.size(), nsimd,
		    {
		      typedef SIMT<T> ACCr;
		      typedef SIMT<outMatrixType> ACCo;
		      auto aa = ACCr::read(*iv.site_ptr(x4d));
		      ACCo::write(*ov.site_ptr(x4d), l(aa) );
		    }
		    );
  return out;
}


template<typename T, typename U, typename Lambda>
auto binop(const CPSmatrixField<T> &a, const CPSmatrixField<U> &b, const Lambda &l)-> 
  CPSmatrixField<typename std::decay<decltype( l( *a.site_ptr(size_t(0)), *b.site_ptr(size_t(0))  )  )>::type>{
  typedef typename std::decay<decltype( l( *a.site_ptr(size_t(0)), *b.site_ptr(size_t(0))  )  )>::type outMatrixType;
  using namespace Grid;
  constexpr int nsimd = T::scalar_type::Nsimd();
  CPSmatrixField<outMatrixType> out(a.getDimPolParams());
  auto ov = out.view();
  auto av = a.view();
  auto bv = b.view();
  
  accelerator_for(x4d, av.size(), nsimd,
		    {
		      typedef SIMT<T> ACCra;
		      typedef SIMT<U> ACCrb;			
		      typedef SIMT<outMatrixType> ACCo;
		      auto aa = ACCra::read(*av.site_ptr(x4d));
		      auto bb = ACCrb::read(*bv.site_ptr(x4d));
		      ACCo::write(*ov.site_ptr(x4d), l(aa,bb) );
		    }
		    );
  return out;
}






CPS_END_NAMESPACE

#endif
