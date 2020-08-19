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
  copyControl::shallow() = true;
  accelerator_for(x4d, in.size(), nsimd,
		    {
		      int lane = Grid::acceleratorSIMTlane(nsimd);
		      l(*out.site_ptr(x4d), *in.site_ptr(x4d), lane);
		    }
		    );
  copyControl::shallow()= false;
  return out;
}

template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _trV{
  typedef typename VectorMatrixType::scalar_type OutputType;
  accelerator_inline void operator()(OutputType &out, const VectorMatrixType &in, const int lane) const{ 
    Trace(out, in, lane);
  }
};


template<typename VectorMatrixType>
inline auto Trace(const CPSmatrixField<VectorMatrixType> &a)->decltype( unop_v(a, _trV<VectorMatrixType>()) ){
  return unop_v(a, _trV<VectorMatrixType>());
}



template<int Index, typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _trIndexV{
  typedef typename _PartialTraceFindReducedType<VectorMatrixType,Index>::type OutputType;
  accelerator_inline void operator()(OutputType &out, const VectorMatrixType &in, const int lane) const{ 
    TraceIndex<Index>(out, in, lane);
  }
};
template<int Index, typename VectorMatrixType>
inline auto TraceIndex(const CPSmatrixField<VectorMatrixType> &a)->decltype( unop_v(a, _trIndexV<Index,VectorMatrixType>()) ){
  return unop_v(a, _trIndexV<Index,VectorMatrixType>());
}

template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<CPSflavorMatrix<ComplexType> > > ColorTrace(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in){
  return TraceIndex<1>(in);
}




template<int Index1, int Index2, typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _trTwoIndicesV{
  typedef typename _PartialDoubleTraceFindReducedType<VectorMatrixType,Index1,Index2>::type OutputType;
  accelerator_inline void operator()(OutputType &out, const VectorMatrixType &in, const int lane) const{ 
    TraceTwoIndices<Index1,Index2>(out, in, lane);
  }
};

//Trace over two indices of a nested matrix. Requires  Index1 < Index2
template<int Index1, int Index2, typename VectorMatrixType>
inline auto TraceTwoIndices(const CPSmatrixField<VectorMatrixType> &a)->decltype( unop_v(a, _trTwoIndicesV<Index1,Index2,VectorMatrixType>()) ){
  return unop_v(a, _trTwoIndicesV<Index1,Index2,VectorMatrixType>());
}

template<typename ComplexType>
inline CPSmatrixField<CPScolorMatrix<ComplexType> > SpinFlavorTrace(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in){
  return TraceTwoIndices<0,2>(in);
}





template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _transposeV{
  typedef VectorMatrixType OutputType;
  accelerator_inline void operator()(VectorMatrixType &out, const VectorMatrixType &in, const int lane) const{ 
    Transpose(out, in, lane);
  }
};

template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> Transpose(const CPSmatrixField<VectorMatrixType> &a){
  return unop_v(a, _transposeV<VectorMatrixType>());
}


template<int Index, typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _transIdx{
  typedef VectorMatrixType OutputType;
  accelerator_inline void operator()(VectorMatrixType &out, const VectorMatrixType &in, const int lane) const{ 
    TransposeOnIndex<Index>(out, in, lane);
  }
};

template<int Index, typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> TransposeOnIndex(CPSmatrixField<VectorMatrixType> &in){
  return unop_v(in, _transIdx<Index,VectorMatrixType>());
}

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > TransposeColor(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in){
  return TransposeOnIndex<1>(in);
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
  copyControl::shallow() = true;
  accelerator_for(x4d, a.size(), nsimd,
		    {
		      int lane = Grid::acceleratorSIMTlane(nsimd);
		      l(*out.site_ptr(x4d), *a.site_ptr(x4d), *b.site_ptr(x4d), lane);
		    }
		    );
  copyControl::shallow()= false;
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
  copyControl::shallow() = true;
  accelerator_for(x4d, m.size(), nsimd,
		    {
		      int lane = Grid::acceleratorSIMTlane(nsimd);
		      l(*m.site_ptr(x4d), lane);
		    }
		    );
  copyControl::shallow()= false;
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



//Binops
#define _MATRIX_FIELD_BINOP(OP) \
  template<typename VectorMatrixType> \
 CPSmatrixField<VectorMatrixType> operator OP(const CPSmatrixField<VectorMatrixType> &a, const CPSmatrixField<VectorMatrixType> &b){ \
 assert(a.size() == b.size());						\
 static const int nsimd = getScalarType<VectorMatrixType, typename MatrixTypeClassify<VectorMatrixType>::type  >::type::Nsimd(); \
 CPSmatrixField<VectorMatrixType> out(a.getDimPolParams());		\
 copyControl::shallow() = true;						\
 using namespace Grid;							\
 accelerator_for(x4d, a.size(), nsimd,					\
 {									\
   typedef SIMT<VectorMatrixType> ACC;					\
   auto aa = ACC::read(*a.site_ptr(x4d));				\
   auto bb = ACC::read(*b.site_ptr(x4d));				\
   ACC::write(*out.site_ptr(x4d), aa OP bb);				\
 }									\
		 );							\
 copyControl::shallow()= false;						\
 return out;								\
}

//Disabled because the binop_v versions are faster
//_MATRIX_FIELD_BINOP(*);
//_MATRIX_FIELD_BINOP(+);
//_MATRIX_FIELD_BINOP(-);

#undef _MATRIX_FIELD_BINOP


//Unop methods
#define _MATRIX_FIELD_UNOP_METHOD(OP)					\
  template<typename VectorMatrixType>			\
 CPSmatrixField<							\
	typename std::decay<decltype( ((VectorMatrixType*)NULL)-> OP () )>::type \
	  >								\
 OP(const CPSmatrixField<VectorMatrixType> &a){				\
    using namespace Grid;						\
    static const int nsimd = VectorMatrixType::scalar_type::Nsimd();	\
    typedef typename std::decay<decltype( ((VectorMatrixType*)NULL)-> OP () )>::type outMatrixType; \
    CPSmatrixField<outMatrixType> out(a.getDimPolParams());		\
    copyControl::shallow() = true;					\
    accelerator_for(x4d, a.size(), nsimd,				\
		    {							\
		      typedef SIMT<VectorMatrixType> ACCr;		\
		      typedef SIMT<outMatrixType> ACCo;			\
		      auto aa = ACCr::read(*a.site_ptr(x4d));		\
		      ACCo::write(*out.site_ptr(x4d), aa . OP () );	\
		    }							\
		    );							\
    copyControl::shallow()= false;					\
    return out;								\
  }

//Disabled because unop_v versions faster
//_MATRIX_FIELD_UNOP_METHOD( Trace );
//_MATRIX_FIELD_UNOP_METHOD( Transpose );

//These are specific to CPSspinColorFlavorMatrix
//_MATRIX_FIELD_UNOP_METHOD( TransposeColor );
//_MATRIX_FIELD_UNOP_METHOD( ColorTrace );
//_MATRIX_FIELD_UNOP_METHOD( SpinFlavorTrace );

#undef _MATRIX_FIELD_UNOP_METHOD

//Unop methods with single template arg
#define _MATRIX_FIELD_UNOP_METHOD_TEMPL1(OP, TEMPL_TYPE, TEMPL_NAME)	\
  template<TEMPL_TYPE TEMPL_NAME, typename VectorMatrixType>		\
 CPSmatrixField<							\
	typename std::decay<decltype( ((VectorMatrixType*)NULL)-> template OP<TEMPL_NAME> () )>::type \
	  >								\
 OP(const CPSmatrixField<VectorMatrixType> &a){				\
    using namespace Grid;					\
    static const int nsimd = VectorMatrixType::scalar_type::Nsimd();	\
    typedef typename std::decay<decltype( ((VectorMatrixType*)NULL)-> template OP<TEMPL_NAME> () )>::type outMatrixType; \
    CPSmatrixField<outMatrixType> out(a.getDimPolParams());		\
    copyControl::shallow() = true;					\
    accelerator_for(x4d, a.size(), nsimd,				\
		    {							\
		      typedef SIMT<VectorMatrixType> ACCr;		\
		      typedef SIMT<outMatrixType> ACCo;			\
		      auto aa = ACCr::read(*a.site_ptr(x4d));		\
		      ACCo::write(*out.site_ptr(x4d), aa . template OP<TEMPL_NAME>() );	\
		    }							\
		    );							\
    copyControl::shallow()= false;					\
    return out;								\
  }


//_MATRIX_FIELD_UNOP_METHOD_TEMPL1( TransposeOnIndex, int, TransposeDepth );

#undef _MATRIX_FIELD_UNOP_METHOD_TEMPL1


//Self-acting methods (no return type)
#define _MATRIX_FIELD_SELFOP_METHOD(OP)		\
  template<typename VectorMatrixType>			\
  CPSmatrixField<VectorMatrixType> & OP(CPSmatrixField<VectorMatrixType> &a){		\
 using namespace Grid;							\
 static const int nsimd = VectorMatrixType::scalar_type::Nsimd();	\
 copyControl::shallow() = true;						\
 accelerator_for(x4d, a.size(), nsimd,					\
 {									\
   typedef SIMT<VectorMatrixType> ACC;					\
   auto aa = ACC::read(*a.site_ptr(x4d));				\
   aa. OP ();								\
   ACC::write(*a.site_ptr(x4d), aa );					\
 }									\
 );               							\
 copyControl::shallow()= false;						\
 return a;								\
}

//_MATRIX_FIELD_SELFOP_METHOD( unit );
//_MATRIX_FIELD_SELFOP_METHOD( timesMinusOne );
//_MATRIX_FIELD_SELFOP_METHOD( timesI );
//_MATRIX_FIELD_SELFOP_METHOD( timesMinusI );

#undef _MATRIX_FIELD_SELFOP_METHOD

//Self-acting methods with one argument
#define _MATRIX_FIELD_SELFOP_METHOD_ARG1(OP, ARG_TYPE, ARG_NAME)	\
  template<typename VectorMatrixType>			\
  CPSmatrixField<VectorMatrixType> & OP(CPSmatrixField<VectorMatrixType> &a, ARG_TYPE ARG_NAME){		\
 using namespace Grid;							\
 static const int nsimd = VectorMatrixType::scalar_type::Nsimd();	\
 copyControl::shallow() = true;						\
 accelerator_for(x4d, a.size(), nsimd,					\
 {									\
   typedef SIMT<VectorMatrixType> ACC;					\
   auto aa = ACC::read(*a.site_ptr(x4d));				\
   aa. OP (ARG_NAME);							\
   ACC::write(*a.site_ptr(x4d), aa );					\
 }									\
 );               							\
 copyControl::shallow()= false;						\
 return a;								\
}

_MATRIX_FIELD_SELFOP_METHOD_ARG1(pl, const FlavorMatrixType, type);
_MATRIX_FIELD_SELFOP_METHOD_ARG1(pr, const FlavorMatrixType, type);
_MATRIX_FIELD_SELFOP_METHOD_ARG1(gl, const int, dir);
_MATRIX_FIELD_SELFOP_METHOD_ARG1(gr, const int, dir);
_MATRIX_FIELD_SELFOP_METHOD_ARG1(glAx, const int, dir);
_MATRIX_FIELD_SELFOP_METHOD_ARG1(grAx, const int, dir);

#undef _MATRIX_FIELD_SELFOP_METHOD_ARG1



template<typename VectorMatrixType>			
CPSmatrixField<typename VectorMatrixType::scalar_type> Trace(const CPSmatrixField<VectorMatrixType> &a, const CPSmatrixField<VectorMatrixType> &b){
  using namespace Grid;
  CPSmatrixField<typename VectorMatrixType::scalar_type> out(a.getDimPolParams());
  static const int nsimd = VectorMatrixType::scalar_type::Nsimd();
  copyControl::shallow() = true;
  accelerator_for(x4d, a.size(), nsimd,
		  {
		    typedef SIMT<VectorMatrixType> ACCi;
		    typedef SIMT<typename VectorMatrixType::scalar_type> ACCo;
		    auto aa = ACCi::read(*a.site_ptr(x4d));
		    auto bb = ACCi::read(*b.site_ptr(x4d));
		    auto cc = Trace(aa,bb);
		    ACCo::write(*out.site_ptr(x4d), cc );
		  }
		  );
  copyControl::shallow()= false;
  return out;
}

template<typename VectorMatrixType>			
CPSmatrixField<typename VectorMatrixType::scalar_type> Trace(const CPSmatrixField<VectorMatrixType> &a, const VectorMatrixType &b){
  using namespace Grid;
  CPSmatrixField<typename VectorMatrixType::scalar_type> out(a.getDimPolParams());
  
  //On GPUs the compiler fails if b is passed by value into the kernel due to "formal parameter space overflow"
  //As we can't assume b is allocated in UVM memory we need to either create a UVM copy or explicitly copy b to the GPU
#ifdef GPU_VEC
  VectorMatrixType* bptr = (VectorMatrixType*)device_alloc_check(sizeof(VectorMatrixType));
  copy_host_to_device(bptr, &b, sizeof(VectorMatrixType));
#else
  VectorMatrixType const* bptr = &b; //need to access by pointer internally to prevent copy-by-value into GPU kernel
#endif  

  static const int nsimd = VectorMatrixType::scalar_type::Nsimd();
  copyControl::shallow() = true;
  accelerator_for(x4d, a.size(), nsimd,
  		  {
  		    typedef SIMT<VectorMatrixType> ACCi;
  		    typedef SIMT<typename VectorMatrixType::scalar_type> ACCo;
  		    auto aa = ACCi::read(*a.site_ptr(x4d));
  		    auto bb = ACCi::read(*bptr);		    
		    auto cc = Trace(aa,bb);
  		    ACCo::write(*out.site_ptr(x4d), cc );
  		  }
  		  );
  copyControl::shallow()= false;

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

  copyControl::shallow() = true;
  accelerator_for(offset, nscalar * field_size/2, nsimd,
		  {
		    typedef SIMT<ScalarType> ACC;
		    //Map offset as  s + nscalar*x
		    size_t s = offset % nscalar;
		    size_t x = offset / nscalar;

		    ScalarType const* aa1_ptr = (ScalarType const *)a.site_ptr(x); //pointer to complex type
		    auto aa1 = ACC::read(aa1_ptr[s]);

		    ScalarType const* aa2_ptr = (ScalarType const *)a.site_ptr(x + field_size/2); //pointer to complex type
		    auto aa2 = ACC::read(aa2_ptr[s]);
		    
		    ACC::write(into[offset], aa1+aa2);
		  }
		  );
  copyControl::shallow()= false;
  
  field_size/=2;

  size_t iter = 0;
  while(field_size % 2 == 0){
    //swap back and forth between the two temp buffers,
    ScalarType const* from = (iter % 2 == 0 ? tmp : tmp2);
    into = (iter % 2 == 0 ? tmp2 : tmp);
    
    accelerator_for(offset, nscalar * field_size/2, nsimd,
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

  copyControl::shallow() = true;
  accelerator_for(offset, nscalar * Lt_loc * field_size_3d/2, nsimd,
		  {
		    typedef SIMT<ScalarType> ACC;
		    //Map offset as  s + nscalar*(t + Lt_loc*x3d)
		    size_t rem = offset;
		    size_t s = rem % nscalar; rem /= nscalar;
		    size_t t = rem % Lt_loc; rem /= Lt_loc;
		    size_t x3d = rem;

		    size_t x4d = a.threeToFour(x3d,t);
		    size_t x4d_shift = a.threeToFour(x3d + field_size_3d/2, t);

		    ScalarType const* aa1_ptr = (ScalarType const *)a.site_ptr(x4d); //pointer to complex type
		    auto aa1 = ACC::read(aa1_ptr[s]);

		    ScalarType const* aa2_ptr = (ScalarType const *)a.site_ptr(x4d_shift); //pointer to complex type
		    auto aa2 = ACC::read(aa2_ptr[s]);
		    
		    ACC::write(into[offset], aa1+aa2);
		  }
		  );
  copyControl::shallow()= false;
  
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
  copyControl::shallow() = true;
  accelerator_for(x4d, in.size(), nsimd,
		    {
		      typedef SIMT<T> ACCr;
		      typedef SIMT<outMatrixType> ACCo;
		      auto aa = ACCr::read(*in.site_ptr(x4d));
		      ACCo::write(*out.site_ptr(x4d), l(aa) );
		    }
		    );
  copyControl::shallow()= false;
  return out;
}


template<typename T, typename U, typename Lambda>
auto binop(const CPSmatrixField<T> &a, const CPSmatrixField<U> &b, const Lambda &l)-> 
  CPSmatrixField<typename std::decay<decltype( l( *a.site_ptr(size_t(0)), *b.site_ptr(size_t(0))  )  )>::type>{
  typedef typename std::decay<decltype( l( *a.site_ptr(size_t(0)), *b.site_ptr(size_t(0))  )  )>::type outMatrixType;
  using namespace Grid;
  constexpr int nsimd = T::scalar_type::Nsimd();
  CPSmatrixField<outMatrixType> out(a.getDimPolParams());
  copyControl::shallow() = true;
  accelerator_for(x4d, a.size(), nsimd,
		    {
		      typedef SIMT<T> ACCra;
		      typedef SIMT<U> ACCrb;			
		      typedef SIMT<outMatrixType> ACCo;
		      auto aa = ACCra::read(*a.site_ptr(x4d));
		      auto bb = ACCrb::read(*b.site_ptr(x4d));
		      ACCo::write(*out.site_ptr(x4d), l(aa,bb) );
		    }
		    );
  copyControl::shallow()= false;
  return out;
}






CPS_END_NAMESPACE

#endif
