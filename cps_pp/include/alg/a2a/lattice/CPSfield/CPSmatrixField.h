#ifndef _CPS_MATRIX_FIELD_H__
#define _CPS_MATRIX_FIELD_H__

#include "CPSfield.h"
#include <alg/a2a/lattice/spin_color_matrices.h>
#include <alg/a2a/lattice/spin_color_matrices_SIMT.h>

//CPSfields of matrices and associated functionality

CPS_START_NAMESPACE 

template<typename VectorMatrixType>
using CPSmatrixField = CPSfield<VectorMatrixType,1, FourDSIMDPolicy<OneFlavorPolicy>, Aligned128AllocPolicy>;


//These structs allow the same interface for operators that are applicable for matrix and complex types
define_test_has_enum(isDerivedFromCPSsquareMatrix);

struct vector_matrix_mark{};
template<typename T>
struct MatrixTypeClassify{
  typedef typename TestElem< is_grid_vector_complex<T>::value, grid_vector_complex_mark,
			     TestElem< has_enum_isDerivedFromCPSsquareMatrix<T>::value, vector_matrix_mark, 
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





//Binops
#define _MATRIX_FIELD_BINOP(OP) \
  template<typename VectorMatrixType> \
 CPSmatrixField<VectorMatrixType> operator OP(const CPSmatrixField<VectorMatrixType> &a, const CPSmatrixField<VectorMatrixType> &b){ \
 assert(a.size() == b.size());						\
 static const int nsimd = getScalarType<VectorMatrixType, typename MatrixTypeClassify<VectorMatrixType>::type  >::type::Nsimd(); \
 CPSmatrixField<VectorMatrixType> out(a.getDimPolParams());		\
 copyControl::shallow() = true;						\
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

_MATRIX_FIELD_BINOP(*);
_MATRIX_FIELD_BINOP(+);
_MATRIX_FIELD_BINOP(-);

#undef _MATRIX_FIELD_BINOP



//Unop methods
#define _MATRIX_FIELD_UNOP_METHOD(OP)					\
  template<typename VectorMatrixType>			\
 CPSmatrixField<							\
	typename std::decay<decltype( ((VectorMatrixType*)NULL)-> OP () )>::type \
	  >								\
 OP(const CPSmatrixField<VectorMatrixType> &a){				\
 static const int nsimd = VectorMatrixType::scalar_type::Nsimd();	\
 typedef typename std::decay<decltype( ((VectorMatrixType*)NULL)-> OP () )>::type outMatrixType; \
 CPSmatrixField<outMatrixType> out(a.getDimPolParams());		\
 copyControl::shallow() = true;						\
 accelerator_for(x4d, a.size(), nsimd,					\
 {									\
   typedef SIMT<VectorMatrixType> ACCr;					\
   typedef SIMT<outMatrixType> ACCo;					\
   auto aa = ACCr::read(*a.site_ptr(x4d));				\
   ACCo::write(*out.site_ptr(x4d), aa . OP () );				\
 }									\
 );               							\
 copyControl::shallow()= false;						\
 return out;								\
}

_MATRIX_FIELD_UNOP_METHOD( Trace );
_MATRIX_FIELD_UNOP_METHOD( Transpose );

//These are specific to CPSspinColorFlavorMatrix
_MATRIX_FIELD_UNOP_METHOD( TransposeColor );
_MATRIX_FIELD_UNOP_METHOD( ColorTrace );
_MATRIX_FIELD_UNOP_METHOD( SpinFlavorTrace );

#undef _MATRIX_FIELD_UNOP_METHOD

//Unop methods with single template arg
#define _MATRIX_FIELD_UNOP_METHOD_TEMPL1(OP, TEMPL_TYPE, TEMPL_NAME)	\
  template<TEMPL_TYPE TEMPL_NAME, typename VectorMatrixType>		\
 CPSmatrixField<							\
	typename std::decay<decltype( ((VectorMatrixType*)NULL)-> template OP<TEMPL_NAME> () )>::type \
	  >								\
 OP(const CPSmatrixField<VectorMatrixType> &a){				\
 static const int nsimd = VectorMatrixType::scalar_type::Nsimd();	\
 typedef typename std::decay<decltype( ((VectorMatrixType*)NULL)-> template OP<TEMPL_NAME> () )>::type outMatrixType; \
 CPSmatrixField<outMatrixType> out(a.getDimPolParams());		\
 copyControl::shallow() = true;						\
 accelerator_for(x4d, a.size(), nsimd,					\
 {									\
   typedef SIMT<VectorMatrixType> ACCr;					\
   typedef SIMT<outMatrixType> ACCo;					\
   auto aa = ACCr::read(*a.site_ptr(x4d));				\
   ACCo::write(*out.site_ptr(x4d), aa . template OP<TEMPL_NAME>() );		\
 }									\
 );               							\
 copyControl::shallow()= false;						\
 return out;								\
}


_MATRIX_FIELD_UNOP_METHOD_TEMPL1( TransposeOnIndex, int, TransposeDepth );

#undef _MATRIX_FIELD_UNOP_METHOD_TEMPL1


//Self-acting methods (no return type)
#define _MATRIX_FIELD_SELFOP_METHOD(OP)		\
  template<typename VectorMatrixType>			\
  CPSmatrixField<VectorMatrixType> & OP(CPSmatrixField<VectorMatrixType> &a){		\
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

_MATRIX_FIELD_SELFOP_METHOD( unit );
_MATRIX_FIELD_SELFOP_METHOD( timesMinusOne );
_MATRIX_FIELD_SELFOP_METHOD( timesI );
_MATRIX_FIELD_SELFOP_METHOD( timesMinusI );

#undef _MATRIX_FIELD_SELFOP_METHOD

//Self-acting methods with one argument
#define _MATRIX_FIELD_SELFOP_METHOD_ARG1(OP, ARG_TYPE, ARG_NAME)	\
  template<typename VectorMatrixType>			\
  CPSmatrixField<VectorMatrixType> & OP(CPSmatrixField<VectorMatrixType> &a, ARG_TYPE ARG_NAME){		\
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
  CPSmatrixField<typename VectorMatrixType::scalar_type> out(a.getDimPolParams());
  static const int nsimd = VectorMatrixType::scalar_type::Nsimd();
  copyControl::shallow() = true;
  accelerator_for(x4d, a.size(), nsimd,
		  {
		    typedef SIMT<VectorMatrixType> ACCi;
		    typedef SIMT<typename VectorMatrixType::scalar_type> ACCo;
		    auto aa = ACCi::read(*a.site_ptr(x4d));
		    auto bb = ACCi::read(b);
		    auto cc = Trace(aa,bb);
		    ACCo::write(*out.site_ptr(x4d), cc );
		  }
		  );
  copyControl::shallow()= false;
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
  VectorMatrixType out; out.zero();
  
  //Want to introduce some paralellism into this thing
  //Parallelize over fundamental complex type
  typedef typename getScalarType<VectorMatrixType, typename MatrixTypeClassify<VectorMatrixType>::type  >::type ScalarType; 

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



CPS_END_NAMESPACE

#endif
