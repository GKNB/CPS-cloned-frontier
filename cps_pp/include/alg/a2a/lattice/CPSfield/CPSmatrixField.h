#ifndef _CPS_MATRIX_FIELD_H__
#define _CPS_MATRIX_FIELD_H__

#include "CPSfield.h"
#include <alg/a2a/lattice/spin_color_matrices.h>
#include <alg/a2a/lattice/spin_color_matrices_SIMT.h>

//CPSfields of matrices and associated functionality

CPS_START_NAMESPACE 

template<typename VectorMatrixType>
using CPSmatrixField = CPSfield<VectorMatrixType,1, FourDSIMDPolicy<OneFlavorPolicy>, Aligned128AllocPolicy>;


//Binops
#define _MATRIX_FIELD_BINOP(OP) \
template<typename VectorMatrixType> \
 CPSmatrixField<VectorMatrixType> operator OP(const CPSmatrixField<VectorMatrixType> &a, const CPSmatrixField<VectorMatrixType> &b){ \
 assert(a.size() == b.size());						\
 static const int nsimd = VectorMatrixType::scalar_type::Nsimd();	\
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
  void OP(CPSmatrixField<VectorMatrixType> &a){		\
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
}

_MATRIX_FIELD_SELFOP_METHOD( unit );
_MATRIX_FIELD_SELFOP_METHOD( timesMinusOne );
_MATRIX_FIELD_SELFOP_METHOD( timesI );
_MATRIX_FIELD_SELFOP_METHOD( timesMinusI );

#undef _MATRIX_FIELD_SELFOP_METHOD

//Self-acting methods with one argument
#define _MATRIX_FIELD_SELFOP_METHOD_ARG1(OP, ARG_TYPE, ARG_NAME)	\
  template<typename VectorMatrixType>			\
  void OP(CPSmatrixField<VectorMatrixType> &a, ARG_TYPE ARG_NAME){		\
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
}

_MATRIX_FIELD_SELFOP_METHOD_ARG1(pl, const FlavorMatrixType, type);
_MATRIX_FIELD_SELFOP_METHOD_ARG1(pr, const FlavorMatrixType, type);
_MATRIX_FIELD_SELFOP_METHOD_ARG1(gl, const int, dir);
_MATRIX_FIELD_SELFOP_METHOD_ARG1(gr, const int, dir);
_MATRIX_FIELD_SELFOP_METHOD_ARG1(glAx, const int, dir);
_MATRIX_FIELD_SELFOP_METHOD_ARG1(grAx, const int, dir);

#undef _MATRIX_FIELD_SELFOP_METHOD_ARG1

CPS_END_NAMESPACE

#endif
