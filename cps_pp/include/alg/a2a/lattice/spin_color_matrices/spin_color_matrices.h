#ifndef _SPIN_COLOR_MATRICES_H
#define _SPIN_COLOR_MATRICES_H

//These are alternatives to Matrix, WilsonMatrix, SpinColorFlavorMatrix
#include<util/flavormatrix.h>
#include <alg/a2a/utils/template_wizardry.h>

CPS_START_NAMESPACE

#include "implementation/spin_color_matrices_meta.tcc"

//A class representing a square matrix
template<typename T, int N>
class CPSsquareMatrix{
protected:
  T v[N][N];
  template<typename U>  //, typename my_enable_if< my_is_base_of<U, CPSsquareMatrix<T,N> >::value, int>::type = 0
  accelerator_inline static void mult(U &into, const U &a, const U &b){
    into.zero();
    for(int i=0;i<N;i++)
      for(int k=0;k<N;k++)
	for(int j=0;j<N;j++)
	  into(i,k) = into(i,k) + a(i,j)*b(j,k);      
  }
public:
  enum { isDerivedFromCPSsquareMatrix=1 };
  enum { Size = N };
  typedef T value_type; //the type of the elements (can be another matrix)
  typedef typename _RecursiveTraceFindScalarType<CPSsquareMatrix<T,N>,cps_square_matrix_mark>::scalar_type scalar_type; //the underlying numerical type
  
  //Get the type were the value_type of this matrix to be replaced by matrix U
  template<typename U>
  struct Rebase{
    typedef CPSsquareMatrix<U,N> type;
  };
  //Get the type were the underlying numerical type (scalar_type) to be replaced by type U
  template<typename U>
  struct RebaseScalarType{
    typedef typename _rebaseScalarType<CPSsquareMatrix<T,N>,cps_square_matrix_mark,U>::type type;
  };

  //The number of fundamental "scalar_type" data in memory
  static constexpr size_t nScalarType(){
    return _RecursiveCountScalarType<CPSsquareMatrix<T,N>, cps_square_matrix_mark>::count();
  }
  //Return a pointer to this object as an array of scalar_type of size nScalarType()
  accelerator_inline scalar_type const* scalarTypePtr() const{ return (scalar_type const*)this; }
  accelerator_inline scalar_type * scalarTypePtr(){ return (scalar_type*)this; }


  accelerator CPSsquareMatrix() = default;
  accelerator CPSsquareMatrix(const CPSsquareMatrix &r) = default;
  
  accelerator_inline T & operator()(const int i, const int j){ return v[i][j]; }
  accelerator_inline const T & operator()(const int i, const int j) const{ return v[i][j]; }

  //*this = Transpose(r)     Transposes on all indices if a nested matrix
  accelerator_inline void equalsTranspose(const CPSsquareMatrix<T,N> &r);

  accelerator_inline CPSsquareMatrix<T,N> Transpose() const;				
  
  //Trace on just the index associated with this matrix
  // T traceIndex() const{
  //   T out; CPSsetZero(out);
  //   for(int i=0;i<N;i++) out = out + v[i][i];
  //   return out;
  // }
  //Trace on all indices recursively
  accelerator_inline scalar_type Trace() const;

  template<int RemoveDepth>
  accelerator_inline typename _PartialTraceFindReducedType<CPSsquareMatrix<T,N>, RemoveDepth>::type TraceIndex() const;

  template<int RemoveDepth1, int RemoveDepth2>
  accelerator_inline typename _PartialDoubleTraceFindReducedType<CPSsquareMatrix<T,N>,RemoveDepth1,RemoveDepth2>::type TraceTwoIndices() const;

  //Set this matrix equal to the transpose of r on its compound tensor index TransposeDepth. If it is not a compound matrix then TranposeDepth=0 does the same thing as equalsTranspose above
  //i.e. in(i,j)(k,l)(m,n)  transpose on index depth 1:   out(i,j)(k,l)(m,n) = in(i,j)(l,k)(m,n)
  template<int TransposeDepth>
  accelerator_inline void equalsTransposeOnIndex(const CPSsquareMatrix<T,N> &r);

  template<int TransposeDepth>
  accelerator_inline CPSsquareMatrix<T,N> TransposeOnIndex() const;
  
  //this = zero
  accelerator_inline CPSsquareMatrix<T,N> & zero();

  //this = unit matrix
  accelerator_inline CPSsquareMatrix<T,N>& unit();

  //this = -this
  accelerator_inline CPSsquareMatrix<T,N> & timesMinusOne();

  //this = i*this
  accelerator_inline CPSsquareMatrix<T,N> & timesI();

  //this = -i*this
  accelerator_inline CPSsquareMatrix<T,N> & timesMinusI();
    
  accelerator_inline CPSsquareMatrix<T,N> & operator+=(const CPSsquareMatrix<T,N> &r);

  accelerator_inline CPSsquareMatrix<T,N> & operator*=(const T &r);

  accelerator_inline CPSsquareMatrix<T,N> & operator-=(const CPSsquareMatrix<T,N> &r);

  accelerator_inline bool operator==(const CPSsquareMatrix<T,N> &r) const;  
};

template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, U>::type operator*(const U &a, const U &b);
template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, U>::type operator*(const U &a, const typename U::scalar_type &b);
template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, U>::type operator*(const typename U::scalar_type &a, const U &b);
template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, U>::type operator+(const U &a, const U &b);
template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, U>::type operator-(const U &a, const U &b);
template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, U>::type operator-(const U &a);

template<typename T, int N>
std::ostream & operator<<(std::ostream &os, const CPSsquareMatrix<T,N> &m);

template<typename U>
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, typename U::scalar_type>::type Trace(const U &a, const U &b);

template<typename T>
accelerator_inline T Transpose(const T& r);

//Perform SIMD reductions of all elements
#ifdef USE_GRID
template<typename VectorMatrixType, typename my_enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
inline typename VectorMatrixType::template RebaseScalarType<typename VectorMatrixType::scalar_type::scalar_type>::type
Reduce(const VectorMatrixType &v);
#endif

//Sum matrix over all nodes
template<typename MatrixType, typename my_enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type = 0>
void globalSum(MatrixType *m){
  globalSum( m->scalarTypePtr(), m->nScalarType() );
}

#include "implementation/spin_color_matrices_macros.tcc"

template<typename T>
class CPSflavorMatrix: public CPSsquareMatrix<T,2>{
public:
  typedef CPSsquareMatrix<T,2> base_type;
  INHERIT_METHODS_AND_TYPES(CPSflavorMatrix<T>, CPSflavorMatrix);
  
  template<typename U>
  struct Rebase{
    typedef CPSflavorMatrix<U> type;
  };
  //Get the type were the underlying numerical type (scalar_type) to be replaced by type U
  template<typename U>
  struct RebaseScalarType{
    typedef typename _rebaseScalarType<CPSflavorMatrix<T>,cps_square_matrix_mark,U>::type type;
  };

  //multiply on left by a flavor matrix
  accelerator_inline CPSflavorMatrix<T> & pl(const FlavorMatrixType &type);
  //multiply on right by a flavor matrix
  accelerator_inline CPSflavorMatrix<T> & pr(const FlavorMatrixType &type);
};


  
  
//  Chiral basis
//  gamma(XUP)    gamma(YUP)    gamma(ZUP)    gamma(TUP)    gamma(FIVE)
//  0  0  0  i    0  0  0 -1    0  0  i  0    0  0  1  0    1  0  0  0
//  0  0  i  0    0  0  1  0    0  0  0 -i    0  0  0  1    0  1  0  0
//  0 -i  0  0    0  1  0  0   -i  0  0  0    1  0  0  0    0  0 -1  0
// -i  0  0  0   -1  0  0  0    0  i  0  0    0  1  0  0    0  0  0 -1

template<typename T>
class CPSspinMatrix: public CPSsquareMatrix<T,4>{
public:
  typedef CPSsquareMatrix<T,4> base_type;
  INHERIT_METHODS_AND_TYPES(CPSspinMatrix<T>, CPSspinMatrix);
  
  template<typename U>
  struct Rebase{
    typedef CPSspinMatrix<U> type;
  };
  //Get the type were the underlying numerical type (scalar_type) to be replaced by type U
  template<typename U>
  struct RebaseScalarType{
    typedef typename _rebaseScalarType<CPSspinMatrix<T>,cps_square_matrix_mark,U>::type type;
  };
  
  //Left Multiplication by Dirac gamma's
  accelerator_inline CPSspinMatrix<T> & gl(int dir);

  //Right Multiplication by Dirac gamma's
  accelerator_inline CPSspinMatrix<T>& gr(int dir);

  //multiply gamma(i)gamma(5) on the left: result = gamma(i)*gamma(5)*from
  accelerator_inline CPSspinMatrix<T>& glAx(const int dir);

  //multiply gamma(i)gamma(5) on the right: result = from*gamma(i)*gamma(5)
  accelerator_inline CPSspinMatrix<T>& grAx(int dir);  
};



template<typename T>
class CPScolorMatrix: public CPSsquareMatrix<T,3>{
public:
  typedef CPSsquareMatrix<T,3> base_type;
  INHERIT_METHODS_AND_TYPES(CPScolorMatrix<T>, CPScolorMatrix);

  template<typename U>
  struct Rebase{
    typedef CPScolorMatrix<U> type;
  };
  //Get the type were the underlying numerical type (scalar_type) to be replaced by type U
  template<typename U>
  struct RebaseScalarType{
    typedef typename _rebaseScalarType<CPScolorMatrix<T>,cps_square_matrix_mark,U>::type type;
  };
};

template<typename ComplexType>
class CPSspinColorFlavorMatrix: public CPSspinMatrix<CPScolorMatrix<CPSflavorMatrix<ComplexType> > >{
public:
  typedef CPSspinMatrix<CPScolorMatrix<CPSflavorMatrix<ComplexType> > > SCFmat;
  typedef CPSspinMatrix<CPScolorMatrix<CPSflavorMatrix<ComplexType> > > base_type;
  INHERIT_METHODS_AND_TYPES(CPSspinColorFlavorMatrix<ComplexType>, CPSspinColorFlavorMatrix);
  
  template<typename U>
  struct Rebase{
    typedef CPSspinMatrix<U> type;
  };
  //Get the type were the underlying numerical type (scalar_type) to be replaced by type U
  template<typename U>
  struct RebaseScalarType{
    typedef CPSspinColorFlavorMatrix<U> type;
  };

  accelerator_inline CPScolorMatrix<ComplexType> SpinFlavorTrace() const;

  accelerator_inline CPSspinMatrix<CPSflavorMatrix<ComplexType> > ColorTrace() const;

  accelerator_inline CPSspinColorFlavorMatrix<ComplexType> TransposeColor() const;

  accelerator_inline void equalsColorTranspose(const CPSspinColorFlavorMatrix<ComplexType> &r);
  
  //multiply on left by a flavor matrix
  accelerator_inline CPSspinColorFlavorMatrix<ComplexType> & pl(const FlavorMatrixType type);

  //multiply on left by a flavor matrix
  accelerator_inline CPSspinColorFlavorMatrix<ComplexType> & pr(const FlavorMatrixType type);
  
};



template<typename ComplexType>
class CPSspinColorMatrix: public CPSspinMatrix<CPScolorMatrix<ComplexType> >{
public:
  typedef CPSspinMatrix<CPScolorMatrix<ComplexType> > base_type;
  INHERIT_METHODS_AND_TYPES(CPSspinColorMatrix<ComplexType>, CPSspinColorMatrix);
  
  template<typename U>
  struct Rebase{
    typedef CPSspinMatrix<U> type;
  };
  //Get the type were the underlying numerical type (scalar_type) to be replaced by type U
  template<typename U>
  struct RebaseScalarType{
    typedef CPSspinColorMatrix<U> type;
  };

  accelerator_inline CPScolorMatrix<ComplexType> SpinTrace() const;

  accelerator_inline CPSspinMatrix<ComplexType> ColorTrace() const;

  accelerator_inline CPSspinColorMatrix<ComplexType> TransposeColor() const;

  accelerator_inline void equalsColorTranspose(const CPSspinColorMatrix<ComplexType> &r);
};

#include "implementation/spin_color_matrices_impl.tcc"
#include "implementation/spin_color_matrices_derived_impl.tcc"

CPS_END_NAMESPACE

#endif
