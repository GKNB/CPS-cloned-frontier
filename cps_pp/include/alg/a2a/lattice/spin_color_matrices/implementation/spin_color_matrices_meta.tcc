//Check if a type is a CPSsquareMatrix
//usage: e.g.  static_assert( ifCPSsquareMatrix<T>::value == 1 );
template<typename T>
class isCPSsquareMatrix{

  template<typename U, U> struct Check;

  template<typename U>
  static char test(Check<int, U::isDerivedFromCPSsquareMatrix> *);

  template<typename U>
  static double test(...);
  
public:

  enum {value = sizeof(test<T>(0)) == sizeof(char) };
};

//A mark for labeling CPSsquareMatrix
struct cps_square_matrix_mark;

//Another way of marking CPSsquareMatrix
// ClassifyMatrixOrNotMatrix<T>::type will be cps_square_matrix_mark or no_mark
template<typename T>
struct ClassifyMatrixOrNotMatrix{
  typedef typename TestElem< isCPSsquareMatrix<T>::value, cps_square_matrix_mark,LastElem >::type type;
};

//Utility for finding the types of partial traces
template<typename T, int RemoveDepth>
struct _PartialTraceFindReducedType{
  typedef typename T::template Rebase< typename _PartialTraceFindReducedType<typename T::value_type,RemoveDepth-1>::type >::type type;
};
template<typename T>
struct _PartialTraceFindReducedType<T,0>{
  typedef typename T::value_type type;
};

template<typename T, int RemoveDepth1, int RemoveDepth2> 
struct _PartialDoubleTraceFindReducedType{
  typedef typename my_enable_if<RemoveDepth1 < RemoveDepth2,int>::type test;
  typedef typename T::template Rebase< typename _PartialDoubleTraceFindReducedType<typename T::value_type,RemoveDepth1-1,RemoveDepth2-1>::type >::type type;
};
template<typename T,int RemoveDepth2>
struct _PartialDoubleTraceFindReducedType<T,0,RemoveDepth2>{
  typedef typename _PartialTraceFindReducedType<typename T::value_type,RemoveDepth2-1>::type type;
};


//Get the type with a different underlying numerical type
template<typename T, typename TypeClass, typename NewNumericalType>
struct _rebaseScalarType{};

template<typename T, typename NewNumericalType>
struct _rebaseScalarType<T, cps_square_matrix_mark, NewNumericalType>{
  typedef typename _rebaseScalarType<typename T::value_type,typename ClassifyMatrixOrNotMatrix<typename T::value_type>::type, NewNumericalType>::type subType;
  typedef typename T::template Rebase<subType>::type type;
};
template<typename T, typename NewNumericalType>
struct _rebaseScalarType<T, no_mark, NewNumericalType>{
  typedef NewNumericalType type;
};

//Find the underlying scalar type
template<typename T, typename TypeClass>
struct _RecursiveTraceFindScalarType{};

template<typename T>
struct _RecursiveTraceFindScalarType<T, cps_square_matrix_mark>{
  typedef typename _RecursiveTraceFindScalarType<typename T::value_type,typename ClassifyMatrixOrNotMatrix<typename T::value_type>::type>::scalar_type scalar_type;
};
template<typename T>
struct _RecursiveTraceFindScalarType<T, no_mark>{
  typedef T scalar_type;
};

//Count the number of fundamental scalars in the nested matrix
template<typename T, typename TypeClass>
struct _RecursiveCountScalarType{};

template<typename T>
struct _RecursiveCountScalarType<T, cps_square_matrix_mark>{
  static constexpr size_t count(){
    return T::Size*T::Size*_RecursiveCountScalarType<typename T::value_type,typename ClassifyMatrixOrNotMatrix<typename T::value_type>::type >::count();
  }
};
template<typename T>
struct _RecursiveCountScalarType<T, no_mark>{
  static constexpr size_t count(){
    return 1;
  }
};