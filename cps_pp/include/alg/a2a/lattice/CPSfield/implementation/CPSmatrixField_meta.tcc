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
