#ifndef _TEMPLATE_WIZARDRY_H
#define _TEMPLATE_WIZARDRY_H

template <int LorR, typename T, typename U>
struct _selectLR{};
template <typename T, typename U>
struct _selectLR<0,T,U>{
  typedef T Type;
};
template <typename T, typename U>
struct _selectLR<1,T,U>{
  typedef U Type;
};

template<typename T,typename U>
struct _equal{
  enum { value = 0 };
};
template<typename T>
struct _equal<T,T>{
  enum { value = 1 };
};


template<int i,int j>
struct intEq{ static const bool val = false; };
template<int i>
struct intEq<i,i>{ static const bool val = true; };

template<bool v, typename T>
struct my_enable_if{};

template<typename T>
struct my_enable_if<true,T>{ typedef T type; };


template<typename T>
struct is_double_or_float{ enum {value = 0}; };

template<>
struct is_double_or_float<double>{ enum {value = 1}; };

template<>
struct is_double_or_float<float>{ enum {value = 1}; };


//A method of asking whether a type is an std::complex<double> or std::complex<float>
template<typename T>
struct is_complex_double_or_float{ enum {value = 0}; };

template<>
struct is_complex_double_or_float<std::complex<double> >{ enum {value = 1}; };

template<>
struct is_complex_double_or_float<std::complex<float> >{ enum {value = 1}; };

#ifdef USE_GRID
//A method of asking a Grid vector type if it's scalar_type is complex, predicated on whether the type is indeed a Grid vector type
template<bool is_grid_vector, typename T>
struct is_scalar_type_complex{};

template<typename T>
struct is_scalar_type_complex<true,T>{
  //safe to call scalar type
  enum {value = Grid::is_complex<typename T::scalar_type>::value };
};
template<typename T>
struct is_scalar_type_complex<false,T>{
  enum {value = 0};
};
  
//A method of asking whether a type is a Grid *complex* vector type
template<typename T>
struct is_grid_vector_complex{
  enum {value = is_scalar_type_complex<Grid::is_simd<T>::value, T>::value };
};
#else

template<typename T>
struct is_grid_vector_complex{
  enum {value = 0};
};

#endif


//A method of providing a list of conditions and associated types for classification
struct no_mark{};

template<bool Condition, typename IfTrue, typename NextTest>
struct TestElem{};

template<typename IfTrue, typename NextTest>
struct TestElem<true,IfTrue,NextTest>{
  typedef IfTrue type;
};
template<typename IfTrue, typename NextTest>
struct TestElem<false,IfTrue,NextTest>{
  typedef typename NextTest::type type;
};

struct LastElem{
  typedef no_mark type;
};


  
//An expandable method of classifying a complex type
struct complex_double_or_float_mark{};
struct grid_vector_complex_mark{};

template<typename T>
struct ComplexClassify{
  typedef typename TestElem< is_complex_double_or_float<T>::value, complex_double_or_float_mark,
			     TestElem< is_grid_vector_complex<T>::value, grid_vector_complex_mark,				       
				       LastElem>
			     >::type type;
};


#endif
