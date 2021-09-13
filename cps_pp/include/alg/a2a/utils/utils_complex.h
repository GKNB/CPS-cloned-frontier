#ifndef _UTILS_COMPLEX_H_
#define _UTILS_COMPLEX_H_

#include <alg/qpropw_arg.h>
#include <util/random.h>
#include "template_wizardry.h"
//Useful functions for complex numbers

CPS_START_NAMESPACE

//Set the complex number at pointer p to a random value of a chosen type
//Uses the current LRG for the given FermionFieldDimension. User should choose the range and the particular site-RNG themselves beforehand
template<typename mf_Float>
class RandomComplex{};

//Only for float and double, hence I have to control its access
template<typename mf_Float>
class RandomComplexBase{
 protected:
  template<typename T> friend class RandomComplex;
  
  static void rand(mf_Float *p, const RandomType type, const FermionFieldDimension frm_dim){
    static const Float PI = 3.14159265358979323846;
    Float theta = LRG.Urand(frm_dim);
  
    switch(type) {
    case UONE:
      p[0] = cos(2. * PI * theta);
      p[1] = sin(2. * PI * theta);
      break;
    case ZTWO:
      p[0] = theta > 0.5 ? 1 : -1;
      p[1] = 0;
      break;
    case ZFOUR:
      if(theta > 0.75) {
	p[0] = 1;
	p[1] = 0;
      }else if(theta > 0.5) {
	p[0] = -1;
	p[1] = 0;
      }else if(theta > 0.25) {
	p[0] = 0;
	p[1] = 1;
      }else {
	p[0] = 0;
	p[1] = -1;
      }
      break;
    default:
      ERR.NotImplemented("RandomComplexBase", "rand(...)");
    }
  }
};

template<typename T>
class RandomComplex<std::complex<T> > : public RandomComplexBase<T>{
public:
  static void rand(std::complex<T> *p, const RandomType &type, const FermionFieldDimension &frm_dim){
    RandomComplexBase<T>::rand( (T*)p, type, frm_dim);
  }
};

#ifdef GRID_CUDA
template<typename T>
class RandomComplex<thrust::complex<T> > : public RandomComplexBase<T>{
public:
  static void rand(thrust::complex<T> *p, const RandomType &type, const FermionFieldDimension &frm_dim){
    std::complex<T> tmp;
    RandomComplexBase<T>::rand( (T*)&tmp, type, frm_dim);
    *p = tmp;
  }
};
#endif

//Wrapper function to multiply a number by +/-i for std::complex and Grid complex

template<typename T, typename T_class>
struct _mult_sgn_times_i_impl{};

template<typename T>
struct _mult_sgn_times_i_impl<T,complex_double_or_float_mark>{
  accelerator_inline static T doit(const int sgn, const T &val){
    return T( -sgn * val.imag(), sgn * val.real() ); // sign * i * val
  }
};

#ifdef USE_GRID
template<typename T>
struct _mult_sgn_times_i_impl<T,grid_vector_complex_mark>{
  accelerator_inline static T doit(const int sgn, const T &val){
    return sgn == -1 ? timesMinusI(val) : timesI(val);
  }
};

template<typename T>
struct _mult_sgn_times_i_impl<T,grid_iscalar_complex_double_or_float_mark>{
  accelerator_inline static T doit(const int sgn, const T &val){
    return sgn == -1 ? timesMinusI(val) : timesI(val);
  }
};

#endif


template<typename T>
accelerator_inline T multiplySignTimesI(const int sgn, const T &val){
  return _mult_sgn_times_i_impl<T,typename ComplexClassify<T>::type>::doit(sgn,val);
}


//Wrapper function to take complex conjugate for std::complex and Grid complex
template<typename T, typename ComplexClass>
struct _cconj{};

template<typename T>
struct _cconj<std::complex<T>,complex_double_or_float_mark>{
  static accelerator_inline std::complex<T> doit(const std::complex<T> &in){ return std::conj(in); }
};

#ifdef USE_GRID

#ifdef GRID_CUDA
//Grid's complex uses thrust
template<typename T>
struct _cconj<Grid::complex<T>,complex_double_or_float_mark>{
  static accelerator_inline Grid::complex<T> doit(const Grid::complex<T> &in){ return Grid::conjugate(in); }
};
#endif

template<typename T>
struct _cconj<T,grid_vector_complex_mark>{
  static accelerator_inline T doit(const T &in){ return Grid::conjugate(in); }
};
#endif

template<typename T>
accelerator_inline T cconj(const T& in){
  return _cconj<T,typename ComplexClassify<T>::type>::doit(in);
}

//Wrapper function for equals for std::complex and Grid complex
template<typename T>
inline bool equals(const T &a, const T &b){ return a == b; }

#ifdef USE_GRID
inline bool equals(const Grid::vComplexD &a, const Grid::vComplexD &b){
  Grid::vComplexD::conv_t ac, bc;
  ac.v = a.v; bc.v = b.v;
  for(int i=0;i<Grid::vComplexD::Nsimd();i++) if(ac.s[i] != bc.s[i]) return false;
  return true;
}
inline bool equals(const Grid::vComplexF &a, const Grid::vComplexF &b){
  Grid::vComplexF::conv_t ac, bc;
  ac.v = a.v; bc.v = b.v;
  for(int i=0;i<Grid::vComplexF::Nsimd();i++) if(ac.s[i] != bc.s[i]) return false;
  return true;
}
#endif

//Convert Grid or std complex number to double precision std::complex
//For vectorized types it will be reduced prior to conversion
template<typename T>
inline std::complex<double> convertComplexD(const std::complex<T> &what){
  return what;
}
#ifdef GRID_CUDA
//Need explicit versions for thrust complex otherwise it will implicitly convert to vComplex and then reduce the result,
//resulting in an overall multiplicative factor of Nsimd being applied!
inline std::complex<double> convertComplexD(const Grid::ComplexD &what){
  return std::complex<double>(what.real(), what.imag());
}
inline std::complex<double> convertComplexD(const Grid::ComplexF &what){
  return std::complex<double>(what.real(), what.imag());
}
#endif

#ifdef USE_GRID
inline std::complex<double> convertComplexD(const Grid::vComplexD &what){
  auto v = Reduce(what);
  return std::complex<double>(v.real(),v.imag());
}
inline std::complex<double> convertComplexD(const Grid::vComplexF &what){
  auto v = Reduce(what);
  return std::complex<double>(v.real(),v.imag());
}
#endif






CPS_END_NAMESPACE


#endif
