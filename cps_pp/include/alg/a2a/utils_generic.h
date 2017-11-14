#ifndef __UTILS_GENERIC_H_
#define __UTILS_GENERIC_H_

#include <cxxabi.h>
#include <util/lattice.h>
#include <alg/fix_gauge_arg.h>

CPS_START_NAMESPACE

//An empty class
class NullObject
{
 public:
  NullObject(){}
};

//A class inheriting from this type must have template parameter T as a double or float
#define EXISTS_IF_DOUBLE_OR_FLOAT(T) public my_enable_if<is_double_or_float<mf_Float>::value,NullObject>::type

//Skip gauge fixing and set all gauge fixing matrices to unity
inline void gaugeFixUnity(Lattice &lat, const FixGaugeArg &fix_gauge_arg){
  FixGaugeType fix = fix_gauge_arg.fix_gauge_kind;
  int start = fix_gauge_arg.hyperplane_start;
  int step = fix_gauge_arg.hyperplane_step;
  int num = fix_gauge_arg.hyperplane_num;

  int h_planes[num];
  for(int i=0; i<num; i++) h_planes[i] = start + step * i;

  lat.FixGaugeAllocate(fix, num, h_planes);
  
#pragma omp parallel for
  for(int sf=0;sf<(GJP.Gparity()+1)*GJP.VolNodeSites();sf++){
    //s + vol*f
    int s = sf % GJP.VolNodeSites();
    int f = sf / GJP.VolNodeSites();
    
    const Matrix* mat = lat.FixGaugeMatrix(s,f);
    if(mat == NULL) continue;
    else{
      Matrix* mm = const_cast<Matrix*>(mat); //evil, I know, but it saves duplicating the accessor (which is overly complicated)
      mm->UnitMatrix();
    }
  }
}


//The base G-parity momentum vector for quark fields with arbitrary sign
inline void GparityBaseMomentum(int p[3], const int sgn){
  for(int i=0;i<3;i++)
    if(GJP.Bc(i) == BND_CND_GPARITY)
      p[i] = sgn;
    else p[i] = 0;
}

//String to anything conversion
template<typename T>
inline T strToAny(const char* a){
  std::stringstream ss; ss << a; T o; ss >> o;
  return o;
}
template<typename T>
inline std::string anyToStr(const T &t){
  std::ostringstream os; os << t;
  return os.str();
}

std::string demangle( const char* mangled_name ) {

  std::size_t len = 0 ;
  int status = 0 ;
  char* unmangled = __cxxabiv1::__cxa_demangle( mangled_name, NULL, &len, &status );
  if(status == 0){  
    std::string out(unmangled);
    free(unmangled);
    return out;
  }else return "<demangling failed>";
  
  /* std::unique_ptr< char, decltype(&std::free) > ptr( */
  /* 						    __cxxabiv1::__cxa_demangle( mangled_name, nullptr, &len, &status ), &std::free ) ; */
  /* return ptr.get() ; */
}

//Print a type as a string (useful for debugging)
template<typename T>
inline std::string printType(){ return demangle(typeid(T).name()); }

CPS_END_NAMESPACE

#endif
