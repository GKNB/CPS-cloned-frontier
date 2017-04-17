#ifndef CK_A2A_UTILS
#define CK_A2A_UTILS

#include <util/lattice.h>
#include <alg/fix_gauge_arg.h>

#include <alg/a2a/utils_array.h>
#include <alg/a2a/utils_floatingpt.h>
#include <alg/a2a/utils_memory.h>
#include <alg/a2a/utils_complex.h>
#include <alg/a2a/utils_matrix.h>
#include <alg/a2a/utils_parallel.h>

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


CPS_END_NAMESPACE

#endif
