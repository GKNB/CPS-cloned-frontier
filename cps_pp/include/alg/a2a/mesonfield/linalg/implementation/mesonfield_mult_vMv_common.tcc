#ifndef MESONFIELD_MULT_VMV_COMMON_H_
#define MESONFIELD_MULT_VMV_COMMON_H_

//Policy classes to get appropriate output matrix type for v*(M)*v operations
struct _mult_vMv_impl_v_GparityPolicy{
  template<typename T> using MatrixType = CPSspinColorFlavorMatrix<T>;
  constexpr static int nf(){ return 2; }
  template<typename T> static inline T & acc(const int s1, const int s2, const int c1, const int c2, const int f1, const int f2, MatrixType<T> &M){ return M(s1,s2)(c1,c2)(f1,f2); }
};
struct _mult_vMv_impl_v_StandardPolicy{
  template<typename T> using MatrixType = CPSspinColorMatrix<T>;
  constexpr static int nf(){ return 1; }
  template<typename T> static inline T & acc(const int s1, const int s2, const int c1, const int c2, const int f1, const int f2, MatrixType<T> &M){ return M(s1,s2)(c1,c2); }
};

//Get appropriate policy based on compile time int (from a2a policy)
template<int isGparity>
struct _mult_vMv_impl_v_getPolicy{};

template<>
struct _mult_vMv_impl_v_getPolicy<0>{
  typedef _mult_vMv_impl_v_StandardPolicy type;
};

template<>
struct _mult_vMv_impl_v_getPolicy<1>{
  typedef _mult_vMv_impl_v_GparityPolicy type;
};





#endif
