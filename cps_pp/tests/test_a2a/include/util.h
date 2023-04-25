#pragma once

CPS_START_NAMESPACE

//Get default CPSfield SIMD parameters based on complex type
template<typename ParamType, typename mf_Complex>
struct defaultFieldParams{
  static void get(ParamType &into){}
};

template<int N, typename mf_Complex>
struct defaultFieldParams< SIMDdims<N>, mf_Complex >{
  static void get(SIMDdims<N> &into){
    SIMDpolicyBase<N>::SIMDdefaultLayout(into, mf_Complex::Nsimd(), 2);
  }
};

//Print float or complex
template<typename T>
struct _printit{
  static void printit(const T d[], const int n){
    for(int i=0;i<n;i++){
      std::cout << d[i] << " ";
    }
    std::cout << std::endl;
  }
};

template<typename T>
struct _printit<std::complex<T> >{
  static void printit(const std::complex<T> d[], const int n){
    for(int i=0;i<n;i++){
      std::cout << '[' << d[i].real() << ',' << d[i].imag() << "] ";
    }
    std::cout << std::endl;
  }
};

  
template<typename T>
void printit(const T d[], const int n){
  _printit<T>::printit(d,n);
}

//Print Grid vector type
template<typename T>
void printvType(const T& v){
  typedef typename T::scalar_type S;
  int Nsimd = T::Nsimd();
  S to[Nsimd];
  vstore(v,to);
  printit(to,Nsimd);
}


//Random Grid vector type
template<typename T>
struct _rand{
  inline static T rand(){
    return LRG.Urand();
  }
};

template<typename T>
struct _rand<std::complex<T> >{
  inline static std::complex<T> rand(){
    return std::complex<T>(LRG.Urand(),LRG.Urand());
  }
};

template<typename T>
T randomvType(){
  T out;
  typedef typename T::scalar_type S;
  int Nsimd = T::Nsimd();
  S r[Nsimd];
  for(int i=0;i<Nsimd;i++) r[i] = _rand<S>::rand();
  vset(out,r);
  return out;
}


//Equality for FlavorMatrixGeneral
template<typename T, typename ComplexClass>
struct _fmatequals{};

template<typename T>
struct _fmatequals<T,complex_double_or_float_mark>{
  static bool equals(const FlavorMatrixGeneral<T> &l, const FlavorMatrixGeneral<T> &r, double tol = 1e-10){
    for(int i=0;i<2;i++){
      for(int j=0;j<2;j++){
	const T &aa = l(i,j);
	const T &bb = r(i,j);	
	if(fabs(aa.real() - bb.real()) > tol)
	  return false;
	if(fabs(aa.imag() - bb.imag()) > tol)
	  return false;
      }
    }
    return true;
  }
};

template<typename T>
struct _fmatequals<T,grid_vector_complex_mark>{
  static bool equals(const FlavorMatrixGeneral<T> &l, const FlavorMatrixGeneral<T> &r, double tol = 1e-10){
    for(int i=0;i<2;i++){
      for(int j=0;j<2;j++){
	auto aa = Reduce(l(i,j));
	auto bb = Reduce(r(i,j));	
	if(fabs(aa.real() - bb.real()) > tol)
	  return false;
	if(fabs(aa.imag() - bb.imag()) > tol)
	  return false;
      }
    }
    return true;
  }
};

template<typename T>
inline bool equals(const FlavorMatrixGeneral<T> &l, const FlavorMatrixGeneral<T> &r, double tol = 1e-10){
  return _fmatequals<T,typename ComplexClassify<T>::type>::equals(l,r,tol);
}


//Compare a2a vectors
//Both should be scalar matrices
template<typename A2Apolicies_1, typename A2Apolicies_2, template<typename> class L, template<typename> class R>
bool compare(const std::vector<A2AmesonField<A2Apolicies_1,L,R> > &M1, const std::vector<A2AmesonField<A2Apolicies_2,L,R> > &M2, double tol){
  if(M1.size() != M2.size()){
    std::cout << "Fail: time vector size mismatch" << std::endl;
    return false;
  }
  for(int t=0;t<M1.size();t++){
    if(M1[t].getNrows() != M2[t].getNrows() ||
       M1[t].getNcols() != M2[t].getNcols() ){
      std::cout << "Fail: matrix size mismatch" << std::endl;
      return false;
    }
    if(M1[t].getRowTimeslice() != M2[t].getRowTimeslice() || 
       M1[t].getColTimeslice() != M2[t].getColTimeslice() ){
      std::cout << "Fail: matrix timeslice mismatch" << std::endl;
      return false;      
    }
      
    for(int i=0;i<M1[t].getNrows();i++){
      for(int j=0;j<M1[t].getNcols();j++){
	auto v1 = M1[t](i,j);
	auto v2 = M2[t](i,j);
	if(fabs(v1.real() - v2.real()) > tol ||
	   fabs(v1.imag() - v2.imag()) > tol){
	  std::cout << "Fail " << i << " " << j << " :  (" << v1.real() << "," << v1.imag() << ")  (" << v2.real() << "," << v2.imag() << ")  diff (" << v1.real()-v2.real() << "," << v1.imag()-v2.imag() << ")" << std::endl;
	  return false;
	}
      }
    }
  }
  return true;
}


//Copy a2a vector, allow for different policies
//Both should be scalar matrices
template<typename A2Apolicies_1, typename A2Apolicies_2, template<typename> class L, template<typename> class R>
void copy(std::vector<A2AmesonField<A2Apolicies_1,L,R> > &Mout, const std::vector<A2AmesonField<A2Apolicies_2,L,R> > &Min){
  assert(Mout.size() == Min.size());
  for(int t=0;t<Min.size();t++){
    assert(Mout[t].getNrows() == Min[t].getNrows() && Min[t].getNcols() == Min[t].getNcols());
    assert(Mout[t].getRowTimeslice() == Min[t].getRowTimeslice() && Mout[t].getColTimeslice() == Min[t].getColTimeslice());
    for(int i=0;i<Min[t].getNrows();i++){
      for(int j=0;j<Min[t].getNcols();j++){
	Mout[t](i,j) = Min[t](i,j);
      }
    }
  }
}


template<typename mf_Complex, typename grid_Complex>
bool compare(const CPSspinColorFlavorMatrix<mf_Complex> &orig, const CPSspinColorFlavorMatrix<grid_Complex> &grid, const double tol){
  bool fail = false;
  
  mf_Complex gd;
  for(int sl=0;sl<4;sl++)
    for(int cl=0;cl<3;cl++)
      for(int fl=0;fl<2;fl++)
	for(int sr=0;sr<4;sr++)
	  for(int cr=0;cr<3;cr++)
	    for(int fr=0;fr<2;fr++){
	      gd = Reduce( grid(sl,sr)(cl,cr)(fl,fr) );
	      const mf_Complex &cp = orig(sl,sr)(cl,cr)(fl,fr);
	      
	      double rdiff = fabs(gd.real()-cp.real());
	      double idiff = fabs(gd.imag()-cp.imag());
	      if(rdiff > tol|| idiff > tol){
		printf("Fail: Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
		fail = true;
	      }
	    }
  return !fail;
}

template<typename mf_Complex>
bool compare(const CPSspinColorFlavorMatrix<mf_Complex> &orig, const CPSspinColorFlavorMatrix<mf_Complex> &newimpl, const double tol){
  bool fail = false;
  
  for(int sl=0;sl<4;sl++)
    for(int cl=0;cl<3;cl++)
      for(int fl=0;fl<2;fl++)
	for(int sr=0;sr<4;sr++)
	  for(int cr=0;cr<3;cr++)
	    for(int fr=0;fr<2;fr++){
	      const mf_Complex &gd = newimpl(sl,sr)(cl,cr)(fl,fr);
	      const mf_Complex &cp = orig(sl,sr)(cl,cr)(fl,fr);
	      
	      double rdiff = fabs(gd.real()-cp.real());
	      double idiff = fabs(gd.imag()-cp.imag());
	      if(rdiff > tol|| idiff > tol){
		printf("Fail: Newimpl (%g,%g) Orig (%g,%g) Diff (%g,%g)\n",gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
		fail = true;
	      }
	    }
  return !fail;
}


template<typename mf_Complex, typename grid_Complex>
bool compare(const CPSspinColorMatrix<mf_Complex> &orig, const CPSspinColorMatrix<grid_Complex> &grid, const double tol){
  bool fail = false;
  
  mf_Complex gd;
  for(int sl=0;sl<4;sl++)
    for(int cl=0;cl<3;cl++)
      for(int sr=0;sr<4;sr++)
	for(int cr=0;cr<3;cr++){
	  gd = Reduce( grid(sl,sr)(cl,cr) );
	  const mf_Complex &cp = orig(sl,sr)(cl,cr);
  
	  double rdiff = fabs(gd.real()-cp.real());
	  double idiff = fabs(gd.imag()-cp.imag());
	  if(rdiff > tol|| idiff > tol){
	    printf("Fail: Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
	    fail = true;
	  }
	}
  return !fail;
}

template<typename mf_Complex>
bool compare(const CPSspinColorMatrix<mf_Complex> &orig, const CPSspinColorMatrix<mf_Complex> &newimpl, const double tol){
  bool fail = false;
  
  for(int sl=0;sl<4;sl++)
    for(int cl=0;cl<3;cl++)
      for(int sr=0;sr<4;sr++)
	for(int cr=0;cr<3;cr++){
	  const mf_Complex &gd = newimpl(sl,sr)(cl,cr);
	  const mf_Complex &cp = orig(sl,sr)(cl,cr);
	  
	  double rdiff = fabs(gd.real()-cp.real());
	  double idiff = fabs(gd.imag()-cp.imag());
	  if(rdiff > tol|| idiff > tol){
	    printf("Fail: Newimpl (%g,%g) Orig (%g,%g) Diff (%g,%g)\n",gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
	    fail = true;
	  }
	}
  return !fail;
}


std::ostream & operator<<(std::ostream &os, const std::pair<int,int> &p){
  os << "(" << p.first << ", " << p.second << ")";
  return os;
}

template<typename T>
void _expect_eq(const T &a, const T &b, const char* file, const int line){
  if(!UniqueID()) std::cout << file << ":" << line << " : Expected equal " << a << " " << b << std::endl;
  if(a!=b) exit(1);
}
#define EXPECT_EQ(A,B) _expect_eq<typename std::decay<decltype(A)>::type>(A,B, __FILE__, __LINE__)


//Print a row of a field in a given direction with other ordinates set to 0, e.g.  (0,0,0,0) (1,0,0,0) (2,0,0,0) ...
template< typename mf_Complex, int SiteSize, typename FlavorPolicy, typename AllocPolicy>
void printRow(const CPSfield<mf_Complex,SiteSize,FourDpolicy<FlavorPolicy>,AllocPolicy> &field, const int dir, const std::string &comment,
	       typename my_enable_if< _equal<typename ComplexClassify<mf_Complex>::type, complex_double_or_float_mark>::value, const int>::type = 0
	       ){
  int L = GJP.Nodes(dir)*GJP.NodeSites(dir);
  std::vector<mf_Complex> buf(L,0.);

  int other_dirs[3]; int aa=0;
  for(int i=0;i<4;i++)
    if(i!=dir) other_dirs[aa++] = i;

  CPSautoView(field_v,field,HostRead);
  if(GJP.NodeCoor(other_dirs[0]) == 0 && GJP.NodeCoor(other_dirs[1]) == 0 && GJP.NodeCoor(other_dirs[2]) == 0){
    for(int x=GJP.NodeCoor(dir)*GJP.NodeSites(dir); x < (GJP.NodeCoor(dir)+1)*GJP.NodeSites(dir); x++){
      int lcoor[4] = {0,0,0,0};
      lcoor[dir] = x - GJP.NodeCoor(dir)*GJP.NodeSites(dir);
      
      mf_Complex const* site_ptr = field_v.site_ptr(lcoor);
      buf[x] = *site_ptr;
    }
  }
  globalSum(buf.data(),L);

  
  if(!UniqueID()){
    printf("%s: (",comment.c_str()); fflush(stdout);
    for(int x=0;x<L;x++){
      if(x % GJP.NodeSites(dir) == 0 && x!=0)
	printf(")(");
      
      printf("[%f,%f] ",buf[x].real(),buf[x].imag());
    }
    printf(")\n"); fflush(stdout);
  }
}

#ifdef USE_GRID
template< typename mf_Complex, int SiteSize, typename FlavorPolicy, typename AllocPolicy>
void printRow(const CPSfield<mf_Complex,SiteSize,FourDSIMDPolicy<FlavorPolicy>,AllocPolicy> &field, const int dir, const std::string &comment,
	       typename my_enable_if< _equal<typename ComplexClassify<mf_Complex>::type, grid_vector_complex_mark>::value, const int>::type = 0
	       ){
  typedef typename mf_Complex::scalar_type ScalarComplex;
  NullObject null_obj;
  CPSfield<ScalarComplex,SiteSize,FourDpolicy<FlavorPolicy>,UVMallocPolicy> tmp(null_obj);
  tmp.importField(field);
  printRow(tmp,dir,comment);
}
#endif


#ifdef USE_GRID

template<typename T>
typename my_enable_if< is_complex_double_or_float<typename T::scalar_type>::value, bool>::type 
vTypeEquals(const T& a, const T &b, const double tolerance = 1e-12, bool verbose = false){
  typedef typename T::scalar_type S;
  int Nsimd = T::Nsimd();
  S ato[Nsimd];
  vstore(a,ato);
  S bto[Nsimd];
  vstore(b,bto);
  
  bool eq = true;
  for(int i=0;i<Nsimd;i++)
    if( fabs(ato[i].real() - bto[i].real()) > tolerance || fabs(ato[i].imag() - bto[i].imag()) > tolerance ){
      if(verbose && !UniqueID()){	
	double rdiff = fabs(ato[i].real() - bto[i].real());
	double idiff = fabs(ato[i].imag() - bto[i].imag());
	printf("Mismatch index %d: (%g,%g) vs (%g,%g) with diffs (%g,%g)\n",i,ato[i].real(),bto[i].real(),ato[i].imag(),bto[i].imag(),rdiff,idiff);
      }
      eq = false; break;
    }

  if(!eq && verbose && !UniqueID()){
    printf("NOT EQUAL:\n");
    printit(ato,Nsimd);
    printit(bto,Nsimd);
  }    
  return eq;
}

#endif //USE_GRID


//Setup SIMD fields
template<typename A2Apolicies, typename ComplexClass>
struct setupFieldParams2{};

template<typename A2Apolicies>
struct setupFieldParams2<A2Apolicies,complex_double_or_float_mark>{
  NullObject params;
};

template<typename A2Apolicies>
struct setupFieldParams2<A2Apolicies, grid_vector_complex_mark>{
  typename FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType params;
  setupFieldParams2(){
    const int nsimd = A2Apolicies::ComplexType::Nsimd();
    FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(params,nsimd,2);
  }
};


//Compare two Grid tensors
#ifdef USE_GRID
template<typename T>
bool GridTensorEquals(const T &a, const T &b){
  typedef typename T::vector_type vtype;
  const int sz = sizeof(T)/sizeof(vtype);

  vtype const* va = (vtype const*)&a;
  vtype const* vb = (vtype const*)&b;
  
  for(int i=0;i<sz;i++){
    if( ! equals(va[i], vb[i])) return false;  
  }
  return true;
}
#endif //USE_GRID

CPS_END_NAMESPACE
