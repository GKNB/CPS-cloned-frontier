#ifndef CK_A2A_UTILS
#define CK_A2A_UTILS

#include <alg/a2a/gsl_wrapper.h>
#include <alg/a2a/template_wizardry.h>
#include <util/spincolorflavormatrix.h>

CPS_START_NAMESPACE

//3x3 complex vector multiplication with different precision matrices and vectors
template<typename VecFloat, typename MatFloat>
void colorMatrixMultiplyVector(VecFloat* y, const MatFloat* u, const VecFloat* x){
	*y     =  *u      * *x     - *(u+1)  * *(x+1) + *(u+2)  * *(x+2)
		- *(u+3)  * *(x+3) + *(u+4)  * *(x+4) - *(u+5)  * *(x+5);
	*(y+1) =  *u      * *(x+1) + *(u+1)  * *x     + *(u+2)  * *(x+3)
		+ *(u+3)  * *(x+2) + *(u+4)  * *(x+5) + *(u+5)  * *(x+4);
	*(y+2) =  *(u+6)  * *x     - *(u+7)  * *(x+1) + *(u+8)  * *(x+2)
		- *(u+9)  * *(x+3) + *(u+10) * *(x+4) - *(u+11) * *(x+5);
	*(y+3) =  *(u+6)  * *(x+1) + *(u+7)  * *x     + *(u+8)  * *(x+3)
		+ *(u+9)  * *(x+2) + *(u+10) * *(x+5) + *(u+11) * *(x+4);
	*(y+4) =  *(u+12) * *x     - *(u+13) * *(x+1) + *(u+14) * *(x+2)
		- *(u+15) * *(x+3) + *(u+16) * *(x+4) - *(u+17) * *(x+5);
	*(y+5) =  *(u+12) * *(x+1) + *(u+13) * *x     + *(u+14) * *(x+3)
		+ *(u+15) * *(x+2) + *(u+16) * *(x+5) + *(u+17) * *(x+4);
}

//Array *= with cps::Float(=double) input and arbitrary precision output
template<typename FloatOut,typename FloatIn>
void VecTimesEquFloat(FloatOut *out, FloatIn *in, const Float fac, const int len) 
{
#pragma omp parallel for
	for(int i = 0; i < len; i++) out[i] = in[i] * fac;
}

inline void getNodeWork(const int work, int &node_work, int &node_off, bool &do_work, const bool node_local = false){
  if(node_local){ node_work = work; node_off = 0; do_work = true; return; } //node does all the work

  int nodes = 1; for(int i=0;i<5;i++) nodes *= GJP.Nodes(i);
  int me = UniqueID();

  //Stolen from BFM :)
  int basework = work/nodes;
  int backfill = nodes-(work % nodes);
  node_work = (work+me)/nodes;
  node_off  = basework * me;
  if ( me > backfill ) 
    node_off+= (me-backfill);
  if(node_work > 0) do_work = true;
}


  // do_work = true;
  // if(nodes > work){
  //   nodes = work; if(UniqueID() >= work) do_work = false; //too many nodes, at least for this parallelization. Might want to consider parallelizing in a different way!
  // }

  // node_work = work/nodes;
  // if(node_work * nodes < work){
  //   int remaining_work = work - node_work * nodes;
  //   if(UniqueID()<remaining_work) node_work++;


  //   node_work += work - node_work * nodes; //node 0 mops up remainder

  //   node_off = UniqueID()*node_work;


inline void compute_overlap(std::vector<bool> &out, const std::vector<bool> &a, const std::vector<bool> &b){
  assert(a.size()==b.size());
  out.resize(a.size());
  for(int i=0;i<a.size();i++) out[i] = a[i] && b[i];
}

class NullObject
{
 public:
  NullObject(){}
};

//A class inheriting from this type must have template parameter T as a double or float
#define EXISTS_IF_DOUBLE_OR_FLOAT(T) public my_enable_if<is_double_or_float<mf_Float>::value,NullObject>::type

//Functions for performing global and timeslice sums of single or double precision quantities. Daiqian had to implement these himself as CPS can only do this with the Float=double type

// My global sum
template <typename T>
void QMP_sum_array(T *result, int len){
#ifdef USE_QMP
  if(sizeof(T) == sizeof(double)) {
    QMP_sum_double_array((double*)result, len);
  } else if(sizeof(T) == sizeof(float)) {
    QMP_sum_float_array((float*)result, len);
  } else {
    QMP_error("QMP_sum_array::data type not supported!\n");
  }
#else
  //CK: This only works for single-node code
  int nodes = 1; for(int i=0;i<4;i++) nodes *= cps::GJP.Nodes(i);
  if(nodes != 1){
    cps::ERR.General("","QMP_sum_array(T *result, int len)","Only implemented for QMP on parallel machines");
  }
  //do nothing!
#endif
}

#ifndef USE_QMP
  inline void QMP_sum_double_array(double *result, int len){
    //CK: This only works for single-node code
    int nodes = 1; for(int i=0;i<4;i++) nodes *= cps::GJP.Nodes(i);
    if(nodes != 1){
      cps::ERR.General("","QMP_sum_double_array fake definition","Not implemented on parallel machines: use QMP!");
    }
  }
  inline void QMP_sum_float_array(float *result, int len){
    //CK: This only works for single-node code
    int nodes = 1; for(int i=0;i<4;i++) nodes *= cps::GJP.Nodes(i);
    if(nodes != 1){
      cps::ERR.General("","QMP_sum_float_array fake definition","Not implemented on parallel machines: use QMP!");
    }
  }
#endif

//Look for contiguous blocks of indices in the idx_map, output a list of start,size pairs
inline void find_contiguous_blocks(std::vector<std::pair<int,int> > &blocks, const int idx_map[], int map_size){
  std::pair<int,int> block(0,1); //start, size
  int prev = idx_map[0];
  for(int j_packed=1;j_packed<map_size;j_packed++){
    int j_unpacked = idx_map[j_packed];
    if(j_unpacked == prev+1){
      ++block.second;
    }else{
      blocks.push_back(block);
      block.first = j_packed;
      block.second = 1;      
    }
    prev = j_unpacked;
  }
  blocks.push_back(block);
}

template<typename T>
inline void resize_2d(std::vector<std::vector<T> > &v, const size_t i, const size_t j){
  v.resize(i);
  for(int a=0;a<i;a++) v[a].resize(j);
}
template<typename T>
inline void resize_3d(std::vector<std::vector<std::vector<T> > > &v, const size_t i, const size_t j, const size_t k){
  v.resize(i);
  for(int a=0;a<i;a++){
    v[a].resize(j);
    for(int b=0;b<j;b++)
      v[a][b].resize(k);
  }
}

inline std::complex<double> GSLtrace(const SpinColorFlavorMatrix& a, const SpinColorFlavorMatrix& b){

  const int scf_size = 24;
  std::complex<double> _a[scf_size][scf_size];
  std::complex<double> _bT[scf_size][scf_size];   //In-place transpose of b so rows are contiguous
  for(int i=0;i<scf_size;i++){
    int rem = i;
    int ci = rem % 3; rem /= 3;
    int si = rem % 4; rem /= 4;
    int fi = rem;
    
    for(int j=0;j<scf_size;j++){
      rem = j;
      int cj = rem % 3; rem /= 3;
      int sj = rem % 4; rem /= 4;
      int fj = rem;
      
      _bT[i][j] = b(sj,cj,fj, si,ci,fi);
      _a[i][j] = a(si,ci,fi, sj,cj,fj);
    }
  }

  double* ad = (double*)&_a[0][0];
  double* bd = (double*)&_bT[0][0];

  gsl_block_complex_struct ablock;
  ablock.size = 24*24;
  ablock.data = ad;

  gsl_vector_complex arow; //single row of a
  arow.block = &ablock;
  arow.owner = 0;
  arow.size = 24;
  arow.stride = 1;
  
  gsl_block_complex_struct bblock;
  bblock.size = 24*24;
  bblock.data = bd;

  gsl_vector_complex bcol; //single col of b
  bcol.block = &bblock;
  bcol.owner = 0;
  bcol.size = 24;
  bcol.stride = 1;

  //gsl_blas_zdotu (const gsl_vector_complex * x, const gsl_vector_complex * y, gsl_complex * dotu)
  //   //  a[0][0]*b[0][0] + a[0][1]*b[1][0] + a[0][2]*b[2][0] + ...
  //   //+ a[1][0]*b[0][1] + a[1][1]*b[1][1] + a[1][2]*b[2][1] + ....
  //   //...

  std::complex<double> out(0.0);
  gsl_complex tmp;
  for(int i=0;i<24;i++){
    arow.data = ad + 24*2*i; //i'th row offset
    bcol.data = bd + 24*2*i; //i'th col offset (remember we transposed it)

    gsl_blas_zdotu(&arow, &bcol, &tmp);
    reinterpret_cast<double(&)[2]>(out)[0] += GSL_REAL(tmp);
    reinterpret_cast<double(&)[2]>(out)[1] += GSL_IMAG(tmp);
  }
  return out;
}


//For a Nrows*Ncols matrix 'to' with elements in the standard order  idx=(Ncols*i + j), poke a submatrix into it with origin (i0,j0) and size (ni,nj)
template<typename T>
void pokeSubmatrix(T* to, const T* sub, const int Nrows, const int Ncols, const int i0, const int j0, const int ni, const int nj, const bool threaded = false){
  #define DOIT \
    for(int row = i0; row < i0+ni; row++){ \
      T* to_block = to + row*Ncols + j0;	  \
      const T* from_block = sub + (row-i0)*nj;	\
      memcpy(to_block,from_block,nj*sizeof(T));	\
    }
  if(threaded){
#pragma omp parallel for
    DOIT;
  }else{
    DOIT;
  }
  #undef DOIT
}
//For a Nrows*Ncols matrix 'from' with elements in the standard order  idx=(Ncols*i + j), get a submatrix with origin (i0,j0) and size (ni,nj) and store in sub
template<typename T>
void getSubmatrix(T* sub, const T* from, const int Nrows, const int Ncols, const int i0, const int j0, const int ni, const int nj, const bool threaded = false){
  #define DOIT \
    for(int row = i0; row < i0+ni; row++){		\
      const T* from_block = from + row*Ncols + j0;	\
      T* to_block = sub + (row-i0)*nj;			\
      memcpy(to_block,from_block,nj*sizeof(T));		\
    }
  if(threaded){
#pragma omp parallel for
    DOIT;
  }else{
    DOIT;
  }
  #undef DOIT
}


//Simple test allocator to find out when memory is allocated
template <typename T>
class mmap_allocator: public std::allocator<T>{
public:
  typedef size_t size_type;
  typedef T* pointer;
  typedef const T* const_pointer;

  template<typename _Tp1>
  struct rebind{
    typedef mmap_allocator<_Tp1> other;
  };

  pointer allocate(size_type n, const void *hint=0){
    fprintf(stderr, "Alloc %d bytes.\n", n*sizeof(T));
    return std::allocator<T>::allocate(n, hint);
  }

  void deallocate(pointer p, size_type n){
    fprintf(stderr, "Dealloc %d bytes (%p).\n", n*sizeof(T), p);
    return std::allocator<T>::deallocate(p, n);
  }

  mmap_allocator() throw(): std::allocator<T>() { fprintf(stderr, "Hello allocator!\n"); }
  mmap_allocator(const mmap_allocator &a) throw(): std::allocator<T>(a) { }
  template <class U>                    
  mmap_allocator(const mmap_allocator<U> &a) throw(): std::allocator<T>(a) { }
  ~mmap_allocator() throw() { }
};


CPS_END_NAMESPACE
#ifdef ARCH_BGQ
#include <spi/include/kernel/memory.h>
#else
#include <sys/sysinfo.h>
#endif
CPS_START_NAMESPACE

inline void printMem(){
#ifdef ARCH_BGQ
  #warning "printMem using ARCH_BGQ"
  uint64_t shared, persist, heapavail, stackavail, stack, heap, guard, mmap;
  Kernel_GetMemorySize(KERNEL_MEMSIZE_SHARED, &shared);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_PERSIST, &persist);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPAVAIL, &heapavail);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_STACKAVAIL, &stackavail);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_STACK, &stack);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAP, &heap);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_GUARD, &guard);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_MMAP, &mmap);

  if(!UniqueID()){
    printf("printMem: Allocated heap: %.2f MB, avail. heap: %.2f MB\n", (double)heap/(1024*1024),(double)heapavail/(1024*1024));
    printf("printMem: Allocated stack: %.2f MB, avail. stack: %.2f MB\n", (double)stack/(1024*1024), (double)stackavail/(1024*1024));
    printf("printMem: Memory: shared: %.2f MB, persist: %.2f MB, guard: %.2f MB, mmap: %.2f MB\n", (double)shared/(1024*1024), (double)persist/(1024*1024), (double)guard/(1024*1024), (double)mmap/(1024*1024));
  }
#else
#warning "printMem using NOARCH"
  /* unsigned long totalram;  /\* Total usable main memory size *\/ */
  /* unsigned long freeram;   /\* Available memory size *\/ */
  /* unsigned long sharedram; /\* Amount of shared memory *\/ */
  /* unsigned long bufferram; /\* Memory used by buffers *\/ */
  /* unsigned long totalswap; /\* Total swap space size *\/ */
  /* unsigned long freeswap;  /\* swap space still available *\/ */
  /* unsigned short procs;    /\* Number of current processes *\/ */
  /* unsigned long totalhigh; /\* Total high memory size *\/ */
  /* unsigned long freehigh;  /\* Available high memory size *\/ */
  /* unsigned int mem_unit;   /\* Memory unit size in bytes *\/ */

  struct sysinfo myinfo;
  sysinfo(&myinfo);
  double total_mem = myinfo.mem_unit * myinfo.totalram;
  total_mem /= (1024.*1024.);
  double free_mem = myinfo.mem_unit * myinfo.freeram;
  free_mem /= (1024.*1024.);
  
  if(!UniqueID()){
    printf("printMem: Memory: total: %.2f MB, avail: %.2f MB, used %.2f MB\n",total_mem, free_mem, total_mem-free_mem);
  }
#endif
}

//Skip gauge fixing and set all gauge fixing matrices to unity
void gaugeFixUnity(Lattice &lat, const FixGaugeArg &fix_gauge_arg){
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

template<typename T, typename T_class>
struct _mult_sgn_times_i_impl{};

template<typename T>
struct _mult_sgn_times_i_impl<T,complex_double_or_float_mark>{
  inline static T doit(const int sgn, const T &val){
    return T( -sgn * val.imag(), sgn * val.real() ); // sign * i * val
  }
};

#ifdef USE_GRID
template<typename T>
struct _mult_sgn_times_i_impl<T,grid_vector_complex_mark>{
  inline static T doit(const int sgn, const T &val){
    return sgn == -1 ? timesMinusI(val) : timesI(val);
  }
};
#endif

// template<typename T, typename my_enable_if<is_complex_double_or_float<T>::value,int>::type = 0> //for standard complex types
// inline T multiplySignTimesI(const int sgn, const T &val){
//   return T( -sgn * val.imag(), sgn * val.real() ); // sign * i * val
// }

// #ifdef USE_GRID
// template<typename T, typename my_enable_if<is_grid_vector_complex<T>::value,int>::type = 0> //for Grid complex types
// inline T multiplySignTimesI(const int sgn, const T &val){
//   return sgn == -1 ? timesMinusI(val) : timesI(val);
// }
// #endif

template<typename T>
inline T multiplySignTimesI(const int sgn, const T &val){
  return _mult_sgn_times_i_impl<T,typename ComplexClassify<T>::type>::doit(sgn,val);
}
CPS_END_NAMESPACE

#endif
