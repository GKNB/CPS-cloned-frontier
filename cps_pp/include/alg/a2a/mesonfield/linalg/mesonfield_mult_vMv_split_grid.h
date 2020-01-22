#ifndef _MULT_VMV_SPLIT_GRID_H
#define _MULT_VMV_SPLIT_GRID_H
#ifdef USE_GRID

#include "mesonfield_mult_vMv_split.h"

CPS_START_NAMESPACE

//Try to save memory at the cost of some performance
#define VMV_SPLIT_GRID_MEM_SAVE
//Don't splat-vectorize packed mesonfield at beginning, instead do it at the point of use
//#define VMV_SPLIT_GRID_STREAMING_SPLAT
//Do blocked matrix multiplication
#define VMV_BLOCKED_MATRIX_MULT


#define STACK_ALLOC_REORD


template<typename ComplexType>
class SCFoperationP{
public:
  virtual void operator()(ComplexType const* M, const int scf, const int rows, const int cols) = 0;
};

//Define common types
template<typename mf_Policies>
struct mult_vMv_split_grid_types{
  typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
  typedef typename mf_Policies::ComplexType SIMDcomplexType;
  typedef typename AlignedVector<SIMDcomplexType>::type AlignedSIMDcomplexVector;
  
#ifdef VMV_SPLIT_GRID_STREAMING_SPLAT
  typedef typename mf_Policies::ScalarComplexType MComplexType;
#else
  typedef typename mf_Policies::ComplexType MComplexType;
#endif

  typedef typename AlignedVector<MComplexType>::type AlignedMComplexVector;

  typedef SCFoperationP<
#ifdef VMV_SPLIT_GRID_STREAMING_SPLAT
			 typename mf_Policies::ScalarComplexType
#else
			 typename mf_Policies::ComplexType
#endif
			 > SCFoperationType;
};
#define INHERIT_BASE_TYPE(A) typedef typename mult_vMv_split_grid_types<mf_Policies>::A A

#define INHERIT_BASE_TYPES				\
  INHERIT_BASE_TYPE(ScalarComplexType);			\
  INHERIT_BASE_TYPE(SIMDcomplexType);			\
  INHERIT_BASE_TYPE(AlignedSIMDcomplexVector);		\
  INHERIT_BASE_TYPE(MComplexType);		\
  INHERIT_BASE_TYPE(AlignedMComplexVector); \
  INHERIT_BASE_TYPE(SCFoperationType)

template<typename T>
struct TwoDArrayWrap{
  const int r;
  const int c;
  T* p;
  TwoDArrayWrap(T* p, const int r, const int c): p(p), r(r), c(c){}  

  T & operator()(const int i, const int j){ return p[j + c*i]; }
  const T & operator()(const int i, const int j) const{ return p[j + c*i]; }
};
template<typename V>
struct TwoDVectorWrap{
  typedef typename V::value_type::value_type T;
  V &v;
  TwoDVectorWrap(V &v): v(v){}
  inline T & operator()(const int i, const int j){ return v[i][j]; }
  inline const T & operator()(const int i, const int j) const{ return v[i][j]; }
};
template<typename V>
struct ThreeDVectorWrap{
  typedef typename V::value_type::value_type::value_type T;
  V &v;
  ThreeDVectorWrap(V &v): v(v){}
  inline T & operator()(const int i, const int j, const int k){ return v[i][j][k]; }
  inline const T & operator()(const int i, const int j, const int k) const{ return v[i][j][k]; }
};

//Single-site matrix vector multiplication operator
template<typename mf_Policies, typename TwoDArrayType>
class multiply_M_r_op_grid: public mult_vMv_split_grid_types<mf_Policies>::SCFoperationType{
  INHERIT_BASE_TYPES;

  TwoDArrayType &Mr; //output
  const TwoDArrayType & rreord;
  std::vector<int> const* i_packed_unmap_all;

public:
  multiply_M_r_op_grid(TwoDArrayType &Mr, const TwoDArrayType &rreord,
		       std::vector<int> const* i_packed_unmap_all): Mr(Mr), rreord(rreord),i_packed_unmap_all(i_packed_unmap_all){
  }
  
  void operator()(SIMDcomplexType const* M_packed, const int scf, const int rows, const int cols){
#ifndef MEMTEST_MODE
    const std::vector<int> &i_packed_unmap = i_packed_unmap_all[scf];
# ifdef VMV_SPLIT_GRID_STREAMING_SPLAT
    SIMDcomplexType tmp;
# endif
  
    for(int i=0;i<rows;i++){
      SIMDcomplexType &into = Mr[scf][i_packed_unmap[i]];
      zeroit(into);
      for(int j=0;j<cols;j++){
	int mpoff = cols*i + j;
# ifdef VMV_SPLIT_GRID_STREAMING_SPLAT
	vsplat(tmp,M_packed[mpoff]);
	into = into + tmp*rreord(scf,j);
# else
    	into = into + M_packed[mpoff]*rreord(scf,j);
# endif
      }
    }
#endif
    
  }
};

//Same as above but with a blocked matrix algorithm
template<typename mf_Policies, typename TwoDArrayType>
class multiply_M_r_op_grid_blocked: public mult_vMv_split_grid_types<mf_Policies>::SCFoperationType{
  INHERIT_BASE_TYPES;

  TwoDArrayType &Mr; //output
  const TwoDArrayType &rreord;
  std::vector<int> const* i_packed_unmap_all;

public:
  multiply_M_r_op_grid_blocked(TwoDArrayType &Mr, const TwoDArrayType &rreord,
			       std::vector<int> const* i_packed_unmap_all): Mr(Mr), rreord(rreord),i_packed_unmap_all(i_packed_unmap_all){
  }
  
  void operator()(SIMDcomplexType const* M_packed, const int scf, const int rows, const int cols){
#ifndef MEMTEST_MODE
    const std::vector<int> &i_packed_unmap = i_packed_unmap_all[scf];
# ifdef VMV_SPLIT_GRID_STREAMING_SPLAT
    SIMDcomplexType tmp;
# endif

    int block_width_max = cols;
    int block_height_max = 32; //4;

    //Blocked matrix multiply
    for(int i0=0; i0<rows; i0+=block_height_max){
      int iblock_size = std::min(rows - i0, block_height_max);
	
      for(int j0=0; j0<cols; j0+=block_width_max){
    	int jblock_size = std::min(cols - j0, block_width_max);
	
    	for(int ii=0;ii<iblock_size;ii++){
    	  SIMDcomplexType &into = Mr(scf,i_packed_unmap[i0+ii]);
    	  for(int jj=0;jj<jblock_size;jj++){
	    int mpoff = cols*(i0+ii) +j0+jj;
#  ifdef VMV_SPLIT_GRID_STREAMING_SPLAT
	    vsplat(tmp,M_packed[mpoff]);
	    into = into + tmp*rreord(scf,j0+jj);
#  else
    	    into = into + M_packed[mpoff]*rreord(scf,j0+jj);
#  endif
	  }
    	}
      }
    }   
#endif  //MEMTEST_MODE
  }
};





//Single spin-color-flavor index but multiple sites
template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR,
	 typename ThreeDArrayType>
class multiply_M_r_singlescf_op_grid: public mult_vMv_split_grid_types<mf_Policies>::SCFoperationType{
  INHERIT_BASE_TYPES;
 
  typedef std::vector<std::vector<AlignedSIMDcomplexVector> >& MrPassType;
  typedef const std::vector<std::vector<AlignedSIMDcomplexVector> >& rreordPassType;

  const int* work;
  const int* off;
  std::vector<int> const* i_packed_unmap_all;
  ThreeDArrayType &Mr; //output
  const ThreeDArrayType &rreord;
  
public:
  multiply_M_r_singlescf_op_grid(const int* _work, const int* _off, 
				 ThreeDArrayType &_Mr, 
				 ThreeDArrayType &_rreord,
				 std::vector<int> const* i_packed_unmap_all): 
    work(_work),off(_off),Mr(_Mr),rreord(_rreord), i_packed_unmap_all(i_packed_unmap_all){}
  
  void operator()(SIMDcomplexType const* M_packed, const int scf, const int rows, const int cols){
#ifndef MEMTEST_MODE

    const std::vector<int> &i_packed_unmap = i_packed_unmap_all[scf];
    
#pragma omp parallel
    {
      int me = omp_get_thread_num();

      //M * r
#ifdef VMV_SPLIT_GRID_STREAMING_SPLAT
      SIMDcomplexType tmp;
#endif

      for(int i=0;i<rows;i++){
	int i_full = i_packed_unmap[i];

	for(int s=off[me];s<off[me]+work[me];s++){
	  SIMDcomplexType &into = Mr(s,scf,i_full);
	  zeroit(into);
	
	  for(int j=0;j<cols;j++){
#  ifdef VMV_SPLIT_GRID_STREAMING_SPLAT
	    vsplat(tmp,M_packed[cols*i+j]);
	    into = into + tmp * rreord[s][scf][j];
#  else
	    into = into + M_packed[cols*i+j] * rreord(s,scf,j);
#  endif
	  }
	}
      } //i
    }//parallel region
#endif //MEMTEST_MODE
  }
};


//Blocked version
template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR,
	 typename ThreeDArrayType>
class multiply_M_r_singlescf_op_grid_blocked: public mult_vMv_split_grid_types<mf_Policies>::SCFoperationType{
  INHERIT_BASE_TYPES;
 
  const int* work;
  const int* off;
  std::vector<int> const* i_packed_unmap_all;
  ThreeDArrayType &Mr; //output
  const ThreeDArrayType &rreord;
  
public:
  multiply_M_r_singlescf_op_grid_blocked(const int* _work, const int* _off, 
					 ThreeDArrayType &_Mr, 
					 const ThreeDArrayType &_rreord,
					 std::vector<int> const* i_packed_unmap_all): 
    work(_work),off(_off),Mr(_Mr),rreord(_rreord), i_packed_unmap_all(i_packed_unmap_all){}
  
  void operator()(SIMDcomplexType const* M_packed, const int scf, const int rows, const int cols){
#ifndef MEMTEST_MODE
    const std::vector<int> &i_packed_unmap = i_packed_unmap_all[scf];
    
#pragma omp parallel
    {
      int me = omp_get_thread_num();

      //M * r
# ifdef VMV_SPLIT_GRID_STREAMING_SPLAT
      SIMDcomplexType tmp;
# endif

      int block_width_max =  cols;
      int block_height_max = 8; //4;
          
      for(int i0=0; i0<rows; i0+=block_height_max){
	int iblock_size = std::min(rows - i0, block_height_max);
      
	for(int j0=0; j0<cols; j0+=block_width_max){
	  int jblock_size = std::min(cols - j0, block_width_max);

	  for(int s=off[me];s<off[me]+work[me];s++){
	    SIMDcomplexType const* base = &rreord(s,scf,j0);
	    for(int i_packed=0;i_packed < iblock_size; i_packed++){
	      SIMDcomplexType &into = Mr(s,scf, i_packed_unmap[i0+i_packed] );
	      zeroit(into);
	    
	      for(int j_packed=0;j_packed<jblock_size;j_packed++){
# ifdef VMV_SPLIT_GRID_STREAMING_SPLAT
		vsplat(tmp, M_packed[cols*(i0+i_packed)+j0+j_packed]);
		into = into + tmp*base[j_packed];
# else
		into = into + M_packed[cols*(i0+i_packed)+j0+j_packed]*base[j_packed];
# endif
	      }
	    }
	  }
	}
      }
 
    }//parallel region
#endif //MEMTEST_MODE
  }
};



//For local outer contraction of meson field by two vectors we can save a lot of time by column reordering the meson field to improve cache use. 
//Save even more time by doing this outside the site loop (it makes no reference to the 3d position, only the time at which the vectors
//are evaluated)
template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR
	 >
class mult_vMv_split_v<mf_Policies, lA2AfieldL, lA2AfieldR, rA2AfieldL, rA2AfieldR, grid_vector_complex_mark>:
public mult_vMv_split_base<mf_Policies,lA2AfieldL,lA2AfieldR,rA2AfieldL,rA2AfieldR>{
  INHERIT_BASE_TYPES;
  
  typedef mult_vMv_split_base<mf_Policies,lA2AfieldL,lA2AfieldR,rA2AfieldL,rA2AfieldR> Base;
  
  //Note:
  //il is the index of l, 
  //ir is the row index of M, 
  //jl is the column index of M and 
  //jr is the index of r
    
  typedef typename Base::iLeftDilutionType iLeftDilutionType;
  typedef typename Base::iRightDilutionType iRightDilutionType;
  
  typedef typename Base::jLeftDilutionType jLeftDilutionType;    
  typedef typename Base::jRightDilutionType jRightDilutionType;

  const static int nscf = 2*3*4;

  std::vector<int> i_packed_unmap_all[nscf];
  int logical_sites_3d; 
  bool setup_called;

  //Packed matrices
#ifdef VMV_SPLIT_GRID_MEM_SAVE
  AlignedMComplexVector mf_reord_buf;
  SIMDcomplexType *mf_reord_lo_lo; //shared nl*nl submatrix
  SIMDcomplexType *mf_reord_lo_hi[nscf]; //the nl * nh[scf] submatrix
  SIMDcomplexType *mf_reord_hi_lo[nscf]; //the nh[scf] * nl submatrix
  SIMDcomplexType *mf_reord_hi_hi[nscf]; //the nh[scf] * nh[scf] submatrix
#else
  std::vector<AlignedMComplexVector> mf_reord; //vector of linearized matrices in packed format where only the rows used are stored. One matrix for each spin/color/flavor combination of the vector r
#endif

#ifndef STACK_ALLOC_REORD
  //Temporary vectors used in parallel region (one per thread)
  int nthr_setup;
  std::vector<std::vector<AlignedSIMDcomplexVector> > lreord; //[thr][scf][reordered mode]
  std::vector<std::vector<AlignedSIMDcomplexVector> > rreord;
  std::vector<std::vector<AlignedSIMDcomplexVector> > Mr; //[thr][scf][M row]
#endif

  int ni_max;
  int nj_max;

  //Stuff needed by constructPackedMloopSCF
#ifdef VMV_SPLIT_GRID_MEM_SAVE
  int nl_row;
  int nl_col;
  bool nj_all_same;

#ifndef STACK_ALLOC_REORD
  std::vector<AlignedMComplexVector> M_packed; //linearized matrix  [thr][j + nj_max*i]. Acts as buffer. Contains enough space for largest nj  //Also one per thread
#endif

#endif

  //Loop over scf, reconstructing packed matrix if necessary, calling operation for each scf
  void constructPackedMloopSCF(SCFoperationType &op){
    int thr = omp_get_thread_num();

#ifdef STACK_ALLOC_REORD
    SIMDcomplexType mpptr[this->nrows_used * nj_max];
#elif defined(VMV_SPLIT_GRID_MEM_SAVE)
    SIMDcomplexType* mpptr = M_packed[thr].data();
#endif

#ifdef VMV_SPLIT_GRID_MEM_SAVE
    if(nj_all_same) pokeSubmatrix<MComplexType>(mpptr, mf_reord_lo_lo, this->nrows_used, nj_max, 0, 0, nl_row, nl_col);
#endif
    
    //M * r
    for(int scf=0;scf<nscf;scf++){

#if !defined(STACK_ALLOC_REORD) && !defined(VMV_SPLIT_GRID_MEM_SAVE)
      SIMDcomplexType* mpptr = mf_reord[scf].data();
#endif

      int nj_this = this->nj[scf]; //vector size
      
#ifdef VMV_SPLIT_GRID_MEM_SAVE
      int nh_row = this->nrows_used - nl_row;
      int nh_col = nj_this - nl_col;
      
      if(!nj_all_same) pokeSubmatrix<MComplexType>(mpptr, mf_reord_lo_lo, this->nrows_used, nj_this, 0, 0, nl_row, nl_col);
      pokeSubmatrix<MComplexType>(mpptr, mf_reord_lo_hi[scf], this->nrows_used, nj_this, 0, nl_col, nl_row, nh_col);
      pokeSubmatrix<MComplexType>(mpptr, mf_reord_hi_lo[scf], this->nrows_used, nj_this, nl_row, 0, nh_row, nl_col);
      pokeSubmatrix<MComplexType>(mpptr, mf_reord_hi_hi[scf], this->nrows_used, nj_this, nl_row, nl_col, nh_row, nh_col);
      op(mpptr, scf, this->nrows_used, nj_this);
#else
      op(mpptr, scf, this->nrows_used, nj_this);
#endif


    }  
  }
   
  //Vector inner product: Multiply l * Mr
  template<typename TwoDArrayType>
  void site_multiply_l_Mr(CPSspinColorFlavorMatrix<SIMDcomplexType> &out, 
			  const TwoDArrayType &lreord,
			  const TwoDArrayType &Mr) const{
    //Vector vector multiplication l*(M*r)
    for(int sl=0;sl<4;sl++){
      for(int sr=0;sr<4;sr++){
	for(int cl=0;cl<3;cl++){
	  for(int cr=0;cr<3;cr++){
	    for(int fl=0;fl<2;fl++){
	      int scfl = fl + 2*(cl + 3*sl);
	      int ni_this = this->ni[scfl];

	      SIMDcomplexType const* lbase = &lreord(scfl,0);

	      const std::vector<std::pair<int,int> > &blocks = this->blocks_scf[scfl];
	      
	      for(int fr=0;fr<2;fr++){
		int scfr = fr + 2*(cr + 3*sr);

		SIMDcomplexType &into = out(sl,sr)(cl,cr)(fl,fr);
		zeroit(into);

		SIMDcomplexType const* Mr_base = &Mr(scfr,0);

		int loff = 0;
		for(int b=0;b<blocks.size();b++){
		  SIMDcomplexType const* Mr_block_ptr = Mr_base + this->irmap[scfl][blocks[b].first]; //Mr is not packed, lreord is. Cycle over blocks of consecutive elements
#ifndef MEMTEST_MODE
		  for(int i=0;i<blocks[b].second;i++)
		    into = into + lbase[loff++] * Mr_block_ptr[i];		  
#endif
		}
	      }
	    }
	  }
	}
      }
    }
  }

  //Setup thread-local temporaries to avoid alloc under threaded loop
  void setup_parallel_temporaries(){
#ifndef STACK_ALLOC_REORD
    lreord.resize(nthr_setup,std::vector<AlignedSIMDcomplexVector>(nscf));
    rreord.resize(nthr_setup,std::vector<AlignedSIMDcomplexVector>(nscf));
#endif

    for(int sc=0;sc<12;sc++){
      for(int f=0;f<2;f++){
	int scf = f + 2*sc;

	//i index
	int ni_this = this->ni[scf];
	const std::vector<int> &ilmap_this = this->ilmap[scf];

#ifndef STACK_ALLOC_REORD
	for(int t=0;t<nthr_setup;t++)
	  lreord[t][scf].resize(ni_this);
#endif

	//j index
	int nj_this = this->nj[scf];
	const std::vector<int> &jrmap_this = this->jrmap[scf]; //jrmap_this.resize(nj_this);

#ifndef STACK_ALLOC_REORD
	for(int t=0;t<nthr_setup;t++)
	  rreord[t][scf].resize(nj_this);
#endif
      }
    }
#ifndef STACK_ALLOC_REORD
    Mr.resize(nthr_setup, std::vector<AlignedSIMDcomplexVector>(nscf, AlignedSIMDcomplexVector(this->Mrows)));
# ifdef VMV_SPLIT_GRID_MEM_SAVE
    M_packed.resize(nthr_setup, AlignedMComplexVector(this->nrows_used * nj_max));
# endif
#endif
  }

  //Unpack the reordered matrix
#ifdef VMV_SPLIT_GRID_MEM_SAVE
  void deconstruct_matrix(AlignedMComplexVector &mf_reord_scf, const int scf, const int nj_this){
    int nh_row = this->nrows_used - nl_row;
    int nh_col = nj_this - nl_col;

    int sz1 = nl_row*nl_col;
    int sz2 = nl_row*nh_col;
    int sz3 = nh_row*nl_col;
    int sz4 = nh_row*nh_col;

    if(scf == 0){
      mf_reord_buf.resize(sz1 + nscf*(sz2 + sz3 + sz4));
      mf_reord_lo_lo = mf_reord_buf.data();
      getSubmatrix<MComplexType >(mf_reord_lo_lo, mf_reord_scf.data(), this->nrows_used, nj_this, 0, 0, nl_row, nl_col);
    }
    mf_reord_lo_hi[scf] = mf_reord_buf.data() + sz1 + scf*sz2;
    getSubmatrix<MComplexType >(mf_reord_lo_hi[scf], mf_reord_scf.data(), this->nrows_used, nj_this, 0, nl_col, nl_row, nh_col);

    mf_reord_hi_lo[scf] = mf_reord_buf.data() + sz1 + nscf*sz2 + scf*sz3;
    getSubmatrix<MComplexType >(mf_reord_hi_lo[scf], mf_reord_scf.data(), this->nrows_used, nj_this, nl_row, 0, nh_row, nl_col);

    mf_reord_hi_hi[scf] = mf_reord_buf.data() + sz1 + nscf*sz2 + nscf*sz3 + scf*sz4;
    getSubmatrix<MComplexType >(mf_reord_hi_hi[scf], mf_reord_scf.data(), this->nrows_used, nj_this, nl_row, nl_col, nh_row, nh_col);
  }
#endif

public:

  mult_vMv_split_v(): setup_called(false){ }

  void free_mem(){
    this->free_mem_base();

#ifdef VMV_SPLIT_GRID_MEM_SAVE
    AlignedMComplexVector().swap(mf_reord_buf);
#else
    std::vector<AlignedMComplexVector>().swap(mf_reord);
#endif

#ifndef STACK_ALLOC_REORD
    std::vector<std::vector<AlignedSIMDcomplexVector> >().swap(lreord);
    std::vector<std::vector<AlignedSIMDcomplexVector> >().swap(rreord);
    std::vector<std::vector<AlignedSIMDcomplexVector> >().swap(Mr);

# ifdef VMV_SPLIT_GRID_MEM_SAVE
    std::vector<AlignedMComplexVector>().swap(M_packed);
# endif

#endif

  }

  //This should be called outside the site loop (and outside any parallel region)
  //top_glb is the time in global lattice coordinates.
  void setup(const lA2AfieldL<mf_Policies> &l,  const A2AmesonField<mf_Policies,lA2AfieldR,rA2AfieldL> &M, const rA2AfieldR<mf_Policies> &r, const int &_top_glb){
    //Precompute index mappings
    ModeContractionIndices<iLeftDilutionType,iRightDilutionType> i_ind(l);
    ModeContractionIndices<jLeftDilutionType,jRightDilutionType> j_ind(r);
    setup(l,M,r,_top_glb,i_ind,j_ind);
  }
  void setup(const lA2AfieldL<mf_Policies> &l,  const A2AmesonField<mf_Policies,lA2AfieldR,rA2AfieldL> &M, const rA2AfieldR<mf_Policies> &r, const int &_top_glb, 
	     const ModeContractionIndices<iLeftDilutionType,iRightDilutionType> &i_ind, const ModeContractionIndices<jLeftDilutionType,jRightDilutionType>& j_ind){
    this->setup_base(l,M,r,_top_glb,i_ind,j_ind);

    logical_sites_3d = l.getMode(0).nodeSites(0)*l.getMode(0).nodeSites(1)*l.getMode(0).nodeSites(2);

#ifndef STACK_ALLOC_REORD
    nthr_setup = omp_get_max_threads();
#endif

#ifdef VMV_SPLIT_GRID_MEM_SAVE
    nl_row = this->Mptr->getRowParams().getNl();
    nl_col = this->Mptr->getColParams().getNl();
    nj_all_same = true;
#endif

    nj_max = 0;
    ni_max = 0;

    for(int scf=0;scf<nscf;scf++){
      if(this->nj[scf] > nj_max) nj_max = this->nj[scf];
      if(this->ni[scf] > ni_max) ni_max = this->ni[scf];
#ifdef VMV_SPLIT_GRID_MEM_SAVE
      if(this->nj[scf] != this->nj[0]) nj_all_same = false;
#endif
    }
#endif

    setup_parallel_temporaries();

    //Not all rows or columns of M are used, so lets use a packed matrix
#ifdef VMV_SPLIT_GRID_MEM_SAVE
    AlignedMComplexVector mf_reord_scf;
#else
    mf_reord.resize(nscf);
#endif
    
    for(int scf=0;scf<nscf;scf++){
      int nj_this = this->nj[scf];
      std::vector<int> &jlmap_this = this->jlmap[scf];

#ifndef VMV_SPLIT_GRID_MEM_SAVE
      AlignedMComplexVector &mf_reord_scf = mf_reord[scf];
#endif

#ifdef VMV_SPLIT_GRID_STREAMING_SPLAT
      M.scalarPackedColReorder(mf_reord_scf, jlmap_this.data(), nj_this, this->rowidx_used);
#else
      M.splatPackedColReorder(mf_reord_scf, jlmap_this.data(), nj_this, this->rowidx_used);
#endif

#ifdef VMV_SPLIT_GRID_MEM_SAVE
      deconstruct_matrix(mf_reord_scf, scf, nj_this);
#endif
             
      //Store the map between packed and full indices
      int i_packed = 0;
      i_packed_unmap_all[scf].resize(this->nrows_used);
      for(int i_full=0;i_full<this->Mrows;i_full++) 
	if(this->rowidx_used[i_full]) i_packed_unmap_all[scf][i_packed++] = i_full;
    }
    
    setup_called = true;
  }

  //Contract on all 3d sites on this node with fixed operator time coord top_glb into a canonically ordered output vector
  void contract(typename AlignedVector<CPSspinColorFlavorMatrix<SIMDcomplexType> >::type &out, const bool conj_l, const bool conj_r){
    const int top = this->top_glb - GJP.TnodeSites()*GJP.TnodeCoor();
    assert(top >= 0 && top < GJP.TnodeSites()); //make sure you use this method on the appropriate node!

    const int sites_3d = logical_sites_3d;

    out.resize(sites_3d);

    std::vector< std::vector<AlignedSIMDcomplexVector> > lreord(sites_3d); //[3d site][scf][reordered mode]
    std::vector< std::vector<AlignedSIMDcomplexVector> > rreord(sites_3d);

    std::vector<  std::vector<AlignedSIMDcomplexVector> > Mr(sites_3d); //[3d site][scf][M row]

    //Run everything in parallel environment to avoid thread creation overheads
    int work[omp_get_max_threads()], off[omp_get_max_threads()];

#pragma omp parallel
    {
      int me = omp_get_thread_num();
      int team = omp_get_num_threads();
   
      //Reorder rows and columns of left and right fields such that they can be accessed sequentially
      thread_work(work[me], off[me], sites_3d, me, team);

      for(int s=off[me];s<off[me]+work[me];s++){
	int site4dop = s + sites_3d*top;
	this->site_reorder_lr(lreord[s],rreord[s],conj_l,conj_r,site4dop);
      }
      
      for(int s=off[me];s<off[me]+work[me];s++){
	Mr[s].resize(nscf); 
	for(int scf=0;scf<nscf;scf++){
	  Mr[s][scf].resize(this->Mrows);
	  for(int i=0;i<this->Mrows;i++)
	    zeroit(Mr[s][scf][i]);
	}
      }
    }

    typedef ThreeDVectorWrap< std::vector<  std::vector<AlignedSIMDcomplexVector> > > ThreeDArrayType;
    typedef TwoDVectorWrap< std::vector<AlignedSIMDcomplexVector> > TwoDArrayType;

    ThreeDArrayType Mr_t(Mr);
    ThreeDArrayType rreord_t(rreord);

#ifdef VMV_BLOCKED_MATRIX_MULT
    multiply_M_r_singlescf_op_grid_blocked<mf_Policies,lA2AfieldL,lA2AfieldR,rA2AfieldL,rA2AfieldR,ThreeDArrayType> op(work,off,Mr_t,rreord_t,i_packed_unmap_all);
#else
    multiply_M_r_singlescf_op_grid<mf_Policies,lA2AfieldL,lA2AfieldR,rA2AfieldL,rA2AfieldR,ThreeDArrayType> op(work,off,Mr_t,rreord_t,i_packed_unmap_all);
#endif
    constructPackedMloopSCF(op);

#pragma omp parallel
    {
      int me = omp_get_thread_num();
      
      //Vector vector multiplication l*(M*r)
      for(int x3d=off[me];x3d<off[me]+work[me];x3d++)
	site_multiply_l_Mr(out[x3d], TwoDArrayType(lreord[x3d]), TwoDArrayType(Mr[x3d]));

    } //end of parallel region

  }//end of method

  //Run inside a threaded/parallelized loop over 3d sites. xop is a 3d coordinate!
  void contract(CPSspinColorFlavorMatrix<SIMDcomplexType> &out, const int &xop, const bool &conj_l, const bool &conj_r){
#ifndef STACK_ALLOC_REORD
    assert(omp_get_num_threads() <= nthr_setup);
#endif

    const int thr_idx = omp_get_thread_num();

    const int top = this->top_glb - GJP.TnodeSites()*GJP.TnodeCoor();
    assert(top >= 0 && top < GJP.TnodeSites()); //make sure you use this method on the appropriate node!

#ifdef STACK_ALLOC_REORD
    typedef TwoDArrayWrap<SIMDcomplexType> ArrayType;

    SIMDcomplexType Mr_d[nscf*this->Mrows];
    SIMDcomplexType lreord_d[nscf*ni_max];
    SIMDcomplexType rreord_d[nscf*nj_max];

    ArrayType Mr_t(Mr_d, nscf, this->Mrows);
    ArrayType lreord_t(lreord_d, nscf, ni_max);
    ArrayType rreord_t(rreord_d, nscf, nj_max);
#else
    typedef TwoDVectorWrap<std::vector<AlignedSIMDcomplexVector> > ArrayType;
    ArrayType Mr_t(Mr[thr_idx]);
    ArrayType lreord_t(lreord[thr_idx]);
    ArrayType rreord_t(rreord[thr_idx]);
#endif    

    for(int scf=0;scf<nscf;scf++)
      for(int i=0;i<this->Mrows;i++)
	zeroit(Mr_t(scf,i));
    
    const int sites_3d = logical_sites_3d;
    const int site4dop = xop + sites_3d*top;

    //Threaded version of site_reorder_lr
    for(int sc=0;sc<12;sc++){
      for(int f=0;f<2;f++){
	const int scf = f + 2*sc;

	//i index
	const int ni_this = this->ni[scf];
	const std::vector<int> &ilmap_this = this->ilmap[scf];

	for(int i = 0; i < ni_this; i++){
	  const SIMDcomplexType &lval_tmp = this->lptr->nativeElem(ilmap_this[i], site4dop, sc, f);
#ifndef MEMTEST_MODE
	  lreord_t(scf,i) = conj_l ? cconj(lval_tmp) : lval_tmp;
#endif
	}

	//j index
	int nj_this = this->nj[scf];
	const std::vector<int> &jrmap_this = this->jrmap[scf]; //jrmap_this.resize(nj_this);

	for(int j = 0; j < nj_this; j++){
	  const SIMDcomplexType &rval_tmp = this->rptr->nativeElem(jrmap_this[j], site4dop, sc, f);
#ifndef MEMTEST_MODE
	  rreord_t(scf,j) = conj_r ? cconj(rval_tmp) : rval_tmp;
#endif
	}

      }
    }

    //M * r
#ifdef VMV_BLOCKED_MATRIX_MULT
    multiply_M_r_op_grid_blocked<mf_Policies,ArrayType> op(Mr_t, rreord_t, this->i_packed_unmap_all);
#else
    multiply_M_r_op_grid<mf_Policies,ArrayType> op(Mr_t, rreord_t, this->i_packed_unmap_all);
#endif
    constructPackedMloopSCF(op);

    //Vector vector multiplication l*(M*r)
    site_multiply_l_Mr(out, lreord_t, Mr_t);
  }


};

#undef INHERIT_BASE_TYPE
#undef INHERIT_BASE_TYPES

#undef STACK_ALLOC_REORD

CPS_END_NAMESPACE

#endif
