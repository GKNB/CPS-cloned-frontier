#ifndef _MULT_VMV_SPLIT_GRID_H
#define _MULT_VMV_SPLIT_GRID_H
#ifdef USE_GRID

#include<alg/a2a/mesonfield_mult_vMv_split.h>

CPS_START_NAMESPACE

//Try to save memory at the cost of some performance
#define VMV_SPLIT_GRID_MEM_SAVE
//Don't splat-vectorize packed mesonfield at beginning, instead do it at the point of use
//#define VMV_SPLIT_GRID_STREAMING_SPLAT
//Do blocked matrix multiplication
#define VMV_BLOCKED_MATRIX_MULT


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

  typedef SCFoperation<typename AlignedVector<
#ifdef VMV_SPLIT_GRID_STREAMING_SPLAT
			 typename mf_Policies::ScalarComplexType
#else
			 typename mf_Policies::ComplexType
#endif
			 >::type > SCFoperationType;
};
#define INHERIT_BASE_TYPE(A) typedef typename mult_vMv_split_grid_types<mf_Policies>::A A

#define INHERIT_BASE_TYPES				\
  INHERIT_BASE_TYPE(ScalarComplexType);			\
  INHERIT_BASE_TYPE(SIMDcomplexType);			\
  INHERIT_BASE_TYPE(AlignedSIMDcomplexVector);		\
  INHERIT_BASE_TYPE(MComplexType);		\
  INHERIT_BASE_TYPE(AlignedMComplexVector)



//Single-site matrix vector multiplication operator
template<typename mf_Policies>
class multiply_M_r_op_grid: public mult_vMv_split_grid_types<mf_Policies>::SCFoperationType{
  INHERIT_BASE_TYPES;

  std::vector<AlignedSIMDcomplexVector>& Mr; //output
  const std::vector<AlignedSIMDcomplexVector>& rreord;
  std::vector<int> const* i_packed_unmap_all;

public:
  multiply_M_r_op_grid(std::vector<AlignedSIMDcomplexVector>& Mr, const std::vector<AlignedSIMDcomplexVector>& rreord,
		       std::vector<int> const* i_packed_unmap_all): Mr(Mr), rreord(rreord),i_packed_unmap_all(i_packed_unmap_all){
  }
  
  void operator()(const AlignedMComplexVector& M_packed, const int scf, const int rows, const int cols){
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
	into = into + tmp*rreord[scf][j];
# else
    	into = into + M_packed[mpoff]*rreord[scf][j];
# endif
      }
    }
#endif
    
  }
};

//Same as above but with a blocked matrix algorithm
template<typename mf_Policies>
class multiply_M_r_op_grid_blocked: public mult_vMv_split_grid_types<mf_Policies>::SCFoperationType{
  INHERIT_BASE_TYPES;

  std::vector<AlignedSIMDcomplexVector>& Mr; //output
  const std::vector<AlignedSIMDcomplexVector>& rreord;
  std::vector<int> const* i_packed_unmap_all;

public:
  multiply_M_r_op_grid_blocked(std::vector<AlignedSIMDcomplexVector>& Mr, const std::vector<AlignedSIMDcomplexVector>& rreord,
			       std::vector<int> const* i_packed_unmap_all): Mr(Mr), rreord(rreord),i_packed_unmap_all(i_packed_unmap_all){
  }
  
  void operator()(const AlignedMComplexVector& M_packed, const int scf, const int rows, const int cols){
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
    	  SIMDcomplexType &into = Mr[scf][i_packed_unmap[i0+ii]];
    	  for(int jj=0;jj<jblock_size;jj++){
	    int mpoff = cols*(i0+ii) +j0+jj;
#  ifdef VMV_SPLIT_GRID_STREAMING_SPLAT
	    vsplat(tmp,M_packed[mpoff]);
	    into = into + tmp*rreord[scf][j0+jj];
#  else
    	    into = into + M_packed[mpoff]*rreord[scf][j0+jj];
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
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR>
class multiply_M_r_singlescf_op_grid: public mult_vMv_split_grid_types<mf_Policies>::SCFoperationType{
  INHERIT_BASE_TYPES;
 
  const int* work;
  const int* off;
  std::vector<int> const* i_packed_unmap_all;
  std::vector< std::vector<AlignedSIMDcomplexVector> > &Mr; //output
  const std::vector< std::vector<AlignedSIMDcomplexVector> > &rreord;
  
public:
  multiply_M_r_singlescf_op_grid(const int* _work, const int* _off, 
				 std::vector<  std::vector<AlignedSIMDcomplexVector> > &_Mr, 
				 const std::vector< std::vector<AlignedSIMDcomplexVector> > &_rreord,
				 std::vector<int> const* i_packed_unmap_all): 
    work(_work),off(_off),Mr(_Mr),rreord(_rreord), i_packed_unmap_all(i_packed_unmap_all){}
  
  void operator()(const AlignedMComplexVector& M_packed, const int scf, const int rows, const int cols){
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
	  SIMDcomplexType &into = Mr[s][scf][i_full];
	  zeroit(into);
	
	  for(int j=0;j<cols;j++){
#  ifdef VMV_SPLIT_GRID_STREAMING_SPLAT
	    vsplat(tmp,M_packed[cols*i+j]);
	    into = into + tmp * rreord[s][scf][j];
#  else
	    into = into + M_packed[cols*i+j] * rreord[s][scf][j];
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
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR>
class multiply_M_r_singlescf_op_grid_blocked: public mult_vMv_split_grid_types<mf_Policies>::SCFoperationType{
  INHERIT_BASE_TYPES;
 
  const int* work;
  const int* off;
  std::vector<int> const* i_packed_unmap_all;
  std::vector< std::vector<AlignedSIMDcomplexVector> > &Mr; //output
  const std::vector< std::vector<AlignedSIMDcomplexVector> > &rreord;
  
public:
  multiply_M_r_singlescf_op_grid_blocked(const int* _work, const int* _off, 
					 std::vector<  std::vector<AlignedSIMDcomplexVector> > &_Mr, 
					 const std::vector< std::vector<AlignedSIMDcomplexVector> > &_rreord,
					 std::vector<int> const* i_packed_unmap_all): 
    work(_work),off(_off),Mr(_Mr),rreord(_rreord), i_packed_unmap_all(i_packed_unmap_all){}
  
  void operator()(const AlignedMComplexVector& M_packed, const int scf, const int rows, const int cols){
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
	    SIMDcomplexType const* base = &rreord[s][scf][j0];
	    for(int i_packed=0;i_packed < iblock_size; i_packed++){
	      SIMDcomplexType &into = Mr[s][scf][ i_packed_unmap[i0+i_packed] ];
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
  AlignedMComplexVector mf_reord_lo_lo; //shared nl*nl submatrix
  AlignedMComplexVector mf_reord_lo_hi[nscf]; //the nl * nh[scf] submatrix
  AlignedMComplexVector mf_reord_hi_lo[nscf]; //the nh[scf] * nl submatrix
  AlignedMComplexVector mf_reord_hi_hi[nscf]; //the nh[scf] * nh[scf] submatrix
#else
  std::vector<AlignedMComplexVector> mf_reord; //vector of linearized matrices in packed format where only the rows used are stored. One matrix for each spin/color/flavor combination of the vector r
#endif

  //Temporary vectors used in parallel region (one per thread)
  int nthr_setup;
  std::vector<std::vector<AlignedSIMDcomplexVector> > lreord; //[thr][scf][reordered mode]
  std::vector<std::vector<AlignedSIMDcomplexVector> > rreord;
  std::vector<std::vector<AlignedSIMDcomplexVector> > Mr; //[thr][scf][M row]

  //Stuff needed by constructPackedMloopSCF
#ifdef VMV_SPLIT_GRID_MEM_SAVE
  int nl_row;
  int nl_col;
  int nj_max;
  bool nj_all_same;
  std::vector<AlignedMComplexVector> M_packed; //linearized matrix  [thr][j + nj_max*i]. Acts as buffer. Contains enough space for largest nj  //Also one per thread
#endif

  //Loop over scf, reconstructing packed matrix if necessary, calling operation for each scf
  void constructPackedMloopSCF(SCFoperation<AlignedMComplexVector> &op){
    int thr = omp_get_thread_num();

#ifdef VMV_SPLIT_GRID_MEM_SAVE
    SIMDcomplexType* mpptr = M_packed[thr].data();
    if(nj_all_same) pokeSubmatrix<MComplexType>(mpptr, mf_reord_lo_lo.data(), this->nrows_used, nj_max, 0, 0, nl_row, nl_col);
#endif
    
    //M * r
    for(int scf=0;scf<nscf;scf++){
      int nj_this = this->nj[scf]; //vector size
      
#ifdef VMV_SPLIT_GRID_MEM_SAVE
      int nh_row = this->nrows_used - nl_row;
      int nh_col = nj_this - nl_col;
      
      if(!nj_all_same) pokeSubmatrix<MComplexType>(mpptr, mf_reord_lo_lo.data(), this->nrows_used, nj_this, 0, 0, nl_row, nl_col);
      pokeSubmatrix<MComplexType>(mpptr, mf_reord_lo_hi[scf].data(), this->nrows_used, nj_this, 0, nl_col, nl_row, nh_col);
      pokeSubmatrix<MComplexType>(mpptr, mf_reord_hi_lo[scf].data(), this->nrows_used, nj_this, nl_row, 0, nh_row, nl_col);
      pokeSubmatrix<MComplexType>(mpptr, mf_reord_hi_hi[scf].data(), this->nrows_used, nj_this, nl_row, nl_col, nh_row, nh_col);
      op(M_packed[thr], scf, this->nrows_used, nj_this);
#else
      const AlignedMComplexVector& M_packed = mf_reord[scf]; //scope for reuse here
      op(M_packed, scf, this->nrows_used, nj_this);
#endif


    }  
  }
   
  //Vector inner product: Multiply l * Mr
  void site_multiply_l_Mr(CPSspinColorFlavorMatrix<SIMDcomplexType> &out, 
			  const std::vector<AlignedSIMDcomplexVector> &lreord,
			  const std::vector<AlignedSIMDcomplexVector> &Mr) const{
    //Vector vector multiplication l*(M*r)
    for(int sl=0;sl<4;sl++){
      for(int sr=0;sr<4;sr++){
	for(int cl=0;cl<3;cl++){
	  for(int cr=0;cr<3;cr++){
	    for(int fl=0;fl<2;fl++){
	      int scfl = fl + 2*(cl + 3*sl);
	      int ni_this = this->ni[scfl];

	      AlignedSIMDcomplexVector const& lbase = lreord[scfl];
	      const std::vector<std::pair<int,int> > &blocks = this->blocks_scf[scfl];
	      
	      for(int fr=0;fr<2;fr++){
		int scfr = fr + 2*(cr + 3*sr);

		SIMDcomplexType &into = out(sl,sr)(cl,cr)(fl,fr);
		zeroit(into);

		SIMDcomplexType const* Mr_base = &Mr[scfr][0];

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
    lreord.resize(nthr_setup,std::vector<AlignedSIMDcomplexVector>(nscf));
    rreord.resize(nthr_setup,std::vector<AlignedSIMDcomplexVector>(nscf));

    for(int sc=0;sc<12;sc++){
      for(int f=0;f<2;f++){
	int scf = f + 2*sc;

	//i index
	int ni_this = this->ni[scf];
	const std::vector<int> &ilmap_this = this->ilmap[scf];
	for(int t=0;t<nthr_setup;t++)
	  lreord[t][scf].resize(ni_this);

	//j index
	int nj_this = this->nj[scf];
	const std::vector<int> &jrmap_this = this->jrmap[scf]; //jrmap_this.resize(nj_this);

	for(int t=0;t<nthr_setup;t++)
	  rreord[t][scf].resize(nj_this);
      }
    }

    Mr.resize(nthr_setup, std::vector<AlignedSIMDcomplexVector>(nscf, AlignedSIMDcomplexVector(this->Mrows)));
    M_packed.resize(nthr_setup, AlignedMComplexVector(this->nrows_used * nj_max));
  }

  //Unpack the reordered matrix
#ifdef VMV_SPLIT_GRID_MEM_SAVE
  void deconstruct_matrix(AlignedMComplexVector &mf_reord_scf, const int scf, const int nj_this){
    int nh_row = this->nrows_used - nl_row;
    int nh_col = nj_this - nl_col;

    if(scf == 0){
      mf_reord_lo_lo.resize(nl_row*nl_col);
      getSubmatrix<MComplexType >(mf_reord_lo_lo.data(), mf_reord_scf.data(), this->nrows_used, nj_this, 0, 0, nl_row, nl_col);
    }
    mf_reord_lo_hi[scf].resize(nl_row*nh_col);
    getSubmatrix<MComplexType >(mf_reord_lo_hi[scf].data(), mf_reord_scf.data(), this->nrows_used, nj_this, 0, nl_col, nl_row, nh_col);

    mf_reord_hi_lo[scf].resize(nh_row*nl_col);
    getSubmatrix<MComplexType >(mf_reord_hi_lo[scf].data(), mf_reord_scf.data(), this->nrows_used, nj_this, nl_row, 0, nh_row, nl_col);

    mf_reord_hi_hi[scf].resize(nh_row*nh_col);
    getSubmatrix<MComplexType >(mf_reord_hi_hi[scf].data(), mf_reord_scf.data(), this->nrows_used, nj_this, nl_row, nl_col, nh_row, nh_col);
  }
#endif

public:

  mult_vMv_split_v(): setup_called(false){ }

  void free_mem(){
    this->free_mem_base();

#ifdef VMV_SPLIT_GRID_MEM_SAVE
    AlignedMComplexVector().swap(mf_reord_lo_lo);
    for(int scf=0;scf<nscf;scf++){
      AlignedMComplexVector().swap(mf_reord_lo_hi[scf]);
      AlignedMComplexVector().swap(mf_reord_hi_lo[scf]);
      AlignedMComplexVector().swap(mf_reord_hi_hi[scf]);
    }
#else
    std::vector<AlignedMComplexVector>().swap(mf_reord);
#endif

    std::vector<std::vector<AlignedSIMDcomplexVector> >().swap(lreord);
    std::vector<std::vector<AlignedSIMDcomplexVector> >().swap(rreord);
    std::vector<std::vector<AlignedSIMDcomplexVector> >().swap(Mr);
#ifdef VMV_SPLIT_GRID_MEM_SAVE
    std::vector<AlignedMComplexVector>().swap(M_packed);
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

    nthr_setup = omp_get_max_threads();

#ifdef VMV_SPLIT_GRID_MEM_SAVE
    nl_row = this->Mptr->getRowParams().getNl();
    nl_col = this->Mptr->getColParams().getNl();
    nj_max = 0;
    nj_all_same = true;
    for(int scf=0;scf<nscf;scf++){
      if(this->nj[scf] > nj_max) nj_max = this->nj[scf];
      if(this->nj[scf] != this->nj[0]) nj_all_same = false;
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
    int top = this->top_glb - GJP.TnodeSites()*GJP.TnodeCoor();
    assert(top >= 0 && top < GJP.TnodeSites()); //make sure you use this method on the appropriate node!

    int sites_3d = logical_sites_3d;

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
#ifdef VMV_BLOCKED_MATRIX_MULT
    multiply_M_r_singlescf_op_grid_blocked<mf_Policies,lA2AfieldL,lA2AfieldR,rA2AfieldL,rA2AfieldR> op(work,off,Mr,rreord,i_packed_unmap_all);
#else
    multiply_M_r_singlescf_op_grid<mf_Policies,lA2AfieldL,lA2AfieldR,rA2AfieldL,rA2AfieldR> op(work,off,Mr,rreord,i_packed_unmap_all);
#endif
    constructPackedMloopSCF(op);

#pragma omp parallel
    {
      int me = omp_get_thread_num();
      
      //Vector vector multiplication l*(M*r)
      for(int x3d=off[me];x3d<off[me]+work[me];x3d++)
	site_multiply_l_Mr(out[x3d], lreord[x3d], Mr[x3d]);
      
    } //end of parallel region

  }//end of method

  //Run inside a threaded/parallelized loop over 3d sites. xop is a 3d coordinate!
  void contract(CPSspinColorFlavorMatrix<SIMDcomplexType> &out, const int &xop, const bool &conj_l, const bool &conj_r){
    assert(omp_get_num_threads() <= nthr_setup);

    const int thr_idx = omp_get_thread_num();

    const int top = this->top_glb - GJP.TnodeSites()*GJP.TnodeCoor();
    assert(top >= 0 && top < GJP.TnodeSites()); //make sure you use this method on the appropriate node!

    for(int scf=0;scf<nscf;scf++)
      for(int i=0;i<this->Mrows;i++)
	zeroit(Mr[thr_idx][scf][i]);
    
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
	  lreord[thr_idx][scf][i] = conj_l ? cconj(lval_tmp) : lval_tmp;
#endif
	}

	//j index
	int nj_this = this->nj[scf];
	const std::vector<int> &jrmap_this = this->jrmap[scf]; //jrmap_this.resize(nj_this);

	for(int j = 0; j < nj_this; j++){
	  const SIMDcomplexType &rval_tmp = this->rptr->nativeElem(jrmap_this[j], site4dop, sc, f);
#ifndef MEMTEST_MODE
	  rreord[thr_idx][scf][j] = conj_r ? cconj(rval_tmp) : rval_tmp;
#endif
	}

      }
    }

    //M * r
#ifdef VMV_BLOCKED_MATRIX_MULT
    multiply_M_r_op_grid_blocked<mf_Policies> op(Mr[thr_idx], rreord[thr_idx], this->i_packed_unmap_all);
#else
    multiply_M_r_op_grid<mf_Policies> op(Mr[thr_idx], rreord[thr_idx], this->i_packed_unmap_all);
#endif
    constructPackedMloopSCF(op);

    //Vector vector multiplication l*(M*r)
    site_multiply_l_Mr(out, lreord[thr_idx], Mr[thr_idx]);
  }


};

#undef INHERIT_BASE_TYPE
#undef INHERIT_BASE_TYPES


CPS_END_NAMESPACE

#endif

#endif
