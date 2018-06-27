#ifndef _MULT_VMV_SPLIT_LITE_SHRBUF
#define _MULT_VMV_SPLIT_LITE_SHRBUF

#include<alg/a2a/mesonfield_mult_vMv_split.h>

CPS_START_NAMESPACE

//This is a version of the lite vMv that shares a common temp buffer between multiple instances
//Each thread has an independent buffer so it doesn't matter if the threads get out of sync

template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR,
	 typename ComplexClass>
class _mult_vMv_split_lite_shrbuf_impl_v{};

template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR	
	 >
class mult_vMv_split_lite_shrbuf: public _mult_vMv_split_lite_shrbuf_impl_v<mf_Policies,lA2AfieldL,lA2AfieldR,rA2AfieldL,rA2AfieldR,typename ComplexClassify<typename mf_Policies::ComplexType>::type>{};

template<typename mf_Policies>
struct vMvLiteSharedBuf{
  typedef typename mf_Policies::ComplexType ComplexType;
  typedef typename AlignedVector<ComplexType>::type ComplexVector;

  //Reorder rows and columns such that they can be accessed sequentially 
  //This is done under the parallel loop and so space is needed for all threads
  std::vector<ComplexVector> reord_buf; //[thread_idx][mode idx]
  std::vector<std::vector<ComplexVector> > Mr_t; //[thr][Mrows][nscf]
        
  void free_mem(){
    std::vector<ComplexVector>().swap(reord_buf);
    std::vector<std::vector<ComplexVector> >().swap(Mr_t);
  }
};

#ifdef USE_GRID

template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR>
class _mult_vMv_split_lite_shrbuf_impl_v<mf_Policies,lA2AfieldL,lA2AfieldR,rA2AfieldL,rA2AfieldR,grid_vector_complex_mark>{ //for SIMD vectorized W and V vectors
  typedef typename mf_Policies::ComplexType SIMDcomplexType;
  typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
  typedef typename AlignedVector<SIMDcomplexType>::type AlignedSIMDcomplexVector;

  typedef typename lA2AfieldL<mf_Policies>::DilutionType iLeftDilutionType;
  typedef typename A2AmesonField<mf_Policies,lA2AfieldR,rA2AfieldL>::LeftDilutionType iRightDilutionType;
  
  typedef typename A2AmesonField<mf_Policies,lA2AfieldR,rA2AfieldL>::RightDilutionType jLeftDilutionType;    
  typedef typename rA2AfieldR<mf_Policies>::DilutionType jRightDilutionType;

  const static int nscf = 2*3*4;
  
  int top_glb;

  int ni[nscf], nj[nscf]; //mapping f+2*(c+3*s)
  std::vector<int> ilmap[nscf], irmap[nscf], jlmap[nscf], jrmap[nscf];
  int Mrows;
  std::vector<bool> rowidx_used; 

  int nthr_setup;

  int nimax;
  int njmax;
  int mmax; //max(nimax, njmax)

  lA2AfieldL<mf_Policies> const* lptr;
  A2AmesonField<mf_Policies,lA2AfieldR,rA2AfieldL> const* Mptr;
  rA2AfieldR<mf_Policies> const* rptr;

  vMvLiteSharedBuf<mf_Policies> *buf_ptr;
  bool free_buf;
public:

  _mult_vMv_split_lite_shrbuf_impl_v(){}

  ~_mult_vMv_split_lite_shrbuf_impl_v(){
    if(free_buf) delete buf_ptr;
  }

  //If buf is NULL it will be created internally
  void setup(const lA2AfieldL<mf_Policies> &l,  const A2AmesonField<mf_Policies,lA2AfieldR,rA2AfieldL> &M, const rA2AfieldR<mf_Policies> &r, const int &_top_glb,
	     vMvLiteSharedBuf<mf_Policies> *buf = NULL){
    //Precompute index mappings
    ModeContractionIndices<iLeftDilutionType,iRightDilutionType> i_ind(l);
    ModeContractionIndices<jLeftDilutionType,jRightDilutionType> j_ind(r);
    setup(l,M,r,_top_glb,i_ind,j_ind, buf);
  }

  void setup(const lA2AfieldL<mf_Policies> &l,  const A2AmesonField<mf_Policies,lA2AfieldR,rA2AfieldL> &M, const rA2AfieldR<mf_Policies> &r, const int &_top_glb, 
	     const ModeContractionIndices<iLeftDilutionType,iRightDilutionType> &i_ind, const ModeContractionIndices<jLeftDilutionType,jRightDilutionType>& j_ind,
	     vMvLiteSharedBuf<mf_Policies> *buf = NULL){
    lptr = &l;
    Mptr = &M;
    rptr = &r;
    if(buf != NULL){
      buf_ptr = buf;
      free_buf = false;
    }else{
      buf_ptr = new vMvLiteSharedBuf<mf_Policies>;
      free_buf = true;
    }

    top_glb = _top_glb;
    assert(l.getMode(0).SIMDlogicalNodes(3) == 1);    

    modeIndexSet ilp, irp, jlp, jrp;
    ilp.time = top_glb;
    irp.time = M.getRowTimeslice();
    
    jlp.time = M.getColTimeslice();
    jrp.time = top_glb;

    Mrows = M.getNrows();

    nthr_setup = omp_get_max_threads();

    //Are particular row indices of M actually used?
    rowidx_used.resize(Mrows); for(int i=0;i<Mrows;i++) rowidx_used[i] = false;
    
    //Note:
    //il is the index of l, 
    //ir is the row index of M, 
    //jl is the column index of M and 
    //jr is the index of r

    nimax = 0; 
    njmax = 0;

    for(int s=0;s<4;s++){
      for(int c=0;c<3;c++){
    	int sc = c + 3*s;
    	ilp.spin_color = jrp.spin_color = sc;
    	for(int f=0;f<2;f++){
    	  ilp.flavor = jrp.flavor = f;

    	  int scf = f + 2*ilp.spin_color;

    	  //i index
    	  int ni_this = i_ind.getNindices(ilp,irp);
    	  ni[scf] = ni_this;
	  
    	  std::vector<int> &ilmap_this = ilmap[scf]; ilmap_this.resize(ni_this);
    	  std::vector<int> &irmap_this = irmap[scf]; irmap_this.resize(ni_this);

    	  for(int i = 0; i < ni_this; i++){
    	    i_ind.getBothIndices(ilmap_this[i],irmap_this[i],i,ilp,irp);
	    rowidx_used[ irmap_this[i] ] = true; //this row index of Mr is used
    	  }

    	  //j index
    	  int nj_this = j_ind.getNindices(jlp,jrp);
    	  nj[scf] = nj_this;
	  
    	  std::vector<int> &jlmap_this = jlmap[scf]; jlmap_this.resize(nj_this);
    	  std::vector<int> &jrmap_this = jrmap[scf]; jrmap_this.resize(nj_this);

    	  for(int j = 0; j < nj_this; j++){
    	    j_ind.getBothIndices(jlmap_this[j],jrmap_this[j],j,jlp,jrp);
    	  }

	  if(ni[scf] > nimax) nimax = ni[scf];
	  if(nj[scf] > njmax) njmax = nj[scf];
     	}
      }
    }

    mmax = nimax > njmax ? nimax : njmax;
    
    if(buf_ptr->reord_buf.size() < nthr_setup) buf_ptr->reord_buf.resize(nthr_setup);
    for(int t=0;t<nthr_setup;t++) if(buf_ptr->reord_buf[t].size() < mmax) buf_ptr->reord_buf[t].resize(mmax);

    if(buf_ptr->Mr_t.size() < nthr_setup) buf_ptr->Mr_t.resize(nthr_setup);
    for(int t=0;t<nthr_setup;t++) if(buf_ptr->Mr_t[t].size() < Mrows) buf_ptr->Mr_t[t].resize(Mrows, AlignedSIMDcomplexVector(nscf));    
  }
  
  void contract(CPSspinColorFlavorMatrix<SIMDcomplexType> &out, const int xop, const bool conj_l, const bool conj_r){
    assert(omp_get_num_threads() <= nthr_setup);

    const int thread_id = omp_get_thread_num();

    out.zero();
    const int top_lcl = top_glb - GJP.TnodeCoor() * GJP.TnodeSites();
    const int site4dop = lptr->getMode(0).threeToFour(xop, top_lcl);

    std::vector<AlignedSIMDcomplexVector> &Mr = buf_ptr->Mr_t[thread_id];

    //Matrix vector multiplication  M*r contracted on mode index j. Only do it for rows that are actually used
    SIMDcomplexType tmp_v;
#ifndef MEMTEST_MODE	       
    for(int scf=0;scf<nscf;scf++){
      int sc = scf/2; int f = scf % 2;

      SIMDcomplexType* rreord_p = buf_ptr->reord_buf[thread_id].data();

      //j index
      int nj_this = nj[scf];	  
      const std::vector<int> &jrmap_this = jrmap[scf];

#ifndef MEMTEST_MODE
      for(int j = 0; j < nj_this; j++){
	const SIMDcomplexType &rval_tmp = rptr->nativeElem(jrmap_this[j], site4dop, sc, f);
	rreord_p[j] = conj_r ? Grid::conjugate(rval_tmp) : rval_tmp;
      }
#endif      

      for(int i=0;i<Mrows;i++){
	if(!rowidx_used[i]) continue;

	zeroit(Mr[i][scf]);

	for(int j=0;j<nj_this;j++){
	  Grid::vsplat(tmp_v, (*Mptr)(i, jlmap[scf][j]) );
	  Mr[i][scf] = Mr[i][scf] + tmp_v * rreord_p[j];
	}
      }
    }

    //Vector vector multiplication l*(M*r)
    for(int sl=0;sl<4;sl++){
      for(int cl=0;cl<3;cl++){
	int scl = cl + 3*sl;
	for(int fl=0;fl<2;fl++){
	  int scfl = fl + 2*scl;	    

	  SIMDcomplexType* lreord_p = buf_ptr->reord_buf[thread_id].data();
  
   	  //i index
	  int ni_this = ni[scfl];
    	  const std::vector<int> &ilmap_this = ilmap[scfl];
	  
#ifndef MEMTEST_MODE
    	  for(int i = 0; i < ni_this; i++){
	    const SIMDcomplexType &lval_tmp = lptr->nativeElem(ilmap_this[i], site4dop, scl, fl);
	    lreord_p[i] = conj_l ? Grid::conjugate(lval_tmp) : lval_tmp;
	  }
#endif

	  for(int sr=0;sr<4;sr++){
	    for(int cr=0;cr<3;cr++){
	      for(int fr=0;fr<2;fr++){
		int scfr = fr + 2*(cr + 3*sr);		
		
		SIMDcomplexType &into = out(sl,sr)(cl,cr)(fl,fr);
		
		for(int i=0;i<ni_this;i++){
		  into = into + lreord_p[i]*Mr[irmap[scfl][i]][scfr];
		}
	      }
	    }
	  }
	}
      }
    }	    
#endif

  }
	      
  //Internally parallelized version
  void contract(typename AlignedVector<CPSspinColorFlavorMatrix<SIMDcomplexType> >::type &out, const bool conj_l, const bool conj_r){
    size_t logical_sites_3d = lptr->getMode(0).nodeSites(0)*lptr->getMode(0).nodeSites(1)*lptr->getMode(0).nodeSites(2);
    out.resize(logical_sites_3d);
#pragma omp parallel for
    for(int x=0;x<logical_sites_3d;x++)
      contract(out[x], x, conj_l, conj_r);
  }

};


#endif //USE_GRID

CPS_END_NAMESPACE

#endif
