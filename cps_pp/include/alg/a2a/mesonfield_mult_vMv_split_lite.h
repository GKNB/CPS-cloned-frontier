#ifndef _MULT_VMV_SPLIT_LITE
#define _MULT_VMV_SPLIT_LITE

#include<alg/a2a/mesonfield_mult_vMv_split.h>

CPS_START_NAMESPACE

//This is a less aggressive version of the split vMv routines that just seeks to avoid memory allocation under the threaded loop
template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR,
	 typename ComplexClass>
class _mult_vMv_split_lite_impl_v{};

template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR	
	 >
class mult_vMv_split_lite: public _mult_vMv_split_lite_impl_v<mf_Policies,lA2AfieldL,lA2AfieldR,rA2AfieldL,rA2AfieldR,typename ComplexClassify<typename mf_Policies::ComplexType>::type>{};


#ifdef USE_GRID

template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR>
class _mult_vMv_split_lite_impl_v<mf_Policies,lA2AfieldL,lA2AfieldR,rA2AfieldL,rA2AfieldR,grid_vector_complex_mark>{ //for SIMD vectorized W and V vectors
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

#define STACK_ALLOC_REORD
  
#ifdef STACK_ALLOC_REORD
  int nimax;
  int njmax;
#else
  //Reorder rows and columns such that they can be accessed sequentially 
  //This is done under the parallel loop and so space is needed for all threads
  std::vector<AlignedSIMDcomplexVector> lreord[nscf]; //[scf][thread_idx]
  std::vector<AlignedSIMDcomplexVector> rreord[nscf];
#endif

  lA2AfieldL<mf_Policies> const* lptr;
  A2AmesonField<mf_Policies,lA2AfieldR,rA2AfieldL> const* Mptr;
  rA2AfieldR<mf_Policies> const* rptr;

public:

  _mult_vMv_split_lite_impl_v(){}

  void setup(const lA2AfieldL<mf_Policies> &l,  const A2AmesonField<mf_Policies,lA2AfieldR,rA2AfieldL> &M, const rA2AfieldR<mf_Policies> &r, const int &_top_glb){
    //Precompute index mappings
    ModeContractionIndices<iLeftDilutionType,iRightDilutionType> i_ind(l);
    ModeContractionIndices<jLeftDilutionType,jRightDilutionType> j_ind(r);
    setup(l,M,r,_top_glb,i_ind,j_ind);
  }

  void setup(const lA2AfieldL<mf_Policies> &l,  const A2AmesonField<mf_Policies,lA2AfieldR,rA2AfieldL> &M, const rA2AfieldR<mf_Policies> &r, const int &_top_glb, 
	     const ModeContractionIndices<iLeftDilutionType,iRightDilutionType> &i_ind, const ModeContractionIndices<jLeftDilutionType,jRightDilutionType>& j_ind){
    lptr = &l;
    Mptr = &M;
    rptr = &r;
     
    top_glb = _top_glb;
    assert(l.getMode(0).SIMDlogicalNodes(3) == 1);    

    modeIndexSet ilp, irp, jlp, jrp;
    ilp.time = top_glb;
    irp.time = M.getRowTimeslice();
    
    jlp.time = M.getColTimeslice();
    jrp.time = top_glb;

    Mrows = M.getNrows();

    nthr_setup = omp_get_max_threads();

#ifndef STACK_ALLOC_REORD    
    for(int scf=0;scf<nscf;scf++){
      lreord[scf].resize(nthr_setup);
      rreord[scf].resize(nthr_setup);      
    }
#endif

    //Are particular row indices of M actually used?
    rowidx_used.resize(Mrows); for(int i=0;i<Mrows;i++) rowidx_used[i] = false;
    
    //Note:
    //il is the index of l, 
    //ir is the row index of M, 
    //jl is the column index of M and 
    //jr is the index of r

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

#ifndef STACK_ALLOC_REORD
	  for(int t=0;t<nthr_setup;t++)
	    lreord[scf][t].resize(ni_this);
#endif

    	  for(int i = 0; i < ni_this; i++){
    	    i_ind.getBothIndices(ilmap_this[i],irmap_this[i],i,ilp,irp);
	    rowidx_used[ irmap_this[i] ] = true; //this row index of Mr is used
    	  }

    	  //j index
    	  int nj_this = j_ind.getNindices(jlp,jrp);
    	  nj[scf] = nj_this;
	  
    	  std::vector<int> &jlmap_this = jlmap[scf]; jlmap_this.resize(nj_this);
    	  std::vector<int> &jrmap_this = jrmap[scf]; jrmap_this.resize(nj_this);

#ifndef STACK_ALLOC_REORD
	  for(int t=0;t<nthr_setup;t++)
	    rreord[scf][t].resize(nj_this);
#endif

    	  for(int j = 0; j < nj_this; j++){
    	    j_ind.getBothIndices(jlmap_this[j],jrmap_this[j],j,jlp,jrp);
    	  }

     	}
      }
    }

#ifdef STACK_ALLOC_REORD
    nimax = 0; 
    njmax = 0;
    for(int scf=0;scf<nscf;scf++){
      if(ni[scf] > nimax) nimax = ni[scf];
      if(nj[scf] > njmax) njmax = nj[scf];
    }
#endif	  
  }
  
  void contract(CPSspinColorFlavorMatrix<SIMDcomplexType> &out, const int xop, const bool conj_l, const bool conj_r){
    assert(omp_get_num_threads() <= nthr_setup);

    const int thread_id = omp_get_thread_num();

    out.zero();
    const int top_lcl = top_glb - GJP.TnodeCoor() * GJP.TnodeSites();
    const int site4dop = lptr->getMode(0).threeToFour(xop, top_lcl);

#ifdef STACK_ALLOC_REORD
    SIMDcomplexType lreord[nscf][nimax];
    SIMDcomplexType rreord[nscf][njmax];
#endif

    for(int s=0;s<4;s++){
      for(int c=0;c<3;c++){
    	int sc = c + 3*s;
    	for(int f=0;f<2;f++){
    	  int scf = f + 2*sc;

#ifdef STACK_ALLOC_REORD
	  SIMDcomplexType* lreord_p = lreord[scf];
	  SIMDcomplexType* rreord_p = rreord[scf];
#else
	  SIMDcomplexType* lreord_p = lreord[scf][thread_id].data();
	  SIMDcomplexType* rreord_p = rreord[scf][thread_id].data();
#endif

    	  //i index
    	  int ni_this = ni[scf];
    	  const std::vector<int> &ilmap_this = ilmap[scf];

#ifndef MEMTEST_MODE
    	  for(int i = 0; i < ni_this; i++){
	    const SIMDcomplexType &lval_tmp = lptr->nativeElem(ilmap_this[i], site4dop, sc, f);
	    lreord_p[i] = conj_l ? Grid::conjugate(lval_tmp) : lval_tmp;
    	  }
#endif

    	  //j index
    	  int nj_this = nj[scf];	  
    	  const std::vector<int> &jrmap_this = jrmap[scf];

#ifndef MEMTEST_MODE
    	  for(int j = 0; j < nj_this; j++){
	    const SIMDcomplexType &rval_tmp = rptr->nativeElem(jrmap_this[j], site4dop, sc, f);
	    rreord_p[j] = conj_r ? Grid::conjugate(rval_tmp) : rval_tmp;
    	  }
#endif
     	}
      }
    }	  


    //Matrix vector multiplication  M*r contracted on mode index j. Only do it for rows that are actually used
    SIMDcomplexType Mr[Mrows][nscf]; //stack
    SIMDcomplexType tmp_v;
#ifndef MEMTEST_MODE	  
    for(int i=0;i<Mrows;i++){
      if(!rowidx_used[i]) continue;
      
      for(int scf=0;scf<nscf;scf++){
	int sc = scf/2; int f = scf % 2;
	int nj_this = nj[scf];
	zeroit(Mr[i][scf]);

#ifdef STACK_ALLOC_REORD
	SIMDcomplexType* rreord_p = rreord[scf];
#else
	SIMDcomplexType* rreord_p = rreord[scf][thread_id].data();
#endif
	
	for(int j=0;j<nj_this;j++){
	  Grid::vsplat(tmp_v, (*Mptr)(i, jlmap[scf][j]) );
	  Mr[i][scf] = Mr[i][scf] + tmp_v * rreord_p[j];
	}
      }
    }

    //Vector vector multiplication l*(M*r)
    for(int sl=0;sl<4;sl++){
      for(int sr=0;sr<4;sr++){
	for(int cl=0;cl<3;cl++){
	  int scl = cl + 3*sl;
	  for(int cr=0;cr<3;cr++){
	    for(int fl=0;fl<2;fl++){
	      int scfl = fl + 2*scl;
	      int ni_this = ni[scfl];

#ifdef STACK_ALLOC_REORD
	      SIMDcomplexType* lreord_p = lreord[scfl];
#else
	      SIMDcomplexType* lreord_p = lreord[scfl][thread_id].data();
#endif

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

};


#undef STACK_ALLOC_REORD

#endif //USE_GRID

CPS_END_NAMESPACE

#endif
