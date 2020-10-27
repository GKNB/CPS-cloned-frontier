#ifndef _MULT_VMV_SPLIT_LITE
#define _MULT_VMV_SPLIT_LITE

CPS_START_NAMESPACE

#include "implementation/mesonfield_mult_vMv_common.tcc"
#include "implementation/mesonfield_mult_vMv_split_lite.tcc"

//Perform site-wise vMv operation more efficiently by moving repeated overhead outside of a threaded loop

//This class contains scratch space for thread writing. It can be shared between multiple instances of mult_vMv_split_lite
template<typename mf_Policies>
struct mult_vMv_split_lite_scratch_space{
  typedef typename mf_Policies::ComplexType SIMDcomplexType;
  typedef typename AlignedVector<SIMDcomplexType>::type AlignedSIMDcomplexVector;
  typedef typename _mult_vMv_impl_v_getPolicy<mf_Policies::GPARITY>::type OutputPolicy;

  int nthr_setup; //max number of threads 
  int Mrows; //number of rows of meson field
  int blocksize;

  //Reorder rows and columns such that they can be accessed sequentially 
  //This is done under the parallel loop and so space is needed for all threads
  std::vector<AlignedSIMDcomplexVector> reord_buf; //[thread_idx][mode idx]
  std::vector<std::vector<AlignedSIMDcomplexVector> > Mr_t; //[thr][Mrows][nscf]

  //Mrows is the number of rows of the meson field
  void setup(const int Mrows){
    const static int nf = OutputPolicy::nf();
    const static int nscf = nf*3*4;
    
    this->blocksize = BlockedSplitvMvArgs::b;
    this->Mrows = Mrows;
    this->nthr_setup = omp_get_max_threads();    

    if(reord_buf.size() < nthr_setup) reord_buf.resize(nthr_setup);
    if(Mr_t.size() < nthr_setup) Mr_t.resize(nthr_setup);
    for(int t=0;t<nthr_setup;t++){
      if(reord_buf[t].size() < blocksize) reord_buf[t].resize(blocksize);
      if(Mr_t[t].size() < Mrows) Mr_t[t].resize(Mrows);
      for(int r=0;r<Mrows;r++)
	Mr_t[t][r].resize(nscf);
    }
  }

  void free_mem(){
    std::vector<AlignedSIMDcomplexVector>().swap(reord_buf);
    std::vector<std::vector<AlignedSIMDcomplexVector> >().swap(Mr_t);
  }
};
  



template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR>
class mult_vMv_split_lite{
  typedef typename ComplexClassify<typename mf_Policies::ComplexType>::type ComplexClass;
  typedef mult_vMv_split_lite_cnum_policy<mf_Policies,ComplexClass> CnumPolicy;
  typedef typename _mult_vMv_impl_v_getPolicy<mf_Policies::GPARITY>::type OutputPolicy;

  typedef typename mf_Policies::ComplexType SIMDcomplexType;
  typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
  typedef typename AlignedVector<SIMDcomplexType>::type AlignedSIMDcomplexVector;

  typedef typename lA2AfieldL<mf_Policies>::DilutionType iLeftDilutionType;
  typedef typename A2AmesonField<mf_Policies,lA2AfieldR,rA2AfieldL>::LeftDilutionType iRightDilutionType;
  
  typedef typename A2AmesonField<mf_Policies,lA2AfieldR,rA2AfieldL>::RightDilutionType jLeftDilutionType;    
  typedef typename rA2AfieldR<mf_Policies>::DilutionType jRightDilutionType;

  typedef typename OutputPolicy::template MatrixType<SIMDcomplexType> OutputMatrixType;

  const static int nf = OutputPolicy::nf();
  const static int nscf = nf*3*4;
  
  int top_glb;

  int ni[nscf], nj[nscf]; //mapping f+nf*(c+3*s)
  std::vector<int> ilmap[nscf], irmap[nscf], jlmap[nscf], jrmap[nscf];
  int Mrows;
  std::vector<bool> rowidx_used; 

  int nimax;
  int njmax;

  lA2AfieldL<mf_Policies> const* lptr;
  A2AmesonField<mf_Policies,lA2AfieldR,rA2AfieldL> const* Mptr;
  rA2AfieldR<mf_Policies> const* rptr;

  mult_vMv_split_lite_scratch_space<mf_Policies> *scratch;
  bool own_scratch;

public:

  mult_vMv_split_lite(): own_scratch(false), scratch(NULL){}

  ~mult_vMv_split_lite(){
    if(scratch && own_scratch) delete scratch;
  }

  //Free the scratch memory (if owned by this instance)
  void free_mem(){
    if(scratch && own_scratch)
      scratch->free_mem();
  }

  //NOTE: The provided time index is the *global* time index. The contract component should only be run on the node in which this timeslice is contained

  //shared_scratch allows the scratch space to be shared between multiple instances of this class
  void setup(const lA2AfieldL<mf_Policies> &l,  const A2AmesonField<mf_Policies,lA2AfieldR,rA2AfieldL> &M, const rA2AfieldR<mf_Policies> &r, const int &_top_glb,
	     mult_vMv_split_lite_scratch_space<mf_Policies> *shared_scratch = NULL){
    //Precompute index mappings
    ModeContractionIndices<iLeftDilutionType,iRightDilutionType> i_ind(l);
    ModeContractionIndices<jLeftDilutionType,jRightDilutionType> j_ind(r);
    setup(l,M,r,_top_glb,i_ind,j_ind, shared_scratch);
  }

  void setup(const lA2AfieldL<mf_Policies> &l,  const A2AmesonField<mf_Policies,lA2AfieldR,rA2AfieldL> &M, const rA2AfieldR<mf_Policies> &r, const int &_top_glb, 
	     const ModeContractionIndices<iLeftDilutionType,iRightDilutionType> &i_ind, const ModeContractionIndices<jLeftDilutionType,jRightDilutionType>& j_ind,
	     mult_vMv_split_lite_scratch_space<mf_Policies> *shared_scratch = NULL){
    lptr = &l;
    Mptr = &M;
    rptr = &r;
     
    top_glb = _top_glb;
    CnumPolicy::checkDecomp(l.getMode(0));

    modeIndexSet ilp, irp, jlp, jrp;
    ilp.time = top_glb;
    irp.time = M.getRowTimeslice();
    
    jlp.time = M.getColTimeslice();
    jrp.time = top_glb;

    Mrows = M.getNrows();

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
    	for(int f=0;f<nf;f++){
    	  ilp.flavor = jrp.flavor = f;

    	  int scf = f + nf*ilp.spin_color;

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

    if(shared_scratch){
      if(this->scratch && own_scratch) delete this->scratch;

      this->scratch = shared_scratch;
      own_scratch = false;
    }else if(!this->scratch){
      scratch = new mult_vMv_split_lite_scratch_space<mf_Policies>();
      own_scratch = true;
    }
    scratch->setup(Mrows);
  }
  
  void contract(OutputMatrixType &out, const int xop, const bool conj_l, const bool conj_r){
    const int thread_id = omp_get_thread_num();

    assert(scratch);
    assert(omp_get_num_threads() <= scratch->nthr_setup);
    assert(Mrows <= scratch->Mrows);

    SIMDcomplexType* rreord_p = scratch->reord_buf[thread_id].data();
    SIMDcomplexType* lreord_p = rreord_p;

    out.zero();
    const int top_lcl = top_glb - GJP.TnodeCoor() * GJP.TnodeSites();
    const int site4dop = lptr->getMode(0).threeToFour(xop, top_lcl);

    if(top_lcl < 0 || top_lcl >= GJP.TnodeSites()) ERR.General("mult_vMv_split_lite","contract","This function should only be run on the node containing the contraction timeslice!");

    std::vector<AlignedSIMDcomplexVector> &Mr = scratch->Mr_t[thread_id];
    for(int i=0;i<Mrows;i++)
      for(int scf=0;scf<nscf;scf++)
	CnumPolicy::zeroit(Mr[i][scf]);

    //Matrix vector multiplication  M*r contracted on mode index j. Only do it for rows that are actually used

    int blocksize = scratch->blocksize;
    int niblock = (Mrows + blocksize - 1)/blocksize;
    int njblock = (njmax + blocksize - 1)/blocksize;
    SIMDcomplexType tmp_v;

#ifndef MEMTEST_MODE	       
    for(int bi=0;bi<niblock;bi++){
      int istart = bi*blocksize;
      int ilessthan = std::min(istart+blocksize, Mrows);

      for(int bj=0;bj<njblock;bj++){
    
	for(int scf=0;scf<nscf;scf++){
	  int sc = scf/nf; int f = scf % nf;
    
	  //j index
	  int nj_this = nj[scf];	  
	  const std::vector<int> &jrmap_this = jrmap[scf];

	  int jstart = bj*blocksize;
	  int jlessthan = std::min(jstart+blocksize, nj_this);

	  //Poke into temp mem
	  for(int j = jstart; j < jlessthan; j++){
	    const SIMDcomplexType &rval_tmp = rptr->nativeElem(jrmap_this[j], site4dop, sc, f);
	    rreord_p[j-jstart] = conj_r ? CnumPolicy::cconj(rval_tmp) : rval_tmp;
	  }
	  

	  //M*r
	  for(int i=istart;i<ilessthan;i++){
	    if(!rowidx_used[i]) continue;
	    
	    for(int j=jstart;j<jlessthan;j++){
	      CnumPolicy::splat(tmp_v, (*Mptr)(i, jlmap[scf][j]) );
	      Mr[i][scf] = Mr[i][scf] + tmp_v * rreord_p[j-jstart];
	    }
	  }

	}//scf
      }//bj
    }//bi


    //Vector vector multiplication l*(M*r)
    for(int sl=0;sl<4;sl++){
      for(int cl=0;cl<3;cl++){
	int scl = cl + 3*sl;
	for(int fl=0;fl<nf;fl++){
	  int scfl = fl + nf*scl;	    
 
   	  //i index
	  int ni_this = ni[scfl];
    	  const std::vector<int> &ilmap_this = ilmap[scfl];

	  int niblock = (ni_this + blocksize - 1)/blocksize;
	  for(int bi=0;bi<niblock;bi++){
	    int istart = bi*blocksize;
	    int ilessthan = std::min(istart + blocksize, ni_this);

	    for(int i = istart; i < ilessthan; i++){
	      const SIMDcomplexType &lval_tmp = lptr->nativeElem(ilmap_this[i], site4dop, scl, fl);
	      lreord_p[i-istart] = conj_l ? CnumPolicy::cconj(lval_tmp) : lval_tmp;
	    }

	    for(int sr=0;sr<4;sr++){
	      for(int cr=0;cr<3;cr++){
		for(int fr=0;fr<nf;fr++){
		  int scfr = fr + nf*(cr + 3*sr);		
		  
		  SIMDcomplexType &into = OutputPolicy::acc(sl,sr,cl,cr,fl,fr,out);
		
		  for(int i=istart;i<ilessthan;i++){
		    into = into + lreord_p[i-istart]*Mr[irmap[scfl][i]][scfr];
		  }
		}
	      }
	    }

	  }//bi

	}//fl
      }//cl
    }//sl
     
#endif //MEMTEST_MODE
  }
	      
  //Internally parallelized version
  void contract(typename AlignedVector<OutputMatrixType>::type &out, const bool conj_l, const bool conj_r){
    size_t logical_sites_3d = lptr->getMode(0).nodeSites(0)*lptr->getMode(0).nodeSites(1)*lptr->getMode(0).nodeSites(2);
    out.resize(logical_sites_3d);
#pragma omp parallel for
    for(int x=0;x<logical_sites_3d;x++)
      contract(out[x], x, conj_l, conj_r);
  }

};




CPS_END_NAMESPACE

#endif
