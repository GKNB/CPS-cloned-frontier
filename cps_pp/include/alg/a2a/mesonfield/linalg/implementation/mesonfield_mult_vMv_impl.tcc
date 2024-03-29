#ifndef _MULT_VMV_IMPL
#define _MULT_VMV_IMPL

#include "mesonfield_mult_vMv_common.tcc"
//Vector mesonfield outer product implementation

template<typename ImplPolicies,
	 typename lA2AfieldView, typename MesonFieldView, typename rA2AfieldView,
	 typename ComplexClass>
class _mult_vMv_impl_v{};

template<typename ImplPolicies,
	 typename lA2AfieldView, typename MesonFieldView, typename rA2AfieldView
	 >
class _mult_vMv_impl_v<ImplPolicies,lA2AfieldView,MesonFieldView,rA2AfieldView,complex_double_or_float_mark>{ //necessary to avoid an annoying ambigous overload when mesonfield friends mult
public:
  typedef typename MesonFieldView::ScalarComplexType ScalarComplexType;
  typedef typename ImplPolicies::template MatrixType<ScalarComplexType> MatrixType;

  //Do a column reorder but where we pack the row indices to exclude those not used (as indicated by input bool array)
  //Output as a GSL matrix. Can reuse previously allocated matrix providing its big enough
  static typename gsl_wrapper<typename ScalarComplexType::value_type>::matrix_complex * GSLpackedColReorder(const MesonFieldView &M, const int idx_map[], int map_size, bool rowidx_used[],
												     typename gsl_wrapper<typename ScalarComplexType::value_type>::matrix_complex *reuse ){
    typedef gsl_wrapper<typename ScalarComplexType::value_type> gw;
    assert(sizeof(typename gw::complex) == sizeof(ScalarComplexType));
    int rows = M.getNrows();
    int cols = M.getNcols();
    
    int nrows_used = 0;
    for(int i_full=0;i_full<rows;i_full++) if(rowidx_used[i_full]) nrows_used++;
    
    typename gw::matrix_complex *M_packed;
    if(reuse!=NULL){
      M_packed = reuse;
      M_packed->size1 = nrows_used;
      M_packed->size2 = M_packed->tda = map_size;
    }else M_packed = gw::matrix_complex_alloc(nrows_used,map_size);
    
    //Look for contiguous blocks in the idx_map we can take advantage of
    std::vector<std::pair<int,int> > blocks;
    find_contiguous_blocks(blocks,idx_map,map_size);

    int i_packed = 0;
    for(int i_full=0;i_full<rows;i_full++){
      if(rowidx_used[i_full]){
	ScalarComplexType const* mf_row_base = M.ptr() + cols*i_full; //meson field are row major so columns are contiguous
	typename gw::complex* row_base = gw::matrix_complex_ptr(M_packed,i_packed,0); //GSL matrix are also row major
	for(int b=0;b<blocks.size();b++){
	  ScalarComplexType const* block_ptr = mf_row_base + idx_map[blocks[b].first];
	  memcpy((void*)row_base,(void*)block_ptr,blocks[b].second*sizeof(ScalarComplexType));
	  row_base += blocks[b].second;
	}
	i_packed++;
      }
    }

    return M_packed;
  }
  
  //Form SpinColorFlavorMatrix prod1 = vL_i(\vec xop, top ; tpi2) [\sum_{\vec xpi2} wL_i^dag(\vec xpi2, tpi2) S2 vL_j(\vec xpi2, tpi2; top)] wL_j^dag(\vec xop,top)

  // l^i(xop,top) M^ij(tl,tr) r^j(xop,top)
  //argument xop is the *local* 3d site index in canonical ordering, top is the *local* time coordinate
  // Node local and unthreaded
  static void mult(MatrixType &out, const lA2AfieldView &l,  const MesonFieldView &M, const rA2AfieldView &r, const int xop, const int top, const bool conj_l, const bool conj_r){
    typedef typename lA2AfieldView::DilutionType iLeftDilutionType;
    typedef typename MesonFieldView::LeftDilutionType iRightDilutionType;

    typedef typename MesonFieldView::RightDilutionType jLeftDilutionType;    
    typedef typename rA2AfieldView::DilutionType jRightDilutionType;

    out.zero();
    constexpr int nf = ImplPolicies::nf();
    int top_glb = top+GJP.TnodeSites()*GJP.TnodeCoor();

    //Precompute index mappings
    ModeContractionIndices<iLeftDilutionType,iRightDilutionType> i_ind(l);
    ModeContractionIndices<jLeftDilutionType,jRightDilutionType> j_ind(r);

    modeIndexSet ilp, irp, jlp, jrp;
    ilp.time = l.tblock(top_glb);
    irp.time = l.tblock(M.getRowTimeslice());
    
    jlp.time = r.tblock(M.getColTimeslice());
    jrp.time = r.tblock(top_glb);

    int site4dop = xop + GJP.VolNodeSites()/GJP.TnodeSites()*top;

    const static int nscf = nf*3*4;
    const static int complex_scf_vect_size = nscf*2;

    int ni[nscf], nj[nscf]; //mapping f+nf*(c+3*s)
    std::vector<int> ilmap[nscf], irmap[nscf], jlmap[nscf], jrmap[nscf];

    //Reorder rows and columns such that they can be accessed sequentially
    std::vector<ScalarComplexType> lreord[nscf];
    std::vector<ScalarComplexType> rreord[nscf];

    int Mrows = M.getNrows();
    int Mcols = M.getNcols();

    //Are particular row and column indices of M actually used?
    bool rowidx_used[Mrows]; for(int i=0;i<Mrows;i++) rowidx_used[i] = false;
    bool colidx_used[Mcols]; for(int i=0;i<Mcols;i++) colidx_used[i] = false;
    
    //Note:
    //il is the index of l, 
    //ir is the row index of M, 
    //jl is the column index of M and 
    //jr is the index of r

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
	    rowidx_used[ irmap_this[i] ] = true; //this row index is used
	  }

	  //M.rowReorder(rowreord[scf], &irmap_this.front(), ni_this);
	  lreord[scf].resize(ni_this);
	  for(int i = 0; i < ni_this; i++){
	    const ScalarComplexType &lval_tmp = l.nativeElem(ilmap_this[i], site4dop, sc, f);
#ifndef MEMTEST_MODE
	    lreord[scf][i] = conj_l ? cconj(lval_tmp) : lval_tmp;
#endif
	  }

	  //j index
	  int nj_this = j_ind.getNindices(jlp,jrp);
	  nj[scf] = nj_this;
	  
	  std::vector<int> &jlmap_this = jlmap[scf]; jlmap_this.resize(nj_this);
	  std::vector<int> &jrmap_this = jrmap[scf]; jrmap_this.resize(nj_this);

	  for(int j = 0; j < nj_this; j++){
	    j_ind.getBothIndices(jlmap_this[j],jrmap_this[j],j,jlp,jrp);
	    colidx_used[ jlmap_this[j] ] = true;
	  }

	  //M.colReorder(colreord[scf], &jlmap_this.front(), nj_this);
	  rreord[scf].resize(nj_this);
	  for(int j = 0; j < nj_this; j++){
	    const ScalarComplexType &rval_tmp = r.nativeElem(jrmap_this[j], site4dop, sc, f);
#ifndef MEMTEST_MODE
	    rreord[scf][j] = conj_r ? cconj(rval_tmp) : rval_tmp;
#endif
	  }

	}
      }
    }	  


    //Matrix vector multiplication  M*r
    ScalarComplexType Mr[Mrows][nscf];

    //Use GSL BLAS
    typedef gsl_wrapper<typename ScalarComplexType::value_type> gw;

    assert(sizeof(typename gw::complex) == sizeof(ScalarComplexType) );

    typename gw::complex tmp;

    //Not all rows or columns of M are used, so lets use a packed matrix
    int nrows_used = 0;
    for(int i_full=0;i_full<Mrows;i_full++) if(rowidx_used[i_full]) nrows_used++;

    typename gw::complex one; GSL_SET_COMPLEX(&one,1.0,0.0);
    typename gw::complex zero; GSL_SET_COMPLEX(&zero,0.0,0.0);

    typename gw::vector_complex *Mr_packed = gw::vector_complex_alloc(nrows_used);
    typename gw::matrix_complex *M_packed_buffer = gw::matrix_complex_alloc(nrows_used,Mcols); //matrix cannot be larger than this so we can reuse the memory

    for(int scf=0;scf<nscf;scf++){
      int nj_this = nj[scf];
      
      std::vector<int> &jlmap_this = jlmap[scf];
      
      typename gw::matrix_complex * M_packed = GSLpackedColReorder(M,&jlmap_this.front(), nj_this, rowidx_used, M_packed_buffer); //packs the GSL matrix
             
      int i_packed = 0;
      int i_packed_unmap[nrows_used];
      for(int i_full=0;i_full<Mrows;i_full++)
    	if(rowidx_used[i_full]) i_packed_unmap[i_packed++] = i_full;
      
      typename ScalarComplexType::value_type* base = (typename ScalarComplexType::value_type*)&rreord[scf][0];
      typename gw::block_complex_struct block;
      block.data = base;
      block.size = nj_this;

      typename gw::vector_complex rgsl;
      rgsl.block = &block;
      rgsl.data = base;
      rgsl.stride = 1;
      rgsl.owner = 1;
      rgsl.size = nj_this;

      gw::blas_gemv(CblasNoTrans, one, M_packed, &rgsl, zero, Mr_packed);

      for(int i_packed=0;i_packed < nrows_used; i_packed++){
    	tmp = gw::vector_complex_get(Mr_packed,i_packed);
    	Mr[ i_packed_unmap[i_packed] ][scf] = ScalarComplexType( GSL_REAL(tmp), GSL_IMAG(tmp) );
      }
    }
    gw::vector_complex_free(Mr_packed);
    gw::matrix_complex_free(M_packed_buffer);

    //Vector vector multiplication l*(M*r)
    for(int sl=0;sl<4;sl++){
      for(int sr=0;sr<4;sr++){
	for(int cl=0;cl<3;cl++){
	  for(int cr=0;cr<3;cr++){
	    for(int fl=0;fl<nf;fl++){
	      int scfl = fl + nf*(cl + 3*sl);
	      int ni_this = ni[scfl];
	      for(int fr=0;fr<nf;fr++){
		int scfr = fr + nf*(cr + 3*sr);
		ScalarComplexType &into = ImplPolicies::acc(sl,sr,cl,cr,fl,fr,out);
		for(int i=0;i<ni_this;i++){
		  int i_full = irmap[scfl][i];
		  into += lreord[scfl][i] * Mr[i_full][scfr];
		}
	      }
	    }
	  }
	}
      }
    }	    

  }

  static void mult_slow(MatrixType &out, const lA2AfieldView &l,  const MesonFieldView &M, const rA2AfieldView &r, const int xop, const int top, const bool conj_l, const bool conj_r){

    int site4dop = xop + GJP.VolNodeSites()/GJP.TnodeSites()*top;

    A2Aparams i_params(l), j_params(r);
    StandardIndexDilution idil(i_params), jdil(j_params);

    constexpr int nf = ImplPolicies::nf();    
    int ni = idil.getNmodes();
    int nj = jdil.getNmodes();

    out.zero();

    for(int sl=0;sl<4;sl++){
      for(int sr=0;sr<4;sr++){
	for(int cl=0;cl<3;cl++){
	  for(int cr=0;cr<3;cr++){
	    for(int fl=0;fl<nf;fl++){	  
	      for(int fr=0;fr<nf;fr++){

		for(int i=0;i<ni;i++){
		  const ScalarComplexType &lval_tmp = l.elem(i,xop,top,cl+3*sl,fl);
		  ScalarComplexType lval = conj_l ? cconj(lval_tmp) : lval_tmp;
		  
  		  for(int j=0;j<nj;j++){
  		    const ScalarComplexType &rval_tmp = r.elem(j,xop,top,cr+3*sr,fr);
  		    ScalarComplexType rval = conj_r ? cconj(rval_tmp) : rval_tmp;

		    const ScalarComplexType &Mval = M.elem(i,j);
		    ScalarComplexType delta = lval * Mval * rval;
		    ImplPolicies::acc(sl,sr,cl,cr,fl,fr,out) += delta;
  		  }
  		}
  	      }
  	    }
  	  }
  	}
      }
    }

  }
};




#ifdef USE_GRID

template<typename ImplPolicies,
	 typename lA2AfieldView, typename MesonFieldView, typename rA2AfieldView
	 >
class _mult_vMv_impl_v<ImplPolicies,lA2AfieldView,MesonFieldView,rA2AfieldView,grid_vector_complex_mark>{ //for SIMD vectorized W and V vectors
public:
  typedef typename MesonFieldView::Policies mf_Policies;
  typedef typename mf_Policies::ComplexType SIMDcomplexType;
  typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
  typedef typename ImplPolicies::template MatrixType<SIMDcomplexType> MatrixType;
  typedef typename AlignedVector<SIMDcomplexType>::type AlignedSIMDcomplexVector;
  
  //Form SpinColorFlavorMatrix prod1 = vL_i(\vec xop, top ; tpi2) [\sum_{\vec xpi2} wL_i^dag(\vec xpi2, tpi2) S2 vL_j(\vec xpi2, tpi2; top)] wL_j^dag(\vec xop,top)

  // l^i(xop,top) M^ij(tl,tr) r^j(xop,top)
  //argument xop is the *local* 3d site index of the reduced (logical) lattice in canonical ordering, top is the *local* time coordinate
  //it is assumed that the lattice is not SIMD vectorized in the time direction
  // Node local and unthreaded
  static void mult(MatrixType &out, const lA2AfieldView &l,  const MesonFieldView &M, const rA2AfieldView &r, const int xop, const int top, const bool conj_l, const bool conj_r){
    typedef typename lA2AfieldView::DilutionType iLeftDilutionType;
    typedef typename MesonFieldView::LeftDilutionType iRightDilutionType;

    typedef typename MesonFieldView::RightDilutionType jLeftDilutionType;    
    typedef typename rA2AfieldView::DilutionType jRightDilutionType;

    assert(l.getMode(0).SIMDlogicalNodes(3) == 1);
    
#ifndef MEMTEST_MODE
    out.zero();
#endif
    
    int top_glb = top+GJP.TnodeSites()*GJP.TnodeCoor();
    constexpr int nf = ImplPolicies::nf();

    //Precompute index mappings
    ModeContractionIndices<iLeftDilutionType,iRightDilutionType> i_ind(l);
    ModeContractionIndices<jLeftDilutionType,jRightDilutionType> j_ind(r);

    modeIndexSet ilp, irp, jlp, jrp;
    ilp.time = l.tblock(top_glb);
    irp.time = l.tblock(M.getRowTimeslice());
    
    jlp.time = l.tblock(M.getColTimeslice());
    jrp.time = l.tblock(top_glb);

    int site4dop = l.getMode(0).threeToFour(xop, top);

    const static int nscf = nf*3*4;
    int ni[nscf], nj[nscf]; //mapping f+nf*(c+3*s)
    std::vector<int> ilmap[nscf], irmap[nscf], jlmap[nscf], jrmap[nscf];

    const int Mrows = M.getNrows();

    //Are particular row indices of M actually used?
    bool rowidx_used[Mrows]; for(int i=0;i<Mrows;i++) rowidx_used[i] = false;

    //Reorder rows and columns such that they can be accessed sequentially    
    AlignedSIMDcomplexVector lreord[nscf];
    AlignedSIMDcomplexVector rreord[nscf];
    
    //Note:
    //il is the index of l, 
    //ir is the row index of M, 
    //jl is the column index of M and 
    //jr is the index of r

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

	  lreord[scf].resize(ni_this);
    	  for(int i = 0; i < ni_this; i++){
    	    i_ind.getBothIndices(ilmap_this[i],irmap_this[i],i,ilp,irp);
	    rowidx_used[ irmap_this[i] ] = true; //this row index of Mr is used
	    
	    const SIMDcomplexType &lval_tmp = l.nativeElem(ilmap_this[i], site4dop, sc, f);
#ifndef MEMTEST_MODE
	    lreord[scf][i] = conj_l ? Grid::conjugate(lval_tmp) : lval_tmp;
#endif
    	  }

    	  //j index
    	  int nj_this = j_ind.getNindices(jlp,jrp);
    	  nj[scf] = nj_this;
	  
    	  std::vector<int> &jlmap_this = jlmap[scf]; jlmap_this.resize(nj_this);
    	  std::vector<int> &jrmap_this = jrmap[scf]; jrmap_this.resize(nj_this);

	  rreord[scf].resize(nj_this);
    	  for(int j = 0; j < nj_this; j++){
    	    j_ind.getBothIndices(jlmap_this[j],jrmap_this[j],j,jlp,jrp);

	    const SIMDcomplexType &rval_tmp = r.nativeElem(jrmap_this[j], site4dop, sc, f);
#ifndef MEMTEST_MODE
	    rreord[scf][j] = conj_r ? Grid::conjugate(rval_tmp) : rval_tmp;
#endif
    	  }

     	}
      }
    }	  


    //Matrix vector multiplication  M*r contracted on mode index j. Only do it for rows that are actually used
    SIMDcomplexType Mr[Mrows][nscf];
    SIMDcomplexType tmp_v;
#ifndef MEMTEST_MODE	  
    for(int i=0;i<Mrows;i++){
      if(!rowidx_used[i]) continue;
      
      for(int scf=0;scf<nscf;scf++){
	int sc = scf/nf; int f = scf % nf;
	int nj_this = nj[scf];
	zeroit(Mr[i][scf]);
	
	for(int j=0;j<nj_this;j++){
	  Grid::vsplat(tmp_v, M(i, jlmap[scf][j]) );
	  //const SIMDcomplexType &relem = r.nativeElem(jrmap[scf][j], site4dop, sc, f);
	  //Mr[i][scf] = Mr[i][scf] + tmp_v * (conj_r ? Grid::conjugate(relem) : relem);
	  Mr[i][scf] = Mr[i][scf] + tmp_v * rreord[scf][j];
	}
      }
    }

    //Vector vector multiplication l*(M*r)
    for(int sl=0;sl<4;sl++){
      for(int sr=0;sr<4;sr++){
	for(int cl=0;cl<3;cl++){
	  int scl = cl + 3*sl;
	  for(int cr=0;cr<3;cr++){
	    for(int fl=0;fl<nf;fl++){
	      int scfl = fl + nf*scl;
	      int ni_this = ni[scfl];
	      for(int fr=0;fr<nf;fr++){
		int scfr = fr + nf*(cr + 3*sr);		

		SIMDcomplexType &into = ImplPolicies::acc(sl,sr,cl,cr,fl,fr,out);

		for(int i=0;i<ni_this;i++){
		  //const SIMDcomplexType lelem = l.nativeElem(ilmap[scfl][i], site4dop, scl, fl);
		  //into = into + (conj_l ? Grid::conjugate(lelem) : lelem) * Mr[irmap[scfl][i]][scfr];
		  
		  into = into + lreord[scfl][i]*Mr[irmap[scfl][i]][scfr];
		}
	      }
	    }
	  }
	}
      }
    }	    
#endif

  }




  static void mult_slow(MatrixType &out, const lA2AfieldView &l,  const MesonFieldView &M, const rA2AfieldView &r, const int xop, const int top, const bool conj_l, const bool conj_r){
    assert(l.getMode(0).SIMDlogicalNodes(3) == 1);
    
    int site4dop = l.getMode(0).threeToFour(xop,top);

    A2Aparams i_params(l), j_params(r);
    StandardIndexDilution idil(i_params), jdil(j_params);

    constexpr int nf = ImplPolicies::nf();    
    int ni = idil.getNmodes();
    int nj = jdil.getNmodes();

    out.zero();

    SIMDcomplexType tmp;
    
    for(int sl=0;sl<4;sl++){
      for(int cl=0;cl<3;cl++){
    	for(int fl=0;fl<nf;fl++){	  

	  for(int sr=0;sr<4;sr++){
	    for(int cr=0;cr<3;cr++){
	      for(int fr=0;fr<nf;fr++){

		for(int i=0;i<ni;i++){

		  const SIMDcomplexType &lval_tmp = l.elem(i,xop,top,cl+3*sl,fl);
		  SIMDcomplexType lval = conj_l ? Grid::conjugate(lval_tmp) : lval_tmp;
		  
  		  for(int j=0;j<nj;j++){
  		    const SIMDcomplexType &rval_tmp = r.elem(j,xop,top,cr+3*sr,fr);
  		    SIMDcomplexType rval = conj_r ? Grid::conjugate(rval_tmp) : rval_tmp;

		    const ScalarComplexType &Mval = M.elem(i,j);
		    Grid::vsplat(tmp, Mval);
		    
		    SIMDcomplexType delta = lval * tmp * rval;
		    ImplPolicies::acc(sl,sr,cl,cr,fl,fr,out) = ImplPolicies::acc(sl,sr,cl,cr,fl,fr,out) + delta;
  		  }
  		}
  	      }
  	    }
  	  }
  	}
      }
    }

  }
};

// l^i(xop,top) M^ij r^j(xop,top)
template<typename lA2AfieldView, typename MesonFieldView, typename rA2AfieldView, 
	 typename std::enable_if<  std::is_same<typename lA2AfieldView::Policies, typename rA2AfieldView::Policies>::value && std::is_same<typename lA2AfieldView::Policies, typename MesonFieldView::Policies>::value, int>::type
	 >
void mult(CPSspinColorFlavorMatrix<typename MesonFieldView::Policies::ComplexType> &out, const lA2AfieldView &l,  const MesonFieldView &M, const rA2AfieldView &r, const int xop, const int top, const bool conj_l, const bool conj_r){
  _mult_vMv_impl_v<_mult_vMv_impl_v_GparityPolicy,lA2AfieldView,MesonFieldView,rA2AfieldView, typename ComplexClassify<typename MesonFieldView::Policies::ComplexType>::type >::mult(out,l,M,r,xop,top,conj_l,conj_r);
}

template<typename lA2AfieldView, typename MesonFieldView, typename rA2AfieldView, 
	 typename std::enable_if<  std::is_same<typename lA2AfieldView::Policies, typename rA2AfieldView::Policies>::value && std::is_same<typename lA2AfieldView::Policies, typename MesonFieldView::Policies>::value, int>::type = 0
	 >
void mult_slow(CPSspinColorFlavorMatrix<typename MesonFieldView::Policies::ComplexType> &out, const lA2AfieldView &l,  const MesonFieldView &M, const rA2AfieldView &r, const int xop, const int top, const bool conj_l, const bool conj_r){
  _mult_vMv_impl_v<_mult_vMv_impl_v_GparityPolicy,lA2AfieldView,MesonFieldView,rA2AfieldView, typename ComplexClassify<typename MesonFieldView::Policies::ComplexType>::type >::mult_slow(out,l,M,r,xop,top,conj_l,conj_r);
}



// l^i(xop,top) M^ij r^j(xop,top)
template<typename lA2AfieldView, typename MesonFieldView, typename rA2AfieldView, 
	 typename std::enable_if<  std::is_same<typename lA2AfieldView::Policies, typename rA2AfieldView::Policies>::value && std::is_same<typename lA2AfieldView::Policies, typename MesonFieldView::Policies>::value, int>::type
	 >
void mult(CPSspinColorMatrix<typename MesonFieldView::Policies::ComplexType> &out, const lA2AfieldView &l,  const MesonFieldView &M, const rA2AfieldView &r, const int xop, const int top, const bool conj_l, const bool conj_r){
  _mult_vMv_impl_v<_mult_vMv_impl_v_StandardPolicy,lA2AfieldView,MesonFieldView,rA2AfieldView, typename ComplexClassify<typename MesonFieldView::Policies::ComplexType>::type >::mult(out,l,M,r,xop,top,conj_l,conj_r);
}
template<typename lA2AfieldView, typename MesonFieldView, typename rA2AfieldView, 
	 typename std::enable_if<  std::is_same<typename lA2AfieldView::Policies, typename rA2AfieldView::Policies>::value && std::is_same<typename lA2AfieldView::Policies, typename MesonFieldView::Policies>::value, int>::type = 0
	 >
void mult_slow(CPSspinColorMatrix<typename MesonFieldView::Policies::ComplexType> &out, const lA2AfieldView &l,  const MesonFieldView &M, const rA2AfieldView &r, const int xop, const int top, const bool conj_l, const bool conj_r){
  _mult_vMv_impl_v<_mult_vMv_impl_v_StandardPolicy,lA2AfieldView,MesonFieldView,rA2AfieldView, typename ComplexClassify<typename MesonFieldView::Policies::ComplexType>::type >::mult_slow(out,l,M,r,xop,top,conj_l,conj_r);
}


#endif



#endif
