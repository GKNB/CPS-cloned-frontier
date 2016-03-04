#ifndef _MULT_VMV_SPLIT_H
#define _MULT_VMV_SPLIT_H

//Try to save memory at the cost of some performance
#define VMV_SPLIT_MEM_SAVE

//For local outer contraction of meson field by two vectors we can save a lot of time by column reordering the meson field to improve cache use. 
//Save even more time by doing this outside the site loop (it makes no reference to the 3d position, only the time at which the vectors
//are evaluated)
template<typename mf_Float, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR
	 >
class mult_vMv_split{
  //Note:
  //il is the index of l, 
  //ir is the row index of M, 
  //jl is the column index of M and 
  //jr is the index of r

  typedef typename lA2AfieldL<mf_Float>::DilutionType iLeftDilutionType;
  typedef typename A2AmesonField<mf_Float,lA2AfieldR,rA2AfieldL>::LeftDilutionType iRightDilutionType;
  
  typedef typename A2AmesonField<mf_Float,lA2AfieldR,rA2AfieldL>::RightDilutionType jLeftDilutionType;    
  typedef typename rA2AfieldR<mf_Float>::DilutionType jRightDilutionType;

  const static int nscf = 2*3*4;

  //Mapping information
  int ni[nscf], nj[nscf]; //mapping f+2*(c+3*s)
  std::vector<int> ilmap[nscf], irmap[nscf];
  std::vector<int> jlmap[nscf], jrmap[nscf];
    
  std::vector< std::vector<std::pair<int,int> > > blocks_scf; //[scf]  contiguous blocks of 'i' indices 

  //Packed matrices
  typedef gsl_wrapper<mf_Float> gw;

#ifdef VMV_SPLIT_MEM_SAVE
  mf_Float *mf_reord_lo_lo; //shared nl*nl submatrix
  mf_Float *mf_reord_lo_hi[nscf]; //the nl * nh[scf] submatrix
  mf_Float *mf_reord_hi_lo[nscf]; //the nh[scf] * nl submatrix
  mf_Float *mf_reord_hi_hi[nscf]; //the nh[scf] * nh[scf] submatrix
#else
  std::vector<typename gw::matrix_complex*> mf_reord; //vector of gsl matrices in packed format where only the rows used are stored. One matrix for each spin/color/flavor combination of the vector r
#endif

  //Info of the row packing
  bool* rowidx_used;
  int nrows_used; //number of packed rows in the output
  std::vector<int> i_packed_unmap_all[nscf];

  const lA2AfieldL<mf_Float> *lptr;
  const A2AmesonField<mf_Float,lA2AfieldR,rA2AfieldL> *Mptr;
  const rA2AfieldR<mf_Float> *rptr;
  int top_glb;

  int Mrows, Mcols;

  bool setup_called;

  void thread_work(int &my_work, int &my_offset, const int total_work, const int me, const int team) const{
    my_work = total_work/team;
    my_offset = me * my_work;

    int rem = total_work - my_work * team;
    if(me < rem){
      ++my_work; //first rem threads mop up the remaining work
      my_offset += me; //each thread before me has gained one extra unit of work
    }else my_offset += rem; //after the first rem threads, the offset shift is uniform
  }

  void site_reorder_lr(std::vector<std::vector<std::complex<mf_Float> > > &lreord,   //[scf][reordered mode]
		       std::vector<std::vector<std::complex<mf_Float> > > &rreord,
		       const bool conj_l, const bool conj_r, const int site4dop) const{    
    lreord.resize(nscf); rreord.resize(nscf);

    for(int sc=0;sc<12;sc++){
      for(int f=0;f<2;f++){
	int scf = f + 2*sc;

	//i index
	int ni_this = ni[scf];
	const std::vector<int> &ilmap_this = ilmap[scf];
	lreord[scf].resize(ni_this);

	for(int i = 0; i < ni_this; i++){
	  const std::complex<mf_Float> &lval_tmp = lptr->nativeElem(ilmap_this[i], site4dop, sc, f);
	  lreord[scf][i] = conj_l ? std::conj(lval_tmp) : lval_tmp;
	}

	//j index
	int nj_this = nj[scf];
	const std::vector<int> &jrmap_this = jrmap[scf]; //jrmap_this.resize(nj_this);

	rreord[scf].resize(nj_this);
	for(int j = 0; j < nj_this; j++){
	  const std::complex<mf_Float> &rval_tmp = rptr->nativeElem(jrmap_this[j], site4dop, sc, f);
	  rreord[scf][j] = conj_r ? std::conj(rval_tmp) : rval_tmp;
	}

      }
    }
  }

  //off is the 3d site offset for the start of the internal site loop, and work is the number of sites to iterate over 
  void multiply_M_r(std::vector<std::vector<std::complex<mf_Float> > >* Mr, const std::vector<std::vector<std::complex<mf_Float> > >* rreord, const int off, const int work) const{
    typename gw::vector_complex* Mr_packed = gw::vector_complex_alloc(nrows_used);
    typename gw::complex one; GSL_SET_COMPLEX(&one,1.0,0.0);
    typename gw::complex zero; GSL_SET_COMPLEX(&zero,0.0,0.0);

#ifdef VMV_SPLIT_MEM_SAVE
    int nl_row = Mptr->getRowParams().getNl();
    int nl_col = Mptr->getColParams().getNl();
    int nj_max = 0;
    for(int scf=0;scf<nscf;scf++) if(nj[scf] > nj_max) nj_max = nj[scf];

    typename gw::matrix_complex* M_packed = gw::matrix_complex_alloc(nrows_used,nj_max);
    pokeSubmatrix<std::complex<mf_Float> >( (std::complex<mf_Float>*)M_packed->data, (const std::complex<mf_Float>*)mf_reord_lo_lo, nrows_used, nj_max, 0, 0, nl_row, nl_col);
#endif

    //M * r
    for(int scf=0;scf<nscf;scf++){
      int nj_this = nj[scf]; //vector size

#ifdef VMV_SPLIT_MEM_SAVE
      int nh_row = nrows_used - nl_row;
      int nh_col = nj_this - nl_col;
      M_packed->size2 = nj_this;

      pokeSubmatrix<std::complex<mf_Float> >( (std::complex<mf_Float>*)M_packed->data, (const std::complex<mf_Float>*)mf_reord_lo_hi[scf], nrows_used, nj_this, 0, nl_col, nl_row, nh_col);
      pokeSubmatrix<std::complex<mf_Float> >( (std::complex<mf_Float>*)M_packed->data, (const std::complex<mf_Float>*)mf_reord_hi_lo[scf], nrows_used, nj_this, nl_row, 0, nh_row, nl_col);
      pokeSubmatrix<std::complex<mf_Float> >( (std::complex<mf_Float>*)M_packed->data, (const std::complex<mf_Float>*)mf_reord_hi_hi[scf], nrows_used, nj_this, nl_row, nl_col, nh_row, nh_col);
#else
      typename gw::matrix_complex* M_packed = mf_reord[scf]; //scope for reuse here
#endif

      const std::vector<int> &i_packed_unmap = i_packed_unmap_all[scf];

      //if(!me) printf("M_packed total rows=%d cols=%d\n",M_packed->size1,M_packed->size2);

      size_t block_width_max =  M_packed->size2;
      size_t block_height_max = 8; //4;

      for(int j0=0; j0<M_packed->size2; j0+=block_width_max){ //columns on outer loop as GSL matrices are row major
	int jblock_size = std::min(M_packed->size2 - j0, block_width_max);
	  
	for(int i0=0; i0<M_packed->size1; i0+=block_height_max){
	  int iblock_size = std::min(M_packed->size1 - i0, block_height_max);
	    
	  //if(!me) printf("i0=%d j0=%d  iblock_size=%d jblock_size=%d total rows=%d cols=%d\n",i0,j0,iblock_size,jblock_size,M_packed->size1,M_packed->size2);
	  typename gw::matrix_complex_const_view submatrix = gw::matrix_complex_const_submatrix(M_packed, i0, j0, iblock_size, jblock_size);
	  
	  for(int s=off;s<off+work;s++){
	    mf_Float* base = (mf_Float*)&rreord[s][scf][j0];
	    typename gw::block_complex_struct block;
	    block.data = base;
	    block.size = jblock_size;
	
	    typename gw::vector_complex rgsl;
	    rgsl.block = &block;
	    rgsl.data = base;
	    rgsl.stride = 1;
	    rgsl.owner = 1;
	    rgsl.size = jblock_size;
	  
	    Mr_packed->size = iblock_size;
	    Mr_packed->block->size = iblock_size;

	    gw::blas_gemv(CblasNoTrans, one, &submatrix.matrix, &rgsl, zero, Mr_packed);

	    typename gw::complex tmp;

	    for(int i_packed=0;i_packed < iblock_size; i_packed++){
	      mf_Float(&tmp)[2] = reinterpret_cast<mf_Float(&)[2]>(Mr[s][scf][ i_packed_unmap[i0+i_packed] ]);
	      mf_Float *t = Mr_packed->data + 2*i_packed*Mr_packed->stride;
	      tmp[0] += *t++; tmp[1] += *t;	       
	    }
	  }	    
	}
      }
    }//end of scf loop
#ifdef VMV_SPLIT_MEM_SAVE
    gw::matrix_complex_free(M_packed);
#endif
    gw::vector_complex_free(Mr_packed);

  }




  //off is the 3d site offset for the start of the internal site loop, and work is the number of sites to iterate over 
  //M_packed is the Mesonfield in packed format.
  void multiply_M_r_singlescf(std::vector<std::vector<std::complex<mf_Float> > >* Mr, const std::vector<std::vector<std::complex<mf_Float> > >* rreord, 
			      typename gw::matrix_complex* M_packed,
			      const int off, const int work, const int scf) const{
    typename gw::vector_complex* Mr_packed = gw::vector_complex_alloc(nrows_used);
    typename gw::complex one; GSL_SET_COMPLEX(&one,1.0,0.0);
    typename gw::complex zero; GSL_SET_COMPLEX(&zero,0.0,0.0);

    //M * r
    int nj_this = nj[scf]; //vector size
    const std::vector<int> &i_packed_unmap = i_packed_unmap_all[scf];
    
    size_t block_width_max =  M_packed->size2;
    size_t block_height_max = 8; //4;
    
    for(int j0=0; j0<M_packed->size2; j0+=block_width_max){ //columns on outer loop as GSL matrices are row major
      int jblock_size = std::min(M_packed->size2 - j0, block_width_max);
      
      for(int i0=0; i0<M_packed->size1; i0+=block_height_max){
	int iblock_size = std::min(M_packed->size1 - i0, block_height_max);
	
	//if(!me) printf("i0=%d j0=%d  iblock_size=%d jblock_size=%d total rows=%d cols=%d\n",i0,j0,iblock_size,jblock_size,M_packed->size1,M_packed->size2);
	typename gw::matrix_complex_const_view submatrix = gw::matrix_complex_const_submatrix(M_packed, i0, j0, iblock_size, jblock_size);
	
	for(int s=off;s<off+work;s++){
	  mf_Float* base = (mf_Float*)&rreord[s][scf][j0];
	  typename gw::block_complex_struct block;
	  block.data = base;
	  block.size = jblock_size;
	  
	  typename gw::vector_complex rgsl;
	  rgsl.block = &block;
	  rgsl.data = base;
	  rgsl.stride = 1;
	  rgsl.owner = 1;
	  rgsl.size = jblock_size;
	  
	  Mr_packed->size = iblock_size;
	  Mr_packed->block->size = iblock_size;
	  
	  gw::blas_gemv(CblasNoTrans, one, &submatrix.matrix, &rgsl, zero, Mr_packed);
	  
	  typename gw::complex tmp;
	  
	  for(int i_packed=0;i_packed < iblock_size; i_packed++){
	    mf_Float(&tmp)[2] = reinterpret_cast<mf_Float(&)[2]>(Mr[s][scf][ i_packed_unmap[i0+i_packed] ]);
	    mf_Float *t = Mr_packed->data + 2*i_packed*Mr_packed->stride;
	    tmp[0] += *t++; tmp[1] += *t;	       
	  }
	}	    
      }
    }

    gw::vector_complex_free(Mr_packed);
  }
  













  void site_multiply_l_Mr(SpinColorFlavorMatrix &out, 
			  const std::vector<std::vector<std::complex<mf_Float> > > &lreord,
			  const std::vector<std::vector<std::complex<mf_Float> > > &Mr,
			  typename gw::vector_complex* Mr_gsl_buffer) const{
    //Vector vector multiplication l*(M*r)
    for(int sl=0;sl<4;sl++){
      for(int cl=0;cl<3;cl++){
	for(int fl=0;fl<2;fl++){
	  int scfl = fl + 2*(cl + 3*sl);
	  int ni_this = ni[scfl];

	  mf_Float* base = (mf_Float*)&lreord[scfl][0];

	  typename gw::block_complex_struct block;
	  block.data = base;
	  block.size = ni_this;
	
	  typename gw::vector_complex lreord_gsl;
	  lreord_gsl.block = &block;
	  lreord_gsl.data = base;
	  lreord_gsl.stride = 1;
	  lreord_gsl.owner = 1;
	  lreord_gsl.size = ni_this;

	  const std::vector<std::pair<int,int> > &blocks = blocks_scf[scfl];

	  for(int sr=0;sr<4;sr++){
	    for(int cr=0;cr<3;cr++){
	      for(int fr=0;fr<2;fr++){
		std::complex<Float> &into = out(sl,cl,fl, sr,cr,fr);

		int scfr = fr + 2*(cr + 3*sr);

		typename gw::vector_complex* Mr_gsl = Mr_gsl_buffer;
		Mr_gsl->size = ni_this;
		Mr_gsl->block->size = ni_this;

		std::complex<mf_Float> const* Mr_base = &Mr[scfr][0];

		for(int b=0;b<blocks.size();b++){
		  std::complex<mf_Float> const* block_ptr = Mr_base + irmap[scfl][blocks[b].first];
		  mf_Float *t = Mr_gsl->data + 2*blocks[b].first*Mr_gsl->stride;
		  memcpy((void*)t, (void*)block_ptr, 2*blocks[b].second*sizeof(mf_Float));
		}
		typename gw::complex dot;
		gw::blas_dotu(&lreord_gsl, Mr_gsl, &dot);
		    
		reinterpret_cast<Float(&)[2]>(into)[0] = GSL_REAL(dot);
		reinterpret_cast<Float(&)[2]>(into)[1] = GSL_IMAG(dot);
	      }
	    }
	  }
	}
      }
    }
  }


public:

  mult_vMv_split(): rowidx_used(NULL), setup_called(false){
#ifdef VMV_SPLIT_MEM_SAVE
    mf_reord_lo_lo = NULL;
    for(int scf=0;scf<nscf;scf++){
      mf_reord_lo_hi[scf] = NULL;
      mf_reord_hi_lo[scf] = NULL;
      mf_reord_hi_hi[scf] = NULL;
    }
#endif
  }

#define FREEIT(A) if(A != NULL){ free(A); A=NULL; }

  void free_mem(){
#ifdef VMV_SPLIT_MEM_SAVE
    FREEIT(mf_reord_lo_lo);

    for(int scf=0;scf<nscf;scf++){
    FREEIT(mf_reord_lo_hi[scf]);
    FREEIT(mf_reord_hi_lo[scf]);
    FREEIT(mf_reord_hi_hi[scf]);
    }
#else
    for(int i=0;i<mf_reord.size();i++) gw::matrix_complex_free(mf_reord[i]);
#endif
    FREEIT(rowidx_used);
  }

  ~mult_vMv_split(){
    free_mem();
  }

  //This should be called outside the site loop (and outside any parallel region)
  //top_glb is the time in global lattice coordinates.
  void setup(const lA2AfieldL<mf_Float> &l,  const A2AmesonField<mf_Float,lA2AfieldR,rA2AfieldL> &M, const rA2AfieldR<mf_Float> &r, const int &_top_glb){
    //Precompute index mappings
    ModeContractionIndices<iLeftDilutionType,iRightDilutionType> i_ind(l);
    ModeContractionIndices<jLeftDilutionType,jRightDilutionType> j_ind(r);
    setup(l,M,r,_top_glb,i_ind,j_ind);
  }

  void setup(const lA2AfieldL<mf_Float> &l,  const A2AmesonField<mf_Float,lA2AfieldR,rA2AfieldL> &M, const rA2AfieldR<mf_Float> &r, const int &_top_glb, 
	     const ModeContractionIndices<iLeftDilutionType,iRightDilutionType> &i_ind, const ModeContractionIndices<jLeftDilutionType,jRightDilutionType>& j_ind){
    lptr = &l; rptr = &r; Mptr = &M; top_glb = _top_glb;

    modeIndexSet ilp, irp, jlp, jrp;
    ilp.time = top_glb;
    irp.time = M.tl;
    
    jlp.time = M.tr;
    jrp.time = top_glb;

    Mrows = M.getNrows();
    Mcols = M.getNcols();

    rowidx_used = (bool*)malloc(Mrows*sizeof(bool)); //Is a particular row of M actually used?
    for(int i=0;i<Mrows;i++) rowidx_used[i] = false;

    const static int nscf = 2*3*4;
#ifndef VMV_SPLIT_MEM_SAVE
    mf_reord.resize(nscf);
#endif

    //Store maps
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
	    rowidx_used[ irmap_this[i] ] = true; //this row index is used
	  }

	  //j index
	  int nj_this = j_ind.getNindices(jlp,jrp);
	  nj[scf] = nj_this;
	  
	  std::vector<int> &jlmap_this = jlmap[scf]; jlmap_this.resize(nj_this);
	  std::vector<int> &jrmap_this = jrmap[scf]; jrmap_this.resize(nj_this);

	  for(int j = 0; j < nj_this; j++)
	    j_ind.getBothIndices(jlmap_this[j],jrmap_this[j],j,jlp,jrp);
	}
      }
    }	  

    //Get contiguous blocks of i indices
    blocks_scf.resize(nscf);
    for(int scfl=0;scfl<nscf;scfl++){
      int ni_this = ni[scfl];
      find_contiguous_blocks(blocks_scf[scfl],&irmap[scfl][0],ni_this);
    }


    //Use GSL BLAS
    
    assert(sizeof(typename gw::complex) == sizeof(std::complex<mf_Float>) ); 

    //Not all rows or columns of M are used, so lets use a packed matrix
    nrows_used = 0;
    for(int i_full=0;i_full<Mrows;i_full++) if(rowidx_used[i_full]) nrows_used++;

    for(int scf=0;scf<nscf;scf++){
      int nj_this = nj[scf];
      std::vector<int> &jlmap_this = jlmap[scf];
      
      typename gw::matrix_complex* mf_scf_reord = M.GSLpackedColReorder(&jlmap_this.front(), nj_this, rowidx_used); //packs the GSL matrix
#ifdef VMV_SPLIT_MEM_SAVE
      int nl_row = M.getRowParams().getNl();
      int nl_col = M.getColParams().getNl();
      int nh_row = nrows_used - nl_row;
      int nh_col = nj_this - nl_col;
      if(scf == 0){
	mf_reord_lo_lo = (mf_Float*)malloc(2*nl_row*nl_col*sizeof(mf_Float));
	getSubmatrix<std::complex<mf_Float> >( (std::complex<mf_Float>*)mf_reord_lo_lo, (const std::complex<mf_Float>*)mf_scf_reord->data, nrows_used, nj_this, 0, 0, nl_row, nl_col);
      }
      mf_reord_lo_hi[scf] = (mf_Float*)malloc(2*nl_row*nh_col*sizeof(mf_Float));
      getSubmatrix<std::complex<mf_Float> >( (std::complex<mf_Float>*)mf_reord_lo_hi[scf], (const std::complex<mf_Float>*)mf_scf_reord->data, nrows_used, nj_this, 0, nl_col, nl_row, nh_col);

      mf_reord_hi_lo[scf] = (mf_Float*)malloc(2*nh_row*nl_col*sizeof(mf_Float));
      getSubmatrix<std::complex<mf_Float> >( (std::complex<mf_Float>*)mf_reord_hi_lo[scf], (const std::complex<mf_Float>*)mf_scf_reord->data, nrows_used, nj_this, nl_row, 0, nh_row, nl_col);

      mf_reord_hi_hi[scf] = (mf_Float*)malloc(2*nh_row*nh_col*sizeof(mf_Float));
      getSubmatrix<std::complex<mf_Float> >( (std::complex<mf_Float>*)mf_reord_hi_hi[scf], (const std::complex<mf_Float>*)mf_scf_reord->data, nrows_used, nj_this, nl_row, nl_col, nh_row, nh_col);

      gw::matrix_complex_free(mf_scf_reord);
#else
      mf_reord[scf] = mf_scf_reord;
#endif
             
      //Store the map between packed and full indices
      int i_packed = 0;
      i_packed_unmap_all[scf].resize(nrows_used);
      for(int i_full=0;i_full<Mrows;i_full++) 
	if(rowidx_used[i_full]) i_packed_unmap_all[scf][i_packed++] = i_full;
    }

    setup_called = true;
  }

public:
  //Contract on all 3d sites on this node with fixed operator time coord top_glb into a canonically ordered output vector
  void contract(std::vector<SpinColorFlavorMatrix> &out, const bool conj_l, const bool conj_r) const{
    int top = top_glb - GJP.TnodeSites()*GJP.TnodeCoor();
    assert(top >= 0 && top < GJP.TnodeSites()); //make sure you use this method on the appropriate node!

    int sites_3d = GJP.VolNodeSites()/GJP.TnodeSites();

    out.resize(sites_3d);

    std::vector< std::vector<std::vector<std::complex<mf_Float> > > > lreord(sites_3d); //[3d site][scf][reordered mode]
    std::vector< std::vector<std::vector<std::complex<mf_Float> > > > rreord(sites_3d);

    assert(sizeof(typename gw::complex) == sizeof(std::complex<mf_Float>) ); 
    typedef gsl_wrapper<mf_Float> gw;
    
    std::vector<  std::vector<std::vector<std::complex<mf_Float> > > > Mr(sites_3d); //[3d site][scf][M row]

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
	site_reorder_lr(lreord[s],rreord[s],conj_l,conj_r,site4dop);
      }
      
      for(int s=off[me];s<off[me]+work[me];s++){
	Mr[s].resize(nscf); 
	for(int scf=0;scf<nscf;scf++){
	  Mr[s][scf].resize(Mrows);
	  for(int i=0;i<Mrows;i++)
	    Mr[s][scf][i] = 0.0;
	}
      }
    }
    
    //If memory saving, generate the packed matrix for each scf outside of threaded region
#ifdef VMV_SPLIT_MEM_SAVE
    int nl_row = Mptr->getRowParams().getNl();
    int nl_col = Mptr->getColParams().getNl();
    int nj_max = 0;
    for(int scf=0;scf<nscf;scf++) if(nj[scf] > nj_max) nj_max = nj[scf];

    typename gw::matrix_complex* M_packed = gw::matrix_complex_alloc(nrows_used,nj_max);
    pokeSubmatrix<std::complex<mf_Float> >( (std::complex<mf_Float>*)M_packed->data, (const std::complex<mf_Float>*)mf_reord_lo_lo, nrows_used, nj_max, 0, 0, nl_row, nl_col,true);
#endif

    for(int scf=0; scf<nscf; scf++){
#ifdef VMV_SPLIT_MEM_SAVE
      int nj_this = nj[scf];
      int nh_row = nrows_used - nl_row;
      int nh_col = nj_this - nl_col;
      M_packed->size2 = nj_this;
      
      pokeSubmatrix<std::complex<mf_Float> >( (std::complex<mf_Float>*)M_packed->data, (const std::complex<mf_Float>*)mf_reord_lo_hi[scf], nrows_used, nj_this, 0, nl_col, nl_row, nh_col, true);
      pokeSubmatrix<std::complex<mf_Float> >( (std::complex<mf_Float>*)M_packed->data, (const std::complex<mf_Float>*)mf_reord_hi_lo[scf], nrows_used, nj_this, nl_row, 0, nh_row, nl_col,true);
      pokeSubmatrix<std::complex<mf_Float> >( (std::complex<mf_Float>*)M_packed->data, (const std::complex<mf_Float>*)mf_reord_hi_hi[scf], nrows_used, nj_this, nl_row, nl_col, nh_row, nh_col, true);
#else
      typename gw::matrix_complex* M_packed = mf_reord[scf]; //scope for reuse here
#endif

#pragma omp parallel
      {
	int me = omp_get_thread_num();
	multiply_M_r_singlescf(&Mr[0],&rreord[0],M_packed,off[me], work[me],scf);
      }
    }

#ifdef VMV_SPLIT_MEM_SAVE
    gw::matrix_complex_free(M_packed);
#endif


    #pragma omp parallel
    {
      int me = omp_get_thread_num();
      //M * r
      //multiply_M_r(&Mr[0],&rreord[0],off,work);
      
      //Vector vector multiplication l*(M*r)
      typename gw::vector_complex* Mr_gsl_buffer = gw::vector_complex_alloc(Mrows);
      for(int x3d=off[me];x3d<off[me]+work[me];x3d++)
	site_multiply_l_Mr(out[x3d], lreord[x3d], Mr[x3d], Mr_gsl_buffer);

      gw::vector_complex_free(Mr_gsl_buffer);
      
    } //end of parallel region



  }//end of method


  //Run inside a threaded/parallelized loop over 3d sites. xop is a 3d coordinate!
  void contract(SpinColorFlavorMatrix &out, const int &xop, const bool &conj_l, const bool &conj_r) const{
    int top = top_glb - GJP.TnodeSites()*GJP.TnodeCoor();
    assert(top >= 0 && top < GJP.TnodeSites()); //make sure you use this method on the appropriate node!

    std::vector<std::vector<std::complex<mf_Float> > > lreord; //[scf][reordered mode]
    std::vector<std::vector<std::complex<mf_Float> > > rreord;

    assert(sizeof(typename gw::complex) == sizeof(std::complex<mf_Float>) ); 
    typedef gsl_wrapper<mf_Float> gw;
    
    std::vector<std::vector<std::complex<mf_Float> > > Mr(nscf); //[scf][M row]
    for(int scf=0;scf<nscf;scf++){
      Mr[scf].resize(Mrows);
      for(int i=0;i<Mrows;i++)
	Mr[scf][i] = 0.0;
    }
    int sites_3d = GJP.VolNodeSites()/GJP.TnodeSites();
    int site4dop = xop + sites_3d*top;
    site_reorder_lr(lreord,rreord,conj_l,conj_r,site4dop);

    //M * r
    multiply_M_r(&Mr,&rreord,0,1);

    //Vector vector multiplication l*(M*r)
    typename gw::vector_complex* Mr_gsl_buffer = gw::vector_complex_alloc(Mrows);
    site_multiply_l_Mr(out, lreord, Mr, Mr_gsl_buffer);

    gw::vector_complex_free(Mr_gsl_buffer);
  }




};






#endif
