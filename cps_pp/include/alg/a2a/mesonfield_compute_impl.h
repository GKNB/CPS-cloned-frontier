//Meson field computation code

#ifndef _MESONFIELD_COMPUTE_IMPL
#define _MESONFIELD_COMPUTE_IMPL

//For all mode indices l_i and r_j, compute the meson field  \sum_p l_i^\dagger(p,t) M(p,t) r_j(p,t)
//It is assumed that A2AfieldL and A2AfieldR are Fourier transformed field containers
//M(p,t) is a completely general momentum-space spin/color/flavor matrix per temporal slice

template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
template<typename InnerProduct>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::compute(const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r, const int &t, bool do_setup){
  if(do_setup) setup(l,r,t,t); //both vectors have same timeslice
  else zero();
  
  if(!UniqueID()) printf("Starting A2AmesonField::compute timeslice %d with %d threads\n",t, omp_get_max_threads());

  double time = -dclock();

  //For W vectors we dilute out the flavor index in-place while performing this contraction
  const typename mf_Policies::FermionFieldType &mode0 = l.getMode(0);
  const int size_3d = mode0.nodeSites(0)*mode0.nodeSites(1)*mode0.nodeSites(2);
  if(mode0.nodeSites(3) != GJP.TnodeSites()) ERR.General("A2AmesonField","compute","Not implemented for fields where node time dimension != GJP.TnodeSites()\n");

  int nl_l = lindexdilution.getNl();
  int nl_r = rindexdilution.getNl();

  int t_lcl = t-GJP.TnodeCoor()*GJP.TnodeSites();
  if(t_lcl >= 0 && t_lcl < GJP.TnodeSites()){ //if timeslice is on-node

#pragma omp parallel for
    for(int i = 0; i < nmodes_l; i++){
      cps::ComplexD mf_accum;

      modeIndexSet i_high_unmapped; if(i>=nl_l) lindexdilution.indexUnmap(i-nl_l,i_high_unmapped);

      for(int j = 0; j < nmodes_r; j++) {
	modeIndexSet j_high_unmapped; if(j>=nl_r) rindexdilution.indexUnmap(j-nl_r,j_high_unmapped);

	mf_accum = 0.;

	for(int p_3d = 0; p_3d < size_3d; p_3d++) {
	  SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> lscf = l.getFlavorDilutedVect(i,i_high_unmapped,p_3d,t_lcl); //dilute flavor in-place if it hasn't been already
	  SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> rscf = r.getFlavorDilutedVect(j,j_high_unmapped,p_3d,t_lcl);

	  mf_accum += M(lscf,rscf,p_3d,t); //produces double precision output by spec
	}
	(*this)(i,j) = mf_accum; //downcast after accumulate      
      }
    }
  }
  sync();
  print_time("A2AmesonField","local compute",time + dclock());
  time = -dclock();

  //Sum over all nodes so all nodes have a copy
  nodeSum();
  print_time("A2AmesonField","nodeSum",time + dclock());
}




//Compute meson fields for all timeslices. This version is more efficient on multi-nodes

template<typename mf_Element, typename InnerProduct, typename FieldSiteType>
struct mf_Element_policy{};

template<typename InnerProduct, typename FieldSiteType>
struct mf_Element_policy< cps::ComplexD, InnerProduct, FieldSiteType>{
  static inline void setZero(cps::ComplexD &mf_accum){
    mf_accum = 0.;
  }  
  static inline void accumulate(cps::ComplexD &mf_accum, const InnerProduct &M, const SCFvectorPtr<FieldSiteType> &lscf, const SCFvectorPtr<FieldSiteType> &rscf, const int p_3d, const int t){
    mf_accum += M(lscf,rscf,p_3d,t); //produces double precision output by spec
  }
};
//For multi-src
template<typename InnerProduct, typename FieldSiteType>
struct mf_Element_policy< std::vector<cps::ComplexD>, InnerProduct, FieldSiteType>{
  static inline void setZero(std::vector<cps::ComplexD> &mf_accum){
    for(int i=0;i<mf_accum.size();i++)
      mf_accum[i] = 0.;
  }  
  static inline void accumulate(std::vector<cps::ComplexD> &mf_accum, const InnerProduct &M, const SCFvectorPtr<FieldSiteType> &lscf, const SCFvectorPtr<FieldSiteType> &rscf, const int p_3d, const int t){
    M(mf_accum,lscf,rscf,p_3d,t);
  }
};



template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename mf_Element>
class MultKernel{
public:
  //Lowest level of blocked matrix mult. Ideally this should fit in L1 cache.
  template<typename InnerProduct>
  static void mult_kernel(std::vector<std::vector<mf_Element> > &mf_accum_m, const InnerProduct &M, const int t,
			  const int i0, const int iup, const int j0, const int jup, const int p0, const int pup,
			  const std::vector<SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> > &base_ptrs_i,
			  const std::vector<SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> > &base_ptrs_j,
			  const std::vector<std::pair<int,int> > &site_offsets_i,
			  const std::vector<std::pair<int,int> > &site_offsets_j){
    typedef mf_Element_policy<mf_Element,InnerProduct,typename mf_Policies::FermionFieldType::FieldSiteType> mfElementPolicy;

    for(int i = i0; i < iup; i++){	      
      for(int j = j0; j < jup; j++) {		
	
	mf_Element &mf_accum = mf_accum_m[i][j];
	mfElementPolicy::setZero(mf_accum);
	
	SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> lscf = base_ptrs_i[i];
	SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> rscf = base_ptrs_j[j];
	lscf.incrementPointers(site_offsets_i[i],p0);
	rscf.incrementPointers(site_offsets_j[j],p0);
	
	for(int p_3d = p0; p_3d < pup; p_3d++) {
	  mfElementPolicy::accumulate(mf_accum, M, lscf, rscf, p_3d, t);
	  lscf.incrementPointers(site_offsets_i[i]);
	  rscf.incrementPointers(site_offsets_j[j]);		  
	}
      }
    }
  }
  //Do a second layer of blocked dgemm to try to fit in the L1 cache
  //note the i0, iup, etc are the low and high range limits from the outer blocking
  template<typename InnerProduct>
  static void inner_block_mult(std::vector<std::vector<mf_Element> > &mf_accum_m, const InnerProduct &M, const int t,
			       const int i0, const int iup, const int j0, const int jup, const int p0, const int pup,
			       const std::vector<SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> > &base_ptrs_i,
			       const std::vector<SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> > &base_ptrs_j,
			       const std::vector<std::pair<int,int> > &site_offsets_i,
			       const std::vector<std::pair<int,int> > &site_offsets_j){
    //int ni = iup - i0, nj = jup - j0, np = pup - p0;
    int bii = 8, bjj = 2, bpp = 4; //inner block sizes

    for(int ii0=i0; ii0 < iup; ii0+=bii){
      int iiup = std::min(ii0+bii,iup);
      for(int jj0=j0; jj0 < jup; jj0+=bjj){
	int jjup = std::min(jj0+bjj,jup);
	for(int pp0=p0; pp0 < pup; pp0+=bpp){
	  int ppup = std::min(pp0+bpp,pup);

	  MultKernel<mf_Policies,A2AfieldL,A2AfieldR,mf_Element>::mult_kernel(mf_accum_m, M, t,
									      ii0, iiup, jj0, jjup, pp0, ppup,
									      base_ptrs_i, base_ptrs_j, site_offsets_i, site_offsets_j);
	}
      }
    }
  }
};

//Policies for single and multi-src outputs
template<typename mfVectorType, typename InnerProduct>
struct mf_Vector_policies{};

//Single src
template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename Allocator, typename InnerProduct>
struct mf_Vector_policies< std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator >, InnerProduct >{
  typedef std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator > mfVectorType;
  typedef cps::ComplexD mf_Element;
  static inline void initializeElement(mf_Element &e){}
  static void initializeMesonFields(mfVectorType &mf_t, const A2AfieldL<mf_Policies> &l, const A2AfieldR<mf_Policies> &r, const int Lt, const bool do_setup){
    mf_t.resize(Lt);
    for(int t=0;t<Lt;t++) 
      if(do_setup) mf_t[t].setup(l,r,t,t); //both vectors have same timeslice (zeroes the starting matrix)
      else mf_t[t].zero();
  }
  static inline void sumThreadedResults(mfVectorType &mf_t, const std::vector<std::vector<std::vector<mf_Element> > > &mf_accum_thr, const int i, const int j, const int t, const int nthread){
    for(int thr=0;thr<nthread;thr++)
	mf_t[t](i,j) += mf_accum_thr[thr][i][j];
  }
  //Used to get information about rows and cols
  static inline const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> & getReferenceMf(const mfVectorType &mf_t, const int t){
    return mf_t[t];
  }
  static inline nodeSum(const mfVectorType &mf_t, const int Lt){
    for(int t=0; t<Lt; t++) mf_t[t].nodeSum();
  }
};

//Multisrc
template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename Allocator, typename InnerProduct>
struct mf_Vector_policies< std::vector< std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator >* >, InnerProduct >{
  enum { nSources = InnerProduct::InnerProductSourceType::nSources };
  typedef std::vector< std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator >* > mfVectorType;  //indexed by [srcidx][t]
  typedef std::vector<cps::ComplexD> mf_Element;
  
  static inline void initializeElement(mf_Element &e){ e.resize(nSources);  }
  static void initializeMesonFields(mfVectorType &mf_st, const A2AfieldL<mf_Policies> &l, const A2AfieldR<mf_Policies> &r, const int Lt, const bool do_setup){
    assert(int(mf_st.size()) == nSources);
    for(int s=0;s<nSources;s++){
      mf_st[s]->resize(Lt);
      for(int t=0;t<Lt;t++) 
	if(do_setup) mf_st[s]->operator[](t).setup(l,r,t,t); //both vectors have same timeslice (zeroes the starting matrix)
	else mf_st[s]->operator[](t).zero();
    }
  }
  static inline void sumThreadedResults(mfVectorType &mf_st, const std::vector<std::vector<std::vector<mf_Element> > > &mf_accum_thr, const int i, const int j, const int t, const int nthread){
    for(int thr=0;thr<nthread;thr++)
      for(int s=0;s<nSources;s++)
	mf_st[s]->operator[](t)(i,j) += mf_accum_thr[thr][i][j][s];
  }

  //Used to get information about rows and cols
  static inline const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> & getReferenceMf(const mfVectorType &mf_st, const int t){
    return mf_st[0]->operator[](t);
  }
  static inline nodeSum(const mfVectorType &mf_st, const int Lt){
    for(int s=0;s<nSources;s++)
      for(int t=0; t<Lt; t++) mf_st[s]->operator[](t).nodeSum();
  }
};



template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename InnerProduct, typename mfVectorType>
struct mfComputeGeneral{
  static void compute(mfVectorType &mf_t, const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r, bool do_setup){
    typedef mf_Vector_policies<mfVectorType, InnerProduct > mfVectorPolicies;
    typedef typename mfVectorPolicies::mf_Element mf_Element;
    
    const int Lt = GJP.Tnodes()*GJP.TnodeSites();
    if(!UniqueID()) printf("Starting A2AmesonField::compute (blocked) for %d timeslices with %d threads\n",Lt, omp_get_max_threads());
#ifdef KNL_OPTIMIZATIONS
    if(!UniqueID()) printf("Using KNL optimizations\n");
#else
    if(!UniqueID()) printf("NOT using KNL optimizations\n");
#endif
    double time = -dclock();
    mfVectorPolicies::initializeMesonFields(mf_t,l,r,Lt,do_setup);
    print_time("A2AmesonField","setup",time + dclock());

    time = -dclock();
    //For W vectors we dilute out the flavor index in-place while performing this contraction
    const typename mf_Policies::FermionFieldType &mode0 = l.getMode(0);
    const int size_3d = mode0.nodeSites(0)*mode0.nodeSites(1)*mode0.nodeSites(2);
    if(mode0.nodeSites(3) != GJP.TnodeSites()) ERR.General("A2AmesonField","compute","Not implemented for fields where node time dimension != GJP.TnodeSites()\n");
  
    //Each node only works on its time block
    for(int t=GJP.TnodeCoor()*GJP.TnodeSites(); t<(GJP.TnodeCoor()+1)*GJP.TnodeSites(); t++){
      const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> & mf_ref = mfVectorPolicies::getReferenceMf(mf_t,t); //assumes all meson fields of the mf_Element type have the same mode parameters
      
      double ttime = -dclock();

      const int nl_l = mf_ref.getRowParams().getNl();
      const int nl_r = mf_ref.getColParams().getNl();
      const int nmodes_l = mf_ref.getNrows();
      const int nmodes_r = mf_ref.getNcols();
      
      int t_lcl = t-GJP.TnodeCoor()*GJP.TnodeSites();

      int bi = BlockedMesonFieldArgs::bi;
      int bj = BlockedMesonFieldArgs::bj;
      int bp = BlockedMesonFieldArgs::bp;

      int nthread = omp_get_max_threads();
      std::vector<std::vector<std::vector<mf_Element> > > mf_accum_thr(nthread); //indexed by [thread][i][j]
      for(int thr=0;thr<nthread;thr++){
	mf_accum_thr[thr].resize(nmodes_l);
	for(int i=0;i<nmodes_l;i++){
	  mf_accum_thr[thr][i].resize(nmodes_r);
	  for(int j=0;j<nmodes_r;j++)
	    mfVectorPolicies::initializeElement(mf_accum_thr[thr][i][j]);
	}
      }

      //Make a table of p base pointers and site offsets for each i,j
      std::vector<SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> > base_ptrs_i(nmodes_l);
      std::vector<SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> > base_ptrs_j(nmodes_r);
      std::vector<std::pair<int,int> > site_offsets_i(nmodes_l);
      std::vector<std::pair<int,int> > site_offsets_j(nmodes_r);

#pragma omp parallel
      {
	int me = omp_get_thread_num();

	//Generate the tables
	int thr_tabwork, thr_taboff;
	thread_work(thr_tabwork, thr_taboff, nmodes_l, me, omp_get_num_threads());
	for(int i=thr_taboff; i<thr_taboff+thr_tabwork;i++){ //i table
	  modeIndexSet i_high_unmapped; if(i>=nl_l) mf_ref.getRowParams().indexUnmap(i-nl_l,i_high_unmapped);
	  base_ptrs_i[i] = l.getFlavorDilutedVect(i,i_high_unmapped,0,t_lcl);
	  site_offsets_i[i] = std::pair<int,int>( l.siteStride3D(i,i_high_unmapped,0), l.siteStride3D(i,i_high_unmapped,1) );
	}
	thread_work(thr_tabwork, thr_taboff, nmodes_r, me, omp_get_num_threads());
	for(int j=thr_taboff; j<thr_taboff+thr_tabwork;j++){ //j table
	  modeIndexSet j_high_unmapped; if(j>=nl_r) mf_ref.getColParams().indexUnmap(j-nl_r,j_high_unmapped);
	  base_ptrs_j[j] = r.getFlavorDilutedVect(j,j_high_unmapped,0,t_lcl);
	  site_offsets_j[j] = std::pair<int,int>( r.siteStride3D(j,j_high_unmapped,0), r.siteStride3D(j,j_high_unmapped,1) );
	}
#pragma omp barrier

	for(int i0 = 0; i0 < nmodes_l; i0+=bi){
	  int iup = std::min(i0+bi,nmodes_l);
	    
	  for(int j0 = 0; j0< nmodes_r; j0+=bj) {
	    int jup = std::min(j0+bj,nmodes_r);
	  
	    for(int p0 = 0; p0 < size_3d; p0+=bp){
	      int pup = std::min(p0+bp,size_3d);
      
	      int thr_pwork, thr_poff;
	      thread_work(thr_pwork, thr_poff, pup-p0, me, omp_get_num_threads());

	      int thr_p0 = p0 + thr_poff;
#ifdef USE_INNER_BLOCKING
	      MultKernel<mf_Policies,A2AfieldL,A2AfieldR,mf_Element>::inner_block_mult(mf_accum_thr[me], M, t,
											  i0, iup, j0, jup, thr_p0, thr_p0+thr_pwork,
											  base_ptrs_i, base_ptrs_j, site_offsets_i, site_offsets_j);
#else
	      MultKernel<mf_Policies,A2AfieldL,A2AfieldR,mf_Element>::mult_kernel(mf_accum_thr[me], M, t,
										     i0, iup, j0, jup, thr_p0, thr_p0+thr_pwork,
										     base_ptrs_i, base_ptrs_j, site_offsets_i, site_offsets_j);
#endif

	    }
	  
	  }
	}
#pragma omp barrier

	const int nthread = omp_get_num_threads();
	const int ijwork = nmodes_l * nmodes_r;
	int thr_ijwork, thr_ijoff;
	thread_work(thr_ijwork, thr_ijoff, ijwork, me, nthread);
	for(int ij=thr_ijoff; ij<thr_ijoff + thr_ijwork; ij++){  //ij = j + mf_t[t].nmodes_r * i
	  int i=ij / nmodes_r;
	  int j=ij % nmodes_r;
	  mfVectorPolicies::sumThreadedResults(mf_t,mf_accum_thr,i,j,t,nthread);
	}		
      
      }//end of parallel region

      std::ostringstream os; os << "timeslice " << t << " from range " << GJP.TnodeCoor()*GJP.TnodeSites() << " to " << (GJP.TnodeCoor()+1)*GJP.TnodeSites()-1 << " : " << nmodes_l << "*" <<  nmodes_r << " modes and inner p loop of size " <<  size_3d <<  " divided over " << omp_get_max_threads() << " threads";
      print_time("A2AmesonField",os.str().c_str(),ttime + dclock());
    }

    print_time("A2AmesonField","local compute",time + dclock());

    time = -dclock();
    sync();
    print_time("A2AmesonField","sync",time + dclock());

    //Accumulate
    time = -dclock();
    mfVectorPolicies::nodeSum(mf_t,Lt);
    print_time("A2AmesonField","nodeSum",time + dclock());
  }
};






template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
template<typename InnerProduct, typename Allocator>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::compute(std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator > &mf_t,
							     const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r, bool do_setup){
  mfComputeGeneral<mf_Policies,A2AfieldL,A2AfieldR,InnerProduct,  std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator > >::compute(mf_t,l,M,r,do_setup);
}

  //Version of the above for multi-src inner products (output vector indexed by [src idx][t]
template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
template<typename InnerProduct, typename Allocator>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::compute(std::vector< std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator >* > &mf_st,
							     const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r, bool do_setup){
  mfComputeGeneral<mf_Policies,A2AfieldL,A2AfieldR,InnerProduct,  std::vector< std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator >* > >::compute(mf_st,l,M,r,do_setup);
}


#endif
