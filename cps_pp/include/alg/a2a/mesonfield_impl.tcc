#ifndef MESON_FIELD_IMPL
#define MESON_FIELD_IMPL

template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::plus_equals(const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &with, const bool parallel){
  if(nmodes_l != with.nmodes_l || nmodes_r != with.nmodes_r || 
     !lindexdilution.paramsEqual(with.lindexdilution) || !rindexdilution.paramsEqual(with.rindexdilution) ){
    ERR.General("A2AmesonField","plus_equals(..)","Second meson field must have the same underlying parameters\n");
  }
  if(parallel){
#pragma omp_parallel for
    for(int i=0;i<fsize;i++) this->ptr()[i] += with.ptr()[i];
  }else{
    for(int i=0;i<fsize;i++) this->ptr()[i] += with.ptr()[i];
  }
}

template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::times_equals(const ScalarComplexType f,const bool parallel){
  if(parallel){
#pragma omp_parallel for
    for(int i=0;i<fsize;i++) this->ptr()[i] *= f;			       
  }else{
    for(int i=0;i<fsize;i++) this->ptr()[i] *= f;
  }
}

template<typename T>
inline std::complex<T> complexAvg(const std::complex<T>&a, const std::complex<T> &b){
  return (a+b)/T(2.0);
}

#if defined(USE_GRID) && defined(GRID_NVCC)
template<typename T>
inline Grid::complex<T> complexAvg(const Grid::complex<T>&a, const Grid::complex<T> &b){
  return (a+b)/T(2.0);
}
#endif


//Replace this meson field with the average of this and a second field, 'with'
template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::average(const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &with, const bool parallel){
  if(nmodes_l != with.nmodes_l || nmodes_r != with.nmodes_r || 
     !lindexdilution.paramsEqual(with.lindexdilution) || !rindexdilution.paramsEqual(with.rindexdilution) ){
    ERR.General("A2AmesonField","average(..)","Second meson field must have the same underlying parameters\n");
  }
  if(parallel){
#pragma omp_parallel for
    for(int i=0;i<fsize;i++) this->ptr()[i] = complexAvg(this->ptr()[i],with.ptr()[i]);//(mf[i] + with.mf[i])/2.0;
  }else{
    for(int i=0;i<fsize;i++) this->ptr()[i] = complexAvg(this->ptr()[i],with.ptr()[i]);//(mf[i] + with.mf[i])/2.0;
  }
}

//Reorder the rows so that all the elements in idx_map are sequential. Indices not in map are ignored. Use at your own risk
template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::rowReorder(A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &into, const int idx_map[], int map_size, bool parallel) const{
  into.setup(lindexdilution, rindexdilution, tl, tr);

#define DOIT \
    int irow = idx_map[i]; \
    for(int j=0;j<nmodes_r;j++) \
      into(i,j) = (*this)(irow,j);

  if(parallel){
#pragma omp parallel for
    for(int i=0;i<map_size;i++){
      DOIT;
    }
  }else{
    for(int i=0;i<map_size;i++){
      DOIT;
    }
  }
#undef DOIT

}
template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::colReorder(A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &into, const int idx_map[], int map_size, bool parallel) const{
  into.setup(lindexdilution, rindexdilution, tl, tr);

#define DOIT \
    for(int j=0;j<map_size;j++){ \
      int jcol = idx_map[j]; \
      into(i,j) = (*this)(i,jcol); \
    }

  if(parallel){
#pragma omp parallel for
    for(int i=0;i<nmodes_l;i++){
      DOIT;
    }
  }else{
    for(int i=0;i<nmodes_l;i++){
      DOIT;
    } 
  }
}



//Do a column reorder but where we pack the row indices to exclude those not used (as indicated by input bool array)
//Output as a GSL matrix
template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
typename gsl_wrapper<typename mf_Policies::ScalarComplexType::value_type>::matrix_complex * A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::GSLpackedColReorder(const int idx_map[], int map_size, bool rowidx_used[], typename gsl_wrapper<typename ScalarComplexType::value_type>::matrix_complex *reuse ) const{
  typedef gsl_wrapper<typename ScalarComplexType::value_type> gw;
  assert(sizeof(typename gw::complex) == sizeof(ScalarComplexType));
  int rows = nmodes_l;
  int cols = nmodes_r;

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
      ScalarComplexType const* mf_row_base = this->ptr() + nmodes_r*i_full; //meson field are row major so columns are contiguous
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

#ifdef USE_GRID
  //Do a column reorder but where we pack the row indices to exclude those not used (as indicated by input bool array)
  //Output to a linearized matrix of Grid SIMD vectors where we have splatted the scalar onto all SIMD lanes
  //Does not set the size of the output vector, allowing reuse of a previously allocated vector providing it's large enough
template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::splatPackedColReorder(typename AlignedVector<typename mf_Policies::ComplexType>::type &into, const int idx_map[], int map_size, bool rowidx_used[], bool do_resize) const{
  typedef typename mf_Policies::ComplexType SIMDcomplexType;
  int full_rows = nmodes_l;
  int full_cols = nmodes_r;

  int nrows_used = 0;
  for(int i_full=0;i_full<full_rows;i_full++) if(rowidx_used[i_full]) nrows_used++;

  if(do_resize) into.resize(nrows_used*map_size);

  //Look for contiguous blocks in the idx_map we can take advantage of
  std::vector<std::pair<int,int> > blocks;
  find_contiguous_blocks(blocks,idx_map,map_size);

  int i_packed = 0;

  for(int i_full=0;i_full<full_rows;i_full++){
    if(rowidx_used[i_full]){
      ScalarComplexType const* mf_row_base = this->ptr() + nmodes_r*i_full;
      SIMDcomplexType* row_base = &into[map_size*i_packed];

      for(int b=0;b<blocks.size();b++){
	ScalarComplexType const* block_ptr = mf_row_base + idx_map[blocks[b].first];
	for(int bb=0;bb<blocks[b].second;bb++)
	  vsplat(row_base[bb],block_ptr[bb]);
	row_base += blocks[b].second;
      }
      i_packed++;
    }
  }
}
template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::scalarPackedColReorder(typename AlignedVector<typename mf_Policies::ComplexType>::type &into, const int idx_map[], int map_size, bool rowidx_used[], bool do_resize) const{
  int full_rows = nmodes_l;
  int full_cols = nmodes_r;

  int nrows_used = 0;
  for(int i_full=0;i_full<full_rows;i_full++) if(rowidx_used[i_full]) nrows_used++;

  if(do_resize) into.resize(nrows_used*map_size);

  //Look for contiguous blocks in the idx_map we can take advantage of
  std::vector<std::pair<int,int> > blocks;
  find_contiguous_blocks(blocks,idx_map,map_size);

  int i_packed = 0;

  for(int i_full=0;i_full<full_rows;i_full++){
    if(rowidx_used[i_full]){
      ScalarComplexType const* mf_row_base = this->ptr() + nmodes_r*i_full;
      ScalarComplexType* row_base = &into[map_size*i_packed];

      for(int b=0;b<blocks.size();b++){
	ScalarComplexType const* block_ptr = mf_row_base + idx_map[blocks[b].first];
	for(int bb=0;bb<blocks[b].second;bb++)
	  row_base[bb] = block_ptr[bb];
	row_base += blocks[b].second;
      }
      i_packed++;
    }
  }
}
#endif




template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::transpose(A2AmesonField<mf_Policies,A2AfieldR,A2AfieldL> &into) const{
  assert( (void*)this != (void*)&into );
  into.setup(rindexdilution, lindexdilution, tr, tl);
#pragma omp parallel for
  for(int i=0;i<nmodes_l;i++)
    for(int j=0;j<nmodes_r;j++)
      into(j,i) = (*this)(i,j);
}

//Compute   l^ij(t1,t2) r^ji(t3,t4)
//Threaded but *node local*
template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR
	 >
typename mf_Policies::ScalarComplexType trace(const A2AmesonField<mf_Policies,lA2AfieldL,lA2AfieldR> &l, const A2AmesonField<mf_Policies,rA2AfieldL,rA2AfieldR> &r){
  //Check the indices match
  if(! l.getRowParams().paramsEqual( r.getColParams() ) || ! l.getColParams().paramsEqual( r.getRowParams() ) )
    ERR.General("","trace(const A2AmesonField<mf_Policies,lA2AfieldL,lA2AfieldR> &, const A2AmesonField<mf_Policies,rA2AfieldL,rA2AfieldR> &)","Illegal matrix product: underlying vector parameters must match\n");
  typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
  ScalarComplexType into(0,0);

  typedef typename A2AmesonField<mf_Policies,lA2AfieldL,lA2AfieldR>::LeftDilutionType DilType0;
  typedef typename A2AmesonField<mf_Policies,lA2AfieldL,lA2AfieldR>::RightDilutionType DilType1;
  typedef typename A2AmesonField<mf_Policies,rA2AfieldL,rA2AfieldR>::LeftDilutionType DilType2;
  typedef typename A2AmesonField<mf_Policies,rA2AfieldL,rA2AfieldR>::RightDilutionType DilType3;

  ModeContractionIndices<DilType0,DilType3> i_ind(l.getRowParams());
  ModeContractionIndices<DilType1,DilType2> j_ind(r.getRowParams());

  const int times[4] = { l.getRowTimeslice(), l.getColTimeslice(), r.getRowTimeslice(), r.getColTimeslice() };

  //W * W is only non-zero when the timeslice upon which we evaluate them are equal
  const int n_threads = omp_get_max_threads();
  std::vector<ScalarComplexType, BasicAlignedAllocator<ScalarComplexType> > ret_vec(n_threads,(0.,0.));
    
  modeIndexSet lip; lip.time = times[0];
  modeIndexSet rip; rip.time = times[3];

  modeIndexSet ljp; ljp.time = times[1];
  modeIndexSet rjp; rjp.time = times[2];

  const int ni = i_ind.getNindices(lip,rip); //how many indices to loop over
  const int nj = j_ind.getNindices(ljp,rjp);

#ifndef MEMTEST_MODE

#pragma omp parallel for schedule(static)
  for(int i = 0; i < ni; i++){
    const int id = omp_get_thread_num();
    const int li = i_ind.getLeftIndex(i,lip,rip);
    const int ri = i_ind.getRightIndex(i,lip,rip);

    for(int j = 0; j < nj; j++){
      const int lj = j_ind.getLeftIndex(j,ljp,rjp);
      const int rj = j_ind.getRightIndex(j,ljp,rjp);
      
      ret_vec[id] += l(li,lj) *  r(rj,ri);
    }
  }

  for(int i=0;i<n_threads;i++) into += ret_vec[i];
	 
#endif
  
  return into;
}




//Compute   l^ij(t1,t2) r^ji(t3,t4) for all t1, t4  and place into matrix element t1,t4
//This is both threaded and distributed over nodes
template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR
	 >
void trace(fMatrix<typename mf_Policies::ScalarComplexType> &into, const std::vector<A2AmesonField<mf_Policies,lA2AfieldL,lA2AfieldR> > &l, const std::vector<A2AmesonField<mf_Policies,rA2AfieldL,rA2AfieldR> > &r){
  //Distribute load over all nodes
  int lsize = l.size();
  int rsize = r.size();

  into.resize(lsize,rsize);

  const int work = lsize*rsize;

  bool do_work;
  int node_work, node_off;
  getNodeWork(work, node_work, node_off, do_work);

#ifndef MEMTEST_MODE
  if(do_work){
    for(int tt=node_off; tt<node_off + node_work; tt++){
      int rem = tt;
      int tsnk = rem % lsize; rem /= lsize; //sink time
      int tsrc = rem; //source time

      into(tsnk,tsrc) = trace(l[tsnk],r[tsrc]);
    }
  }
  into.nodeSum(); //give all nodes a copy
#endif
}


//Compute   m^ii(t1,t2)
//Threaded but *node local*
template<typename mf_Policies, 
	 template <typename> class A2AfieldL,  template <typename> class A2AfieldR
	 >
typename mf_Policies::ScalarComplexType trace(const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &m){
  //Check the indices match
  if(! m.getRowParams().paramsEqual( m.getColParams() ) )
    ERR.General("","trace(const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &)","Illegal trace: underlying index parameters must match\n");
  
  typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
  ScalarComplexType into(0,0);

  typedef typename A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::LeftDilutionType DilType0;
  typedef typename A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::RightDilutionType DilType1;

  ModeContractionIndices<DilType0,DilType1> i_ind(m.getRowParams());

  const int times[2] = { m.getRowTimeslice(), m.getColTimeslice() };

  const int n_threads = omp_get_max_threads();
  std::vector<ScalarComplexType, BasicAlignedAllocator<ScalarComplexType> > ret_vec(n_threads,(0.,0.));
    
  modeIndexSet lip; lip.time = times[0];
  modeIndexSet rip; rip.time = times[1];

  const int ni = i_ind.getNindices(lip,rip); //how many indices to loop over

#ifndef MEMTEST_MODE

#pragma omp parallel for schedule(static)
  for(int i = 0; i < ni; i++){
    const int id = omp_get_thread_num();
    const int li = i_ind.getLeftIndex(i,lip,rip);
    const int ri = i_ind.getRightIndex(i,lip,rip);

    ret_vec[id] += m(li,ri);
  }

  for(int i=0;i<n_threads;i++) into += ret_vec[i];
	 
#endif
  
  return into;
}




//Basic implementation for testing
template<typename mf_Policies, 
	 template <typename> class A2AfieldL,  template <typename> class A2AfieldR
	 >
typename mf_Policies::ScalarComplexType trace_slow(const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &m){
  //Check the indices match
  if(! m.getRowParams().paramsEqual( m.getColParams() ) )
    ERR.General("","trace(const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &)","Illegal trace: underlying index parameters must match\n");
  
  typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
  ScalarComplexType into(0,0);

  const int nv = m.getRowParams().getNv();

  const int n_threads = omp_get_max_threads();
  std::vector<ScalarComplexType, BasicAlignedAllocator<ScalarComplexType> > ret_vec(n_threads,(0.,0.));
  
#pragma omp parallel for schedule(static)
  for(int i = 0; i < nv; i++){
    const int id = omp_get_thread_num();
    ret_vec[id] += m.elem(i,i);
  }
  for(int i=0;i<n_threads;i++) into += ret_vec[i];  
  return into;
}




//Compute   m^ii(t1,t2)  for an arbitrary vector of meson fields
//This is both threaded and distributed over nodes
template<typename mf_Policies, 
	 template <typename> class A2AfieldL,  template <typename> class A2AfieldR
	 >
void trace(std::vector<typename mf_Policies::ScalarComplexType> &into, const std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> > &m){
  //Distribute load over all nodes
  const int nmf = m.size();
  into.resize(nmf);
  for(int i=0;i<nmf;i++) into[i] = typename mf_Policies::ScalarComplexType(0);

  const int work = nmf;
  bool do_work;
  int node_work, node_off;
  getNodeWork(work, node_work, node_off, do_work);

#ifndef MEMTEST_MODE
  if(do_work){
    for(int t=node_off; t<node_off + node_work; t++){
      into[t] = trace(m[t]);
    }
  }
  QMP_sum_array( (typename mf_Policies::ScalarComplexType::value_type*)&into[0],2*nmf); //give all nodes a copy
#endif
}



//Delete all the data associated with this meson field apart from on node with UniqueID 'node'. The node index is saved so that the data can be later retrieved.
template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::nodeDistribute(){
  this->distribute();
}




//Get back the data. After the call, all nodes will have a complete copy
template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::nodeGet(bool require){
  this->gather(require);
}


template<typename S>
struct _nodeGetManyPerf{ 
  static void reset(){}
  static void print(){}; 
};

template<>
struct _nodeGetManyPerf<DistributedMemoryStorage>{ 
  static void reset(){ DistributedMemoryStorage::perf().reset(); }; 
  static void print(){ 
    DistributedMemoryStorage::perf().print(); 
#ifdef DISTRIBUTED_MEMORY_STORAGE_REUSE_MEMORY
    if(!UniqueID()) DistributedMemoryStorage::block_allocator().stats(std::cout);
#endif
  }; 
};


//Handy helpers for gather and distribute of length Lt vectors of meson fields
template<typename T>
void nodeGetMany(const int n, std::vector<T> *a, ...){
  _nodeGetManyPerf<MesonFieldDistributedStorageType>::reset();
  cps::sync();
  
  double time = -dclock();

  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  for(int t=0;t<Lt;t++){
    a->operator[](t).nodeGet();
  }

  va_list vl;
  va_start(vl,a);
  for(int i=1; i<n; i++){
    std::vector<T>* val=va_arg(vl,std::vector<T>*);

    for(int t=0;t<Lt;t++){
      val->operator[](t).nodeGet();
    }
  }
  va_end(vl);

  print_time("nodeGetMany","Meson field gather",time+dclock());
  _nodeGetManyPerf<MesonFieldDistributedStorageType>::print();
}


template<typename T>
void nodeDistributeMany(const int n, std::vector<T> *a, ...){
  cps::sync();
  
  double time = -dclock();

  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  for(int t=0;t<Lt;t++){
    a->operator[](t).nodeDistribute();
  }

  va_list vl;
  va_start(vl,a);
  for(int i=1; i<n; i++){
    std::vector<T>* val=va_arg(vl,std::vector<T>*);

    for(int t=0;t<Lt;t++){
      val->operator[](t).nodeDistribute();
    }
  }
  va_end(vl);

  print_time("nodeDistributeMany","Meson field distribute",time+dclock());
}


//Same as above but the user can pass in a set of bools that tell the gather whether the MF on that timeslice is required. If not it is internally deleted, freeing memory
template<typename T>
void nodeGetMany(const int n, std::vector<T> *a, std::vector<bool> const* a_timeslice_mask,  ...){
  _nodeGetManyPerf<MesonFieldDistributedStorageType>::reset();
  cps::sync();

  double time = -dclock();

  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  for(int t=0;t<Lt;t++){
    a->operator[](t).nodeGet(a_timeslice_mask->at(t));
  }

  va_list vl;
  va_start(vl,a_timeslice_mask);
  for(int i=1; i<n; i++){
    std::vector<T>* val=va_arg(vl,std::vector<T>*);
    std::vector<bool> const* timeslice_mask = va_arg(vl,std::vector<bool> const*);
    
    for(int t=0;t<Lt;t++){
      val->operator[](t).nodeGet(timeslice_mask->at(t));
    }
  }
  va_end(vl);

  print_time("nodeGetMany","Meson field gather",time+dclock());
  _nodeGetManyPerf<MesonFieldDistributedStorageType>::print();
}


//Distribute only meson fields in 'from' that are *not* present in any of the sets 'notina' and following
template<typename T>
void nodeDistributeUnique(std::vector<T> &from, const int n, std::vector<T> const* notina, ...){
  cps::sync();
  
  double time = -dclock();
  
  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  
  std::set<T const*> exclude;

  for(int t=0;t<Lt;t++)
    exclude.insert(& notina->operator[](t) );

  va_list vl;
  va_start(vl,notina);
  for(int i=1; i<n; i++){
    std::vector<T> const* val=va_arg(vl,std::vector<T> const*);

    for(int t=0;t<Lt;t++)
      exclude.insert(& val->operator[](t) );
  }
  va_end(vl);

  for(int t=0;t<Lt;t++){
    if(exclude.count(&from[t]) == 0) from[t].nodeDistribute();
  }

  print_time("nodeDistributeUnique","Meson field distribute",time+dclock());
}

#endif








