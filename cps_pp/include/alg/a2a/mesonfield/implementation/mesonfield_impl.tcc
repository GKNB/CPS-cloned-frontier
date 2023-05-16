#ifndef MESON_FIELD_IMPL
#define MESON_FIELD_IMPL

template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::testRandom(const Float hi, const Float lo){
  CPSautoView(t_v,(*this),HostWrite);
#ifndef USE_C11_RNG    
  static UniformRandomGenerator urng(hi,lo);
  static bool init = false;
  if(!init){ urng.Reset(1234); init = true; }
  for(int i=0;i<this->fsize;i++) t_v.ptr()[i] = ScalarComplexType(urng.Rand(hi,lo), urng.Rand(hi,lo) );
#else
  static CPS_RNG eng(1234);
  std::uniform_real_distribution<Float> urand(lo,hi);
  for(int i=0;i<this->fsize;i++) t_v.ptr()[i] = ScalarComplexType(urand(eng), urand(eng) );
#endif
}


template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::plus_equals(const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &with, const bool parallel){
  if(nmodes_l != with.nmodes_l || nmodes_r != with.nmodes_r || 
     !lindexdilution.paramsEqual(with.lindexdilution) || !rindexdilution.paramsEqual(with.rindexdilution) ){
    ERR.General("A2AmesonField","plus_equals(..)","Second meson field must have the same underlying parameters\n");
  }
  CPSautoView(t_v,(*this),HostReadWrite);
  CPSautoView(with_v,with,HostRead);

  if(parallel){
#pragma omp_parallel for
    for(int i=0;i<fsize;i++) t_v.ptr()[i] += with_v.ptr()[i];
  }else{
    for(int i=0;i<fsize;i++) t_v.ptr()[i] += with_v.ptr()[i];
  }
}

template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::times_equals(const ScalarComplexType f,const bool parallel){
  CPSautoView(t_v,(*this),HostReadWrite);

  if(parallel){
#pragma omp_parallel for
    for(int i=0;i<fsize;i++) t_v.ptr()[i] *= f;			       
  }else{
    for(int i=0;i<fsize;i++) t_v.ptr()[i] *= f;
  }
}

template<typename T>
inline std::complex<T> complexAvg(const std::complex<T>&a, const std::complex<T> &b){
  return (a+b)/T(2.0);
}

#if defined(USE_GRID) && (defined(GRID_CUDA) || defined(GRID_HIP))
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
  CPSautoView(t_v,(*this),HostReadWrite);
  CPSautoView(with_v,with,HostRead);

  if(parallel){
#pragma omp_parallel for
    for(int i=0;i<fsize;i++) t_v.ptr()[i] = complexAvg(t_v.ptr()[i],with_v.ptr()[i]);//(mf[i] + with.mf[i])/2.0;
  }else{
    for(int i=0;i<fsize;i++) t_v.ptr()[i] = complexAvg(t_v.ptr()[i],with_v.ptr()[i]);//(mf[i] + with.mf[i])/2.0;
  }
}

//Reorder the rows so that all the elements in idx_map are sequential. Indices not in map are ignored. Use at your own risk
template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::rowReorder(A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &into, const int idx_map[], int map_size, bool parallel) const{
  into.setup(lindexdilution, rindexdilution, tl, tr);
  CPSautoView(t_v,(*this),HostRead);
  CPSautoView(into_v,into,HostWrite)

#define DOIT \
    int irow = idx_map[i]; \
    for(int j=0;j<nmodes_r;j++) \
      into_v(i,j) = t_v(irow,j);

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
  CPSautoView(t_v,(*this),HostRead);
  CPSautoView(into_v,into,HostWrite)

#define DOIT \
    for(int j=0;j<map_size;j++){ \
      int jcol = idx_map[j]; \
      into_v(i,j) = t_v(i,jcol);			\
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
  CPSautoView(t_v,(*this),HostRead);
  for(int i_full=0;i_full<full_rows;i_full++){
    if(rowidx_used[i_full]){
      ScalarComplexType const* mf_row_base = t_v.ptr() + nmodes_r*i_full;
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
  CPSautoView(t_v,(*this),HostRead);
  for(int i_full=0;i_full<full_rows;i_full++){
    if(rowidx_used[i_full]){
      ScalarComplexType const* mf_row_base = t_v.ptr() + nmodes_r*i_full;
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
  CPSautoView(t_v,(*this),HostRead);
  CPSautoView(into_v,into,HostWrite);
#pragma omp parallel for
  for(int i=0;i<nmodes_l;i++)
    for(int j=0;j<nmodes_r;j++)
      into_v(j,i) = t_v(i,j);
}

//Take the complex conjugate of the meson field
template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::conj(A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &into) const{
  assert( (void*)this != (void*)&into );
  into.setup(lindexdilution, rindexdilution, tl, tr);
  CPSautoView(t_v,(*this),HostRead);
  CPSautoView(into_v,into,HostWrite);
#pragma omp parallel for
  for(int i=0;i<nmodes_l;i++)
    for(int j=0;j<nmodes_r;j++)
      into_v(i,j) = std::conj(t_v(i,j));
}
//Take the hermitian conjugate of the meson field
template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::hconj(A2AmesonField<mf_Policies,A2AfieldR,A2AfieldL> &into) const{
  assert( (void*)this != (void*)&into );
  into.setup(rindexdilution, lindexdilution, tr, tl);
  CPSautoView(t_v,(*this),HostRead);
  CPSautoView(into_v,into,HostWrite);
#pragma omp parallel for
  for(int i=0;i<nmodes_l;i++)
    for(int j=0;j<nmodes_r;j++)
      into_v(j,i) = conj(t_v(i,j));
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


template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::View::View(ViewMode mode, const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &mf): nmodes_l(mf.nmodes_l), nmodes_r(mf.nmodes_r), fsize(mf.fsize),
																     tl(mf.tl), tr(mf.tr), parent(&mf), alloc_view(mf.mf_Policies::MesonFieldDistributedStorageType::view(mode)){
  data = (ScalarComplexType *)alloc_view();
}

template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::View::free(){
  alloc_view.free();
}

template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::unpack(typename mf_Policies::ScalarComplexType* into) const{
  memset(into, 0, getNrowsFull()*getNcolsFull()*sizeof(ScalarComplexType));
  int lnl = getRowParams().getNl(),  lnf = getRowParams().getNflavors(), lnsc = getRowParams().getNspinColor() , lnt = getRowParams().getNtBlocks();
  int rnl = getColParams().getNl(),  rnf = getColParams().getNflavors(), rnsc = getColParams().getNspinColor() , rnt = getColParams().getNtBlocks();
  int ncolsfull = getNcolsFull();
  CPSautoView(t_v,(*this),HostRead);
#pragma omp parallel for
  for(int i=0;i<getNrows();i++){
    int iinto = mesonFieldConvertDilution<LeftDilutionType>::unpack(i, getRowParams().tblock(getRowTimeslice()), lnl, lnf, lnsc, lnt);
    for(int j=0;j<getNcols();j++){
      int jinto = mesonFieldConvertDilution<RightDilutionType>::unpack(j, getColParams().tblock(getColTimeslice()), rnl, rnf, rnsc, rnt);

      into[jinto + ncolsfull*iinto] = t_v(i,j);
    }
  }
  
}

template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::unpack_device(typename mf_Policies::ScalarComplexType* into, A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::View const* view) const{
#if !defined(USE_GRID) || ( !defined(GPU_VEC) && !defined(FORCE_A2A_OFFLOAD) )
  unpack(into); //into must be a host pointer here
#else
  A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::View *vp = const_cast<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::View *>(view);
  bool delete_view = false;
  if(vp == nullptr){
    vp = new A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::View(this->view(DeviceRead)); //copies to device
    delete_view = true;
  }

  A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::View const& v = *vp;
  
  device_memset(into, 0, getNrowsFull()*getNcolsFull()*sizeof(ScalarComplexType));
  int lnl = getRowParams().getNl(),  lnf = getRowParams().getNflavors(), lnsc = getRowParams().getNspinColor() , lnt = getRowParams().getNtBlocks();
  int rnl = getColParams().getNl(),  rnf = getColParams().getNflavors(), rnsc = getColParams().getNspinColor() , rnt = getColParams().getNtBlocks();
  int ncolsfull = getNcolsFull();
  int nrows= getNrows(), ncols = getNcols();
  int tl = getRowTimeslice(), tr = getColTimeslice();
  int tlblock = getRowParams().tblock(tl), trblock = getColParams().tblock(tr);
  
  {
    using namespace Grid;
    accelerator_for2d(j, ncols, i, nrows, 1, {
	int iinto = mesonFieldConvertDilution<LeftDilutionType>::unpack(i, tlblock, lnl, lnf, lnsc, lnt);
	int jinto = mesonFieldConvertDilution<RightDilutionType>::unpack(j, trblock, rnl, rnf, rnsc, rnt);

	into[jinto + ncolsfull*iinto] = v(i,j);
      });
  }

  if(delete_view){
    vp->free();
    delete vp;
  }
    
#endif
}


template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::pack(typename mf_Policies::ScalarComplexType const* from){
  int lnl = getRowParams().getNl(),  lnf = getRowParams().getNflavors(), lnsc = getRowParams().getNspinColor() , lnt = getRowParams().getNtBlocks();
  int rnl = getColParams().getNl(),  rnf = getColParams().getNflavors(), rnsc = getColParams().getNspinColor() , rnt = getColParams().getNtBlocks();
  int nrowsfull = getNrowsFull(), ncolsfull = getNcolsFull();
  CPSautoView(t_v,(*this),HostWrite);
#pragma omp parallel for  
  for(int i=0;i<nrowsfull;i++){
    auto iinto = mesonFieldConvertDilution<LeftDilutionType>::pack(i, getRowParams().tblock(getRowTimeslice()), lnl, lnf, lnsc, lnt);
    if(iinto.second){
      for(int j=0;j<getNcols();j++){
	auto jinto = mesonFieldConvertDilution<RightDilutionType>::pack(j, getColParams().tblock(getColTimeslice()), rnl, rnf, rnsc, rnt);
	if(jinto.second){
	  t_v(iinto.first,jinto.first) = from[j + ncolsfull*i];
	}
      }
    }
  }
}


template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::pack_device(typename mf_Policies::ScalarComplexType const* from){
#if !defined(USE_GRID) || ( !defined(GPU_VEC) && !defined(FORCE_A2A_OFFLOAD) )
  pack(from);
#else  
  int lnl = getRowParams().getNl(),  lnf = getRowParams().getNflavors(), lnsc = getRowParams().getNspinColor() , lnt = getRowParams().getNtBlocks();
  int rnl = getColParams().getNl(),  rnf = getColParams().getNflavors(), rnsc = getColParams().getNspinColor() , rnt = getColParams().getNtBlocks();
  int nrowsfull = getNrowsFull(), ncolsfull = getNcolsFull();
  int nrows= getNrows(), ncols = getNcols(); 
  int tl = getRowTimeslice(), tr = getColTimeslice();
  int tlblock = getRowParams().tblock(tl), trblock = getColParams().tblock(tr);
  
  //Create a staging post on the device
  size_t stage_size = nrows*ncols*sizeof(ScalarComplexType);
  ScalarComplexType *stage = (ScalarComplexType*)device_alloc_check(stage_size);
  
  {
    using namespace Grid;
    accelerator_for2d(j, ncolsfull, i, nrowsfull, 1, {
	auto iinto = mesonFieldConvertDilution<LeftDilutionType>::pack(i, tlblock, lnl, lnf, lnsc, lnt);
	auto jinto = mesonFieldConvertDilution<RightDilutionType>::pack(j, trblock, rnl, rnf, rnsc, rnt);
	if(iinto.second && jinto.second)
	  stage[jinto.first + ncols*iinto.first] = from[j + ncolsfull * i];
      });
  }
  CPSautoView(t_v,(*this),HostWrite);
  copy_device_to_host(t_v.ptr(),stage,stage_size);
  device_free(stage);
#endif
}
  

#endif








