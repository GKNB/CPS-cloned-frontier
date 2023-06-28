//Cyclic permutation of *4D* and *3D* CPSfield with std::complex type and FourDpolicy dimension policy
//Conventions are direction of *data flow*: For shift n in direction +1   f'(x) = f(x-\hat i)  so data is sent in the +x direction.

#ifdef USE_QMP

//-------------------------------------------------------------------------------------------------------------------------------------------------------
//QMP, non-SIMD data
//-------------------------------------------------------------------------------------------------------------------------------------------------------

#define CONDITION _equal<typename ComplexClassify<mf_Complex>::type, complex_double_or_float_mark>::value && (_equal<MappingPolicy,FourDpolicy<typename MappingPolicy::FieldFlavorPolicy> >::value || _equal<MappingPolicy,SpatialPolicy<typename MappingPolicy::FieldFlavorPolicy> >::value)

//Version with QMP and no Grid
template< typename mf_Complex, int SiteSize, typename MappingPolicy, typename AllocPolicy>
void cyclicPermuteImpl(CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy> &to, const CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy> &from,
		   const int dir, const int pm, const int n,
		   typename my_enable_if<CONDITION , const int>::type dummy = 0){
  enum {Dimension = MappingPolicy::EuclideanDimension};
  assert(dir < Dimension);
  assert(n < GJP.NodeSites(dir));
  assert(pm == 1 || pm == -1);
	   
  if(&to == &from){
    if(n==0) return;    
    CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy> tmpfrom(from);
    return cyclicPermuteImpl(to,tmpfrom,dir,pm,n);
  }
  if(n == 0){
    to = from;
    return;
  }

  QMP_barrier();
  
  //Prepare face to send. If we send in the + direction we need to collect the slice starting {L-n ... L-1} (inclusive), and if we send in the - dir we collect the slice {0... n-1}
  int bsites = n; //sites on boundary
  int bsizes[Dimension]; bsizes[dir] = n;
  int boff[Dimension]; boff[dir] = (pm == 1 ? GJP.NodeSites(dir)-n : 0);
		 
  for(int i=0;i<Dimension;i++)
    if(i != dir){
      bsizes[i] = GJP.NodeSites(i);
      bsites *= bsizes[i];
      boff[i] = 0;
    }
  int flav_off = from.flav_offset();
  int nf = from.nflavors();
  
  int bufsz = bsites * SiteSize * nf;
  int halfbufsz = bufsz/2;

  QMP_mem_t *recv_mem = QMP_allocate_memory(bufsz * sizeof(mf_Complex));
  mf_Complex *recv_buf = (mf_Complex *)QMP_get_memory_pointer(recv_mem);

  QMP_mem_t *send_mem = QMP_allocate_memory(bufsz * sizeof(mf_Complex));
  mf_Complex *send_buf = (mf_Complex *)QMP_get_memory_pointer(send_mem);
  
  {
    CPSautoView(from_v,from,HostRead);
#pragma omp parallel for
    for(int i=0;i<bsites;i++){
      int rem = i;
      int coor[Dimension];
      for(int d=0;d<Dimension;d++){ coor[d] = rem % bsizes[d] + boff[d]; rem/=bsizes[d]; }

      mf_Complex const* site_ptr = from_v.site_ptr(coor);
      mf_Complex* bp = send_buf + i*SiteSize;
      memcpy(bp,site_ptr,SiteSize*sizeof(mf_Complex));
      if(nf == 2){
	site_ptr += flav_off;
	bp += halfbufsz;
	memcpy(bp,site_ptr,SiteSize*sizeof(mf_Complex));
      }
    }
  }
  QMP_barrier();
 
  //Copy remaining sites from on-node data with shift
  int rsizes[Dimension]; rsizes[dir] = GJP.NodeSites(dir) - n;
  int rsites = GJP.NodeSites(dir) - n;
  //if we sent in the + direction we need to shift the remaining L-n sites {0...L-n-1} forwards by n to make way for a new slice at the left side
  //if we sent in the - direction we need to shift the remaining L-n sites {n ... L-1} backwards by n to make way for a new slice at the right side
  
  int roff[Dimension]; roff[dir] = (pm == 1 ? 0 : n);  
  for(int i=0;i<Dimension;i++)
    if(i != dir){
      rsizes[i] = GJP.NodeSites(i);
      rsites *= rsizes[i];
      roff[i] = 0;
    }

  {
    CPSautoView(to_v,to,HostWrite);
    CPSautoView(from_v,from,HostRead);
    
#pragma omp parallel for
    for(int i=0;i<rsites;i++){
      int rem = i;
      int from_coor[Dimension];
      for(int d=0;d<Dimension;d++){ from_coor[d] = rem % rsizes[d] + roff[d]; rem/=rsizes[d]; }
    
      int to_coor[Dimension]; memcpy(to_coor,from_coor,Dimension*sizeof(int));
      to_coor[dir] = (pm == +1 ? from_coor[dir] + n : from_coor[dir] - n);
    
      mf_Complex const* from_ptr = from_v.site_ptr(from_coor);
      mf_Complex * to_ptr = to_v.site_ptr(to_coor);

      memcpy(to_ptr,from_ptr,SiteSize*sizeof(mf_Complex));
      if(nf == 2){
	from_ptr += flav_off;
	to_ptr += flav_off;
	memcpy(to_ptr,from_ptr,SiteSize*sizeof(mf_Complex));
      }
    }
  }
  
  //Send/receive
  QMP_msgmem_t send_msg = QMP_declare_msgmem(send_buf,bufsz * sizeof(mf_Complex));
  QMP_msgmem_t recv_msg = QMP_declare_msgmem(recv_buf,bufsz * sizeof(mf_Complex));
  
  QMP_msghandle_t send = QMP_declare_send_relative(send_msg, dir, pm, 0);
  QMP_msghandle_t recv = QMP_declare_receive_relative(recv_msg, dir, -pm, 0);
  QMP_start(recv);
  QMP_start(send);
  
  QMP_status_t send_status = QMP_wait(send);
  if (send_status != QMP_SUCCESS) 
    QMP_error("Send failed in cyclicPermuteImpl: %s\n", QMP_error_string(send_status));
  QMP_status_t rcv_status = QMP_wait(recv);
  if (rcv_status != QMP_SUCCESS) 
    QMP_error("Receive failed in PassDataT: %s\n", QMP_error_string(rcv_status));

  //Copy received face into position. For + shift the origin we copy into is the left-face {0..n-1}, for a - shift its the right-face {L-n .. L-1}
  boff[dir] = (pm == 1 ? 0 : GJP.NodeSites(dir)-n);
  {
    CPSautoView(to_v,to,HostWrite);

#pragma omp parallel for
    for(int i=0;i<bsites;i++){
      int rem = i;
      int coor[Dimension];
      for(int d=0;d<Dimension;d++){ coor[d] = rem % bsizes[d] + boff[d]; rem/=bsizes[d]; }
    
      mf_Complex * site_ptr = to_v.site_ptr(coor);
      mf_Complex const* bp = recv_buf + i*SiteSize;
      memcpy(site_ptr,bp,SiteSize*sizeof(mf_Complex));
      if(nf == 2){
	site_ptr += flav_off;
	bp += halfbufsz;
	memcpy(site_ptr,bp,SiteSize*sizeof(mf_Complex));
      }
    }
  }

  QMP_free_msghandle(send);
  QMP_free_msghandle(recv);
  QMP_free_msgmem(send_msg);
  QMP_free_msgmem(recv_msg);
  QMP_free_memory(send_mem);
  QMP_free_memory(recv_mem);
  QMP_barrier();
}
#undef CONDITION

# ifdef USE_GRID

//-------------------------------------------------------------------------------------------------------------------------------------------------------
//QMP, SIMD data
//-------------------------------------------------------------------------------------------------------------------------------------------------------

#define CONDITION _equal<typename ComplexClassify<mf_Complex>::type, grid_vector_complex_mark>::value && (_equal<MappingPolicy,FourDSIMDPolicy<typename MappingPolicy::FieldFlavorPolicy> >::value || _equal<MappingPolicy,ThreeDSIMDPolicy<typename MappingPolicy::FieldFlavorPolicy> >::value)

template< typename mf_Complex, int SiteSize, typename MappingPolicy, typename AllocPolicy>
void cyclicPermuteImpl(CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy> &to, const CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy> &from,
		   const int dir, const int pm, const int n,
		   typename my_enable_if<CONDITION, const int>::type dummy = 0){
  enum {Dimension = MappingPolicy::EuclideanDimension};
  assert(dir < Dimension);
  assert(n < GJP.NodeSites(dir));
  assert(pm == 1 || pm == -1);
  
  if(&to == &from){
    if(n==0) return;    
    CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy> tmpfrom(from);
    return cyclicPermuteImpl(to,tmpfrom,dir,pm,n);
  }
  if(n == 0){
    to = from;
    return;
  }

  const int nsimd = mf_Complex::Nsimd();
  
  //Use notation c (combined index), o (outer index) i (inner index)
  
  int bcsites = n; //sites on boundary
  int bcsizes[Dimension]; bcsizes[dir] = n;
  int bcoff[Dimension]; bcoff[dir] = (pm == 1 ? GJP.NodeSites(dir)-n : 0);
  int bcoff_postcomms[Dimension]; bcoff_postcomms[dir] = (pm == 1 ? 0 : GJP.NodeSites(dir)-n);
  
  for(int i=0;i<Dimension;i++)
    if(i != dir){
      bcsizes[i] = GJP.NodeSites(i);
      bcsites *= bcsizes[i];
      bcoff[i] = 0;
      bcoff_postcomms[i] = 0;
    }

  //Build table of points on face (both outer and inner index)
  int nf = from.nflavors();
  int flav_off = from.flav_offset();

  typedef typename Grid::GridTypeMapper<mf_Complex>::scalar_type scalarType;
  
  int bufsz = bcsites * SiteSize * nf;

  QMP_mem_t *recv_mem = QMP_allocate_memory(bufsz * sizeof(scalarType));
  scalarType *recv_buf = (scalarType *)QMP_get_memory_pointer(recv_mem);

  QMP_mem_t *send_mem = QMP_allocate_memory(bufsz * sizeof(scalarType));
  scalarType *send_buf = (scalarType *)QMP_get_memory_pointer(send_mem);

  int osites = from.nsites();
  std::vector<int> to_oi_buf_map(nf * osites * nsimd); //map from outer and inner index of destination site to offset within buffer, used *after* comms.
  //map i + nsimd*(o + osites*f) as index
  {
    CPSautoView(from_v,from,HostRead);

#pragma omp parallel for
    for(int c=0;c<bcsites;c++){
      int rem = c;
      int coor[Dimension];
      for(int d=0;d<Dimension;d++){ coor[d] = rem % bcsizes[d]; rem/=bcsizes[d]; }

      int coor_dest[Dimension];
      for(int d=0;d<Dimension;d++){
	coor_dest[d] = coor[d] + bcoff_postcomms[d];
	coor[d] += bcoff[d];
      }
    
      int i = from.SIMDmap(coor);
      int o = from.siteMap(coor);

      int i_dest = from.SIMDmap(coor_dest);
      int o_dest = from.siteMap(coor_dest);

      typename AlignedVector<scalarType>::type ounpacked(nsimd);
      for(int f=0;f<nf;f++){
	mf_Complex const *osite_ptr = from_v.site_ptr(o,f);
	int send_buf_off = (c + bcsites*f)*SiteSize;
	scalarType* bp = send_buf + send_buf_off;
	to_oi_buf_map[ i_dest + nsimd*(o_dest+osites*f) ] = send_buf_off;
      
	for(int s=0;s<SiteSize;s++){
	  vstore(*(osite_ptr++), ounpacked.data());
	  *(bp++) = ounpacked[i];
	}      
      }
    }
  }
  //Send/receive
  QMP_msgmem_t send_msg = QMP_declare_msgmem(send_buf,bufsz * sizeof(scalarType));
  QMP_msgmem_t recv_msg = QMP_declare_msgmem(recv_buf,bufsz * sizeof(scalarType));
  
  QMP_msghandle_t send = QMP_declare_send_relative(send_msg, dir, pm, 0);
  QMP_msghandle_t recv = QMP_declare_receive_relative(recv_msg, dir, -pm, 0);
  QMP_start(recv);
  QMP_start(send);
  
  QMP_status_t send_status = QMP_wait(send);
  if (send_status != QMP_SUCCESS) 
    QMP_error("Send failed in cyclicPermuteImpl: %s\n", QMP_error_string(send_status));
  QMP_status_t rcv_status = QMP_wait(recv);
  if (rcv_status != QMP_SUCCESS) 
    QMP_error("Receive failed in PassDataT: %s\n", QMP_error_string(rcv_status));


  
  //Copy remaining sites from on-node data with shift and pull in data from buffer simultaneously
  //if we sent in the + direction we need to shift the remaining L-n sites {0...L-n-1} forwards by n to make way for a new slice at the left side
  //if we sent in the - direction we need to shift the remaining L-n sites {n ... L-1} backwards by n to make way for a new slice at the right side
  //Problem is we don't want two threads writing to the same AVX register at the same time. Therefore we thread the loop over the destination SIMD vectors and work back
  std::vector< std::vector<int> > lane_offsets(nsimd,  std::vector<int>(Dimension) );
  for(int i=0;i<nsimd;i++) from.SIMDunmap(i, lane_offsets[i].data() );
  
  {
    CPSautoView(to_v,to,HostWrite);
    CPSautoView(from_v,from,HostRead);
    
#pragma omp parallel for
    for(int oto = 0;oto < osites; oto++){
      int oto_base_coor[Dimension]; to.siteUnmap(oto,oto_base_coor);

      //For each destination lane compute the source site index and lane
      int from_lane[nsimd];
      int from_osite_idx[nsimd]; //also use for recv_buf offsets for sites pulled over boundary
      for(int lane = 0; lane < nsimd; lane++){
	int offrom_coor[Dimension];
	for(int d=0;d<Dimension;d++) offrom_coor[d] = oto_base_coor[d] + lane_offsets[lane][d];
	offrom_coor[dir] += (pm == 1 ? -n : n);

	if(offrom_coor[dir] < 0 || offrom_coor[dir] >= GJP.NodeSites(dir)){
	  from_lane[lane] = -1; //indicates data is in recv_buf	
	  from_osite_idx[lane] = to_oi_buf_map[ lane + nsimd*oto ]; //here is for flavor 0 - remember to offset for second flav
	}else{
	  from_lane[lane] = from.SIMDmap(offrom_coor);
	  from_osite_idx[lane] = from.siteMap(offrom_coor);
	}
      }

      //Now loop over flavor and element within the site as well as SIMD lanes of the destination vector and gather what we need to poke - then poke it
      typename AlignedVector<scalarType>::type towrite(nsimd);
      typename AlignedVector<scalarType>::type unpack(nsimd);
    
      for(int f=0;f<nf;f++){
	for(int s=0;s<SiteSize;s++){
	  for(int tolane=0;tolane<nsimd;tolane++){	  
	    if(from_lane[tolane] != -1){
	      mf_Complex const* from_osite_ptr = from_v.site_ptr(from_osite_idx[tolane], f) + s;
	      vstore(*from_osite_ptr,unpack.data());
	      towrite[tolane] = unpack[ from_lane[tolane] ];
	    }else{
	      //data is in buffer
	      towrite[tolane] = recv_buf[ from_osite_idx[tolane] + s + f*bcsites*SiteSize ];
	    }
	    
	  }
	  mf_Complex* to_osite_ptr = to_v.site_ptr(oto,f) + s;
	  vset(*to_osite_ptr, towrite.data());	
	}
      }
    }
  }
  QMP_free_msghandle(send);
  QMP_free_msghandle(recv);
  QMP_free_msgmem(send_msg);
  QMP_free_msgmem(recv_msg);
  QMP_free_memory(send_mem);
  QMP_free_memory(recv_mem);
  QMP_barrier();
}
#undef CONDITION

# endif //ifdef USE_GRID

#else //ifdef USE_QMP


//-------------------------------------------------------------------------------------------------------------------------------------------------------
//No QMP (local), non-SIMD data
//-------------------------------------------------------------------------------------------------------------------------------------------------------

#define CONDITION _equal<typename ComplexClassify<mf_Complex>::type, complex_double_or_float_mark>::value && (_equal<MappingPolicy,FourDpolicy<typename MappingPolicy::FieldFlavorPolicy> >::value || _equal<MappingPolicy,SpatialPolicy<typename MappingPolicy::FieldFlavorPolicy> >::value)

//Version without comms (local) and without Grid
template< typename mf_Complex, int SiteSize, typename MappingPolicy, typename AllocPolicy>
void cyclicPermuteImpl(CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy> &to, const CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy> &from,
		       const int dir, const int pm, const int n,
		       typename my_enable_if<CONDITION , const int>::type dummy = 0){
  enum {Dimension = MappingPolicy::EuclideanDimension};
  assert(dir < Dimension);
  assert(n < GJP.NodeSites(dir));
  assert(pm == 1 || pm == -1);
  
  if(&to == &from){
    if(n==0) return;    
    CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy> tmpfrom(from);
    return cyclicPermuteImpl(to,tmpfrom,dir,pm,n);
  }
  if(n == 0){
    to = from;
    return;
  }
  const int nodes = GJP.Xnodes()*GJP.Ynodes()*GJP.Znodes()*GJP.Tnodes()*GJP.Snodes();
  if(nodes != 1) ERR.General("","cyclicPermuteImpl","Parallel implementation requires QMP\n");

  CPSautoView(from_v,from,HostRead);
  CPSautoView(to_v,to,HostWrite);
  
#pragma omp parallel for
  for(int i=0;i<from.nfsites();i++){
    int f; int x[Dimension];
    from.fsiteUnmap(i,x,f);
    x[dir] = (x[dir] + pm * n + 5*GJP.NodeSites(dir) ) % GJP.NodeSites(dir);
    const mf_Complex* from_ptr = from_v.fsite_ptr(i);
    mf_Complex* to_ptr = to_v.site_ptr(x,f);
    memcpy(to_ptr,from_ptr,SiteSize*sizeof(mf_Complex));
  }
}
#undef CONDITION

# ifdef USE_GRID


//-------------------------------------------------------------------------------------------------------------------------------------------------------
//non-QMP (local), SIMD data
//-------------------------------------------------------------------------------------------------------------------------------------------------------

#define CONDITION _equal<typename ComplexClassify<mf_Complex>::type, grid_vector_complex_mark>::value && (_equal<MappingPolicy,FourDSIMDPolicy<typename MappingPolicy::FieldFlavorPolicy> >::value || _equal<MappingPolicy,ThreeDSIMDPolicy<typename MappingPolicy::FieldFlavorPolicy> >::value)

//Version without comms (local) and with SIMD vectorized data
template< typename mf_Complex, int SiteSize, typename MappingPolicy, typename AllocPolicy>
void cyclicPermuteImpl(CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy> &to, const CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy> &from,
		       const int dir, const int pm, const int n,
		       typename my_enable_if<CONDITION, const int>::type dummy = 0){
  enum {Dimension = MappingPolicy::EuclideanDimension};
  assert(dir < Dimension);
  assert(n < GJP.NodeSites(dir));
  assert(pm == 1 || pm == -1);
  
  if(&to == &from){
    if(n==0) return;    
    CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy> tmpfrom(from);
    return cyclicPermuteImpl(to,tmpfrom,dir,pm,n);
  }
  if(n == 0){
    to = from;
    return;
  }
  const int nodes = GJP.Xnodes()*GJP.Ynodes()*GJP.Znodes()*GJP.Tnodes()*GJP.Snodes();
  if(nodes != 1) ERR.General("","cyclicPermuteImpl","Parallel implementation requires QMP\n");
  
  const int nsimd = mf_Complex::Nsimd();

  typedef typename mf_Complex::scalar_type scalar_type;
  const int nthr = omp_get_max_threads();
  scalar_type* tmp_store_thr[nthr]; for(int i=0;i<nthr;i++) tmp_store_thr[i] = (scalar_type*)memalign_check(128,nsimd*sizeof(scalar_type));
  
  CPSautoView(to_v,to,HostWrite);
  CPSautoView(from_v,from,HostRead);

#pragma omp parallel for
  for(int ofto=0;ofto<to.nfsites();ofto++){ //loop over outer site index
    const int me = omp_get_thread_num();
    int f; int oxto[Dimension];
    to.fsiteUnmap(ofto,oxto,f);

    mf_Complex* to_base_ptr = to_v.fsite_ptr(ofto);
    
    scalar_type* tmp_store = tmp_store_thr[me];

    //indexed by destination lane
    mf_Complex const* from_base_ptrs[nsimd];
    int from_lane_idx[nsimd];
      
    for(int tolane = 0; tolane < nsimd; tolane++){
      int ixto_off[Dimension];
      to.SIMDunmap(tolane,ixto_off); //get offset of inner site on tolane

      int xfrom[Dimension]; for(int d=0;d<Dimension;d++) xfrom[d] = oxto[d] + ixto_off[d]; //full coord corresponding to tolane + outer site
      xfrom[dir] = (xfrom[dir] - pm * n + 5*GJP.NodeSites(dir) ) % GJP.NodeSites(dir);

      from_base_ptrs[tolane] = from_v.site_ptr(xfrom,f);
      from_lane_idx[tolane] = from.SIMDmap(xfrom);
    }

    for(int s=0;s<SiteSize;s++){
      for(int tolane = 0; tolane < nsimd; tolane++)
	tmp_store[tolane] = *( (scalar_type*)(from_base_ptrs[tolane] + s) + from_lane_idx[tolane] ); //cast SIMD type to scalar type pointer
      vset(*(to_base_ptr + s), tmp_store);
    }
  }            
  for(int i=0;i<nthr;i++) free(tmp_store_thr[i]);  
}
#undef CONDITION
  
# endif //ifdef USE_GRID

#endif //ifdef USE_QMP
