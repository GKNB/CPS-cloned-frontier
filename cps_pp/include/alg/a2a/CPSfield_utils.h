#ifndef CPS_FIELD_UTILS_H
#define CPS_FIELD_UTILS_H

CPS_START_NAMESPACE

inline void compareFermion(const CPSfermion5D<ComplexD> &A, const CPSfermion5D<ComplexD> &B, const std::string &descr = "Ferms", const double tol = 1e-9){
  double fail = 0.;
  for(int i=0;i<GJP.VolNodeSites()*GJP.SnodeSites();i++){
    int x[5]; int rem = i;
    for(int ii=0;ii<5;ii++){ x[ii] = rem % GJP.NodeSites(ii); rem /= GJP.NodeSites(ii); }
    
    for(int f=0;f<GJP.Gparity()+1;f++){
      for(int sc=0;sc<24;sc++){
	double vbfm = *((double*)A.site_ptr(i,f) + sc);
	double vgrid = *((double*)B.site_ptr(i,f) + sc);
	    
	double diff_rat = fabs( 2.0 * ( vbfm - vgrid )/( vbfm + vgrid ) );
	double rat_grid_bfm = vbfm/vgrid;
	if(vbfm == 0.0 && vgrid == 0.0){ diff_rat = 0.;	 rat_grid_bfm = 1.; }
	if( (vbfm == 0.0 && fabs(vgrid) < 1e-50) || (vgrid == 0.0 && fabs(vbfm) < 1e-50) ){ diff_rat = 0.;	 rat_grid_bfm = 1.; }

	if(diff_rat > tol){
	  printf("Fail: (%d,%d,%d,%d,%d; %d; %d) A %g B %g rat_A_B %g fracdiff %g\n",x[0],x[1],x[2],x[3],x[4],f,sc,vbfm,vgrid,rat_grid_bfm,diff_rat);
	  fail = 1.0;
	}//else printf("Pass: (%d,%d,%d,%d,%d; %d; %d) A %g B %g rat_A_B %g fracdiff %g\n",x[0],x[1],x[2],x[3],x[4],f,sc,vbfm,vgrid,rat_grid_bfm,diff_rat);
      }
    }
  }
  glb_max(&fail);
  
  if(fail!=0.0){
    if(!UniqueID()){ printf("Failed %s check\n", descr.c_str()); fflush(stdout); } 
    exit(-1);
  }else{
    if(!UniqueID()){ printf("Passed %s check\n", descr.c_str()); fflush(stdout); }
  }
}

template<typename FieldType, typename my_enable_if<_equal<typename ComplexClassify<typename FieldType::FieldSiteType>::type, complex_double_or_float_mark>::value,int>::type = 0>
inline void compareField(const FieldType &A, const FieldType &B, const std::string &descr = "Field", const double tol = 1e-9, bool print_all = false){
  typedef typename FieldType::FieldSiteType::value_type value_type;
  
  double fail = 0.;
  for(int xf=0;xf<A.nfsites();xf++){
    int f; int x[FieldType::FieldDimensionPolicy::EuclideanDimension];
    A.fsiteUnmap(xf, x,f);

    for(int i=0;i<FieldType::FieldSiteSize;i++){
      value_type const* av = (value_type const*)(A.fsite_ptr(xf)+i);
      value_type const* bv = (value_type const*)(B.fsite_ptr(xf)+i);
      for(int reim=0;reim<2;reim++){
	value_type diff_rat = (av[reim] == 0.0 && bv[reim] == 0.0) ? 0.0 : fabs( 2.*(av[reim]-bv[reim])/(av[reim]+bv[reim]) );
	if(diff_rat > tol || print_all){
	  if(!print_all) std::cout << "Fail: (";
	  else std::cout << "Pass: (";
	  
	  for(int xx=0;xx<FieldType::FieldDimensionPolicy::EuclideanDimension-1;xx++)
	    std::cout << x[xx] << ", ";
	  std::cout << x[FieldType::FieldDimensionPolicy::EuclideanDimension-1];

	  std::cout << ") f=" << f << " reim " << reim << " A " << av[reim] << " B " << bv[reim] << " fracdiff " << diff_rat << std::endl;
	  if(!print_all) fail = 1.;
	}
      }
    }
  }
  glb_max(&fail);
  
  if(fail!=0.0){
    if(!UniqueID()){ printf("Failed %s check\n", descr.c_str()); fflush(stdout); } 
    exit(-1);
  }else{
    if(!UniqueID()){ printf("Passed %s check\n", descr.c_str()); fflush(stdout); }
  }
}








#ifdef USE_BFM
inline void exportBFMcb(CPSfermion5D<ComplexD> &into, Fermion_t from, bfm_evo<double> &dwf, int cb, bool singleprec_evec = false){
  Fermion_t zero_a = dwf.allocFermion();
#pragma omp parallel
  {   
    dwf.set_zero(zero_a); 
  }
  Fermion_t etmp = dwf.allocFermion(); 
  Fermion_t tmp[2];
  tmp[!cb] = zero_a;
  if(singleprec_evec){
    const int len = 24 * dwf.node_cbvol * (1 + dwf.gparity) * dwf.cbLs;
#pragma omp parallel for
    for(int j = 0; j < len; j++) {
      ((double*)etmp)[j] = ((float*)(from))[j];
    }
    tmp[cb] = etmp;
  }else tmp[cb] = from;

  dwf.cps_impexFermion(into.ptr(),tmp,0);
  dwf.freeFermion(zero_a);
  dwf.freeFermion(etmp);
}
#endif

#ifdef USE_GRID
template<typename GridPolicies>
inline void exportGridcb(CPSfermion5D<ComplexD> &into, typename GridPolicies::GridFermionField &from, typename GridPolicies::FgridFclass &latg){
  Grid::GridCartesian *FGrid = latg.getFGrid();
  typename GridPolicies::GridFermionField tmp_g(FGrid);
  tmp_g = Grid::zero;

  setCheckerboard(tmp_g, from);
  latg.ImportFermion((Vector*)into.ptr(), tmp_g);
}
#endif

#ifdef USE_QMP

//Cyclic permutation of *4D* CPSfield with std::complex type and FourDpolicy dimension policy
//Conventions are direction of *data flow*: For shift n in direction +1   f'(x) = f(x-\hat i)  so data is sent in the +x direction. 
template< typename mf_Complex, int SiteSize, typename FlavorPolicy, typename AllocPolicy>
void cyclicPermute(CPSfield<mf_Complex,SiteSize,FourDpolicy,FlavorPolicy,AllocPolicy> &to, const CPSfield<mf_Complex,SiteSize,FourDpolicy,FlavorPolicy,AllocPolicy> &from,
		   const int dir, const int pm, const int n,
		   typename my_enable_if< _equal<typename ComplexClassify<mf_Complex>::type, complex_double_or_float_mark>::value, const int>::type dummy = 0){
  if(&to == &from){
    if(n==0) return;    
    CPSfield<mf_Complex,SiteSize,FourDpolicy,FlavorPolicy,AllocPolicy> tmpfrom(from);
    return cyclicPermute(to,tmpfrom,dir,pm,n);
  }
  if(n == 0){
    to = from;
    return;
  }
  assert(n < GJP.NodeSites(dir));

  QMP_barrier();
  
  //Prepare face to send. If we send in the + direction we need to collect the slice starting {L-n ... L-1} (inclusive), and if we send in the - dir we collect the slice {0... n-1}
  int bsites = n; //sites on boundary
  int bsizes[4]; bsizes[dir] = n;
  int boff[4]; boff[dir] = (pm == 1 ? GJP.NodeSites(dir)-n : 0);
		 
  for(int i=0;i<4;i++)
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

#pragma omp parallel for
  for(int i=0;i<bsites;i++){
    int rem = i;
    int coor[4];
    coor[0] = rem % bsizes[0] + boff[0]; rem/=bsizes[0];
    coor[1] = rem % bsizes[1] + boff[1]; rem/=bsizes[1];
    coor[2] = rem % bsizes[2] + boff[2]; rem/=bsizes[2];
    coor[3] = rem + boff[3];

    mf_Complex const* site_ptr = from.site_ptr(coor);
    mf_Complex* bp = send_buf + i*SiteSize;
    memcpy(bp,site_ptr,SiteSize*sizeof(mf_Complex));
    if(nf == 2){
      site_ptr += flav_off;
      bp += halfbufsz;
      memcpy(bp,site_ptr,SiteSize*sizeof(mf_Complex));
    }
  }
  QMP_barrier();
 
  //Copy remaining sites from on-node data with shift
  int rsizes[4]; rsizes[dir] = GJP.NodeSites(dir) - n;
  int rsites = GJP.NodeSites(dir) - n;
  //if we sent in the + direction we need to shift the remaining L-n sites {0...L-n-1} forwards by n to make way for a new slice at the left side
  //if we sent in the - direction we need to shift the remaining L-n sites {n ... L-1} backwards by n to make way for a new slice at the right side
  
  int roff[4]; roff[dir] = (pm == 1 ? 0 : n);  
  for(int i=0;i<4;i++)
    if(i != dir){
      rsizes[i] = GJP.NodeSites(i);
      rsites *= rsizes[i];
      roff[i] = 0;
    }

#pragma omp parallel for
  for(int i=0;i<rsites;i++){
    int rem = i;
    int from_coor[4];
    from_coor[0] = rem % rsizes[0] + roff[0]; rem/=rsizes[0];
    from_coor[1] = rem % rsizes[1] + roff[1]; rem/=rsizes[1];
    from_coor[2] = rem % rsizes[2] + roff[2]; rem/=rsizes[2];
    from_coor[3] = rem + roff[3];
    
    int to_coor[4]; memcpy(to_coor,from_coor,4*sizeof(int));
    to_coor[dir] = (pm == +1 ? from_coor[dir] + n : from_coor[dir] - n);
    
    mf_Complex const* from_ptr = from.site_ptr(from_coor);
    mf_Complex * to_ptr = to.site_ptr(to_coor);

    memcpy(to_ptr,from_ptr,SiteSize*sizeof(mf_Complex));
    if(nf == 2){
      from_ptr += flav_off;
      to_ptr += flav_off;
      memcpy(to_ptr,from_ptr,SiteSize*sizeof(mf_Complex));
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
    QMP_error("Send failed in cyclicPermute: %s\n", QMP_error_string(send_status));
  QMP_status_t rcv_status = QMP_wait(recv);
  if (rcv_status != QMP_SUCCESS) 
    QMP_error("Receive failed in PassDataT: %s\n", QMP_error_string(rcv_status));

  //Copy received face into position. For + shift the origin we copy into is the left-face {0..n-1}, for a - shift its the right-face {L-n .. L-1}
  boff[dir] = (pm == 1 ? 0 : GJP.NodeSites(dir)-n);
#pragma omp parallel for
  for(int i=0;i<bsites;i++){
    int rem = i;
    int coor[4];
    coor[0] = rem % bsizes[0] + boff[0]; rem/=bsizes[0];
    coor[1] = rem % bsizes[1] + boff[1]; rem/=bsizes[1];
    coor[2] = rem % bsizes[2] + boff[2]; rem/=bsizes[2];
    coor[3] = rem + boff[3];

    mf_Complex * site_ptr = to.site_ptr(coor);
    mf_Complex const* bp = recv_buf + i*SiteSize;
    memcpy(site_ptr,bp,SiteSize*sizeof(mf_Complex));
    if(nf == 2){
      site_ptr += flav_off;
      bp += halfbufsz;
      memcpy(site_ptr,bp,SiteSize*sizeof(mf_Complex));
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

#ifdef USE_GRID

//Version with SIMD vectorized data
template< typename mf_Complex, int SiteSize, typename FlavorPolicy, typename AllocPolicy>
void cyclicPermute(CPSfield<mf_Complex,SiteSize,FourDSIMDPolicy,FlavorPolicy,AllocPolicy> &to, const CPSfield<mf_Complex,SiteSize,FourDSIMDPolicy,FlavorPolicy,AllocPolicy> &from,
		   const int dir, const int pm, const int n,
		   typename my_enable_if< _equal<typename ComplexClassify<mf_Complex>::type, grid_vector_complex_mark>::value, const int>::type dummy = 0){
  if(&to == &from){
    if(n==0) return;    
    CPSfield<mf_Complex,SiteSize,FourDSIMDPolicy,FlavorPolicy,AllocPolicy> tmpfrom(from);
    return cyclicPermute(to,tmpfrom,dir,pm,n);
  }
  if(n == 0){
    to = from;
    return;
  }
  assert(n < GJP.NodeSites(dir));

  const int nsimd = mf_Complex::Nsimd();
  
  //Use notation c (combined index), o (outer index) i (inner index)
  
  int bcsites = n; //sites on boundary
  int bcsizes[4]; bcsizes[dir] = n;
  int bcoff[4]; bcoff[dir] = (pm == 1 ? GJP.NodeSites(dir)-n : 0);
  int bcoff_postcomms[4]; bcoff_postcomms[dir] = (pm == 1 ? 0 : GJP.NodeSites(dir)-n);
  
  for(int i=0;i<4;i++)
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
  
#pragma omp parallel for
  for(int c=0;c<bcsites;c++){
    int rem = c;
    int coor[4];
    coor[0] = rem % bcsizes[0]; rem/=bcsizes[0];
    coor[1] = rem % bcsizes[1]; rem/=bcsizes[1];
    coor[2] = rem % bcsizes[2]; rem/=bcsizes[2];
    coor[3] = rem;

    int coor_dest[4];
    for(int d=0;d<4;d++){
      coor_dest[d] = coor[d] + bcoff_postcomms[d];
      coor[d] += bcoff[d];
    }
    
    int i = from.SIMDmap(coor);
    int o = from.siteMap(coor);

    int i_dest = from.SIMDmap(coor_dest);
    int o_dest = from.siteMap(coor_dest);

    Grid::Vector<scalarType> ounpacked(nsimd);
    for(int f=0;f<nf;f++){
      mf_Complex const *osite_ptr = from.site_ptr(o,f);
      int send_buf_off = (c + bcsites*f)*SiteSize;
      scalarType* bp = send_buf + send_buf_off;
      to_oi_buf_map[ i_dest + nsimd*(o_dest+osites*f) ] = send_buf_off;
      
      for(int s=0;s<SiteSize;s++){
	vstore(*(osite_ptr++), ounpacked.data());
	*(bp++) = ounpacked[i];
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
    QMP_error("Send failed in cyclicPermute: %s\n", QMP_error_string(send_status));
  QMP_status_t rcv_status = QMP_wait(recv);
  if (rcv_status != QMP_SUCCESS) 
    QMP_error("Receive failed in PassDataT: %s\n", QMP_error_string(rcv_status));


  
  //Copy remaining sites from on-node data with shift and pull in data from buffer simultaneously
  //if we sent in the + direction we need to shift the remaining L-n sites {0...L-n-1} forwards by n to make way for a new slice at the left side
  //if we sent in the - direction we need to shift the remaining L-n sites {n ... L-1} backwards by n to make way for a new slice at the right side
  //Problem is we don't want two threads writing to the same AVX register at the same time. Therefore we thread the loop over the destination SIMD vectors and work back
  std::vector< std::vector<int> > lane_offsets(nsimd,  std::vector<int>(4) );
  for(int i=0;i<nsimd;i++) from.SIMDunmap(i, lane_offsets[i].data() );

#pragma omp parallel for
  for(int oto = 0;oto < osites; oto++){
    int oto_base_coor[4]; to.siteUnmap(oto,oto_base_coor);

    //For each destination lane compute the source site index and lane
    int from_lane[nsimd];
    int from_osite_idx[nsimd]; //also use for recv_buf offsets for sites pulled over boundary
    for(int lane = 0; lane < nsimd; lane++){
      int offrom_coor[4];
      for(int d=0;d<4;d++) offrom_coor[d] = oto_base_coor[d] + lane_offsets[lane][d];
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
    Grid::Vector<scalarType> towrite(nsimd);
    Grid::Vector<scalarType> unpack(nsimd);
    
    for(int f=0;f<nf;f++){
      for(int s=0;s<SiteSize;s++){
	for(int tolane=0;tolane<nsimd;tolane++){	  
	  if(from_lane[tolane] != -1){
	    mf_Complex const* from_osite_ptr = from.site_ptr(from_osite_idx[tolane], f) + s;
	    vstore(*from_osite_ptr,unpack.data());
	    towrite[tolane] = unpack[ from_lane[tolane] ];
	  }else{
	    //data is in buffer
	    towrite[tolane] = recv_buf[ from_osite_idx[tolane] + f*bcsites*SiteSize ];
	  }
	    
	}
	mf_Complex* to_osite_ptr = to.site_ptr(oto,f) + s;
	vset(*to_osite_ptr, towrite.data());	
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

#endif





#endif


template<typename CPSfieldType>
void fft(CPSfieldType &into, const CPSfieldType &from, const bool* do_dirs,
	 typename my_enable_if<_equal<typename ComplexClassify<typename CPSfieldType::FieldSiteType>::type, complex_double_or_float_mark>::value, const int>::type = 0
	 ){
  typedef typename LocalToGlobalInOneDirMap<typename CPSfieldType::FieldDimensionPolicy>::type DimPolGlobalInOneDir;
  typedef CPSfieldGlobalInOneDir<typename CPSfieldType::FieldSiteType, CPSfieldType::FieldSiteSize, DimPolGlobalInOneDir, typename CPSfieldType::FieldFlavorPolicy, typename CPSfieldType::FieldAllocPolicy> CPSfieldTypeGlobalInOneDir;

  int dcount = 0;
  
  for(int mu=0;mu<CPSfieldType::FieldDimensionPolicy::EuclideanDimension;mu++)
    if(do_dirs[mu]){
      CPSfieldTypeGlobalInOneDir tmp_dbl(mu);
      tmp_dbl.gather( dcount==0 ? from : into );
      tmp_dbl.fft();
      tmp_dbl.scatter(into);
      dcount ++;
    }
}
template<typename CPSfieldType>
void fft(CPSfieldType &fftme, const bool* do_dirs,
	 typename my_enable_if<_equal<typename ComplexClassify<typename CPSfieldType::FieldSiteType>::type, complex_double_or_float_mark>::value, const int>::type = 0
	 ){
  fft(fftme,fftme,do_dirs);
}


CPS_END_NAMESPACE
#endif
