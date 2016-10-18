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
  assert(&to != &from);
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
  
  //Send/receive (note QMP direction convention opposite to mine)
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

#endif





CPS_END_NAMESPACE
#endif
