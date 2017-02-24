#ifndef CPS_FIELD_IMPL
#define CPS_FIELD_IMPL

//Implementations of CPSfield.h


//Real-reduce for norm2
template<typename T>
struct normdefs{};

template<>
struct normdefs<double>{
  inline static double real_reduce(const double in){ return in; }
  inline static double conjugate(const double in){ return in; }
};
template<>
struct normdefs<float>{
  inline static double real_reduce(const float in){ return in; }
  inline static float conjugate(const float in){ return in; }
};
template<typename T>
struct normdefs<std::complex<T> >{
  inline static double real_reduce(const std::complex<T> in){ return in.real(); }
  inline static std::complex<T> conjugate(const std::complex<T> in){ return std::conj(in); }
};
#ifdef USE_GRID
template<>
struct normdefs<Grid::vRealD>{
  inline static double real_reduce(const Grid::vRealD in){ return Reduce(in); }
  inline static Grid::vRealD conjugate(const Grid::vRealD in){ return in; }
};
template<>
struct normdefs<Grid::vRealF>{
  inline static double real_reduce(const Grid::vRealF in){ return Reduce(in); }
  inline static Grid::vRealF conjugate(const Grid::vRealF in){ return in; }
};
template<>
struct normdefs<Grid::vComplexD>{
  inline static double real_reduce(const Grid::vComplexD in){ return std::real(Reduce(in)); }
  inline static Grid::vComplexD conjugate(const Grid::vComplexD in){ return Grid::conjugate(in); }
};
template<>
struct normdefs<Grid::vComplexF>{
  inline static double real_reduce(const Grid::vComplexF in){ return std::real(Reduce(in)); }
  inline static Grid::vComplexF conjugate(const Grid::vComplexF in){ return Grid::conjugate(in); }
};
#endif

template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy>
double CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy>::norm2() const{
  SiteType accum[omp_get_max_threads()];
  memset(accum, 0, omp_get_max_threads()*sizeof(SiteType));
#pragma omp parallel for schedule(static)  
  for(int i=0;i<this->nfsites();i++){
    SiteType const *site = this->fsite_ptr(i);
    for(int s=0;s<SiteSize;s++)
      accum[omp_get_thread_num()] = accum[omp_get_thread_num()] + normdefs<SiteType>::conjugate(site[s])*site[s];
  }
  SiteType total;
  memset(&total, 0, sizeof(SiteType));

  for(int i=0;i<omp_get_max_threads();i++)
    total = total + accum[i];

  double final = normdefs<SiteType>::real_reduce(total);
  glb_sum_five(&final);
  return final;
}

template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy>
double CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy>::norm2(const IncludeSite<MappingPolicy::EuclideanDimension> & restrictsites) const{
  SiteType accum[omp_get_max_threads()];
  memset(accum, 0, omp_get_max_threads()*sizeof(SiteType));
#pragma omp parallel for schedule(static)  
  for(int i=0;i<this->nfsites();i++){
    SiteType const *site = this->fsite_ptr(i);
    int x[MappingPolicy::EuclideanDimension]; int f;
    this->fsiteUnmap(i,x,f);
    if(!restrictsites.query(x,f)) continue;
    for(int s=0;s<SiteSize;s++)
      accum[omp_get_thread_num()] = accum[omp_get_thread_num()] + normdefs<SiteType>::conjugate(site[s])*site[s];
  }
  SiteType total;
  memset(&total, 0, sizeof(SiteType));

  for(int i=0;i<omp_get_max_threads();i++)
    total = total + accum[i];

  double final = normdefs<SiteType>::real_reduce(total);
  glb_sum_five(&final);
  return final;
}

template<typename SiteType>
class _testRandom{
public:
  static void rand(SiteType* f, int fsize, const Float hi, const Float lo){
    for(int i=0;i<fsize;i++) f[i] = LRG.Urand(hi,lo,FOUR_D);
  }
};
template<typename T>
class _testRandom<std::complex<T> >{
public:
  static void rand(std::complex<T>* f, int fsize, const Float hi, const Float lo){
    assert(sizeof(std::complex<T>) == 2*sizeof(T));
    T* ff = (T*)f;
    for(int i=0;i<2*fsize;i++) ff[i] = LRG.Urand(hi,lo,FOUR_D);
  }
};


//Set each float to a uniform random number in the specified range.
//WARNING: Uses only the current RNG in LRG, and does not change this based on site. This is therefore only useful for testing*
template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy>
void CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy>::testRandom(const Float hi, const Float lo){
  _testRandom<SiteType>::rand(this->f,this->fsize,hi,lo);
}
template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy>
//Set this field to the average of this and a second field, r
void CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy>::average(const CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy> &r, const bool &parallel){
  //The beauty of having the ordering baked into the policy class is that we implicitly *know* the ordering of the second field, so we can just loop over the floats in a dumb way
  if(parallel){
#pragma omp parallel for
    for(int i=0;i<fsize;i++) f[i] = (f[i] + r.f[i])/2.0;
  }else{
    for(int i=0;i<fsize;i++) f[i] = (f[i] + r.f[i])/2.0;
  }
}


template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy, typename ComplexClass>
struct _gauge_fix_site_op_impl;
  
template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
struct _gauge_fix_site_op_impl<mf_Complex,MappingPolicy,AllocPolicy,complex_double_or_float_mark>{
  inline static void gauge_fix_site_op(CPSfermion<mf_Complex,MappingPolicy,AllocPolicy> &field, const int x4d[], const int &f, Lattice &lat, const bool dagger){
    typedef typename mf_Complex::value_type mf_Float;
    int i = x4d[0] + GJP.XnodeSites()*( x4d[1] + GJP.YnodeSites()* ( x4d[2] + GJP.ZnodeSites()*x4d[3] ) );
    mf_Complex tmp[3];
    const Matrix* gfmat = lat.FixGaugeMatrix(i,f);
    mf_Complex* sc_base = (mf_Complex*)field.site_ptr(x4d,f); //if Dimension < 4 the site_ptr method will ignore the remaining indices. Make sure this is what you want
    for(int s=0;s<4;s++){
      memcpy(tmp, sc_base + 3 * s, 3 * sizeof(mf_Complex));
      if(!dagger)
	colorMatrixMultiplyVector<mf_Float,Float>( (mf_Float*)(sc_base + 3*s), (Float*)gfmat, (mf_Float*)tmp);
      else
	colorMatrixDaggerMultiplyVector<mf_Float,Float>( (mf_Float*)(sc_base + 3*s), (Float*)gfmat, (mf_Float*)tmp);      
    }
  }
};


#ifdef USE_GRID
template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
struct _gauge_fix_site_op_impl<mf_Complex,MappingPolicy,AllocPolicy,grid_vector_complex_mark>{
  inline static void gauge_fix_site_op(CPSfermion<mf_Complex,MappingPolicy,AllocPolicy> &field, const int x4d[], const int &f, Lattice &lat, const bool dagger){
    //x4d is an outer site index
    int nsimd = field.Nsimd();
    int ndim = MappingPolicy::EuclideanDimension;
    assert(ndim == 4);

    //Assemble pointers to the GF matrices for each lane
    std::vector<cps::Complex*> gf_base_ptrs(nsimd);
    int x4d_lane[4];
    int lane_off[4];
    
    for(int lane=0;lane<nsimd;lane++){
      field.SIMDunmap(lane, lane_off);		      
      for(int xx=0;xx<4;xx++) x4d_lane[xx] = x4d[xx] + lane_off[xx];
      int gf_off = x4d_lane[0] + GJP.XnodeSites()*( x4d_lane[1] + GJP.YnodeSites()* ( x4d_lane[2] + GJP.ZnodeSites()*x4d_lane[3] ) );
      gf_base_ptrs[lane] = (cps::Complex*)lat.FixGaugeMatrix(gf_off,f);
    }


    //Poke the GFmatrix elements into SIMD vector objects
    typedef typename mf_Complex::scalar_type stype;
    stype* buf = (stype*)memalign(128, nsimd*sizeof(stype));

    mf_Complex gfmat[3][3];
    for(int i=0;i<3;i++){
      for(int j=0;j<3;j++){

	for(int lane=0;lane<nsimd;lane++)
	  buf[lane] = *(gf_base_ptrs[lane] + j + 3*i);
	vset(gfmat[i][j], buf);
      }
    }

    free(buf);

    //Do the matrix multiplication
    mf_Complex* tmp = (mf_Complex*)memalign(128, 3*sizeof(mf_Complex));
    mf_Complex* sc_base = field.site_ptr(x4d,f); 
    for(int s=0;s<4;s++){
      mf_Complex* s_base = sc_base + 3 * s;
      memcpy(tmp, s_base, 3 * sizeof(mf_Complex));
      if(!dagger)
	for(int i=0;i<3;i++)
	  s_base[i] = gfmat[i][0]*tmp[0] + gfmat[i][1]*tmp[1] + gfmat[i][2]*tmp[2];
      else
	for(int i=0;i<3;i++)
	  s_base[i] = conjugate(gfmat[0][i])*tmp[0] + conjugate(gfmat[1][i])*tmp[1] + conjugate(gfmat[2][i])*tmp[2];
    }
    free(tmp);
  }
};
#endif


//Apply gauge fixing matrices to the field
template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
void CPSfermion<mf_Complex,MappingPolicy,AllocPolicy>::gauge_fix_site_op(const int x4d[], const int &f, Lattice &lat, const bool dagger){
  _gauge_fix_site_op_impl<mf_Complex,MappingPolicy,AllocPolicy,typename ComplexClassify<mf_Complex>::type>::gauge_fix_site_op(*this, x4d, f, lat,dagger);
}

template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
void CPSfermion<mf_Complex,MappingPolicy,AllocPolicy>::getMomentumUnits(double punits[3]){
  for(int i=0;i<3;i++){
    int fac;
    if(GJP.Bc(i) == BND_CND_PRD) fac = 1;
    else if(GJP.Bc(i) == BND_CND_APRD) fac = 2;
    else if(GJP.Bc(i) == BND_CND_GPARITY) fac = 4;
    else{ ERR.General("CPSfermion","getMomentumUnits","Unknown boundary condition"); }

    punits[i] = 6.283185308/(GJP.NodeSites(i)*GJP.Nodes(i)*fac); // 2pi/(fac*L)
  }
}

//Apply the phase exp(-ip.x) to each site of this vector, where p is a *three momentum*
//The units of the momentum are 2pi/L for periodic BCs, pi/L for antiperiodic BCs and pi/2L for G-parity BCs
//x_lcl is the site in node lattice coords. 3 or more dimensions (those after 3 are ignored)

template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy, typename ComplexClass>
struct _apply_phase_site_op_impl{};

template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
struct _apply_phase_site_op_impl<mf_Complex,MappingPolicy,AllocPolicy,complex_double_or_float_mark>{
  inline static void apply_phase_site_op(CPSfermion<mf_Complex,MappingPolicy,AllocPolicy> &field, const int x_lcl[], const int &flav, const int p[], const double punits[]){
    StaticAssert<MappingPolicy::EuclideanDimension >= 3> check;
    
    int x_glb[MappingPolicy::EuclideanDimension]; for(int i=0;i<MappingPolicy::EuclideanDimension;i++) x_glb[i] = x_lcl[i] + GJP.NodeCoor(i)*GJP.NodeSites(i);
    
    double phi = 0;
    for(int i=0;i<3;i++) phi += p[i]*punits[i]*x_glb[i];
    std::complex<double> phase( cos(phi), -sin(phi) );
    mf_Complex phase_prec(phase);

    mf_Complex *base = field.site_ptr(x_lcl,flav);
    for(int sc=0;sc<12;sc++){
      mf_Complex* v = base + sc;
      (*v) *= phase_prec;
    }
  }
};

#ifdef USE_GRID
  
template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
struct _apply_phase_site_op_impl<mf_Complex,MappingPolicy,AllocPolicy,grid_vector_complex_mark>{
  inline static void apply_phase_site_op(CPSfermion<mf_Complex,MappingPolicy,AllocPolicy> &field, const int x_lcl[], const int &flav, const int p[], const double punits[]){
    StaticAssert<MappingPolicy::EuclideanDimension >= 3> check;

    int nsimd = field.Nsimd();

    typedef typename mf_Complex::scalar_type stype;
    stype* buf = (stype*)memalign(128, nsimd*sizeof(stype));

    int lane_off[MappingPolicy::EuclideanDimension];
    int x_gbl_lane[MappingPolicy::EuclideanDimension];
    
    for(int lane = 0; lane < nsimd; lane++){
      field.SIMDunmap(lane, lane_off);
      for(int xx=0;xx<MappingPolicy::EuclideanDimension;xx++) x_gbl_lane[xx] = x_lcl[xx] + lane_off[xx] + GJP.NodeCoor(xx)*GJP.NodeSites(xx);
      
      double phi = 0;
      for(int i=0;i<3;i++) phi += p[i]*punits[i]*x_gbl_lane[i];

      buf[lane] = stype( cos(phi), -sin(phi) );
    }

    mf_Complex vphase;
    vset(vphase, buf);
    free(buf);

    mf_Complex* base = field.site_ptr(x_lcl,flav);
    for(int sc=0;sc<12;sc++){
      mf_Complex* v = base + sc;
      *v = vphase * (*v);
    }
  }
};
#endif



template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
void CPSfermion<mf_Complex,MappingPolicy,AllocPolicy>::apply_phase_site_op(const int x_lcl[], const int &flav, const int p[], const double punits[]){
  _apply_phase_site_op_impl<mf_Complex,MappingPolicy,AllocPolicy,typename ComplexClassify<mf_Complex>::type>::apply_phase_site_op(*this, x_lcl, flav, p, punits);
}  


//Apply gauge fixing matrices to the field
template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
void CPSfermion4D<mf_Complex,MappingPolicy,AllocPolicy>::gauge_fix_site_op(int fi, Lattice &lat,const bool dagger){
  int x4d[4]; int f; this->fsiteUnmap(fi,x4d,f);
  CPSfermion<mf_Complex,MappingPolicy,AllocPolicy>::gauge_fix_site_op(x4d,f,lat,dagger);
}
template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
void CPSfermion4D<mf_Complex,MappingPolicy,AllocPolicy>::gaugeFix(Lattice &lat, const bool parallel, const bool dagger){
  if(parallel){
#pragma omp parallel for
    for(int fi=0;fi<this->nfsites();fi++)
      gauge_fix_site_op(fi,lat,dagger);
  }else{
    for(int fi=0;fi<this->nfsites();fi++)
      gauge_fix_site_op(fi,lat,dagger);
  }
}


//Apply the phase exp(-ip.x) to each site of this vector, where p is a *three momentum*
//The units of the momentum are 2pi/L for periodic BCs, pi/L for antiperiodic BCs and pi/2L for G-parity BCs
template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
void CPSfermion4D<mf_Complex,MappingPolicy,AllocPolicy>::apply_phase_site_op(int sf,const int p[],double punits[]){
  int x[MappingPolicy::EuclideanDimension]; int f; this->fsiteUnmap(sf,x,f);
  CPSfermion<mf_Complex,MappingPolicy,AllocPolicy>::apply_phase_site_op(x,f,p,punits);
}

template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
void CPSfermion4D<mf_Complex,MappingPolicy,AllocPolicy>::applyPhase(const int p[], const bool &parallel){
  const char *fname = "apply_phase(int p[])";

  double punits[3];
  CPSfermion<mf_Complex,MappingPolicy,AllocPolicy>::getMomentumUnits(punits);
  
  if(parallel){
#pragma omp parallel for
    for(int sf=0;sf<this->nfsites();sf++)
      apply_phase_site_op(sf,p,punits);
  }else{
    for(int sf=0;sf<this->nfsites();sf++)
      apply_phase_site_op(sf,p,punits);
  }
}


//Set the real and imaginary parts to uniform random numbers drawn from the appropriate local RNGs
template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
void CPSfermion4D<mf_Complex,MappingPolicy,AllocPolicy>::setUniformRandom(const Float &hi, const Float &lo){
  typedef typename mf_Complex::value_type mf_Float;
  LRG.SetInterval(hi,lo);
  for(int i = 0; i < this->nsites()*this->nflavors(); ++i) {
    int flav = i / this->nsites();
    int st = i % this->nsites();

    LRG.AssignGenerator(st,flav);
    mf_Float *p = (mf_Float*)this->site_ptr(st,flav);

    for(int site_lcl_off=0;site_lcl_off<2*FieldSiteSize;site_lcl_off++)
      *(p++) = LRG.Urand(FOUR_D);
  }
}

template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
void CPSfermion4D<mf_Complex,MappingPolicy,AllocPolicy>::setGaussianRandom(){
  typedef typename mf_Complex::value_type mf_Float;
  for(int i = 0; i < this->sites*this->flavors; ++i) {
    int flav = i / this->sites;
    int st = i % this->sites;

    LRG.AssignGenerator(st,flav);
    mf_Float *p = (mf_Float*)this->site_ptr(st,flav);

    for(int site_lcl_off=0;site_lcl_off<2*FieldSiteSize;site_lcl_off++)
      *(p++) = LRG.Grand(FOUR_D);
  }
}

template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
void CPSfermion5D<mf_Complex,MappingPolicy,AllocPolicy>::setGaussianRandom(){
  typedef typename mf_Complex::value_type mf_Float;
  for(int i = 0; i < this->nsites()*this->nflavors(); ++i) {
    int flav = i / this->nsites();
    int st = i % this->nsites();

    LRG.AssignGenerator(st,flav);
    mf_Float *p = (mf_Float*)this->site_ptr(st,flav);

    for(int site_lcl_off=0;site_lcl_off<2*FieldSiteSize;site_lcl_off++)
      *(p++) = LRG.Grand(FIVE_D);
  }
}


#ifdef USE_BFM

template<typename FloatExt,  typename mf_Complex, typename MappingPolicy, typename AllocPolicy, typename ComplexClass>
struct _bfm_fermion_impex{};

template<typename FloatExt,  typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
struct _bfm_fermion_impex<FloatExt,mf_Complex,MappingPolicy,AllocPolicy, complex_double_or_float_mark>{
  static void impexFermion(CPSfermion5D<mf_Complex,MappingPolicy,AllocPolicy> &cps_field, Fermion_t bfm_field, const int cb, const int do_import, bfm_qdp<FloatExt> &dwf){
    if(cps_field.nflavors() == 2) assert(dwf.gparity);

    const int sc_incr = dwf.simd() * 2; //stride between spin-color indices
    FloatExt * bb = (FloatExt*)bfm_field;

    typedef typename mf_Complex::value_type mf_Float;

#pragma omp parallel for
    for(int fs=0;fs<cps_field.nfsites();fs++){
      int x[5], f; cps_field.fsiteUnmap(fs, x, f);
      if( (x[0]+x[1]+x[2]+x[3] + (dwf.precon_5d ? x[4] : 0)) % 2 == cb){
	mf_Float* cps_base = (mf_Float*)cps_field.fsite_ptr(fs);

#ifdef USE_NEW_BFM_GPARITY
	int bidx_off = dwf.bagel_idx5d(x, x[4], 0, 0, 12, 1, f);
#else
	int bidx_off = dwf.gparity ? 
	  dwf.bagel_gparity_idx5d(x, x[4], 0, 0, 12, 1, f) :
	  dwf.bagel_idx5d(x, x[4], 0, 0, 12, 1);
#endif
	
	FloatExt * bfm_base = bb + bidx_off;

	for(int i=0;i<12;i++)
	  for(int reim=0;reim<2;reim++)
	    if(do_import) *(cps_base + 2*i + reim) = *(bfm_base + sc_incr*i + reim);	    
	    else *(bfm_base + sc_incr*i + reim) = *(cps_base + 2*i + reim);
      }
    }
  }
};




template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
template<typename FloatExt>
void CPSfermion5D<mf_Complex,MappingPolicy,AllocPolicy>::importFermion(const Fermion_t bfm_field, const int cb, bfm_qdp<FloatExt> &dwf){
  _bfm_fermion_impex<FloatExt,mf_Complex,MappingPolicy,AllocPolicy,typename ComplexClassify<mf_Complex>::type>::impexFermion(*this, const_cast<Fermion_t>(bfm_field), cb, 1, dwf);
}
template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
template<typename FloatExt>
void CPSfermion5D<mf_Complex,MappingPolicy,AllocPolicy>::exportFermion(const Fermion_t bfm_field, const int cb, bfm_qdp<FloatExt> &dwf) const{
  _bfm_fermion_impex<FloatExt,mf_Complex,MappingPolicy,AllocPolicy,typename ComplexClassify<mf_Complex>::type>::impexFermion(*const_cast<CPSfermion5D<mf_Complex,MappingPolicy,AllocPolicy>*>(this), const_cast<Fermion_t>(bfm_field), cb, 0, dwf);
}

#endif













//Gauge fix 3D fermion field with dynamic info type
template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
struct _ferm3d_gfix_impl{

  static void gaugeFix(CPSfermion3D<mf_Complex,MappingPolicy,AllocPolicy> &field, Lattice &lat, const typename GaugeFix3DInfo<typename MappingPolicy::FieldFlavorPolicy>::InfoType &t, const bool &parallel){
    if(GJP.Gparity() && field.nflavors() == 1) ERR.General("CPSfermion3D","gaugeFix(Lattice &, const int &, const bool &)","For one flavor fields with G-parity enabled, to gauge fix we need to know the flavor of this field\n");

#define LOOP								\
    for(int fi=0;fi<field.nfsites();fi++){				\
      int x4d[4]; int f; field.fsiteUnmap(fi,x4d,f);			\
      x4d[3] = t;							\
      field.CPSfermion<mf_Complex,MappingPolicy>::gauge_fix_site_op(x4d,f,lat); \
    }

    if(parallel){
#pragma omp parallel for
      LOOP;
    }else{
      LOOP;
    }
#undef LOOP
  }

};
//Partial specialization for one flavor. We must provide the flavor index for the gauge fixing matrix, i.e. the flavor that this field represents
template< typename mf_Complex, template<typename> typename DimensionPolicy, typename AllocPolicy>
struct _ferm3d_gfix_impl<mf_Complex,DimensionPolicy<FixedFlavorPolicy<1> >,AllocPolicy>{
  static void gaugeFix(CPSfermion3D<mf_Complex,DimensionPolicy<FixedFlavorPolicy<1> >,AllocPolicy> &field, Lattice &lat, const typename GaugeFix3DInfo<FixedFlavorPolicy<1> >::InfoType &time_flav, const bool &parallel){
    printf("_ferm3d_gfix_impl::gauge_fix with time=%d, flav=%d\n",time_flav.first,time_flav.second);
    typedef typename mf_Complex::value_type mf_Float;

#define SITE_OP								\
    int x4d[4]; field.siteUnmap(i,x4d);		\
    x4d[3] = time_flav.first;						\
    int gfmat_site = x4d[0] + GJP.XnodeSites()*( x4d[1] + GJP.YnodeSites()* ( x4d[2] + GJP.ZnodeSites()*x4d[3] )); \
    mf_Complex tmp[3];							\
    const Matrix* gfmat = lat.FixGaugeMatrix(gfmat_site,time_flav.second);	\
    mf_Complex* sc_base = field.site_ptr(x4d);			\
    for(int s=0;s<4;s++){						\
      memcpy(tmp, sc_base + 3 * s, 3 * sizeof(mf_Complex));		\
      colorMatrixMultiplyVector<mf_Float,Float>( (mf_Float*)(sc_base + 3*s), (Float*)gfmat, (mf_Float*)tmp); \
    }									

    if(parallel){
#pragma omp parallel for
      for(int i=0;i<field.nsites();i++){
	SITE_OP;
      }
    }else{
      for(int i=0;i<field.nsites();i++){
	SITE_OP;
      }
    }
#undef SITE_OP

  }


};


template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
void CPSfermion3D<mf_Complex,MappingPolicy,AllocPolicy>::gaugeFix(Lattice &lat, const typename GaugeFix3DInfo<typename MappingPolicy::FieldFlavorPolicy>::InfoType &t, const bool &parallel){
  _ferm3d_gfix_impl<mf_Complex,MappingPolicy,AllocPolicy>::gaugeFix(*this,lat,t,parallel);
}


//Apply the phase exp(-ip.x) to each site of this vector, where p is a *three momentum*
//The units of the momentum are 2pi/L for periodic BCs, pi/L for antiperiodic BCs and pi/2L for G-parity BCs
template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
void CPSfermion3D<mf_Complex,MappingPolicy,AllocPolicy>::apply_phase_site_op(const int &sf,const int p[],double punits[]){
  int x[MappingPolicy::EuclideanDimension]; int f; this->fsiteUnmap(sf,x,f);
  CPSfermion<mf_Complex,MappingPolicy,AllocPolicy>::apply_phase_site_op(x,f,p,punits);
}

template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
void CPSfermion3D<mf_Complex,MappingPolicy,AllocPolicy>::applyPhase(const int p[], const bool &parallel){
  const char *fname = "apply_phase(int p[])";

  double punits[3];
  CPSfermion<mf_Complex,MappingPolicy,AllocPolicy>::getMomentumUnits(punits);
  
  if(parallel){
#pragma omp parallel for
    for(int sf=0;sf<this->nfsites();sf++)
      apply_phase_site_op(sf,p,punits);
  }else{
    for(int sf=0;sf<this->nfsites();sf++)
      apply_phase_site_op(sf,p,punits);
  }
}



















//Make a random complex scalar field of type
template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
void CPScomplex4D<mf_Complex,MappingPolicy,AllocPolicy>::setRandom(const RandomType &type){
  LRG.SetInterval(1, 0);
  for(int i = 0; i < this->sites*this->nflavors(); ++i) {
    int flav = i / this->nsites();
    int st = i % this->nsites();

    LRG.AssignGenerator(st,flav);
    mf_Complex *p = this->site_ptr(st,flav);
    RandomComplex<mf_Complex>::rand(p,type,FOUR_D);
  }
}

//Set the real and imaginary parts to uniform random numbers drawn from the appropriate local RNGs
template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
void CPScomplex4D<mf_Complex,MappingPolicy,AllocPolicy>::setUniformRandom(const Float &hi, const Float &lo){
  typedef typename mf_Complex::value_type mf_Float;
  LRG.SetInterval(hi,lo);
  for(int i = 0; i < this->nsites()*this->nflavors(); ++i) {
    int flav = i / this->nsites();
    int st = i % this->nsites();

    LRG.AssignGenerator(st,flav);
    mf_Float *p = (mf_Float*)this->site_ptr(st,flav);

    for(int i=0;i<2;i++)
      *(p++) = LRG.Urand(FOUR_D);
  }
}

 
//Perform the FFT
template< typename mf_Complex, typename FlavorPolicy, typename AllocPolicy>
void CPSglobalComplexSpatial<mf_Complex,FlavorPolicy,AllocPolicy>::fft(){
  typedef typename mf_Complex::value_type mf_Float;
  const int fft_dim[3] = {this->glb_size[2], this->glb_size[1], this->glb_size[0]};
  const int size_3d_glb = fft_dim[0] * fft_dim[1] * fft_dim[2];

  size_t this_floatsize = this->size() * 2;
  
  typename FFTWwrapper<mf_Float>::complexType* fft_mem = FFTWwrapper<mf_Float>::alloc_complex(this_floatsize);
  
  memcpy((void *)fft_mem, this->ptr(), this_floatsize*sizeof(mf_Float));

  //Plan creation is expensive, so make it static
  static typename FFTWwrapper<mf_Float>::planType plan_src;
  static bool init = false;
  if(!init){
    plan_src = FFTWwrapper<mf_Float>::plan_many_dft(3, fft_dim, 1,
						    fft_mem, NULL, 1, size_3d_glb,
						    fft_mem, NULL, 1, size_3d_glb,
						    FFTW_FORWARD, FFTW_ESTIMATE);
    init = true;
  }

  for(int f = 0; f < this->nflavors(); f++) {
    int off = f * size_3d_glb;
    FFTWwrapper<mf_Float>::execute_dft(plan_src, fft_mem + off, fft_mem + off);
  }

  memcpy((void *)this->ptr(), (void*)fft_mem, this_floatsize*sizeof(mf_Float));
  FFTWwrapper<mf_Float>::free(fft_mem);

  //FFTWwrapper<mf_Float>::cleanup(); //Don't need to cleanup, it doesn't have the function I initially thought
}
  
  

//Scatter to a local field

template< typename mf_Complex, typename FlavorPolicy, typename AllocPolicy,
	  typename extComplex, typename extMapPolicy, typename extAllocPolicy,
	  typename complex_class, int extEuclDim, typename dummy>
struct _CPSglobalComplexSpatial_scatter_impl{};

//Standard implementation for std::complex
template< typename mf_Complex, typename FlavorPolicy, typename AllocPolicy,
	  typename extComplex, typename extMapPolicy, typename extAllocPolicy, typename dummy>
struct _CPSglobalComplexSpatial_scatter_impl<mf_Complex,FlavorPolicy,AllocPolicy,  extComplex, extMapPolicy, extAllocPolicy, complex_double_or_float_mark, 3, dummy>{
  static void doit(CPSfield<extComplex,1,extMapPolicy,extAllocPolicy> &to, const CPSglobalComplexSpatial<mf_Complex,FlavorPolicy,AllocPolicy> &from){
    const char *fname = "scatter(...)";
    int orig[3]; for(int i=0;i<3;i++) orig[i] = GJP.NodeSites(i)*GJP.NodeCoor(i);

#pragma omp parallel for
    for(int i=0;i<to.nfsites();i++){
      int x[3]; int flavor;  to.fsiteUnmap(i,x,flavor); //unmap the target coordinate
      for(int j=0;j<3;j++) x[j] += orig[j]; //global coord

      extComplex* tosite = to.fsite_ptr(i);
      mf_Complex const* fromsite = from.site_ptr(x,flavor);

      *tosite = *fromsite;
    }	
  }
};

#ifdef USE_GRID

//Implementation for Grid vector complex types
template< typename mf_Complex, typename FlavorPolicy, typename AllocPolicy,
	  typename extComplex, typename extMapPolicy, typename extAllocPolicy, typename dummy>
struct _CPSglobalComplexSpatial_scatter_impl<mf_Complex,FlavorPolicy,AllocPolicy,  extComplex, extMapPolicy, extAllocPolicy, grid_vector_complex_mark, 3, dummy>{
  static void doit(CPSfield<extComplex,1,extMapPolicy,extAllocPolicy> &to, const CPSglobalComplexSpatial<mf_Complex,FlavorPolicy,AllocPolicy> &from){
    const char *fname = "scatter(...)";
    int orig[3]; for(int i=0;i<3;i++) orig[i] = GJP.NodeSites(i)*GJP.NodeCoor(i);

    const int ndim = 3;
    int nsimd = extComplex::Nsimd();
    std::vector<std::vector<int> > packed_offsets(nsimd,std::vector<int>(ndim)); //get the vector offsets for the different SIMD packed sites
    for(int i=0;i<nsimd;i++) to.SIMDunmap(i,&packed_offsets[i][0]);
    
#pragma omp parallel for
    for(int i=0;i<to.nfsites();i++){
      int x[3]; int flavor;  to.fsiteUnmap(i,x,flavor); //unmap the target coordinate. This is a root coordinate, we need to construct the other offsets
      for(int j=0;j<3;j++) x[j] += orig[j]; //global coord

      extComplex* toptr = to.fsite_ptr(i); 

      //x is the root coordinate corresponding to SIMD packed index 0      
      std::vector<mf_Complex const*> ptrs(nsimd);
      ptrs[0] = from.site_ptr(x,flavor);
      
      int xx[ndim];
      for(int i=1;i<nsimd;i++){
	for(int d=0;d<ndim;d++)
	  xx[d] = x[d] + packed_offsets[i][d];  //xx = x + offset
	ptrs[i] = from.site_ptr(xx,flavor);
      }
      to.SIMDpack(toptr, ptrs, 1);
    }	
  }
};

#endif


template< typename mf_Complex, typename FlavorPolicy, typename AllocPolicy>
template<typename extComplex, typename extMapPolicy, typename extAllocPolicy>
void CPSglobalComplexSpatial<mf_Complex,FlavorPolicy,AllocPolicy>::scatter(CPSfield<extComplex,1,extMapPolicy,extAllocPolicy> &to) const{
  _CPSglobalComplexSpatial_scatter_impl<mf_Complex,FlavorPolicy,AllocPolicy,
					extComplex,extMapPolicy,extAllocPolicy,
					typename ComplexClassify<extComplex>::type,
					extMapPolicy::EuclideanDimension,
					typename my_enable_if<_equal<typename extMapPolicy::FieldFlavorPolicy,FlavorPolicy>::value, int>::type
					>::doit(to,*this);
}


//When the local field has the right dimension but is not the same as the equivalent local dimension policy, do an intermediate impex
template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy,
	  typename extSiteType, typename extMapPol, typename extAllocPol,
	  typename my_enable_if<intEq<MappingPolicy::EuclideanDimension,extMapPol::EuclideanDimension>::val && _equal<typename MappingPolicy::FieldFlavorPolicy,typename extMapPol::FieldFlavorPolicy>::value, int>::type = 0>
struct _gather_scatter_impl{
  typedef typename MappingPolicy::EquivalentLocalPolicy EquivalentLocalPolicy;
  
  static void gather(CPSfieldGlobalInOneDir<SiteType,SiteSize,MappingPolicy,AllocPolicy> &into, const CPSfield<extSiteType,SiteSize,extMapPol,extAllocPol> &from){
    NullObject n;
    CPSfield<SiteType,SiteSize,EquivalentLocalPolicy,AllocPolicy> tmp(n);
    tmp.importField(from);
    _gather_scatter_impl<SiteType,SiteSize,MappingPolicy,AllocPolicy,
			 SiteType, EquivalentLocalPolicy, AllocPolicy>::gather(into, tmp);    
  }
  static void scatter(CPSfield<extSiteType,SiteSize,extMapPol,extAllocPol> &to, const CPSfieldGlobalInOneDir<SiteType,SiteSize,MappingPolicy,AllocPolicy> &from){
    NullObject n;
    CPSfield<SiteType,SiteSize,EquivalentLocalPolicy,AllocPolicy> tmp(n);
    _gather_scatter_impl<SiteType,SiteSize,MappingPolicy,AllocPolicy,
			 SiteType, EquivalentLocalPolicy, AllocPolicy>::scatter(tmp, from);
    to.importField(tmp);
  }
    
};

//When the local field has DimensionPolicy equal to the EquivalentLocalPolicy
template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy,
	  typename my_enable_if<intEq<MappingPolicy::EuclideanDimension,MappingPolicy::EquivalentLocalPolicy::EuclideanDimension>::val, int>::type test>
struct _gather_scatter_impl<SiteType,SiteSize,MappingPolicy,AllocPolicy,
		    SiteType, typename MappingPolicy::EquivalentLocalPolicy, AllocPolicy, test>{
  typedef typename MappingPolicy::EquivalentLocalPolicy LocalMappingPolicy;

  static void gather(CPSfieldGlobalInOneDir<SiteType,SiteSize,MappingPolicy,AllocPolicy> &into, const CPSfield<SiteType,SiteSize,LocalMappingPolicy,AllocPolicy> &from){
    const int dir = into.getDir();

    const char *fname = "gather(...)";
    NullObject nullobj;
    CPSfield<SiteType,SiteSize,LocalMappingPolicy,AllocPolicy> tmp1(nullobj);
    CPSfield<SiteType,SiteSize,LocalMappingPolicy,AllocPolicy> tmp2(nullobj);
    CPSfield<SiteType,SiteSize,LocalMappingPolicy,AllocPolicy>* send = const_cast<CPSfield<SiteType,SiteSize,LocalMappingPolicy,AllocPolicy>* >(&from);
    CPSfield<SiteType,SiteSize,LocalMappingPolicy,AllocPolicy>* recv = &tmp2;

    int cur_dir_origin = GJP.NodeSites(dir)*GJP.NodeCoor(dir);    
    int size_in_Float = from.size() * sizeof(SiteType) / sizeof(IFloat); //getPlusData measures the send/recv size in units of sizeof(IFloat)

    int nshift = GJP.Nodes(dir);

    for(int shift = 0; shift < nshift; shift++){
#pragma omp parallel for
      for(int i=0;i<send->nfsites();i++){
	int x[MappingPolicy::EuclideanDimension]; int flavor;  send->fsiteUnmap(i,x,flavor); //unmap the buffer coordinate
	x[dir] += cur_dir_origin; //now a global coordinate in the dir direction

	SiteType* tosite = into.site_ptr(x,flavor);
	SiteType* fromsite = send->fsite_ptr(i);

	memcpy((void*)tosite, (void*)fromsite, into.siteSize()*sizeof(SiteType));
      }	

      if(shift != nshift-1){
	getPlusData((IFloat*)recv->ptr(), (IFloat*)send->ptr(), size_in_Float, dir);
	cur_dir_origin += GJP.NodeSites(dir);
	cur_dir_origin %= (GJP.NodeSites(dir)*GJP.Nodes(dir));

	if(shift == 0){
	  recv = &tmp1;
	  send = &tmp2;
	}else std::swap(send,recv);
      }
    }    
  }

  static void scatter(CPSfield<SiteType,SiteSize,LocalMappingPolicy,AllocPolicy> &to, const CPSfieldGlobalInOneDir<SiteType,SiteSize,MappingPolicy,AllocPolicy> &from){
    const int dir = from.getDir();
    
    const char *fname = "scatter(...)";
    int cur_dir_origin = GJP.NodeSites(dir)*GJP.NodeCoor(dir);

#pragma omp parallel for
    for(int i=0;i<to.nfsites();i++){
      int x[MappingPolicy::EuclideanDimension]; int flavor;  to.fsiteUnmap(i,x, flavor); //unmap the target coordinate
      x[dir] += cur_dir_origin; //now a global coordinate in the dir direction
      
      SiteType* tosite = to.fsite_ptr(i);
      SiteType const* fromsite = from.site_ptr(x,flavor);
      
      memcpy((void*)tosite, (void*)fromsite, from.siteSize()*sizeof(SiteType));
    }
  }
  
};




//Gather up the row. Involves internode communication
template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy>
template<typename extSiteType, typename extMapPol, typename extAllocPol>
void CPSfieldGlobalInOneDir<SiteType,SiteSize,MappingPolicy,AllocPolicy>::gather(const CPSfield<extSiteType,SiteSize,extMapPol,extAllocPol> &from){
  _gather_scatter_impl<SiteType,SiteSize,MappingPolicy,AllocPolicy,
	       extSiteType, extMapPol, extAllocPol>::gather(*this, from);  
}


//Scatter back out. Involves no communication
template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy>
template<typename extSiteType, typename extMapPol, typename extAllocPol>
void CPSfieldGlobalInOneDir<SiteType,SiteSize,MappingPolicy,AllocPolicy>::scatter(CPSfield<extSiteType,SiteSize,extMapPol,extAllocPol> &to) const{
  _gather_scatter_impl<SiteType,SiteSize,MappingPolicy,AllocPolicy,
		       extSiteType, extMapPol, extAllocPol>::scatter(to, *this);
}

#define FFT_MULTI

#ifndef FFT_MULTI

//Perform a fast Fourier transform along the principal direction
//NOTE: This won't work correctly if the DimensionPolicy does not use canonical ordering: FIXME
//Assumes SiteType is a std::complex type
template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy>
void CPSfieldGlobalInOneDir<SiteType,SiteSize,MappingPolicy,AllocPolicy>::fft(const bool inverse_transform){
  const int dir = this->getDir();
  const char* fname = "fft()";
  
  //We do a large number of simple linear FFTs. This field has its principal direction as the fastest changing index so this is nice and easy
  int sc_size = this->siteSize(); //we have to assume the sites comprise complex numbers
  int size_1d_glb = GJP.NodeSites(dir) * GJP.Nodes(dir);
  const int n_fft = this->nsites() / GJP.NodeSites(dir) * sc_size * this->nflavors();

  //Plan creation is expensive, so make it static and only re-create if the field size changes
  //Create a plan for each direction because we can have non-cubic spatial volumes
  static FFTplanContainer<typename SiteType::value_type> plan_f[4];
  static bool plan_init = false;
  static int plan_sc_size;
  static bool plan_inv_trans;
  if(!plan_init || sc_size != plan_sc_size || inverse_transform != plan_inv_trans){ //recreate/create
    typename FFTWwrapper<typename SiteType::value_type>::complexType *tmp_f; //I don't think it actually does anything with this

    for(int i=0;i<4;i++){
      int size_i = GJP.NodeSites(i) * GJP.Nodes(i);

      plan_f[i].setPlan(1, &size_i, 1, 
			tmp_f, NULL, sc_size, size_i * sc_size,
			tmp_f, NULL, sc_size, size_i * sc_size,
			inverse_transform ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE);  
    }
    plan_sc_size = sc_size;
    plan_inv_trans = inverse_transform;
    plan_init = true;
  }
    
  typename FFTWwrapper<typename SiteType::value_type>::complexType *fftw_mem = FFTWwrapper<typename SiteType::value_type>::alloc_complex(size_1d_glb * n_fft);
    
  memcpy((void *)fftw_mem, this->ptr(), this->size()*sizeof(SiteType));
#pragma omp parallel for
  for(int n = 0; n < n_fft; n++) {
    int sc_id = n % sc_size;
    int chunk_id = n / sc_size; //3d block index
    int off = size_1d_glb * sc_size * chunk_id + sc_id;
    FFTWwrapper<typename SiteType::value_type>::execute_dft(plan_f[dir].getPlan(), fftw_mem + off, fftw_mem + off); 
  }

  //FFTWwrapper<SiteType>::cleanup(); //I think this actually destroys existing plans!

  if(!inverse_transform) memcpy(this->ptr(), (void *)fftw_mem, this->size()*sizeof(SiteType));
  else for(int i=0;i<this->size();i++) this->ptr()[i] = *( (SiteType*)fftw_mem+i )/double(size_1d_glb);
  
  FFTWwrapper<typename SiteType::value_type>::free(fftw_mem);
}

#else

template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy>
void CPSfieldGlobalInOneDir<SiteType,SiteSize,MappingPolicy,AllocPolicy>::fft(const bool inverse_transform){
  const int dir = this->getDir();
  const char* fname = "fft()";
  
  //We do a large number of simple linear FFTs. This field has its principal direction as the fastest changing index so this is nice and easy
  int sc_size = this->siteSize(); //we have to assume the sites comprise complex numbers
  int size_1d_glb = GJP.NodeSites(dir) * GJP.Nodes(dir);
  const int n_fft = this->nsites() / GJP.NodeSites(dir) * this->nflavors();

  //Plan creation is expensive, so make it static and only re-create if the field size changes
  //Create a plan for each direction because we can have non-cubic spatial volumes
  static FFTplanContainer<typename SiteType::value_type> plan_f[4];
  static bool plan_init = false;
  static int plan_sc_size;
  static bool plan_inv_trans;
  if(!plan_init || sc_size != plan_sc_size || inverse_transform != plan_inv_trans){ //recreate/create
    typename FFTWwrapper<typename SiteType::value_type>::complexType *tmp_f; //I don't think it actually does anything with this

    for(int i=0;i<4;i++){
      int size_i = GJP.NodeSites(i) * GJP.Nodes(i);

      plan_f[i].setPlan(1, &size_i, sc_size, 
			tmp_f, NULL, sc_size, 1,
			tmp_f, NULL, sc_size, 1,
			inverse_transform ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE);  
    }
    plan_sc_size = sc_size;
    plan_inv_trans = inverse_transform;
    plan_init = true;
  }

  typename FFTWwrapper<typename SiteType::value_type>::complexType *fftw_mem = FFTWwrapper<typename SiteType::value_type>::alloc_complex(size_1d_glb * n_fft * sc_size);
    
  memcpy((void *)fftw_mem, this->ptr(), this->size()*sizeof(SiteType));
#pragma omp parallel for
  for(int n = 0; n < n_fft; n++) {
    int chunk_id = n; //3d block index
    int off = size_1d_glb * sc_size * chunk_id;
    FFTWwrapper<typename SiteType::value_type>::execute_dft(plan_f[dir].getPlan(), fftw_mem + off, fftw_mem + off); 
  }

  if(!inverse_transform) memcpy(this->ptr(), (void *)fftw_mem, this->size()*sizeof(SiteType));
  else for(int i=0;i<this->size();i++) this->ptr()[i] = *( (SiteType*)fftw_mem+i )/double(size_1d_glb);
  
  FFTWwrapper<typename SiteType::value_type>::free(fftw_mem);
}

#endif

#endif
