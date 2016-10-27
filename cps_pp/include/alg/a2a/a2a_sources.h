#ifndef _A2A_SOURCES_H
#define _A2A_SOURCES_H

CPS_START_NAMESPACE

//Spatial source structure in *momentum-space*. Should assign the same value to both flavors if G-parity

//3D complex field. Defined for a *single flavor* if GPBC
template<typename mf_Complex,typename DimensionPolicy = SpatialPolicy, typename FieldAllocPolicy = StandardAllocPolicy, typename my_enable_if<DimensionPolicy::EuclideanDimension == 3, int>::type = 0>
class A2Asource{
public:
  typedef CPSfield<mf_Complex,1,DimensionPolicy,OneFlavorPolicy,FieldAllocPolicy> FieldType;  
protected:
  FieldType *src;
public:
  A2Asource(const typename FieldType::InputParamType &params){
    setup(params);
  }
  inline void setup(const typename FieldType::InputParamType &params){
    src = new FieldType(params);
  }
  
  A2Asource(): src(NULL){
  }
  ~A2Asource(){
    if(src != NULL) delete src;
  }

  
  inline const mf_Complex & siteComplex(const int site) const{ return *src->site_ptr(site); }
  inline const int nsites() const{ return src->nsites(); }

  template< typename extComplexType, typename extDimPol, typename extAllocPol>
  void importSource(const A2Asource<extComplexType,extDimPol,extAllocPol> &from){
    src->importField(*from.src);
  }
  FieldType & getSource(){ return *src; } //For testing

  inline static int pmod(const int x, const int Lx){
    //return x >= Lx/2 ? x : Lx-x;
    return (x + Lx/2) % Lx - Lx/2; //same as above
  }
  
  static Float pmodr(const int r[3], const int glb_size[3]){
    Float ssq =0.;
    for(int i=0;i<3;i++){
      int sr = pmod(r[i],glb_size[i]);
      ssq += sr*sr;
    }
    return sqrt(ssq);
  }
};


//Use CRTP for 'setSite' method which should be specialized according to the source type
template<typename FieldPolicies, typename Child>
class A2AsourceBase: public A2Asource<typename FieldPolicies::ComplexType, typename FieldPolicies::DimensionPolicy, typename FieldPolicies::AllocPolicy>{
public:
  typedef FieldPolicies Policies;
  typedef typename A2Asource<typename FieldPolicies::ComplexType, typename FieldPolicies::DimensionPolicy, typename FieldPolicies::AllocPolicy>::FieldType::InputParamType FieldParamType;
  
  A2AsourceBase(const FieldParamType &p): A2Asource<typename FieldPolicies::ComplexType, typename FieldPolicies::DimensionPolicy, typename FieldPolicies::AllocPolicy>(p){};
  A2AsourceBase(): A2Asource<typename FieldPolicies::ComplexType, typename FieldPolicies::DimensionPolicy, typename FieldPolicies::AllocPolicy>(){}; //SOURCE IS NOT SETUP
  
  void fft_source(){
    assert(this->src != NULL);
    int glb_size[3]; for(int i=0;i<3;i++) glb_size[i] = GJP.Nodes(i)*GJP.NodeSites(i);

    //Generate a global 4d source
    CPSglobalComplexSpatial<cps::ComplexD,OneFlavorPolicy> glb; //always of this type
    glb.zero();
         
#pragma omp_parallel for
    for(int i=0;i<glb.nsites();i++){
      int x[3]; glb.siteUnmap(i,x); 
      *glb.site_ptr(i) = static_cast<Child const*>(this)->value(x,glb_size);
    }
    //Perform the FFT and pull out this nodes subvolume
    glb.fft();
    glb.scatter<typename FieldPolicies::ComplexType, typename FieldPolicies::DimensionPolicy, typename FieldPolicies::AllocPolicy>(*this->src);
  }
};


//Exponential (hydrogen wavefunction) source
//SrcParams is just a Float for the radius
template<typename FieldPolicies = StandardSourcePolicies>
class A2AexpSource: public A2AsourceBase<FieldPolicies, A2AexpSource<FieldPolicies> >{
  Float radius;

public:
  typedef FieldPolicies Policies;
  typedef typename A2AsourceBase<FieldPolicies, A2AexpSource<FieldPolicies> >::FieldParamType FieldParamType;
  typedef typename Policies::ComplexType ComplexType;

  inline ComplexD value(const int site[3], const int glb_size[3]) const{
    Float v = pmodr(site,glb_size)/radius;
    v = exp(-v)/(glb_size[0]*glb_size[1]*glb_size[2]);
    return ComplexD(v,0);
  }
    
  A2AexpSource(const Float _radius, const FieldParamType &field_params): radius(_radius), A2AsourceBase<FieldPolicies, A2AexpSource<FieldPolicies> >(field_params){
    this->fft_source();
  }
  A2AexpSource(const Float _radius): radius(_radius), A2AsourceBase<FieldPolicies, A2AexpSource<FieldPolicies> >(NullObject()){
    this->fft_source();
  } //syntactic sugar to avoid having to provide a NullObject instance where appropriate

  A2AexpSource(): radius(0.), A2AsourceBase<FieldPolicies, A2AexpSource<FieldPolicies> >(){} //src is not setup

  //Setup the source if the default constructor was used
  void setup(const Float _radius, const FieldParamType &field_params){
    this->A2AsourceBase<FieldPolicies, A2AexpSource<FieldPolicies> >::setup(field_params);
    radius = _radius;
    this->fft_source();
  }
  void setup(const Float radius){
    return setup(radius, NullObject());
  }
    
  inline void siteFmat(FlavorMatrixGeneral<typename Policies::ComplexType> &out, const int site) const{
    out(0,0) = out(1,1) = this->siteComplex(site);
    out(0,1) = out(1,0) = typename Policies::ComplexType(0);    
  }
};

//Box source. Unflavored so ignore second flav
//SrcParams is std::vector<Float> for the extents x,y,z . *These must be even numbers* (checked)
template<typename FieldPolicies = StandardSourcePolicies>
class A2AboxSource: public A2AsourceBase<FieldPolicies, A2AboxSource<FieldPolicies> >{
  int box_size[3];
public:
  typedef FieldPolicies Policies;
  typedef typename A2AsourceBase<FieldPolicies, A2AboxSource<FieldPolicies> >::FieldParamType FieldParamType;
  typedef typename Policies::ComplexType ComplexType;
  
  ComplexD value(const int site[3], const int glb_size[3]) const{
    bool inbox = true;
    int V = glb_size[0]*glb_size[1]*glb_size[2];
    for(int i=0;i<3;i++){ 
      int bdist = pmod(site[i],glb_size[i]);
      
      if(bdist > box_size[i]){
	inbox = false; break;
      }
    }
    if(inbox)
      return ComplexD(1./V);
  }
  
  A2AboxSource(const int _box_size[3],const FieldParamType &field_params): A2AsourceBase<FieldPolicies, A2AboxSource<FieldPolicies> >(field_params){
    this->setup(_box_size);
  }
  A2AboxSource(const int _box_size[3]): A2AsourceBase<FieldPolicies, A2AboxSource<FieldPolicies> >(NullObject()){
    this->setup(_box_size);
  }//syntatic sugar to avoid creating a NullObject

  void setup(const int _box_size[3]){
    for(int i=0;i<3;i++){
      if(_box_size[i] % 2 == 1){
	ERR.General("A2AboxSource","A2AboxSource","box size must be multiple of 2");
      }
      box_size[i] = _box_size[i];
    }
    this->fft_source();
  }

  
  inline void siteFmat(FlavorMatrixGeneral<typename Policies::ComplexType> &out, const int site) const{
    out(0,0) = out(1,1) = this->siteComplex(site);
    out(0,1) = out(1,0) = typename Policies::ComplexType(0);    
  }
};

//Splat a cps::ComplexD onto a SIMD type. Just a plain copy for non-SIMD complex types
#ifdef USE_GRID
template<typename ComplexType>
inline void SIMDsplat(ComplexType &to, const cps::ComplexD &from, typename my_enable_if< _equal<  typename ComplexClassify<ComplexType>::type, grid_vector_complex_mark  >::value, int>::type = 0){
  return vsplat(to,from);
}
#endif
template<typename ComplexType>
inline void SIMDsplat(ComplexType &to, const cps::ComplexD &from, typename my_enable_if< !_equal<  typename ComplexClassify<ComplexType>::type, grid_vector_complex_mark  >::value, int>::type = 0){
  return to = ComplexType(from.real(),from.imag());
}
  

//Daiqian's original implementation sets the (1 +/- sigma_2) flavor projection on G-parity fields to unity when the two fermion fields coincide.
//I'm not sure this is actually necessary, but I need to be able to reproduce his numbers
template<typename SourceType>
class A2AflavorProjectedSource{
public:
  typedef typename SourceType::FieldParamType FieldParamType;
  typedef typename SourceType::Policies::ComplexType ComplexType;
protected:
  int sign;

  //Derived class should setup the sources
  SourceType src;
  ComplexType val000;
public:

  //Assumes momenta are in units of \pi/2L, and must be *odd integer* (checked)
  inline static int getProjSign(const int p[3]){
    if(!GJP.Gparity()){ ERR.General("A2AflavorProjectedSource","getProjSign","Requires GPBC in at least one direction\n"); }

    //Sign is exp(i\pi n_p)
    //where n_p is the solution to  p_j = \pi/2L( 1 + 2n_p )
    //Must be consistent for all j

    int np;
    for(int j=0;j<3;j++){
      if(GJP.Bc(j)!=BND_CND_GPARITY) continue;

      if(abs(p[j]) %2 != 1){ ERR.General("A2AflavorProjectedSource","getProjSign","Component %d of G-parity momentum (%d,%d,%d) is invalid as it is not an odd integer!\n",j,p[0],p[1],p[2]); }
      int npj = (p[j] - 1)/2;
      if(j == 0) np = npj;
      else if(abs(npj)%2 != abs(np)%2){ 
	ERR.General("A2AflavorProjectedSource","getProjSign","Momentum component %d of G-parity momentum (%d,%d,%d) is invalid because it doesn't differ from component 0 by multiple of 2pi (4 in these units). Got np(0)=%d, np(j)=%d\n",j,p[0],p[1],p[2],np,npj); 
      }
    }
    int sgn = (abs(np) % 2 == 0 ? 1 : -1); //exp(i\pi n_p) = exp(-i\pi n_p)  for all integer n_p
    if(!UniqueID()){ printf("A2AflavorProjectedSource::getProjSign got sign %d (np = %d) for p=(%d,%d,%d)pi/2L\n",sgn,np,p[0],p[1],p[2]); fflush(stdout); }

    return sgn;
  }
  A2AflavorProjectedSource(const int p[3]): sign(getProjSign(p)){
    int zero[3] = {0,0,0}; int L[3] = {GJP.NodeSites(0)*GJP.Nodes(0), GJP.NodeSites(1)*GJP.Nodes(1), GJP.NodeSites(2)*GJP.Nodes(2) };
    cps::ComplexD v = src.value(zero,L);
    SIMDsplat(val000,v);    
  }

  int nsites() const{ return src.nsites(); }
  
  inline void siteFmat(FlavorMatrixGeneral<ComplexType> &out, const int site) const{
    //Matrix is FFT of  (1 + [sign]*sigma_2) when |x-y| !=0 or 1 when |x-y| == 0
    //It is always 1 on the diagonals
    const ComplexType &val = src.siteComplex(site);
    
    out(0,0) = out(1,1) = val;
    //and has \pm i on the diagonals with a momentum structure that is computed by omitting site 0,0,0
    out(1,0) = multiplySignTimesI(sign,val - val000);
    out(0,1) = -out(1,0); //-1 from sigma2
  }
};


template<typename FieldPolicies = StandardSourcePolicies>
class A2AflavorProjectedExpSource : public A2AflavorProjectedSource<A2AexpSource<FieldPolicies> >{
public:
  typedef typename A2AflavorProjectedSource<A2AexpSource<FieldPolicies> >::FieldParamType FieldParamType;
  typedef typename A2AflavorProjectedSource<A2AexpSource<FieldPolicies> >::ComplexType ComplexType;
  
  A2AflavorProjectedExpSource(const Float &radius, const int p[3], const FieldParamType &src_field_params = NullObject()): A2AflavorProjectedSource<A2AexpSource<FieldPolicies> >(p){
    this->src.setup(radius,src_field_params);
  }

};






CPS_END_NAMESPACE

#endif
