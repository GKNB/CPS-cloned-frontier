#ifndef _A2A_SOURCES_H
#define _A2A_SOURCES_H

CPS_START_NAMESPACE

//Spatial source structure in *momentum-space*. Should assign the same value to both flavors if G-parity

//3D single-precision complex field. Defined for a *single flavor* if GPBC
class A2Asource{
protected:
  CPScomplexSpatial<Float,OneFlavorPolicy> src;
public:
  inline const std::complex<Float> & siteComplex(const int &site) const{ return reinterpret_cast<std::complex<Float> const&>(*src.site_ptr(site)); }
  inline const int nsites() const{ return src.nsites(); }
};

//Use CRTP for 'setSite' method which should be specialized according to the source type
template<typename SrcParams, typename Child>
class A2AsourceBase: public A2Asource{
public:
  void set(const SrcParams &srcp){
    int glb_size[3]; for(int i=0;i<3;i++) glb_size[i] = GJP.Nodes(i)*GJP.NodeSites(i);

    //Generate a global 4d exponential source
    CPSglobalComplexSpatial<Float,OneFlavorPolicy> glb;
    glb.zero();
         
#pragma omp_parallel for
    for(int i=0;i<glb.nsites();i++)
      static_cast<Child const*>(this)->setSite(glb,i,srcp,glb_size);

    //Perform the FFT and pull out this nodes subvolume
    glb.fft();
    glb.scatter(src);
  }
};


//Exponential (hydrogen wavefunction) source
//SrcParams is just a Float for the radius
class A2AexpSource: public A2AsourceBase<Float, A2AexpSource>{
  friend class A2AsourceBase<Float, A2AexpSource>;
  bool omit_000; //set source to zero at spatial site 0,0,0

private:
  void setSite(CPSglobalComplexSpatial<Float,OneFlavorPolicy> &glb, const int ss, const Float &radius, const int glb_size[3]) const{
    int site[3]; glb.siteUnmap(ss,site); //global site

    Float ssq = 0.0;
    for(int i=0;i<3;i++){
      int sr = (site[i] + glb_size[i]/2) % glb_size[i] - glb_size[i]/2; //center at zero
      ssq += sr*sr;
    }
    Float v = sqrt(ssq)/radius;
    v = exp(-v)/glb.nsites();

    if(omit_000 && ss==0) v = 0;

    glb.site_ptr(ss,0)[0] = v; //real part only
  }

public:

  A2AexpSource(const Float &radius, bool _omit_000 = false): omit_000(_omit_000){
    set(radius);
  }
  A2AexpSource(): omit_000(false){}
  
  void setup(const Float &radius, bool _omit_000 = false){
    omit_000 = _omit_000;
    set(radius);
  }
};

//Box source. Unflavored so ignore second flav
//SrcParams is std::vector<Float> for the extents x,y,z . *These must be even numbers* (checked)
class A2AboxSource: public A2AsourceBase<std::vector<int>, A2AboxSource>{
  friend class A2AsourceBase<std::vector<int>, A2AboxSource>;

private:
  void setSite(CPSglobalComplexSpatial<Float,OneFlavorPolicy> &glb, const int ss, const std::vector<int> &box_size, const int glb_size[3]) const{
    int site[3]; glb.siteUnmap(ss,site); //global site

    bool inbox = true;
    for(int i=0;i<3;i++){ 
      //Compute distance to closest boundary
      int bdist = site[i];
      if(glb_size[i]-site[i] < bdist) bdist = glb_size[i]-site[i]; 

      if(bdist > glb_size[i]/2){
	inbox = false; break;
      }
    }
    if(inbox)
      glb.site_ptr(ss,0)[0] = 1./glb.nsites(); //real part only    
  }

public:

  A2AboxSource(const int box_size[3]){
    std::vector<int> ss(3);
    for(int i=0;i<3;i++){
      if(box_size[i] % 2 == 1){
	ERR.General("A2AboxSource","A2AboxSource","box size must be multiple of 2");
      }
      ss[i] = box_size[i];
    }
    set(ss);
  }
};


//Daiqian's original implementation sets the (1 +/- sigma_2) flavor projection on G-parity fields to unity when the two fermion fields coincide.
//I'm not sure this is actually necessary, but I need to be able to reproduce his numbers
template<typename SourceType>
class A2AflavorProjectedSource{
protected:
  int sign;

  //Derived class should setup the sources
  SourceType src_allsites;
  SourceType src_omit000; //same source structure, only the site 0,0,0 has been set to zero during the FFT

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
  A2AflavorProjectedSource(const int p[3]): sign(getProjSign(p)){}

  int nsites() const{ return src_allsites.nsites(); }
  
  inline void siteFmat(FlavorMatrix &out, const int site) const{
    //Matrix is FFT of  (1 + [sign]*sigma_2) when |x-y| !=0 or 1 when |x-y| == 0
    //It is always 1 on the diagonals
    out(0,0) = out(1,1) = src_allsites.siteComplex(site);
    //and has \pm i on the diagonals with a momentum structure that is computed by omitting site 0,0,0
    const std::complex<Float> &val = src_omit000.siteComplex(site);

    out(1,0) = std::complex<Float>( -sign * std::imag(val), sign * std::real(val) ); // sign * i * val
    out(0,1) = -out(1,0); //-1 from sigma2
  }
};



class A2AflavorProjectedExpSource : public A2AflavorProjectedSource<A2AexpSource>{
public:
  A2AflavorProjectedExpSource(const Float &radius, const int p[3]): A2AflavorProjectedSource<A2AexpSource>(p){
    src_allsites.setup(radius,false);
    src_omit000.setup(radius,true);
  }

};



CPS_END_NAMESPACE

#endif
