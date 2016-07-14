#ifndef _CK_INNER_PRODUCT_H
#define _CK_INNER_PRODUCT_H

#include<alg/a2a/scfvectorptr.h>
#include<alg/a2a/a2a_sources.h>
#include<alg/a2a/gsl_wrapper.h>
#include<alg/a2a/inner_product_grid.h>

#ifdef USE_GRID
#define USE_GRID_SCCON //switch on inner_product_grid code
#endif

CPS_START_NAMESPACE

//Classes that perform the inner product of two spin-color-flavor vectors on a given (momentum-space) site
//Class must have an 
//complex<double> operator()(const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int &p, const int &t) const
//p is the *local* 3-momentum coordinate in canonical ordering 
//t is the local time coordinate
//mf_Complex is the base complex type for the vectors
//Output should be *double precision complex* even if the vectors are stored in single precision. Do this to avoid finite prec errors on spatial sum

template<typename mf_Complex, bool conj_left, bool conj_right>
struct Mconj{};

// (re*re - im*im, re*im + im*re )
template<typename mf_Complex>
struct Mconj<mf_Complex,false,false>{
  static inline std::complex<double> doit(const mf_Complex *const l, const mf_Complex *const r){
    return std::complex<double>(l->real()*r->real() - l->imag()*r->imag(), l->real()*r->imag() + l->imag()*r->real());
  }
};
template<typename mf_Complex>
struct Mconj<mf_Complex,false,true>{
  static inline std::complex<double> doit(const mf_Complex *const l, const mf_Complex *const r){
    return std::complex<double>(l->real()*r->real() + l->imag()*r->imag(), l->imag()*r->real() - l->real()*r->imag());
  }
};
template<typename mf_Complex>
struct Mconj<mf_Complex,true,false>{
  static inline std::complex<double> doit(const mf_Complex *const l, const mf_Complex *const r){
    return std::complex<double>(l->real()*r->real() + l->imag()*r->imag(), l->real()*r->imag() - l->imag()*r->real());
  }
};
template<typename mf_Complex>
struct Mconj<mf_Complex,true,true>{
  static inline std::complex<double> doit(const mf_Complex *const l, const mf_Complex *const r){
    return std::complex<double>(l->real()*r->real() - l->imag()*r->imag(), -l->real()*r->imag() - l->imag()*r->real());
  }
};



//Simple inner product of a momentum-space scalar source function and a constant spin matrix
//Assumed diagonal matrix in flavor space if G-parity
template<typename mf_Complex, typename SourceType, bool conj_left = true, bool conj_right=false>
class SCmatrixInnerProduct{
  const WilsonMatrix &sc;
  const SourceType &src;
  bool conj[2];
public:
  SCmatrixInnerProduct(const WilsonMatrix &_sc, const SourceType &_src): sc(_sc), src(_src){ }
    
  std::complex<double> operator()(const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
    std::complex<double> out(0.0,0.0);
    for(int f=0;f<1+GJP.Gparity();f++){
      // Mr
      std::complex<double> tvec[4][3];
      for(int s1=0;s1<4;s1++)
	for(int c1=0;c1<3;c1++){
	  tvec[s1][c1] = std::complex<double>(0.0,0.0);
	  for(int s2=0;s2<4;s2++)
	    for(int c2=0;c2<3;c2++)
	      tvec[s1][c1] += sc(s1,c1,s2,c2) * ( conj_right ? std::conj(r(s2,c2,f)) : r(s2,c2,f) );	    
	}      
      //l.(Mr)
      std::complex<double> outf(0.0,0.0);
      for(int s1=0;s1<4;s1++)
	for(int c1=0;c1<3;c1++)
	  outf += ( conj_left ? std::conj(l(s1,c1,f)) : l(s1,c1,f) ) * tvec[s1][c1];
	
      out += outf;
    }
    //Multiply by momentum-space source structure
    return out * src.siteComplex(p);
  }
};

//Compute a^T Mb  for spin-color vectors a,b and spin-color matrix M  in optimal fashion
//If a,b have flavor structure the indices must be provided
//Option to complex conjugate left/right vector components

template<typename mf_Complex, bool conj_left, bool conj_right>
class OptimizedSpinColorContract{
public:
  inline static std::complex<double> g5(const mf_Complex *const l, const mf_Complex *const r){
    const static int sc_size =12;
    const static int half_sc = 6;
    
    std::complex<double> v3(0,0);

#if defined(USE_GRID_SCCON)
    grid_g5contract<mf_Complex,conj_left,conj_right>::doit(v3,l,r);
#else
    for(int i = half_sc; i < sc_size; i++){ 
      v3 += Mconj<mf_Complex,conj_left,conj_right>::doit(l+i,r+i);
    }
    v3 *= -1;
      
    for(int i = 0; i < half_sc; i ++){ 
      v3 += Mconj<mf_Complex,conj_left,conj_right>::doit(l+i,r+i);
    }
#endif

    return v3;
  }
  inline static std::complex<double> unit(const mf_Complex *const l, const mf_Complex *const r){
    const static int sc_size =12;
    std::complex<double> v3(0,0);

#ifdef USE_GSL_SCCON    
    typedef gsl_wrapper<typename mf_Complex::value_type> gw;
    
    typename gw::block_complex_struct lblock;
    typename gw::vector_complex lgsl;
    typename gw::block_complex_struct rblock;
    typename gw::vector_complex rgsl;
    typename gw::complex result;

    lblock.size = sc_size;
    lgsl.block = &lblock;
    lgsl.size = sc_size;
    lgsl.stride = 1;
    lgsl.owner = 0;
      
    rblock.size = sc_size;
    rgsl.block = &rblock;
    rgsl.size = sc_size;
    rgsl.stride = 1;
    rgsl.owner = 0;

    lblock.data = lgsl.data = l;
    rblock.data = rgsl.data = r;

    gsl_dotproduct<typename mf_Complex::value_type,conj_left,conj_right>::doit(&lgsl,&rgsl,&result);
    double(&v3_a)[2] = reinterpret_cast<double(&)[2]>(v3);
    v3_a[0] = GSL_REAL(result);
    v3_a[1] = GSL_IMAG(result);
 
#else
    for(int i = 0; i < sc_size; i ++){
      v3 += Mconj<mf_Complex,conj_left,conj_right>::doit(l+i,r+i);
    }
#endif

    return v3;
  }

};


//Optimized gamma^5 inner product with unit flavor matrix
template<typename mf_Complex, typename SourceType, bool conj_left = true, bool conj_right=false>
class SCg5InnerProduct{
  const SourceType &src;
public:
  SCg5InnerProduct(const SourceType &_src): src(_src){ }
    
  std::complex<double> operator()(const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
    std::complex<double> out(0.0,0.0);
    for(int f=0;f<1+GJP.Gparity();f++)
      out += OptimizedSpinColorContract<mf_Complex,conj_left,conj_right>::g5(l.getPtr(f),r.getPtr(f));    
    return out * src.siteComplex(p);
  }
};

//Optimized inner product for general spin matrix
//Spin matrix indexed in range (0..15) following QDP convention: integer representation of binary(n4 n3 n2 n1) for spin structure  gamma1^n1 gamma2^n2 gamma3^n3 gamma4^n4
  //Thus   idx   matrix
  //       0      Unit
  //       1      gamma1
  //       2      gamma2
  //       3      gamma1 gamma2
  //       4      gamma3
  //       5      gamma1 gamma3
  //       6      gamma2 gamma3
  //       7      gamma1 gamma2 gamma3        =  gamma5 gamma4
  //       8      gamma4
  //       9      gamma1 gamma4
  //       10     gamma2 gamma4
  //       11     gamma1 gamma2 gamma4        = -gamma5 gamma3
  //       12     gamma3 gamma4
  //       13     gamma1 gamma3 gamma4        =  gamma5 gamma2
  //       14     gamma2 gamma3 gamma4        = -gamma5 gamma1
  //       15     gamma1 gamma2 gamma3 gamma4 =  gamma5
template<typename mf_Complex, typename SourceType, bool conj_left = true, bool conj_right=false>
class SCspinInnerProduct{
  const SourceType &src;
  int smatidx;
  
  // inline std::complex<double> do_op(const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r,const int &f1, const int &f3) const{
  //   if(smatidx == 15) return OptimizedSpinColorContract<mf_Complex,conj_left,conj_right>::g5(l.getPtr(f1),r.getPtr(f3));
  //   else if(smatidx == 0) return OptimizedSpinColorContract<mf_Complex,conj_left,conj_right>::unit(l.getPtr(f1),r.getPtr(f3));
  //   else{ ERR.General("SCFspinflavorInnerProduct","do_op","Spin matrix with idx %d not yet implemented\n",smatidx); }
  // }
public:
  SCspinInnerProduct(const int _smatidx, const SourceType &_src): smatidx(_smatidx), src(_src){ }
    
  std::complex<double> operator()(const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const;

  // {
  //   std::complex<double> out(0.0,0.0);
  //   for(int f=0;f<1+GJP.Gparity();f++)
  //     out += do_op(l,r,f,f);
  //   return out * src.siteComplex(p);
  // }
};

template<typename mf_Complex, typename SourceType, bool conj_left, bool conj_right, typename ComplexClass>
struct _SCspinInnerProduct_impl{};

template<typename mf_Complex, typename SourceType, bool conj_left, bool conj_right>
struct _SCspinInnerProduct_impl<mf_Complex,SourceType,conj_left,conj_right, complex_double_or_float_mark>{
  inline static std::complex<double> do_op(const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r,const int f1, const int f3, const int smatidx){
    if(smatidx == 15) return OptimizedSpinColorContract<mf_Complex,conj_left,conj_right>::g5(l.getPtr(f1),r.getPtr(f3));
    else if(smatidx == 0) return OptimizedSpinColorContract<mf_Complex,conj_left,conj_right>::unit(l.getPtr(f1),r.getPtr(f3));
    else{ ERR.General("SCFspinflavorInnerProduct","do_op","Spin matrix with idx %d not yet implemented\n",smatidx); }
  }
  static std::complex<double> doit(const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t, const int smatidx,const SourceType &src){
    std::complex<double> out(0.0,0.0);
    for(int f=0;f<1+GJP.Gparity();f++)
      out += do_op(l,r,f,f,smatidx);
    return out * src.siteComplex(p);
  }
};
template<typename mf_Complex, typename SourceType, bool conj_left, bool conj_right>
struct _SCspinInnerProduct_impl<mf_Complex,SourceType,conj_left,conj_right, grid_vector_complex_mark>{
  static std::complex<double> doit(const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t, const int smatidx,const SourceType &src){
    assert(0); //NOT YET IMPLEMENTED
  }
};
  
template<typename mf_Complex, typename SourceType, bool conj_left, bool conj_right>
std::complex<double> SCspinInnerProduct<mf_Complex,SourceType,conj_left,conj_right>::operator()(const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
  _SCspinInnerProduct_impl<mf_Complex,SourceType,conj_left,conj_right, typename ComplexClassify<mf_Complex>::type>::doit(l,r,p,t,smatidx,src);
}



//Constant spin-color-flavor matrix source structure with position-dependent flavor matrix from source
// l M N r    where l,r are the vectors, M is the constant matrix and N the position-dependent
//For use with GPBC
template<typename mf_Complex, typename SourceType, bool conj_left = true, bool conj_right=false>
class SCFfmatSrcInnerProduct{
  const SourceType &src;
  const SpinColorFlavorMatrix &scf;
public:
  SCFfmatSrcInnerProduct(const SpinColorFlavorMatrix &_scf, const SourceType &_src): 
    scf(_scf), src(_src){ 
    if(!GJP.Gparity()) ERR.General("SCFfmatSrcInnerProduct","SCFfmatSrcInnerProduct","Only for G-parity BCs");
  }

  std::complex<double> operator()(const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
    //Get source flavor matrix structure for this momentum site
    FlavorMatrix N = src.siteFmat(p);
    
    //Nr
    std::complex<double> rvec[4][3][2];
    for(int s1=0;s1<4;s1++)
      for(int c1=0;c1<3;c1++)
	for(int f1=0;f1<2;f1++){
	  rvec[s1][c1][f1] = std::complex<double>(0.0,0.0);
	  for(int f2=0;f2<2;f2++){
	    std::complex<double> rr = ( conj_right ? std::conj(r(s1,c1,f2)) : r(s1,c1,f2) );
	    rvec[s1][c1][f1] += N(f1,f2) * rr;	    
	  }
	}  
    //lM
    std::complex<double> lvec[4][3][2];
    for(int s1=0;s1<4;s1++)
      for(int c1=0;c1<3;c1++)
	for(int f1=0;f1<2;f1++){
	  lvec[s1][c1][f1] = std::complex<double>(0.0,0.0);
	  for(int s2=0;s2<4;s2++)
	    for(int c2=0;c2<3;c2++)
	      for(int f2=0;f2<2;f2++){
		std::complex<double> ll = ( conj_left ? std::conj(l(s2,c2,f2)) : l(s2,c2,f2) );
		lvec[s1][c1][f1] += ll * scf(s2,c2,f2,s1,c1,f1);
	      }
	}     
    std::complex<double> out(0.0,0.0);
    for(int s1=0;s1<4;s1++)
      for(int c1=0;c1<3;c1++)
	for(int f1=0;f1<2;f1++)
	  out += lvec[s1][c1][f1] * rvec[s1][c1][f1];
    return out;
  }
};




//Optimized gamma^5*sigma_i inner product with flavor projection
template<typename mf_Complex, typename SourceType, bool conj_left = true, bool conj_right=false>
class SCg5sigmaInnerProduct{
  const SourceType &src;
  FlavorMatrixType sigma;
public:
  SCg5sigmaInnerProduct(const FlavorMatrixType &_sigma, const SourceType &_src): sigma(_sigma),src(_src){ }
    
  // l[sc1,f1]^T g5[sc1,sc2] s3[f1,f2] phi[f2,f3] r[sc2,f3]
  //where phi has flavor structure
  std::complex<double> operator()(const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
    //Tie together the spin-color structure to form a flavor matrix   lg5r[f1,f3] =  l[sc1,f1]^T g5[sc1,sc2] r[sc2,f3]
    FlavorMatrix lg5r;
    for(int f1=0;f1<2;f1++)
      for(int f3=0;f3<2;f3++)
	lg5r(f1,f3) = OptimizedSpinColorContract<mf_Complex,conj_left,conj_right>::g5(l.getPtr(f1),r.getPtr(f3)); 

    //Compute   lg5r[f1,f3] s3[f1,f2] phi[f2,f3]  =   lg5r^T[f3,f1] s3[f1,f2] phi[f2,f3] 
    FlavorMatrix phi;
    src.siteFmat(phi,p);
    phi.pl(sigma);

    //return (lg5r.transpose() * phi).Trace();
    return Trace(lg5r.transpose() , phi);
  }
};

//Optimized inner product for general spin and flavor matrix
//Spin matrix indexed in QDP convention, see comments for SCspinInnerProduct
template<typename mf_Complex, typename SourceType, bool conj_left = true, bool conj_right=false>
class SCFspinflavorInnerProduct{
  const SourceType &src;
  FlavorMatrixType sigma;
  int smatidx;
public:

  SCFspinflavorInnerProduct(const FlavorMatrixType &_sigma, const int &_smatidx, const SourceType &_src): 
  sigma(_sigma), smatidx(_smatidx), src(_src){}

  // l[sc1,f1]^T g5[sc1,sc2] s3[f1,f2] phi[f2,f3] r[sc2,f3]
  //where phi has flavor structure
  //p is the momentum 'site' index
  std::complex<double> operator()(const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const;
};



template<typename mf_Complex, typename SourceType, bool conj_left, bool conj_right, typename Dummy>
struct _SCFspinflavorInnerProduct_impl{};

template<typename mf_Complex, typename SourceType, bool conj_left, bool conj_right>
struct _SCFspinflavorInnerProduct_impl<mf_Complex,SourceType,conj_left,conj_right,  complex_double_or_float_mark>{
  static std::complex<double> doit(const SourceType &src, const FlavorMatrixType sigma, const int smatidx, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t){
    //Tie together the spin-color structure to form a flavor matrix   lg5r[f1,f3] =  l[sc1,f1]^T M[sc1,sc2] r[sc2,f3]
    const static std::complex<double> zero(0.,0.);
    FlavorMatrix lMr;

    //Not all are yet supported!
    switch(smatidx){
    case 15:
#ifdef USE_GRID_SCCON
      grid_scf_contract<mf_Complex,conj_left,conj_right>::grid_g5con(lMr,l,r);
#else
      for(int f1=0;f1<2;f1++)
	for(int f3=0;f3<2;f3++)
	  lMr(f1,f3) = l.isZero(f1) || r.isZero(f3) ? zero : OptimizedSpinColorContract<mf_Complex,conj_left,conj_right>::g5(l.getPtr(f1),r.getPtr(f3));
#endif
      break;
    case 0: 
#ifdef USE_GRID_SCCON
      grid_scf_contract<mf_Complex,conj_left,conj_right>::grid_unitcon(lMr,l,r);
#else
      for(int f1=0;f1<2;f1++)
	for(int f3=0;f3<2;f3++)
	  lMr(f1,f3) = l.isZero(f1) || r.isZero(f3) ? zero : OptimizedSpinColorContract<mf_Complex,conj_left,conj_right>::unit(l.getPtr(f1),r.getPtr(f3));
#endif
      break;
    default:
      ERR.General("SCFspinflavorInnerProduct","do_op","Spin matrix with idx %d not yet implemented\n",smatidx);
    }
    
    //Compute   lg5r[f1,f3] s3[f1,f2] phi[f2,f3]  =   lg5r^T[f3,f1] s3[f1,f2] phi[f2,f3] 
    FlavorMatrix phi;
    src.siteFmat(phi,p);
    phi.pl(sigma);

    //return Trace(lMr.transpose(), phi);
    return TransLeftTrace(lMr, phi);
  }
};


#ifdef USE_GRID

template<typename vComplexType, bool conj_left, bool conj_right>
struct MconjGrid{};

template<typename vComplexType>
struct MconjGrid<vComplexType,false,false>{
  static inline vComplexType doit(const vComplexType *const l, const vComplexType *const r){
    return (*l) * (*r);
  }
};
template<typename vComplexType>
struct MconjGrid<vComplexType,false,true>{
  static inline vComplexType doit(const vComplexType *const l, const vComplexType *const r){
    return (*l) * conjugate(*r);
  }
};
template<typename vComplexType>
struct MconjGrid<vComplexType,true,false>{
  static inline vComplexType doit(const vComplexType *const l, const vComplexType *const r){
    return conjugate(*l) * (*r);
  }
};
template<typename vComplexType>
struct MconjGrid<vComplexType,true,true>{
  static inline vComplexType doit(const vComplexType *const l, const vComplexType *const r){
    return conjugate(*l)*conjugate(*r);
  }
};


template<typename vComplexType, bool conj_left, bool conj_right>
class GridVectorizedSpinColorContract{
public:
  inline static vComplexType g5(const vComplexType *const l, const vComplexType *const r){
    const static int sc_size =12;
    const static int half_sc = 6;

    vComplexType v3; zeroit(v3);

    for(int i = half_sc; i < sc_size; i++){ 
      v3 -= MconjGrid<vComplexType,conj_left,conj_right>::doit(l+i,r+i);
    }
    for(int i = 0; i < half_sc; i ++){ 
      v3 += MconjGrid<vComplexType,conj_left,conj_right>::doit(l+i,r+i);
    }
    return v3;
  }
  inline static vComplexType unit(const vComplexType *const l, const vComplexType *const r){
    const static int sc_size =12;
    vComplexType v3; zeroit(v3);

    for(int i = 0; i < sc_size; i ++){
      v3 += MconjGrid<vComplexType,conj_left,conj_right>::doit(l+i,r+i);
    }
    return v3;
  }

};

template<typename mf_Complex, typename SourceType, bool conj_left, bool conj_right>
struct _SCFspinflavorInnerProduct_impl<mf_Complex,SourceType,conj_left,conj_right,  grid_vector_complex_mark>{
  static std::complex<double> doit(const SourceType &src, const FlavorMatrixType sigma, const int smatidx, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t){
    //Tie together the spin-color structure to form a flavor matrix   lg5r[f1,f3] =  l[sc1,f1]^T M[sc1,sc2] r[sc2,f3]
    const mf_Complex zero(0.);
    FlavorMatrixGeneral<mf_Complex> lMr; //is vectorized 

    //Not all are yet supported!
    switch(smatidx){
    case 15:
      for(int f1=0;f1<2;f1++)
	for(int f3=0;f3<2;f3++)
	  lMr(f1,f3) = l.isZero(f1) || r.isZero(f3) ? zero : GridVectorizedSpinColorContract<mf_Complex,conj_left,conj_right>::g5(l.getPtr(f1),r.getPtr(f3));
      break;
    case 0: 
      for(int f1=0;f1<2;f1++)
	for(int f3=0;f3<2;f3++)
	  lMr(f1,f3) = l.isZero(f1) || r.isZero(f3) ? zero : GridVectorizedSpinColorContract<mf_Complex,conj_left,conj_right>::unit(l.getPtr(f1),r.getPtr(f3));
      break;
    default:
      ERR.General("SCFspinflavorInnerProduct","do_op","Spin matrix with idx %d not yet implemented\n",smatidx);
    }
    
    //Compute   lg5r[f1,f3] s3[f1,f2] phi[f2,f3]  =   lg5r^T[f3,f1] s3[f1,f2] phi[f2,f3] 
    FlavorMatrixGeneral<mf_Complex> phi;
    src.siteFmat(phi,p);
    phi.pl(sigma);

    mf_Complex tlt = TransLeftTrace(lMr, phi);

    //Do the sum over the SIMD vectorized sites
    return Reduce(tlt);
  }
};

#endif


template<typename mf_Complex, typename SourceType, bool conj_left, bool conj_right>
std::complex<double> SCFspinflavorInnerProduct<mf_Complex,SourceType,conj_left,conj_right>::operator()(const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
  return _SCFspinflavorInnerProduct_impl<mf_Complex,SourceType,conj_left,conj_right, typename ComplexClassify<mf_Complex>::type>::doit(src,sigma,smatidx,  l,r,p,t);
}







CPS_END_NAMESPACE

#endif
