#ifndef _CK_INNER_PRODUCT_H
#define _CK_INNER_PRODUCT_H

#include<alg/a2a/scfvectorptr.h>
#include<alg/a2a/a2a_sources.h>
#include<alg/a2a/gsl_wrapper.h>
#include<alg/a2a/conj_zmul.h>
#include<alg/a2a/inner_product_spincolorcontract.h>
#include<alg/a2a/inner_product_fmatspincolorcontract.h>

CPS_START_NAMESPACE

//Classes that perform the inner product of two spin-color-flavor vectors on a given (momentum-space) site
//Class must have an 
//complex<double> operator()(const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int &p, const int &t) const
//p is the *local* 3-momentum coordinate in canonical ordering 
//t is the local time coordinate
//mf_Complex is the base complex type for the vectors
//Output should be *double precision complex* even if the vectors are stored in single precision. Do this to avoid finite prec errors on spatial sum


//Simple inner product of a momentum-space scalar source function and a constant spin matrix
//Assumed diagonal matrix in flavor space if G-parity
template<typename mf_Complex, typename SourceType, bool conj_left = true, bool conj_right=false>
class SCmatrixInnerProduct{
  const WilsonMatrix &sc;
  const SourceType &src;
  bool conj[2];
public:
  typedef SourceType InnerProductSourceType;
  
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


//Optimized gamma^5 inner product with unit flavor matrix
template<typename mf_Complex, typename SourceType, bool conj_left = true, bool conj_right=false>
class SCg5InnerProduct{
  const SourceType &src;
public:
  typedef SourceType InnerProductSourceType;
  
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
public:
  typedef SourceType InnerProductSourceType;
  
  SCspinInnerProduct(const int _smatidx, const SourceType &_src): smatidx(_smatidx), src(_src){ }
    
  std::complex<double> operator()(const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const;
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
  typedef SourceType InnerProductSourceType;
  
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


template<typename SourceType, int Remaining, int Idx=0>
struct _siteFmatRecurseStd{
  static inline void doit(std::vector<std::complex<double> > &into, const SourceType &src, const FlavorMatrixType sigma, const int p, const FlavorMatrix &lMr){
    FlavorMatrix phi;
    src.template getSource<Idx>().siteFmat(phi,p);
    phi.pl(sigma);
    into[Idx] += TransLeftTrace(lMr, phi);
    _siteFmatRecurseStd<SourceType,Remaining-1,Idx+1>::doit(into,src,sigma,p,lMr);
  }
};
template<typename SourceType, int Idx>
struct _siteFmatRecurseStd<SourceType,0,Idx>{
  static inline void doit(std::vector<std::complex<double> > &into, const SourceType &src, const FlavorMatrixType sigma, const int p, const FlavorMatrix &lMr){}
};

#ifdef USE_GRID
template<typename SourceType, typename mf_Complex, int Remaining, int Idx=0>
struct _siteFmatRecurseGrid{
  static inline void doit(std::vector<std::complex<double> > &into, const SourceType &src, const FlavorMatrixType sigma, const int p, const FlavorMatrixGeneral<mf_Complex> &lMr){
    FlavorMatrixGeneral<mf_Complex> phi;
    src.template getSource<Idx>().siteFmat(phi,p);
    phi.pl(sigma);
    
    mf_Complex tlt = TransLeftTrace(lMr, phi);
    into[Idx] += Reduce(tlt);
    _siteFmatRecurseGrid<SourceType,mf_Complex,Remaining-1,Idx+1>::doit(into,src,sigma,p,lMr);
  }
};
template<typename SourceType, typename mf_Complex, int Idx>
struct _siteFmatRecurseGrid<SourceType,mf_Complex,0,Idx>{
  static inline void doit(std::vector<std::complex<double> > &into, const SourceType &src, const FlavorMatrixType sigma, const int p, const FlavorMatrixGeneral<mf_Complex> &lMr){}
};
#endif


//All of the inner products for G-parity can be separated into a part involving only the spin-color structure of the source and a part involving the flavor and smearing function.
//This case class implements the flavor/smearing function part and leaves the spin-color part to the derived class
template<typename mf_Complex, typename SourceType, typename SpinColorContractPolicy>
class GparityInnerProductBase: public SpinColorContractPolicy{
  SourceType &src;
  FlavorMatrixType sigma;
public:
  typedef SourceType InnerProductSourceType;

  GparityInnerProductBase(const FlavorMatrixType &_sigma, SourceType &_src): sigma(_sigma),src(_src){ }

  SourceType & getSrc(){ return src; }
  const SourceType & getSrc() const{ return src; }
  
  //std::complex   single source
  template<typename ComplexType = mf_Complex>
  inline typename my_enable_if< _equal<typename ComplexClassify<ComplexType>::type,complex_double_or_float_mark>::value, std::complex<double> >::type
    operator()(const SCFvectorPtr<ComplexType> &l, const SCFvectorPtr<ComplexType> &r, const int p, const int t) const{
    FlavorMatrix lMr;
    this->spinColorContract(lMr,l,r);
    
    //Compute   lMr[f1,f3] s3[f1,f2] phi[f2,f3]  =   lMr^T[f3,f1] s3[f1,f2] phi[f2,f3] 
    FlavorMatrix phi;
    src.siteFmat(phi,p);
    phi.pl(sigma);
    
    return TransLeftTrace(lMr, phi);
  }
  
#ifdef USE_GRID
  //Grid vector type single source
  template<typename ComplexType = mf_Complex>
  inline typename my_enable_if< _equal<typename ComplexClassify<ComplexType>::type,grid_vector_complex_mark>::value, std::complex<double> >::type
    operator()(const SCFvectorPtr<ComplexType> &l, const SCFvectorPtr<ComplexType> &r, const int p, const int t) const{
    FlavorMatrixGeneral<ComplexType> lMr; //is vectorized
    this->spinColorContract(lMr,l,r);
    
    //Compute   lMr[f1,f3] s3[f1,f2] phi[f2,f3]  =   lMr^T[f3,f1] s3[f1,f2] phi[f2,f3] 
    FlavorMatrixGeneral<ComplexType> phi;
    src.siteFmat(phi,p);
    phi.pl(sigma);
    
    ComplexType tlt = TransLeftTrace(lMr, phi);

    //Do the sum over the SIMD vectorized sites
    return Reduce(tlt);
  }
#endif

  //std::complex multi source
  //Does out += op(l,r,p,t);
  template<typename ComplexType = mf_Complex>
  inline typename my_enable_if<_equal<typename ComplexClassify<ComplexType>::type,complex_double_or_float_mark>::value, void>::type
  operator()(std::vector< std::complex<double> > &out, const SCFvectorPtr<ComplexType> &l, const SCFvectorPtr<ComplexType> &r, const int p, const int t) const{
    FlavorMatrix lMr;
    this->spinColorContract(lMr,l,r);
    
    _siteFmatRecurseStd<SourceType,SourceType::nSources>::doit(out,src,sigma,p,lMr);
  }

#ifdef USE_GRID
  template<typename ComplexType = mf_Complex>
  inline typename my_enable_if< _equal<typename ComplexClassify<ComplexType>::type,grid_vector_complex_mark>::value, void>::type
  operator()(std::vector< std::complex<double> > &out, const SCFvectorPtr<ComplexType> &l, const SCFvectorPtr<ComplexType> &r, const int p, const int t) const{
    FlavorMatrixGeneral<ComplexType> lMr; //is vectorized
    this->spinColorContract(lMr,l,r);

    _siteFmatRecurseGrid<SourceType,ComplexType,SourceType::nSources>::doit(out,src,sigma,p,lMr);
  }
#endif
  
};

template<int smatidx, typename mf_Complex, typename SourceType, bool conj_left = true, bool conj_right=false>
class SCFspinflavorInnerProduct: public GparityInnerProductBase<mf_Complex, SourceType, flavorMatrixSpinColorContract<smatidx,mf_Complex,conj_left,conj_right> >{
public:
  typedef SourceType InnerProductSourceType;
  
  SCFspinflavorInnerProduct(const FlavorMatrixType &_sigma, SourceType &_src):
    GparityInnerProductBase<mf_Complex, SourceType, flavorMatrixSpinColorContract<smatidx,mf_Complex,conj_left,conj_right> >(_sigma,_src){}
};







CPS_END_NAMESPACE

#endif
