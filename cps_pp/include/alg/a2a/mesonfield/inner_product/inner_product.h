#ifndef _CK_INNER_PRODUCT_H
#define _CK_INNER_PRODUCT_H

#include<alg/a2a/mesonfield/a2a_sources.h>
#include<alg/a2a/a2a_fields/field_array.h>
#include "inner_product_spincolorcontract.h"
#include "inner_product_fmatspincolorcontract.h"


CPS_START_NAMESPACE

//Classes that perform the inner product of two spin-color-flavor vectors on a given (momentum-space) site
//Class must have an 
//void operator()(AccumType &into, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const
//p is the *local* 3-momentum coordinate in canonical ordering 
//t is the local time coordinate
//mf_Complex is the base complex type for the vectors
//AccumType is implementation dependent. Generally it is a complex type but may also be an array of complex.

inline void doAccum(std::complex<double> &to, const std::complex<double> &from){
  to += from;
}
#ifdef USE_GRID

#ifdef GRID_CUDA
accelerator_inline void doAccum(Grid::ComplexD &to, const Grid::ComplexD &from){
  to += from;
}
#endif

accelerator_inline void doAccum(Grid::ComplexD &to, const Grid::vComplexD &from){
  to += Reduce(from);
}
// accelerator_inline void doAccum(Grid::vComplexD &to, const Grid::vComplexD &from){
//   to += from;
// }

//SIMT version
accelerator_inline void doAccum(Grid::vComplexD &to, const typename SIMT<Grid::vComplexD>::value_type &from){
  auto v = SIMT<Grid::vComplexD>::read(to);
  v += from;
  SIMT<Grid::vComplexD>::write(to, v);
}

#endif

//Simple inner product of a momentum-space scalar source function and a constant spin matrix
//Assumed diagonal matrix in flavor space if G-parity
//Will not work with Grid vectorized types
template<typename mf_Complex, typename SourceType, bool conj_left = true, bool conj_right=false>
class SCmatrixInnerProduct{
  const WilsonMatrix &sc;
  const SourceType &src;
  bool conj[2];
public:
  typedef SourceType InnerProductSourceType;
  
  SCmatrixInnerProduct(const WilsonMatrix &_sc, const SourceType &_src): sc(_sc), src(_src){ }

  void operator()(std::complex<double> &into, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
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
    into += out * src.siteComplex(p);
  }
};


//Optimized gamma^5 inner product with unit flavor matrix
//Will not work with Grid vectorized types
template<typename mf_Complex, typename SourceType, bool conj_left = true, bool conj_right=false>
class SCg5InnerProduct{
  const SourceType &src;
public:
  typedef SourceType InnerProductSourceType;
  
  SCg5InnerProduct(const SourceType &_src): src(_src){ }
    
  void operator()(std::complex<double> &into, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
    std::complex<double> out(0.0,0.0);
    for(int f=0;f<1+GJP.Gparity();f++)
      out += OptimizedSpinColorContract<mf_Complex,conj_left,conj_right>::g5(l.getPtr(f),r.getPtr(f));    
    into += out * src.siteComplex(p);
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

template<int smatidx,typename mf_Complex, typename SourceType, bool conj_left = true, bool conj_right=false>
class SCspinInnerProduct{
  const SourceType src;
  
  template<typename S>
  static inline typename my_enable_if<!has_enum_nSources<S>::value, int>::type _mfPerTimeSlice(){ return 1; }

  //When running with a multisrc type this returns the number of meson fields per timeslice = nSources
  template<typename S>
  static inline typename my_enable_if<has_enum_nSources<S>::value, int>::type _mfPerTimeSlice(){ return S::nSources; }

public:
  typedef SourceType InnerProductSourceType;
  
  SCspinInnerProduct(const SourceType &_src): src(_src){ }

  static inline int mfPerTimeSlice(){ return _mfPerTimeSlice<SourceType>(); }

  template<typename AccumType>  
  inline void operator()(AccumType &into, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
    assert(mfPerTimeSlice() == 1); //not yet generalized to multi-src types
    auto out = GeneralSpinColorContractSelect<smatidx,mf_Complex,conj_left,conj_right, typename ComplexClassify<mf_Complex>::type>::doit(l.getPtr(0),r.getPtr(0));
    auto site_val = src.siteComplex(p);
    doAccum(into,out * site_val);
  }
  
  class View{
    typename SourceType::View src;
   
  public:
    template<typename AccumType>  
    accelerator_inline void operator()(AccumType &into, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
      auto out = GeneralSpinColorContractSelect<smatidx,mf_Complex,conj_left,conj_right, typename ComplexClassify<mf_Complex>::type>::doit(l.getPtr(0),r.getPtr(0));
      auto site_val = SIMT<mf_Complex>::read(src.siteComplex(p));
      doAccum(into,out * site_val);
    }

    View(const SCspinInnerProduct &r): src(r.src.view()){};

    View() = default;
    View(const View &r) = default;
  };

  View view() const{ return View(*this); }

};


//Constant spin-color-flavor matrix source structure with position-dependent flavor matrix from source
// l M N r    where l,r are the vectors, M is the constant matrix and N the position-dependent
//For use with GPBC
//Will not work for Grid vectorized types
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

  void operator()(std::complex<double> &into, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
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
    into += out;
  }
};


//All of the inner products for G-parity can be separated into a part involving only the spin-color structure of the source and a part involving the flavor and smearing function.
//For many sources that share the same spin structure, the source flavor and momentum-space structure differ between sources but the spin-color contract, which is the most expensive part, is common.
//Thus we allow for operation on containers that contain either one or many sources that share spin structure allowing re-use of the spin-color contraction.

//The GparityInnerProduct class implements the flavor/smearing function part and leaves the spin-color part to the derived class. 

//Helper structs to iterate recursively over Remaining sources, where Remaining is known at compile time
template<typename SourceType, typename mf_Complex, int Remaining, int Idx=0>
struct _siteFmatRecurse{
  template<typename AccumVtype>
  static accelerator_inline void doit(AccumVtype &into, const SourceType &src, const FlavorMatrixType sigma, const int p, const FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> &lMr){
    FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> phi;
    src.template getSource<Idx>().siteFmat(phi,p);
    phi.pl(sigma);
    
    doAccum(into[Idx], TransLeftTrace(lMr, phi));
    _siteFmatRecurse<SourceType,mf_Complex,Remaining-1,Idx+1>::doit(into,src,sigma,p,lMr);
  }
};
template<typename SourceType, typename mf_Complex, int Idx>
struct _siteFmatRecurse<SourceType,mf_Complex,0,Idx>{
  template<typename AccumVtype>
  static accelerator_inline void doit(AccumVtype &into, const SourceType &src, const FlavorMatrixType sigma, const int p, const FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> &lMr){}
};

//SpinColorContractPolicy is a policy class that has a static method
//spinColorContract(FlavorMatrixGeneral<ComplexType> &lMr, const SCFvectorPtr<ComplexType> &l, const SCFvectorPtr<ComplexType> &r)
//that performs the spin-color contraction for each flavor component of l, r

template<typename mf_Complex, typename SourceType, typename SpinColorContractPolicy>
class GparityInnerProduct: public SpinColorContractPolicy{
  SourceType src;
  FlavorMatrixType sigma;

  //When running with a multisrc type this returns the number of meson fields per timeslice = nSources
  template<typename S>
  inline typename my_enable_if<has_enum_nSources<S>::value, int>::type _mfPerTimeSlice() const{ return S::nSources; }
  
  template<typename S>
  inline typename my_enable_if<!has_enum_nSources<S>::value, int>::type _mfPerTimeSlice() const{ return 1; }

  //Single source
  template<typename AccumType, typename S>
  inline typename my_enable_if<!has_enum_nSources<S>::value, void>::type
  do_op(AccumType &out, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
#ifndef MEMTEST_MODE
    FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> lMr; //scalar on GPU, vector otherwise
    this->spinColorContract(lMr,l,r);
    
    //Compute   lMr[f1,f3] s3[f1,f2] phi[f2,f3]  =   lMr^T[f3,f1] s3[f1,f2] phi[f2,f3] 
    FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> phi;
    src.siteFmat(phi,p);
    phi.pl(sigma);

    //Do the sum over the SIMD vectorized sites
    doAccum(out, TransLeftTrace(lMr, phi));
#endif
  }  
  
  //Multi source
  //Does out += op(l,r,p,t);
  template<typename AccumVtype, typename S>
  inline typename my_enable_if<has_enum_nSources<S>::value, void>::type
  do_op(AccumVtype &out, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
#ifndef MEMTEST_MODE
    FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> lMr; //scalar on GPU, vectorized otherwise
    this->spinColorContract(lMr,l,r);

    _siteFmatRecurse<SourceType,mf_Complex,SourceType::nSources>::doit(out,src,sigma,p,lMr);
#endif
  }
  
public:
  typedef SourceType InnerProductSourceType;

  GparityInnerProduct(const FlavorMatrixType &_sigma, const SourceType &_src): sigma(_sigma),src(_src){ }

  //When running with a multisrc type this returns the number of meson fields per timeslice = nSources
  inline int mfPerTimeSlice() const{ return _mfPerTimeSlice<SourceType>(); }
  
  template<typename AccumType>
  inline void operator()(AccumType &out, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
    do_op<AccumType,SourceType>(out,l,r,p,t);
  }

  inline SourceType * getSource(){ return &src; }

  class View: public SpinColorContractPolicy{
    typename SourceType::View src;
    FlavorMatrixType sigma;

      //Single source
    template<typename AccumType, typename S>
    accelerator_inline typename my_enable_if<!has_enum_nSources<S>::value, void>::type
    do_op(AccumType &out, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
#ifndef MEMTEST_MODE
      FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> lMr; //scalar on GPU, vector otherwise
      this->spinColorContract(lMr,l,r);
    
      //Compute   lMr[f1,f3] s3[f1,f2] phi[f2,f3]  =   lMr^T[f3,f1] s3[f1,f2] phi[f2,f3] 
      FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> phi;
      src.siteFmat(phi,p);
      phi.pl(sigma);

      //Do the sum over the SIMD vectorized sites
      doAccum(out, TransLeftTrace(lMr, phi));
#endif
    }  
  
    //Multi source
    //Does out += op(l,r,p,t);
    template<typename AccumVtype, typename S>
    accelerator_inline typename my_enable_if<has_enum_nSources<S>::value, void>::type
    do_op(AccumVtype &out, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
#ifndef MEMTEST_MODE
      FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> lMr; //scalar on GPU, vectorized otherwise
      this->spinColorContract(lMr,l,r);

      _siteFmatRecurse<typename SourceType::View,mf_Complex,SourceType::nSources>::doit(out,src,sigma,p,lMr);
#endif
    }
    
  public:
    template<typename AccumType>
    accelerator_inline void operator()(AccumType &out, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
      do_op<AccumType,SourceType>(out,l,r,p,t);
    }

    View(const GparityInnerProduct &r): src(r.src.view()), sigma(r.sigma), SpinColorContractPolicy(r){};  //: src(r.src.view()), sigma(r.sigma), SpinColorContractPolicy(r){}

    View() = default;
    View(const View &r) = default;
  };

  View view() const{ return View(*this); }
};

template<int smatidx, typename mf_Complex, typename SourceType, bool conj_left = true, bool conj_right=false>
class SCFspinflavorInnerProduct: public GparityInnerProduct<mf_Complex, SourceType, flavorMatrixSpinColorContract<smatidx,conj_left,conj_right> >{
public:
  typedef SourceType InnerProductSourceType;
  
  SCFspinflavorInnerProduct(const FlavorMatrixType &_sigma, SourceType &_src):
    GparityInnerProduct<mf_Complex, SourceType, flavorMatrixSpinColorContract<smatidx,conj_left,conj_right> >(_sigma,_src){}
};




//Helper structs to iterate recursively over Remaining sources, where Remaining is known at compile time
template<typename SourceType, FlavorMatrixType F, typename mf_Complex, int Remaining, int Idx=0>
struct _siteFmatRecurseCT{
  template<typename AccumVtype>
  static accelerator_inline void doit(AccumVtype &into, const SourceType &src, const int p, const FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> &lMr){
    FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> phi;
    src.template getSource<Idx>().siteFmat(phi,p);
    plCT<typename SIMT<mf_Complex>::value_type, F>::action(phi);
    
    doAccum(into[Idx], TransLeftTrace(lMr, phi));
    _siteFmatRecurse<SourceType,mf_Complex,Remaining-1,Idx+1>::doit(into,src,p,lMr);
  }
};
template<typename SourceType, FlavorMatrixType F, typename mf_Complex, int Idx>
struct _siteFmatRecurseCT<SourceType,F,mf_Complex,0,Idx>{
  template<typename AccumVtype>
  static accelerator_inline void doit(AccumVtype &into, const SourceType &src, const int p, const FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> &lMr){}
};


template<typename mf_Complex, typename SourceType, typename SpinColorContractPolicy, FlavorMatrixType F>
class GparityInnerProductCT: public SpinColorContractPolicy{
  SourceType src;


  //When running with a multisrc type this returns the number of meson fields per timeslice = nSources
  template<typename S>
  inline typename my_enable_if<has_enum_nSources<S>::value, int>::type _mfPerTimeSlice() const{ return S::nSources; }
  
  template<typename S>
  inline typename my_enable_if<!has_enum_nSources<S>::value, int>::type _mfPerTimeSlice() const{ return 1; }

  //Single source
  template<typename AccumType, typename S>
  accelerator_inline typename my_enable_if<!has_enum_nSources<S>::value, void>::type
  do_op(AccumType &out, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
#ifndef MEMTEST_MODE
    FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> lMr; //scalar on GPU, vector otherwise
    this->spinColorContract(lMr,l,r);
    
    //Compute   lMr[f1,f3] s3[f1,f2] phi[f2,f3]  =   lMr^T[f3,f1] s3[f1,f2] phi[f2,f3] 
    FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> phi;
    src.siteFmat(phi,p);
    plCT<typename SIMT<mf_Complex>::value_type, F>::action(phi);

    //Do the sum over the SIMD vectorized sites
    doAccum(out, TransLeftTrace(lMr, phi));
#endif
  }  
  
  //Multi source
  //Does out += op(l,r,p,t);
  template<typename AccumVtype, typename S>
  accelerator_inline typename my_enable_if<has_enum_nSources<S>::value, void>::type
  do_op(AccumVtype &out, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
#ifndef MEMTEST_MODE
    FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> lMr; //scalar on GPU, vectorized otherwise
    this->spinColorContract(lMr,l,r);

    _siteFmatRecurseCT<SourceType,F,mf_Complex,SourceType::nSources>::doit(out,src,p,lMr);
#endif
  }
  
public:
  typedef SourceType InnerProductSourceType;

  GparityInnerProductCT(const SourceType &_src): src(_src){ }

  //When running with a multisrc type this returns the number of meson fields per timeslice = nSources
  inline int mfPerTimeSlice() const{ return _mfPerTimeSlice<SourceType>(); }
  
  template<typename AccumType>
  accelerator_inline void operator()(AccumType &out, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
    do_op<AccumType,SourceType>(out,l,r,p,t);
  }  
};

template<int smatidx, FlavorMatrixType F, typename mf_Complex, typename SourceType, bool conj_left = true, bool conj_right=false>
class SCFspinflavorInnerProductCT: public GparityInnerProductCT<mf_Complex, SourceType, flavorMatrixSpinColorContract<smatidx,conj_left,conj_right>, F >{
public:
  typedef SourceType InnerProductSourceType;
  
  SCFspinflavorInnerProductCT(SourceType &_src):
    GparityInnerProductCT<mf_Complex, SourceType, flavorMatrixSpinColorContract<smatidx,conj_left,conj_right>, F >(_src){}
};







//Just a simple spin/color/flavor matrix inner product with no source. (Equivalent to source whose Fourier transform is unity on all momentum sites)
template<typename mf_Complex, typename SpinColorContractPolicy>
class GparityNoSourceInnerProduct: public SpinColorContractPolicy{
  FlavorMatrixType sigma;

  template<typename AccumType>
  accelerator_inline void 
  do_op(AccumType &out, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
#ifndef MEMTEST_MODE
    FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> lMr; //scalar on GPU, vector otherwise
    this->spinColorContract(lMr,l,r);
    
    //Compute   lMr[f1,f3] sigma[f1,f3]  =   lMr^T[f3,f1] sigma[f1,f3]
    FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> phi;
    phi.unit();
    phi.pl(sigma);

    //Do the sum over the SIMD vectorized sites
    doAccum(out, TransLeftTrace(lMr, phi));
#endif
  }  

public:
  GparityNoSourceInnerProduct(const FlavorMatrixType &_sigma): sigma(_sigma){ }
 
  template<typename AccumType>
  accelerator_inline void operator()(AccumType &out, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
    do_op<AccumType>(out,l,r,p,t);
  }  

  inline int mfPerTimeSlice() const{ return 1; }
};









//The class GparitySourceShiftInnerProduct generalizes the concept of multi-source to allow for a series of momentum-space vector shifts applied to each source, reducing memory usage by performing those shifts
//on-the-fly

template<typename Vtype, int Idx>
struct _getSource{};

template<typename SourceType, int Idx>
struct _getSource< std::vector<SourceType*>, Idx >{
  static auto  doit(const int shift_idx, const std::vector<SourceType*> &shifted_sources)-> const decltype(shifted_sources[shift_idx]->template getSource<Idx>()) & {
    return shifted_sources[shift_idx]->template getSource<Idx>();
  }
};
template<typename SourceViewType, int Idx>
struct _getSource< ViewArray<SourceViewType>, Idx >{
  static auto doit(const int shift_idx, const ViewArray<SourceViewType> &shifted_sources)-> const decltype(shifted_sources[shift_idx].template getSource<Idx>()) &  {
    return shifted_sources[shift_idx].template getSource<Idx>();
  }
};
  

template<typename SourceType, typename mf_Complex, int Remaining, int Idx=0>
struct _siteFmatRecurseShift{

  template<typename AccumVtype, typename ShiftedSourceArrayType>
  static accelerator_inline void doit(AccumVtype &into, const ShiftedSourceArrayType &shifted_sources, const FlavorMatrixType sigma, const int p,
				      const FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> &lMr){
    FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> phi;
    for(int i=0;i<shifted_sources.size();i++){
      auto const &src_i = _getSource<ShiftedSourceArrayType, Idx>::doit(i, shifted_sources);
      src_i.siteFmat(phi,p);
      phi.pl(sigma);
      doAccum(into[Idx+SourceType::nSources*i], TransLeftTrace(lMr, phi));
    }
    _siteFmatRecurseShift<SourceType,mf_Complex,Remaining-1,Idx+1>::doit(into,shifted_sources,sigma,p,lMr);
  }
};
template<typename SourceType, typename mf_Complex, int Idx>
struct _siteFmatRecurseShift<SourceType,mf_Complex,0,Idx>{
  template<typename AccumVtype, typename ShiftedSourceArrayType>
  static accelerator_inline void doit(AccumVtype &into, const ShiftedSourceArrayType &shifted_sources, const FlavorMatrixType sigma, const int p,
				      const FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> &lMr){}
};


template<typename SourceType,int Remaining, int Idx=0>
struct _shiftRecurse{
  static void inline doit(SourceType &what, const std::vector<int> &shift){
    shiftPeriodicField(  what.template getSource<Idx>().getSource(), what.template getSource<Idx>().getSource(), shift);
    _shiftRecurse<SourceType,Remaining-1,Idx+1>::doit(what,shift);
  }
};
template<typename SourceType, int Idx>
struct _shiftRecurse<SourceType,0,Idx>{
  static void inline doit(SourceType &what, const std::vector<int> &shift){}
};

template<typename mf_Complex, typename SourceType, typename SpinColorContractPolicy>
class GparitySourceShiftInnerProduct: public SpinColorContractPolicy{
  const SourceType &src;
  FlavorMatrixType sigma;
  std::vector< std::vector<int> > shifts; //should be a set of 3-vectors
  std::vector<SourceType*> shifted_sources;
  std::vector<int> cur_shift;

  template<typename S>
  inline typename my_enable_if<!has_enum_nSources<S>::value, void>::type
  shiftTheSource(S &what, const std::vector<int> &shift){ shiftPeriodicField(what.getSource(),what.getSource(), shift); }

  template<typename S>
  inline typename my_enable_if<has_enum_nSources<S>::value, void>::type
  shiftTheSource(S &what, const std::vector<int> &shift){ _shiftRecurse<S,S::nSources>::doit(what, shift); }
  
  void shiftSource(SourceType &what, const std::vector<int> &shift){
    std::vector<int> actual_shift(shift);
    for(int i=0;i<3;i++) actual_shift[i] -= cur_shift[i]; //remove current shift in process
    shiftTheSource<SourceType>(what,actual_shift);
  }

  //When running with a multisrc type this returns the number of meson fields per timeslice = nSources * nshift
  template<typename S>
  inline typename my_enable_if<has_enum_nSources<S>::value, int>::type _mfPerTimeSlice() const{ return shifts.size() * S::nSources; } //indexed by  source_idx + nSources*shift_idx

  //When running with a single src type this returns the number of meson fields per timeslice = nshift
  template<typename S>
  inline typename my_enable_if<!has_enum_nSources<S>::value, int>::type _mfPerTimeSlice() const{ return shifts.size(); }


  //Single src, output vector indexed by src shift index
  template<typename AccumVtype, typename S>
  accelerator_inline typename my_enable_if<!has_enum_nSources<S>::value, void>::type
  do_op(AccumVtype &out, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
#ifndef MEMTEST_MODE
    FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> lMr;
    this->spinColorContract(lMr,l,r);

    FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> phi;
    for(int i=0;i<shifted_sources.size();i++){
      shifted_sources[i]->siteFmat(phi,p);
      phi.pl(sigma);
      doAccum(out[i],TransLeftTrace(lMr, phi));
    }
#endif
  }  

  //Multi src. output indexed by source_idx + nSources*shift_idx
  template<typename AccumVtype,typename S>
  accelerator_inline typename my_enable_if<has_enum_nSources<S>::value, void>::type
  do_op(AccumVtype &out, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
#ifndef MEMTEST_MODE
    FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> lMr;
    this->spinColorContract(lMr,l,r);
    _siteFmatRecurseShift<S,mf_Complex,S::nSources>::doit(out,shifted_sources,sigma,p,lMr);
#endif
  }
  
public: 
  typedef SourceType InnerProductSourceType;
  
  GparitySourceShiftInnerProduct(const FlavorMatrixType &_sigma, const SourceType &_src): sigma(_sigma),src(_src), shifts(0), cur_shift(3,0){ }
  GparitySourceShiftInnerProduct(const FlavorMatrixType &_sigma, const SourceType &_src, const std::vector< std::vector<int> > &_shifts): sigma(_sigma),src(_src), shifts(_shifts), cur_shift(3,0){ }

  GparitySourceShiftInnerProduct(const GparitySourceShiftInnerProduct &cp): sigma(cp.sigma), src(cp.src), shifts(cp.shifts), cur_shift(cp.cur_shift), shifted_sources(cp.shifted_sources.size()){
    for(int i=0;i<cp.shifted_sources.size();i++)
      shifted_sources[i] = new SourceType(*cp.shifted_sources[i]);
  }

  ~GparitySourceShiftInnerProduct(){
    for(int i=0;i<shifted_sources.size();i++){
      delete shifted_sources[i];
    }      
  }
  
  inline int mfPerTimeSlice() const{ return _mfPerTimeSlice<SourceType>(); } //indexed by  source_idx + nSources*shift_idx

  //Set the shift vector (std::vector<int>) for each source
  void setShifts(const std::vector< std::vector<int> > &to_shifts){
    shifts = to_shifts;
    for(int i=0;i<shifted_sources.size();i++) delete shifted_sources[i];
    shifted_sources.resize(shifts.size());

    for(int i=0;i<shifts.size();i++){
      shifted_sources[i] = new SourceType(src);
      shiftSource(*shifted_sources[i], shifts[i]);
    }    
  }
  
  template<typename AccumVtype>
  accelerator_inline void operator()(AccumVtype &out, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
    return do_op<AccumVtype,SourceType>(out,l,r,p,t);
  }
 
  class View: public SpinColorContractPolicy{
    ViewArray<typename SourceType::View> shifted_sources;
    FlavorMatrixType sigma;

    //Single src, output vector indexed by src shift index
    template<typename AccumVtype, typename S>
    accelerator_inline typename my_enable_if<!has_enum_nSources<S>::value, void>::type
    do_op(AccumVtype &out, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
#ifndef MEMTEST_MODE
      FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> lMr;
      this->spinColorContract(lMr,l,r);

      FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> phi;
      for(int i=0;i<shifted_sources.size();i++){
	shifted_sources[i].siteFmat(phi,p);
	phi.pl(sigma);
	doAccum(out[i],TransLeftTrace(lMr, phi));
      }
#endif
    }  

    //Multi src. output indexed by source_idx + nSources*shift_idx
    template<typename AccumVtype,typename S>
    accelerator_inline typename my_enable_if<has_enum_nSources<S>::value, void>::type
    do_op(AccumVtype &out, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
#ifndef MEMTEST_MODE
      FlavorMatrixGeneral<typename SIMT<mf_Complex>::value_type> lMr;
      this->spinColorContract(lMr,l,r);
      _siteFmatRecurseShift<S,mf_Complex,S::nSources>::doit(out,shifted_sources,sigma,p,lMr);
#endif
    }
    
  public:    
    View(const GparitySourceShiftInnerProduct &p): shifted_sources(p.shifted_sources), sigma(p.sigma), SpinColorContractPolicy(p){
    }

    template<typename AccumVtype>
    accelerator_inline void operator()(AccumVtype &out, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r, const int p, const int t) const{
      return do_op<AccumVtype,SourceType>(out,l,r,p,t);
    }
    
    void free(){ shifted_sources.free(); }
  };
    
  View view() const{ return View(*this); }
    
};




CPS_END_NAMESPACE

#endif
