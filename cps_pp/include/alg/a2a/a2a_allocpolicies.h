#ifndef _A2A_ALLOC_POLICIES_H
#define _A2A_ALLOC_POLICIES_H

#include<alg/a2a/utils.h>
#include<alg/a2a/a2a_fft.h>

CPS_START_NAMESPACE

template< typename mf_Policies> class A2AvectorV;
template< typename mf_Policies> class A2AvectorVfftw;
template< typename mf_Policies> class A2AvectorW;
template< typename mf_Policies> class A2AvectorWfftw;

struct ManualAllocStrategy{};
struct AutomaticAllocStrategy{};

template<typename mf_Policies>
class A2AvectorV_autoAllocPolicies{
  typedef typename mf_Policies::FermionFieldType FermionFieldType;
protected:
  void allocInitializeFields(std::vector<PtrWrapper<FermionFieldType> > &v, const typename FermionFieldType::InputParamType &field_setup_params){
    for(int i=0;i<v.size();i++){
      v[i].set(new FermionFieldType(field_setup_params));
      v[i]->zero(); //initialize to zero
    }
  }
public:
  typedef AutomaticAllocStrategy FieldAllocStrategy;
};

template<typename mf_Policies>
class A2AvectorV_manualAllocPolicies{
  typedef typename mf_Policies::FermionFieldType FermionFieldType;
  typename FermionFieldType::InputParamType field_setup_params;
  std::vector<PtrWrapper<FermionFieldType> > *vptr;
protected:
  void allocInitializeFields(std::vector< PtrWrapper<FermionFieldType> > &v, const typename FermionFieldType::InputParamType &_field_setup_params){
    vptr = &v; field_setup_params = _field_setup_params;
    for(int i=0;i<v.size();i++) v[i].set(NULL);
  }
public:
  void allocMode(const int i){
    if(! (*vptr)[i].assigned() ){
      (*vptr)[i].set(new FermionFieldType(field_setup_params));
      if(!UniqueID()) printf("V allocMode %d %p\n",i,(*vptr)[i].operator->());
    }
    (*vptr)[i]->zero();
  }
  void freeMode(const int i){
    if(!UniqueID()) printf("V freeMode %d %p\n",i,(*vptr)[i].operator->());
    (*vptr)[i].free();
  }
  void allocModes(){
    for(int i=0;i<vptr->size();i++) allocMode(i); 
  }
  void freeModes(){
    for(int i=0;i<vptr->size();i++) freeMode(i);
  }
  typedef ManualAllocStrategy FieldAllocStrategy;
};


template<typename mf_Policies>
class A2AvectorVfftw_autoAllocPolicies{
  typedef typename mf_Policies::FermionFieldType FermionFieldType;
protected:
  void allocInitializeFields(std::vector<PtrWrapper<FermionFieldType> > &v, const typename FermionFieldType::InputParamType &field_setup_params){
    for(int i=0;i<v.size();i++){
      v[i].set(new FermionFieldType(field_setup_params));
      v[i]->zero(); //initialize to zero
    }
  }
public:
  typedef AutomaticAllocStrategy FieldAllocStrategy;
};

template<typename mf_Policies>
class A2AvectorVfftw_manualAllocPolicies{
  typedef typename mf_Policies::FermionFieldType FermionFieldType;
  typename FermionFieldType::InputParamType field_setup_params;
  std::vector<PtrWrapper<FermionFieldType> > *vptr;
protected:
  void allocInitializeFields(std::vector<PtrWrapper<FermionFieldType> > &v, const typename FermionFieldType::InputParamType &_field_setup_params){
    vptr = &v; field_setup_params = _field_setup_params;
    for(int i=0;i<v.size();i++) v[i].set(NULL);
  }
  
public:
  void allocMode(const int i){
    if(! (*vptr)[i].assigned() ){
      (*vptr)[i].set(new FermionFieldType(field_setup_params));
      if(!UniqueID()) printf("VFFT allocMode %d %p\n",i,(*vptr)[i].operator->());
    }
  }
  void freeMode(const int i){
    if(!UniqueID()) printf("VFFT freeMode %d %p\n",i,(*vptr)[i].operator->());
    (*vptr)[i].free();
  }
  void allocModes(){
    for(int i=0;i<vptr->size();i++) allocMode(i); 
  }
  void freeModes(){
    for(int i=0;i<vptr->size();i++) freeMode(i);
  }
  typedef ManualAllocStrategy FieldAllocStrategy;

  //Allocates Vfft modes and deallocates V along the way to minimize memory usage
  void destructivefft(A2AvectorV<mf_Policies> &from, fieldOperation<typename mf_Policies::FermionFieldType>* mode_preop = NULL){
    _V_fft_impl<A2AvectorVfftw<mf_Policies>, A2AvectorV<mf_Policies>, VFFTfieldPolicyAllocFree>::fft(static_cast<A2AvectorVfftw<mf_Policies>&>(*this),from,mode_preop);    
  }
  
  void destructiveInversefft(A2AvectorV<mf_Policies> &to, fieldOperation<typename mf_Policies::FermionFieldType>* mode_postop = NULL){
    _V_invfft_impl<A2AvectorV<mf_Policies>, A2AvectorVfftw<mf_Policies>, VFFTfieldPolicyAllocFree>::inversefft(to,static_cast<A2AvectorVfftw<mf_Policies>&>(*this),mode_postop);
  }

  void destructiveGaugeFixTwistFFT(A2AvectorV<mf_Policies> &from, const int _p[3], Lattice &_lat ){
    gaugeFixAndTwist<typename mf_Policies::FermionFieldType> op(_p,_lat); destructivefft(from, &op);
  }

  void destructiveUnapplyGaugeFixTwistFFT(A2AvectorV<mf_Policies> &to, const int _p[3], Lattice &_lat ){
    reverseGaugeFixAndTwist<typename mf_Policies::FermionFieldType> op(_p,_lat); destructiveInversefft(to, &op);
  }
};




template<typename mf_Policies>
class A2AvectorW_autoAllocPolicies{
  typedef typename mf_Policies::FermionFieldType FermionFieldType;
  typedef typename mf_Policies::ComplexFieldType ComplexFieldType;
protected:
  void allocInitializeLowModeFields(std::vector<PtrWrapper<FermionFieldType> > &v, const typename FermionFieldType::InputParamType &field_setup_params){
    for(int i=0;i<v.size();i++) v[i].set(new FermionFieldType(field_setup_params));
  }
  void allocInitializeHighModeFields(std::vector<PtrWrapper<ComplexFieldType> > &v, const typename ComplexFieldType::InputParamType &field_setup_params){
    for(int i=0;i<v.size();i++) v[i].set(new ComplexFieldType(field_setup_params));
  }  
public:
  typedef AutomaticAllocStrategy FieldAllocStrategy;
};

template<typename mf_Policies>
class A2AvectorW_manualAllocPolicies{
  typedef typename mf_Policies::FermionFieldType FermionFieldType;
  typedef typename mf_Policies::ComplexFieldType ComplexFieldType;
  
  typename FermionFieldType::InputParamType lfield_setup_params;
  typename ComplexFieldType::InputParamType hfield_setup_params;
  
  std::vector<PtrWrapper<FermionFieldType> > *lptr;
  std::vector<PtrWrapper<ComplexFieldType> > *hptr;
protected:
  void allocInitializeLowModeFields(std::vector<PtrWrapper<FermionFieldType> > &v, const typename FermionFieldType::InputParamType &_field_setup_params){
    lptr = &v; lfield_setup_params = _field_setup_params;
    for(int i=0;i<v.size();i++) v[i].set(NULL);
  }
  void allocInitializeHighModeFields(std::vector<PtrWrapper<ComplexFieldType> > &v, const typename ComplexFieldType::InputParamType &_field_setup_params){
    hptr = &v; hfield_setup_params = _field_setup_params;
    for(int i=0;i<v.size();i++) v[i].set(NULL);
  }
public:
  void allocLowMode(const int i){    
    if(! (*lptr)[i].assigned() ) (*lptr)[i].set(new FermionFieldType(lfield_setup_params)); 
  }
  void allocHighMode(const int i){
    if(! (*hptr)[i].assigned() ) (*hptr)[i].set(new ComplexFieldType(hfield_setup_params)); 
  }
  
  void freeLowMode(const int i){
    (*lptr)[i].free();
  }
  void freeHighMode(const int i){
    (*hptr)[i].free();
  }
  
  void allocModes(){
    for(int i=0;i<lptr->size();i++) allocLowMode(i);
    for(int i=0;i<hptr->size();i++) allocHighMode(i);
  }
  void freeModes(){
    for(int i=0;i<lptr->size();i++) freeLowMode(i);
    for(int i=0;i<hptr->size();i++) freeHighMode(i);
  }
  typedef ManualAllocStrategy FieldAllocStrategy;
};



template<typename mf_Policies>
class A2AvectorWfftw_autoAllocPolicies{
  typedef typename mf_Policies::FermionFieldType FermionFieldType;
protected:
  void allocInitializeLowModeFields(std::vector<PtrWrapper<FermionFieldType> > &v, const typename FermionFieldType::InputParamType &field_setup_params){
    for(int i=0;i<v.size();i++) v[i].set(new FermionFieldType(field_setup_params));
  }
  void allocInitializeHighModeFields(std::vector<PtrWrapper<FermionFieldType> > &v, const typename FermionFieldType::InputParamType &field_setup_params){
    for(int i=0;i<v.size();i++) v[i].set(new FermionFieldType(field_setup_params));
  }  
  
public:
  typedef AutomaticAllocStrategy FieldAllocStrategy;
};

template<typename mf_Policies>
class A2AvectorWfftw_manualAllocPolicies{
  typedef typename mf_Policies::FermionFieldType FermionFieldType;
  
  typename FermionFieldType::InputParamType field_setup_params;
  
  std::vector<PtrWrapper<FermionFieldType> > *lptr;
  std::vector<PtrWrapper<FermionFieldType> > *hptr;
protected:
  void allocInitializeLowModeFields(std::vector<PtrWrapper<FermionFieldType> > &v, const typename FermionFieldType::InputParamType &_field_setup_params){
    lptr = &v; field_setup_params = _field_setup_params;
    for(int i=0;i<v.size();i++) v[i].set(NULL);
  }
  void allocInitializeHighModeFields(std::vector<PtrWrapper<FermionFieldType> > &v, const typename FermionFieldType::InputParamType &_field_setup_params){
    hptr = &v; 
    for(int i=0;i<v.size();i++) v[i].set(NULL);
  }
  
public:
  void allocLowMode(const int i){
    if(! (*lptr)[i].assigned() ) (*lptr)[i].set(new FermionFieldType(field_setup_params)); 
  }
  void allocHighMode(const int i){
    if(! (*hptr)[i].assigned() ) (*hptr)[i].set(new FermionFieldType(field_setup_params)); 
  }
  
  void freeLowMode(const int i){
    (*lptr)[i].free();
  }
  void freeHighMode(const int i){
    (*hptr)[i].free();
  }
  
  void allocModes(){
    for(int i=0;i<lptr->size();i++) allocLowMode(i);
    for(int i=0;i<hptr->size();i++) allocHighMode(i);
  }
  void freeModes(){
    for(int i=0;i<lptr->size();i++) freeLowMode(i);
    for(int i=0;i<hptr->size();i++) freeHighMode(i);
  }
  typedef ManualAllocStrategy FieldAllocStrategy;

  void destructivefft(A2AvectorW<mf_Policies> &from, fieldOperation<typename mf_Policies::FermionFieldType>* mode_preop = NULL){
    _W_fft_impl<A2AvectorWfftw<mf_Policies>, A2AvectorW<mf_Policies>, WFFTfieldPolicyAllocFree>::fft(static_cast<A2AvectorWfftw<mf_Policies>&>(*this),from,mode_preop);
  }

  void destructiveInversefft(A2AvectorW<mf_Policies> &to, fieldOperation<typename mf_Policies::FermionFieldType>* mode_postop = NULL){
    _W_invfft_impl<A2AvectorW<mf_Policies>, A2AvectorWfftw<mf_Policies>, WFFTfieldPolicyAllocFree>::inversefft(to,static_cast<A2AvectorWfftw<mf_Policies>&>(*this),mode_postop);
  }

  void destructiveGaugeFixTwistFFT(A2AvectorW<mf_Policies> &from, const int _p[3], Lattice &_lat){
    gaugeFixAndTwist<typename mf_Policies::FermionFieldType> op(_p,_lat); destructivefft(from, &op);
  }

  void destructiveUnapplyGaugeFixTwistFFT(A2AvectorW<mf_Policies> &to, const int _p[3], Lattice &_lat){
    reverseGaugeFixAndTwist<typename mf_Policies::FermionFieldType> op(_p,_lat); destructiveInversefft(to, &op);
  }
};





#define SET_A2AVECTOR_AUTOMATIC_ALLOC(PolicyType) \
  typedef A2AvectorV_autoAllocPolicies<PolicyType> A2AvectorVpolicies; \
  typedef A2AvectorW_autoAllocPolicies<PolicyType> A2AvectorWpolicies; \
  typedef A2AvectorVfftw_autoAllocPolicies<PolicyType> A2AvectorVfftwPolicies; \
  typedef A2AvectorWfftw_autoAllocPolicies<PolicyType> A2AvectorWfftwPolicies


#define SET_A2AVECTOR_MANUAL_ALLOC(PolicyType) \
  typedef A2AvectorV_manualAllocPolicies<PolicyType> A2AvectorVpolicies; \
  typedef A2AvectorW_manualAllocPolicies<PolicyType> A2AvectorWpolicies; \
  typedef A2AvectorVfftw_manualAllocPolicies<PolicyType> A2AvectorVfftwPolicies; \
  typedef A2AvectorWfftw_manualAllocPolicies<PolicyType> A2AvectorWfftwPolicies

CPS_END_NAMESPACE

#endif
