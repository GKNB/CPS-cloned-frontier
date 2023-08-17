#pragma once

#include <alg/a2a/a2a_fields/field_vectors/field_array.h>
#include <alg/a2a/a2a_fields/field_vectors/field_utils.h>

CPS_START_NAMESPACE

template<typename mf_Policies>
class A2AvectorVfftw_autoAllocPolicies{
  typedef typename mf_Policies::FermionFieldType FermionFieldType;
protected:
  void allocInitializeFields(CPSfieldArray<FermionFieldType> &v, const typename FermionFieldType::InputParamType &field_setup_params){
    for(int i=0;i<v.size();i++){
      v[i].emplace(field_setup_params);
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
  CPSfieldArray<FermionFieldType> *vptr;
protected:
  void allocInitializeFields(CPSfieldArray<FermionFieldType> &v, const typename FermionFieldType::InputParamType &_field_setup_params){
    //if(!UniqueID()){ printf("A2AvectorVfftw_manualAllocPolicies::allocInitializeFields called\n"); fflush(stdout); }
    vptr = &v; field_setup_params = _field_setup_params;
    for(int i=0;i<v.size();i++) v[i].free();
  }
  
public:
  void allocMode(const int i){
    if(! (*vptr)[i].assigned() ){
      (*vptr)[i].emplace(field_setup_params);
      //if(!UniqueID()) printf("VFFT allocMode %d %p\n",i,(*vptr)[i].operator->());
    }
  }
  void freeMode(const int i){
    //if(!UniqueID()) printf("VFFT freeMode %d %p\n",i,(*vptr)[i].operator->());
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

  inline bool modeIsAllocated(const int i) const{ return vptr->operator[](i).assigned(); }
};

CPS_END_NAMESPACE
