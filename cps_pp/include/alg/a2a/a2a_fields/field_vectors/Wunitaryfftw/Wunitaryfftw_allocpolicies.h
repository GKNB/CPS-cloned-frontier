#pragma once

#include <alg/a2a/a2a_fields/field_vectors/Wfftw/Wfftw_allocpolicies.h>

CPS_START_NAMESPACE

template<typename mf_Policies>
using A2AvectorWunitaryfftw_autoAllocPolicies = A2AvectorWfftw_autoAllocPolicies<mf_Policies>;

template<typename mf_Policies>
class A2AvectorWunitaryfftw_manualAllocPolicies{
  typedef typename mf_Policies::FermionFieldType FermionFieldType;
  
  typename FermionFieldType::InputParamType field_setup_params;
  
  CPSfieldArray<FermionFieldType> *lptr;
  CPSfieldArray<FermionFieldType> *hptr;
protected:
  void allocInitializeLowModeFields(CPSfieldArray<FermionFieldType> &v, const typename FermionFieldType::InputParamType &_field_setup_params){
    lptr = &v; field_setup_params = _field_setup_params;
    for(int i=0;i<v.size();i++) v[i].free();
  }
  void allocInitializeHighModeFields(CPSfieldArray<FermionFieldType> &v, const typename FermionFieldType::InputParamType &_field_setup_params){
    hptr = &v; 
    for(int i=0;i<v.size();i++) v[i].free();
  }
  
public:
  void allocLowMode(const int i){
    if(! (*lptr)[i].assigned() ) (*lptr)[i].emplace(field_setup_params); 
  }
  void allocHighMode(const int i){
    if(! (*hptr)[i].assigned() ) (*hptr)[i].emplace(field_setup_params); 
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

  void destructivefft(A2AvectorWunitary<mf_Policies> &from, fieldOperation<typename mf_Policies::FermionFieldType>* mode_preop = NULL){
    _Wunitary_fft_impl<A2AvectorWunitaryfftw<mf_Policies>, A2AvectorWunitary<mf_Policies>, WFFTfieldPolicyAllocFree>::fft(static_cast<A2AvectorWunitaryfftw<mf_Policies>&>(*this),from,mode_preop);
  }

  void destructiveInversefft(A2AvectorWunitary<mf_Policies> &to, fieldOperation<typename mf_Policies::FermionFieldType>* mode_postop = NULL){
    _Wunitary_invfft_impl<A2AvectorWunitary<mf_Policies>, A2AvectorWunitaryfftw<mf_Policies>, WFFTfieldPolicyAllocFree>::inversefft(to,static_cast<A2AvectorWunitaryfftw<mf_Policies>&>(*this),mode_postop);
  }

  void destructiveGaugeFixTwistFFT(A2AvectorWunitary<mf_Policies> &from, const int _p[3], Lattice &_lat){
    gaugeFixAndTwist<typename mf_Policies::FermionFieldType> op(_p,_lat); destructivefft(from, &op);
  }

  void destructiveUnapplyGaugeFixTwistFFT(A2AvectorWunitary<mf_Policies> &to, const int _p[3], Lattice &_lat){
    reverseGaugeFixAndTwist<typename mf_Policies::FermionFieldType> op(_p,_lat); destructiveInversefft(to, &op);
  }

  void destructivefft(A2AvectorWtimePacked<mf_Policies> &from, fieldOperation<typename mf_Policies::FermionFieldType>* mode_preop = NULL){
    _Wtimepacked_fft_impl<A2AvectorWunitaryfftw<mf_Policies>, A2AvectorWtimePacked<mf_Policies>, WFFTfieldPolicyAllocFree>::fft(static_cast<A2AvectorWunitaryfftw<mf_Policies>&>(*this),from,mode_preop);
  }

  void destructiveInversefft(A2AvectorWtimePacked<mf_Policies> &to, fieldOperation<typename mf_Policies::FermionFieldType>* mode_postop = NULL){
    _Wtimepacked_invfft_impl<A2AvectorWtimePacked<mf_Policies>, A2AvectorWunitaryfftw<mf_Policies>, WFFTfieldPolicyAllocFree>::inversefft(to,static_cast<A2AvectorWunitaryfftw<mf_Policies>&>(*this),mode_postop);
  }

  void destructiveGaugeFixTwistFFT(A2AvectorWtimePacked<mf_Policies> &from, const int _p[3], Lattice &_lat){
    gaugeFixAndTwist<typename mf_Policies::FermionFieldType> op(_p,_lat); destructivefft(from, &op);
  }

  void destructiveUnapplyGaugeFixTwistFFT(A2AvectorWtimePacked<mf_Policies> &to, const int _p[3], Lattice &_lat){
    reverseGaugeFixAndTwist<typename mf_Policies::FermionFieldType> op(_p,_lat); destructiveInversefft(to, &op);
  }


  inline bool lowModeIsAllocated(const int i) const{ return lptr->operator[](i).assigned(); }
  inline bool highModeIsAllocated(const int i) const{ return hptr->operator[](i).assigned(); }
};

CPS_END_NAMESPACE
