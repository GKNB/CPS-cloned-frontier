#pragma once

#include <alg/a2a/a2a_fields/field_vectors/field_array.h>
#include <alg/a2a/a2a_fields/field_vectors/field_utils.h>

CPS_START_NAMESPACE

template<typename mf_Policies>
class A2AvectorWfftw_autoAllocPolicies{
  typedef typename mf_Policies::FermionFieldType FermionFieldType;
protected:
  void allocInitializeLowModeFields(CPSfieldArray<FermionFieldType> &v, const typename FermionFieldType::InputParamType &field_setup_params){
    for(int i=0;i<v.size();i++) v[i].emplace(field_setup_params);
  }
  void allocInitializeHighModeFields(CPSfieldArray<FermionFieldType> &v, const typename FermionFieldType::InputParamType &field_setup_params){
    for(int i=0;i<v.size();i++) v[i].emplace(field_setup_params);
  }  
  
public:
  typedef AutomaticAllocStrategy FieldAllocStrategy;
};

template<typename mf_Policies>
class A2AvectorWfftw_manualAllocPolicies{
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

  inline bool lowModeIsAllocated(const int i) const{ return lptr->operator[](i).assigned(); }
  inline bool highModeIsAllocated(const int i) const{ return hptr->operator[](i).assigned(); }
};

CPS_END_NAMESPACE
