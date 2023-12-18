#pragma once

#include <alg/a2a/a2a_fields/field_vectors/field_array.h>
#include <alg/a2a/a2a_fields/field_vectors/field_utils.h>

CPS_START_NAMESPACE

template<typename mf_Policies>
class A2AvectorWtimePacked_autoAllocPolicies{
  typedef typename mf_Policies::FermionFieldType FermionFieldType;
protected:
  void allocInitializeFields(CPSfieldArray<FermionFieldType> &v, int _nl, const typename FermionFieldType::InputParamType &field_setup_params){
    for(int i=0;i<v.size();i++) v[i].emplace(field_setup_params);
  }
public:
  typedef AutomaticAllocStrategy FieldAllocStrategy;
};

template<typename mf_Policies>
class A2AvectorWtimePacked_manualAllocPolicies{
  typedef typename mf_Policies::FermionFieldType FermionFieldType;
  
  typename FermionFieldType::InputParamType field_setup_params;
  
  CPSfieldArray<FermionFieldType> *ptr;
  int nl;
protected:
  void allocInitializeFields(CPSfieldArray<FermionFieldType> &v, int _nl, const typename FermionFieldType::InputParamType &_field_setup_params){
    nl = _nl;  ptr = &v; field_setup_params = _field_setup_params;
    for(int i=0;i<v.size();i++) v[i].free();
  }
public:
  void allocMode(const int i){    
    if(! (*ptr)[i].assigned() ) (*ptr)[i].emplace(field_setup_params); 
  }
  void freeMode(const int i){
    (*ptr)[i].free();
  }
  inline void allocLowMode(const int i){  allocMode(i); }
  inline void allocHighMode(const int i){  allocMode(nl+i); }
  inline void freeLowMode(const int i){ freeMode(i); }
  inline void freeHighMode(const int i){ freeMode(nl+i); }
  
  void allocModes(){
    for(int i=0;i<ptr->size();i++) allocMode(i);
  }
  void freeModes(){
    for(int i=0;i<ptr->size();i++) freeMode(i);
  }
  typedef ManualAllocStrategy FieldAllocStrategy;

  inline bool modeIsAllocated(const int i) const{ return ptr->operator[](i).assigned(); }
};

CPS_END_NAMESPACE
