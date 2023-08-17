#pragma once

#include <alg/a2a/a2a_fields/field_vectors/field_array.h>
#include <alg/a2a/a2a_fields/field_vectors/field_utils.h>

CPS_START_NAMESPACE

template<typename mf_Policies>
class A2AvectorW_autoAllocPolicies{
  typedef typename mf_Policies::FermionFieldType FermionFieldType;
  typedef typename mf_Policies::ComplexFieldType ComplexFieldType;
protected:
  void allocInitializeLowModeFields(CPSfieldArray<FermionFieldType> &v, const typename FermionFieldType::InputParamType &field_setup_params){
    for(int i=0;i<v.size();i++) v[i].emplace(field_setup_params);
  }
  void allocInitializeHighModeFields(CPSfieldArray<ComplexFieldType> &v, const typename ComplexFieldType::InputParamType &field_setup_params){
    for(int i=0;i<v.size();i++) v[i].emplace(field_setup_params);
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
  
  CPSfieldArray<FermionFieldType> *lptr;
  CPSfieldArray<ComplexFieldType> *hptr;
protected:
  void allocInitializeLowModeFields(CPSfieldArray<FermionFieldType> &v, const typename FermionFieldType::InputParamType &_field_setup_params){
    lptr = &v; lfield_setup_params = _field_setup_params;
    for(int i=0;i<v.size();i++) v[i].free();
  }
  void allocInitializeHighModeFields(CPSfieldArray<ComplexFieldType> &v, const typename ComplexFieldType::InputParamType &_field_setup_params){
    hptr = &v; hfield_setup_params = _field_setup_params;
    for(int i=0;i<v.size();i++) v[i].free();
  }
public:
  void allocLowMode(const int i){    
    if(! (*lptr)[i].assigned() ) (*lptr)[i].emplace(lfield_setup_params); 
  }
  void allocHighMode(const int i){
    if(! (*hptr)[i].assigned() ) (*hptr)[i].emplace(hfield_setup_params); 
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

  inline bool lowModeIsAllocated(const int i) const{ return lptr->operator[](i).assigned(); }
  inline bool highModeIsAllocated(const int i) const{ return hptr->operator[](i).assigned(); }
};

CPS_END_NAMESPACE
