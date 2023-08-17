#pragma once

#include <alg/a2a/a2a_fields/field_vectors/field_array.h>
#include <alg/a2a/a2a_fields/field_vectors/field_utils.h>

CPS_START_NAMESPACE

template<typename mf_Policies>
class A2AvectorV_autoAllocPolicies{
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
class A2AvectorV_manualAllocPolicies{
  typedef typename mf_Policies::FermionFieldType FermionFieldType;
  typename FermionFieldType::InputParamType field_setup_params;
  CPSfieldArray<FermionFieldType> *vptr;
protected:
  void allocInitializeFields(CPSfieldArray<FermionFieldType> &v, const typename FermionFieldType::InputParamType &_field_setup_params){
    vptr = &v; field_setup_params = _field_setup_params;
    for(int i=0;i<v.size();i++) v[i].free();
  }
public:
  void allocMode(const int i){
    if(! (*vptr)[i].assigned() ){
      (*vptr)[i].emplace(field_setup_params);
      //if(!UniqueID()) printf("V allocMode %d %p\n",i,(*vptr)[i].operator->());
    }
    (*vptr)[i]->zero();
  }
  void freeMode(const int i){
    //if(!UniqueID()) printf("V freeMode %d %p\n",i,(*vptr)[i].operator->());
    (*vptr)[i].free();
  }
  void allocModes(){
    for(int i=0;i<vptr->size();i++) allocMode(i); 
  }
  void freeModes(){
    for(int i=0;i<vptr->size();i++) freeMode(i);
  }
  typedef ManualAllocStrategy FieldAllocStrategy;

  inline bool modeIsAllocated(const int i) const{ return vptr->operator[](i).assigned(); }
};


CPS_END_NAMESPACE
