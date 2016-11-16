#ifndef _A2A_ALLOC_POLICIES_H
#define _A2A_ALLOC_POLICIES_H

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
  void allocInitializeFields(std::vector<FermionFieldType*> &v, const typename FermionFieldType::InputParamType &field_setup_params){
    for(int i=0;i<v.size();i++){
      v[i] = new FermionFieldType(field_setup_params);
      v[i]->zero(); //initialize to zero
    }
  }
  void freeFields(std::vector<FermionFieldType*> &v){
    for(int i=0;i<v.size();i++) delete v[i];
  }
public:
  typedef AutomaticAllocStrategy FieldAllocStrategy;
};

template<typename mf_Policies>
class A2AvectorV_manualAllocPolicies{
  typedef typename mf_Policies::FermionFieldType FermionFieldType;
  typename FermionFieldType::InputParamType field_setup_params;
  std::vector<FermionFieldType*> *vptr;
protected:
  void allocInitializeFields(std::vector<FermionFieldType*> &v, const typename FermionFieldType::InputParamType &_field_setup_params){
    vptr = &v; field_setup_params = _field_setup_params;
    for(int i=0;i<v.size();i++) v[i] = NULL;
  }
  void freeFields(std::vector<FermionFieldType*> &v){
    for(int i=0;i<v.size();i++) if(v[i] != NULL) delete v[i];
  }
  
public:
  void allocMode(const int i){
    if( (*vptr)[i] == NULL ) (*vptr)[i] = new FermionFieldType(field_setup_params); 
    (*vptr)[i]->zero();
  }
  void freeMode(const int i){
    if((*vptr)[i] != NULL){
      delete  (*vptr)[i];
      (*vptr)[i] = NULL;
    }
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
  void allocInitializeFields(std::vector<FermionFieldType*> &v, const typename FermionFieldType::InputParamType &field_setup_params){
    for(int i=0;i<v.size();i++){
      v[i] = new FermionFieldType(field_setup_params);
      v[i]->zero(); //initialize to zero
    }
  }
  void freeFields(std::vector<FermionFieldType*> &v){
    for(int i=0;i<v.size();i++) delete v[i];
  }
public:
  typedef AutomaticAllocStrategy FieldAllocStrategy;
};

template<typename mf_Policies>
class A2AvectorVfftw_manualAllocPolicies{
  typedef typename mf_Policies::FermionFieldType FermionFieldType;
  typename FermionFieldType::InputParamType field_setup_params;
  std::vector<FermionFieldType*> *vptr;
protected:
  void allocInitializeFields(std::vector<FermionFieldType*> &v, const typename FermionFieldType::InputParamType &_field_setup_params){
    vptr = &v; field_setup_params = _field_setup_params;
    for(int i=0;i<v.size();i++) v[i] = NULL;
  }
  void freeFields(std::vector<FermionFieldType*> &v){
    for(int i=0;i<v.size();i++) if(v[i] != NULL) delete v[i];
  }
  
public:
  void allocMode(const int i){
    if( (*vptr)[i] == NULL ) (*vptr)[i] = new FermionFieldType(field_setup_params); 
  }
  void freeMode(const int i){
    if((*vptr)[i] != NULL){
      delete  (*vptr)[i];
      (*vptr)[i] = NULL;
    }
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
class A2AvectorW_autoAllocPolicies{
  typedef typename mf_Policies::FermionFieldType FermionFieldType;
  typedef typename mf_Policies::ComplexFieldType ComplexFieldType;
protected:
  void allocInitializeLowModeFields(std::vector<FermionFieldType*> &v, const typename FermionFieldType::InputParamType &field_setup_params){
    for(int i=0;i<v.size();i++) v[i] = new FermionFieldType(field_setup_params);
  }
  void allocInitializeHighModeFields(std::vector<ComplexFieldType*> &v, const typename ComplexFieldType::InputParamType &field_setup_params){
    for(int i=0;i<v.size();i++) v[i] = new ComplexFieldType(field_setup_params);
  }  
  void freeLowModeFields(std::vector<FermionFieldType*> &v){
    for(int i=0;i<v.size();i++) delete v[i];
  }
  void freeHighModeFields(std::vector<ComplexFieldType*> &v){
    for(int i=0;i<v.size();i++) delete v[i];
  }
  
public:
  typedef AutomaticAllocStrategy FieldAllocStrategy;
};

template<typename mf_Policies>
class A2AvectorW_manualAllocPolicies{
  typedef typename mf_Policies::FermionFieldType FermionFieldType;
  typedef typename mf_Policies::ComplexFieldType ComplexFieldType;
  
  typename FermionFieldType::InputParamType lfield_setup_params;
  typename FermionFieldType::InputParamType hfield_setup_params;
  
  std::vector<FermionFieldType*> *lptr;
  std::vector<ComplexFieldType*> *hptr;
protected:
  void allocInitializeLowModeFields(std::vector<FermionFieldType*> &v, const typename FermionFieldType::InputParamType &_field_setup_params){
    lptr = &v; lfield_setup_params = _field_setup_params;
    for(int i=0;i<v.size();i++) v[i] = NULL;
  }
  void allocInitializeHighModeFields(std::vector<ComplexFieldType*> &v, const typename ComplexFieldType::InputParamType &_field_setup_params){
    hptr = &v; hfield_setup_params = _field_setup_params;
    for(int i=0;i<v.size();i++) v[i] = NULL;
  }
  
  void freeLowModeFields(std::vector<FermionFieldType*> &v){
    for(int i=0;i<v.size();i++) if(v[i] != NULL) delete v[i];
  }
  void freeHighModeFields(std::vector<ComplexFieldType*> &v){
    for(int i=0;i<v.size();i++) if(v[i] != NULL) delete v[i];
  }
  
public:
  void allocLowMode(const int i){
    if( (*lptr)[i] == NULL ) (*lptr)[i] = new FermionFieldType(lfield_setup_params); 
  }
  void allocHighMode(const int i){
    if( (*hptr)[i] == NULL ) (*lptr)[i] = new ComplexFieldType(hfield_setup_params); 
  }
  
  void freeLowMode(const int i){
    if((*lptr)[i] != NULL){ delete  (*lptr)[i]; (*lptr)[i] = NULL; }
  }
  void freeHighMode(const int i){
    if((*hptr)[i] != NULL){ delete  (*hptr)[i]; (*hptr)[i] = NULL; }
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
  void allocInitializeLowModeFields(std::vector<FermionFieldType*> &v, const typename FermionFieldType::InputParamType &field_setup_params){
    for(int i=0;i<v.size();i++) v[i] = new FermionFieldType(field_setup_params);
  }
  void allocInitializeHighModeFields(std::vector<FermionFieldType*> &v, const typename FermionFieldType::InputParamType &field_setup_params){
    for(int i=0;i<v.size();i++) v[i] = new FermionFieldType(field_setup_params);
  }  
  void freeLowModeFields(std::vector<FermionFieldType*> &v){
    for(int i=0;i<v.size();i++) delete v[i];
  }
  void freeHighModeFields(std::vector<FermionFieldType*> &v){
    for(int i=0;i<v.size();i++) delete v[i];
  }
  
public:
  typedef AutomaticAllocStrategy FieldAllocStrategy;
};

template<typename mf_Policies>
class A2AvectorWfftw_manualAllocPolicies{
  typedef typename mf_Policies::FermionFieldType FermionFieldType;
  
  typename FermionFieldType::InputParamType field_setup_params;
  
  std::vector<FermionFieldType*> *lptr;
  std::vector<FermionFieldType*> *hptr;
protected:
  void allocInitializeLowModeFields(std::vector<FermionFieldType*> &v, const typename FermionFieldType::InputParamType &_field_setup_params){
    lptr = &v; field_setup_params = _field_setup_params;
    for(int i=0;i<v.size();i++) v[i] = NULL;
  }
  void allocInitializeHighModeFields(std::vector<FermionFieldType*> &v, const typename FermionFieldType::InputParamType &_field_setup_params){
    hptr = &v; 
    for(int i=0;i<v.size();i++) v[i] = NULL;
  }
  
  void freeLowModeFields(std::vector<FermionFieldType*> &v){
    for(int i=0;i<v.size();i++) if(v[i] != NULL) delete v[i];
  }
  void freeHighModeFields(std::vector<FermionFieldType*> &v){
    for(int i=0;i<v.size();i++) if(v[i] != NULL) delete v[i];
  }
  
public:
  void allocLowMode(const int i){
    if( (*lptr)[i] == NULL ) (*lptr)[i] = new FermionFieldType(field_setup_params); 
  }
  void allocHighMode(const int i){
    if( (*hptr)[i] == NULL ) (*lptr)[i] = new FermionFieldType(field_setup_params); 
  }
  
  void freeLowMode(const int i){
    if((*lptr)[i] != NULL){ delete  (*lptr)[i]; (*lptr)[i] = NULL; }
  }
  void freeHighMode(const int i){
    if((*hptr)[i] != NULL){ delete  (*hptr)[i]; (*hptr)[i] = NULL; }
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
