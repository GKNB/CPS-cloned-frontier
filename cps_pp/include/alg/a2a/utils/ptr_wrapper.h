#ifndef _PTR_WRAPPER_H
#define _PTR_WRAPPER_H

#include <tuple>
#include "utils_malloc.h"

CPS_START_NAMESPACE

//A class that owns data via a pointer that has an assignment and copy constructor which does a deep copy.
template<typename T>
class PtrWrapper{
  T* t;
public:
  inline T* ptr(){ return t; }
  inline T const* ptr() const{ return t; }
  inline T& operator*(){ return *t; }
  inline T* operator->(){ return t; }
  inline T const& operator*() const{ return *t; }
  inline T const* operator->() const{ return t; }
  
  inline PtrWrapper(): t(NULL){};
  inline PtrWrapper(T* _t): t(_t){}
  inline ~PtrWrapper(){ if(t!=NULL) delete t; }

  inline bool assigned() const{ return t != NULL; }
  
  inline void set(T* _t){
    if(t!=NULL) delete t;
    t = _t;
  }
  inline void free(){
    if(t!=NULL) delete t;
    t = NULL;
  }
  
  //Deep copies
  inline PtrWrapper(const PtrWrapper &r): t(NULL){
    if(r.t != NULL) t = new T(*r.t);
  }

  inline PtrWrapper(PtrWrapper &&r){
    t = r.t; r.t = NULL;
  }
  
  inline PtrWrapper & operator=(const PtrWrapper &r){
    if(t!=NULL){ delete t; t = NULL; }
    if(r.t!=NULL) t = new T(*r.t);
    return *this;
  }

  inline PtrWrapper & operator=(PtrWrapper &&r){
    if(t!=NULL){ delete t; t = NULL; }
    t = r.t; r.t = NULL;
    return *this;
  }
  
  //Construct the object in-place using memory assigned internally
  template<typename... ConstructArgs>
  inline void emplace(ConstructArgs&&... construct_args){
    if(t) delete t;
    t = new T(std::forward<ConstructArgs>(construct_args)...);
  }

};


//A variant of the above where the pointer is to managed memory
template<typename T>
class ManagedPtrWrapper{
  T* t;
public:
  inline T* ptr(){ return t; }
  inline T const* ptr() const{ return t; }
  inline T& operator*(){ return *t; }
  inline T* operator->(){ return t; }
  inline T const& operator*() const{ return *t; }
  inline T const* operator->() const{ return t; }

  class View{
    T* t;
  public:
    accelerator_inline T* ptr() const{ return t; }
    accelerator_inline T& operator*() const{ return *t; }
    accelerator_inline T* operator->() const{ return t; }

    View(ViewMode mode, const ManagedPtrWrapper &r): t(r.t){} //mode is unimportant for managed memory
  };

  View view(ViewMode mode) const{ return View(mode, *this); }
  
  inline ManagedPtrWrapper(): t(NULL){};

  //Construct an instance of T in place with arguments provided
  template<typename... ConstructArgs>
  inline ManagedPtrWrapper(ConstructArgs&&... construct_args): t(NULL){
    emplace(std::forward<ConstructArgs>(construct_args)...);
  }

  inline ManagedPtrWrapper(ManagedPtrWrapper &&r): t(r.t){
    r.t = NULL;
  }

  inline ManagedPtrWrapper(const ManagedPtrWrapper &r): t(NULL){
    if(r.t != NULL) emplace(*r.t);
  }
  //Need non-const version to avoid the arg being caught by the variadic constructor
  inline ManagedPtrWrapper(ManagedPtrWrapper &r): t(NULL){
    if(r.t != NULL) emplace(*r.t);
  }

  inline ~ManagedPtrWrapper(){ 
    free(); 
  }

  inline bool assigned() const{ return t != NULL; }
  
  //Construct the object in-place using memory assigned internally
  template<typename... ConstructArgs>
  inline void emplace(ConstructArgs&&... construct_args){
    if(t!=NULL){
      t->~T();
    }else{
      t = (T*)managed_alloc_check(sizeof(T));
    }
    t = new(t) T(std::forward<ConstructArgs>(construct_args)...);
  }
  //If the memory is owned, free it and set ptr to NULL. If not owned, just set pointer to NULL
  inline void free(){
    if(t!=NULL){ 
      t->~T();
      managed_free(t);
    }
    t = NULL;
  }
  
  //Always deep copy
  inline ManagedPtrWrapper & operator=(const ManagedPtrWrapper &r){
    if(r.t !=NULL) emplace(*r.t);
    else free();
    return *this;
  } 
};


CPS_END_NAMESPACE

#endif
