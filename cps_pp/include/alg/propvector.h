CPS_START_NAMESPACE
#ifndef PROP_VECTOR_H
#define PROP_VECTOR_H
CPS_END_NAMESPACE

#include<config.h>
#include <alg/propagatorcontainer.h>

CPS_START_NAMESPACE

//container for multiple props with destructor that deletes all props
template<typename T>
class PointerArray{
public:
  const static int MAX_SIZE = 100;  

  T & operator[](const int &idx);

  void set(const int &idx, T* to);

  T& append(T* v);

  void clear();
  const int &size() const;
  
  PointerArray();
  virtual ~PointerArray();

private:
  T *ptrs[MAX_SIZE];
  int sz;
};

class PropVector: public PointerArray<PropagatorContainer>{
public:
  PropagatorContainer & addProp(PropagatorArg &arg);
  PropVector();
};
class LanczosVector: public PointerArray<LanczosContainer>{
public:
  LanczosContainer & add(LanczosContainerArg &arg);
  LanczosVector();
};

template<typename T>
PointerArray<T>::PointerArray(): sz(0){ for(int i=0;i<MAX_SIZE;i++) ptrs[i] = NULL; }
template<typename T>
PointerArray<T>::~PointerArray(){ for(int i=0;i<MAX_SIZE;i++) if(ptrs[i]!=NULL) delete ptrs[i]; }

template<typename T>
T & PointerArray<T>::operator[](const int &idx){ return *ptrs[idx]; }

template<typename T>
void PointerArray<T>::clear(){
  for(int i=0;i<MAX_SIZE;i++)
    if(ptrs[i]!=NULL){
      delete ptrs[i];
      ptrs[i]=NULL;
    }
  sz = 0;
}
template<typename T>
const int &PointerArray<T>::size() const{ return sz; }

template<typename T>
void PointerArray<T>::set(const int &idx, T* to){
  if(ptrs[idx]==NULL) ERR.General("PointerArray","set(..)","Element out of bounds\n");
  delete ptrs[idx]; ptrs[idx] = to;
}

template<typename T>
T& PointerArray<T>::append(T* v){
  if(sz==MAX_SIZE){ ERR.General("PointerArray","append(..)","Reached maximum number of allowed pointers: %d\n",MAX_SIZE); }
  ptrs[sz++] = v;
  return *v;
}


#endif
CPS_END_NAMESPACE
