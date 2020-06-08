#ifndef _UTILS_ARRAY_H_
#define _UTILS_ARRAY_H_

#include <vector>
#include <cassert>
#include <util/gjp.h>

#include "utils_malloc.h"
//Utilities for arrays

CPS_START_NAMESPACE

//out[i] = a[i] && b[i]
inline void compute_overlap(std::vector<bool> &out, const std::vector<bool> &a, const std::vector<bool> &b){
  assert(a.size()==b.size());
  out.resize(a.size());
  for(int i=0;i<a.size();i++) out[i] = a[i] && b[i];
}


//Look for contiguous blocks of indices in the idx_map, output a list of start,size pairs
//The start index here is the 'packed' (smaller) index that indexes the array idx_map
inline void find_contiguous_blocks(std::vector<std::pair<int,int> > &blocks, const int idx_map[], int map_size){
  blocks.resize(0);
  std::pair<int,int> block(0,1); //start, size
  int prev = idx_map[0];
  for(int j_packed=1;j_packed<map_size;j_packed++){
    int j_unpacked = idx_map[j_packed];
    if(j_unpacked == prev+1){
      ++block.second;
    }else{
      blocks.push_back(block);
      block.first = j_packed;
      block.second = 1;      
    }
    prev = j_unpacked;
  }
  blocks.push_back(block);

  int sum = 0;
  for(int b=0;b<blocks.size();b++){
    //printf("Block %d, start %d, size %d\n",b,blocks[b].first,blocks[b].second);
    sum += blocks[b].second;
  }
  if(sum != map_size)
    ERR.General("find_contiguous_blocks","","Sum of block sizes %d, expect %d\n",sum,map_size);
}

//Vector resize
template<typename T>
inline void resize_2d(std::vector<std::vector<T> > &v, const size_t i, const size_t j){
  v.resize(i);
  for(int a=0;a<i;a++) v[a].resize(j);
}
template<typename T>
inline void resize_3d(std::vector<std::vector<std::vector<T> > > &v, const size_t i, const size_t j, const size_t k){
  v.resize(i);
  for(int a=0;a<i;a++){
    v[a].resize(j);
    for(int b=0;b<j;b++)
      v[a][b].resize(k);
  }
}



//A vector class that uses managed memory (if available) such that the internal pointer is valid on host and device
//The copy-constructor deep-copies the data if copyControl::shallow() == false, otherwise it does a shallow copy
template<typename T>
class ManagedVector{
  T* v;
  size_t sz;
  bool own;
public:
  //Destructive resize
  void resize(const size_t n){
    if(!own) ERR.General("ManagedVector","resize","Cannot resize a shallow copy");

    if(v) managed_free(v);
    if(n==0){ v = NULL; sz = n; return; }
    v = (T*)managed_alloc_check(n*sizeof(T));
    sz = n;
    for(int i=0;i<sz;i++) T* s = new (v + i) T();
  }
  ManagedVector(): v(NULL), sz(0), own(true){}
  ManagedVector(const int n): v(NULL), sz(0), own(true){
    this->resize(n);
  }
  ManagedVector(const ManagedVector &r){
    if(copyControl::shallow()){
      v = r.v; sz = r.sz; own = false;
      //std::cout << "ManagedVector shallow copy" << std::endl;
    }else{
      if(!own){ own = true; v = NULL; }
      std::cout << "ManagedVector deep copy" << std::endl;
      this->resize(r.sz);
      for(int i=0;i<sz;i++) T* s = new (v + i) T(r.v[i]);
    }
  }   
  ManagedVector(ManagedVector &&r): v(r.v), sz(r.sz), own(r.own){
    r.v = NULL; r.own = false;
  }   
  ~ManagedVector(){
    if(v && own) managed_free(v);
  }
  accelerator_inline size_t size() const{
    return sz;
  }
  //Same as resize(0)
  inline void free(){ resize(0); }

  ManagedVector & operator=(const ManagedVector &r){
    if(!own){ own = true; v = NULL; } //relinquish shallow copy status    
    this->resize(r.sz);
    for(int i=0;i<sz;i++) T* s = new (v + i) T(r.v[i]);
    return *this;
  }

  ManagedVector & operator=(ManagedVector &&r){
    if(!own){ own = true; v = NULL; } //relinquish shallow copy status    
    if(v) managed_free(v);
    v = r.v;
    sz = r.sz;
    r.v = NULL;
    return *this;
  }

  accelerator_inline T & operator[](const size_t i){ return v[i]; }
  accelerator_inline const T & operator[](const size_t i) const{ return v[i]; }
};




//A vector class that uses managed memory (if available) such that the internal pointer is valid on host and device
//The copy-constructor copies the pointer *by value* (for use within host/device lambdas); if you want a complete copy use the deepcopy method
template<typename T>
class ShallowCopyManagedVector{
  T* v;
  size_t sz;
  bool own_memory; //am I the original and thus responsible for deallocating?
public:
  //Destructive resize
  void resize(const size_t n){
    if(v && own_memory) managed_free(v);
    if(n==0){ v = NULL; sz = n; return; }
    v = (T*)managed_alloc_check(n*sizeof(T));
    sz = n;
    own_memory = true;
  }
  ShallowCopyManagedVector(): v(NULL), sz(0), own_memory(true){}
  ShallowCopyManagedVector(const int n): v(NULL), sz(0){
    this->resize(n);
  }
  ShallowCopyManagedVector(const ShallowCopyManagedVector &r): v(r.v), sz(r.sz), own_memory(false){ //I am a copy!
  }   
  ShallowCopyManagedVector(ShallowCopyManagedVector &&r): v(r.v), sz(r.sz), own_memory(true){ //I assume responsibility from the rvalue
    r.own_memory = false;
  }   

  ~ShallowCopyManagedVector(){
    if(v && own_memory) managed_free(v);
  }
  accelerator_inline size_t size() const{
    return sz;
  }

  ShallowCopyManagedVector & operator=(const ShallowCopyManagedVector &r){
    if(v && own_memory) managed_free(v);
    v = r.v;
    sz = r.sz;
    own_memory = false; //I am a copy
    return *this;
  }

  ShallowCopyManagedVector & operator=(ShallowCopyManagedVector &&r){
    if(v && own_memory) managed_free(v);
    v = r.v;
    sz = r.sz;
    own_memory = true; //I assume responsibility for the malloc
    r.own_memory = false;
    return *this;
  }

  void deepcopy(const ShallowCopyManagedVector &r){
    this->resize(r.sz);
    for(int i=0;i<sz;i++) T* s = new (v + i) T(r.v[i]);
  }   

  accelerator_inline T & operator[](const size_t i){ return v[i]; }
  accelerator_inline const T & operator[](const size_t i) const{ return v[i]; }
};







CPS_END_NAMESPACE

#endif
