#ifndef _A2A_MESONFIELD_DISTRIBUTED_STORAGE_H
#define _A2A_MESONFIELD_DISTRIBUTED_STORAGE_H

#include<vector>
#include<cstdarg>

#include<alg/a2a/utils/utils_memstorage.h>

CPS_START_NAMESPACE

//If using the NODE_DISTRIBUTE_MESONFIELDS option, we can choose whether to distribute over the memory of the nodes or to read/write from disk
#define MESONFIELD_USE_DISTRIBUTED_STORAGE

#ifdef MESONFIELD_USE_BURSTBUFFER
typedef BurstBufferMemoryStorage MesonFieldDistributedStorageType;
#elif defined(MESONFIELD_USE_NODE_SCRATCH)
typedef IndependentDiskWriteStorage MesonFieldDistributedStorageType;
#elif defined(MESONFIELD_USE_DISTRIBUTED_STORAGE)
typedef DistributedMemoryStorage MesonFieldDistributedStorageType;
#else
#error "Meson field invalid storage type"
#endif


template<typename T>
void nodeGetMany(const int n, std::vector<T> *a, ...);

//Same as above but the user can pass in a set of bools that tell the gather whether the MF on that timeslice is required. If not it is internally deleted, freeing memory
template<typename T>
void nodeGetMany(const int n, std::vector<T> *a, std::vector<bool> const* a_timeslice_mask,  ...);

template<typename T>
void nodeDistributeMany(const int n, std::vector<T> *a, ...);

//Distribute only meson fields in 'from' that are *not* present in any of the sets 'notina' and following
template<typename T>
void nodeDistributeUnique(std::vector<T> &from, const int n, std::vector<T> const* notina, ...);


template<typename T>
bool mesonFieldsOnNode(const std::vector<T> &mf){
  for(int i=0;i<mf.size();i++) if(!mf[i].isOnNode()) return false;
  return true;
}

#include "implementation/mesonfield_distributed_storage_impl.tcc"


CPS_END_NAMESPACE

#endif
