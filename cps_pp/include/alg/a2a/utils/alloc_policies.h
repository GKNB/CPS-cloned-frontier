#pragma once
CPS_START_NAMESPACE

inline void writePolicyName(std::ostream &file, const std::string &policy, const std::string &name, bool newline = true){
  file << policy << " = " << name << (newline ? "\n" : "");
}

inline void checkPolicyName(std::istream &file, const std::string &policy, const std::string &name){
  std::string tmp; getline(file,tmp);
  std::ostringstream expect; writePolicyName(expect,policy,name,false);
  if(tmp != expect.str()){ printf("checkPolicyName expected \"%s\" got \"%s\"\n",expect.str().c_str(), tmp.c_str()); fflush(stdout); exit(-1); }
}

//Where relevant, this enum allows specifying whether the allocation is performed initially on the device or the host
enum class AllocLocationPref { Host, Device };

class UVMallocPolicy{
  void* _ptr;
  size_t _byte_size;

protected:
  struct AllocView{
    void* ptr;
    accelerator_inline void* operator()(){ return ptr; }

    AllocView() = default;
    AllocView(const AllocView &r) = default;
    AllocView(AllocView &&r) = default;
    AllocView(void* ptr): ptr(ptr){}

    inline void free(){}
  };

  inline void _alloc(const size_t byte_size, AllocLocationPref loc = AllocLocationPref::Host){ //location pref irrelevant
    _ptr = managed_alloc_check(128,byte_size); //note CUDA ignores alignment
    _byte_size = byte_size;
  }
  inline void _free(){
    if(_ptr) managed_free(_ptr);
  }
  inline void writeParams(std::ostream &file) const{
    writePolicyName(file, "ALLOCPOLICY", "UVMallocPolicy");
  }
  inline void readParams(std::istream &file){
    checkPolicyName(file, "ALLOCPOLICY", "UVMallocPolicy");
  }

  inline AllocView _getAllocView(ViewMode mode) const{
    //Unified memory
    return AllocView(_ptr);
  }

  inline void _move(UVMallocPolicy &into){
    into._ptr = _ptr;
    into._byte_size = _byte_size;
    _ptr = nullptr;
  }
 
public: 
  UVMallocPolicy(): _ptr(nullptr){}

  inline void deviceSetAdviseUVMreadOnly(const bool to) const{
    if(to) device_UVM_advise_readonly(_ptr, _byte_size);
    else device_UVM_advise_unset_readonly(_ptr, _byte_size);
  }

  inline void enqueuePrefetch(ViewMode mode) const{
    if(_ptr){
      if(mode == HostRead || mode == HostReadWrite){
	device_UVM_prefetch_host(_ptr,_byte_size);
      }else if(mode == DeviceRead || mode == DeviceReadWrite){
	device_UVM_prefetch_device(_ptr,_byte_size);
      }
    }
  }

  static inline void startPrefetches(){ }

  static inline void waitPrefetches(){
    device_synchronize_copies();
  }
  
  enum { UVMenabled = 1 }; //supports UVM
};

//Not usable on device
class HostAllocPolicy{
  void* _ptr;
  size_t _byte_size;

protected:
  struct AllocView{
    void* ptr;
    accelerator_inline void* operator()() const{ return ptr; }

    AllocView() = default;
    AllocView(const AllocView &r) = default;
    AllocView(AllocView &&r) = default;
    AllocView(void* ptr): ptr(ptr){}

    inline void free(){}
  };

  inline void _alloc(const size_t byte_size, AllocLocationPref loc = AllocLocationPref::Host){ //loc pref ignored
    _ptr = memalign_check(128,byte_size); //note CUDA ignores alignment
    _byte_size = byte_size;
  }
  inline void _free(){
    if(_ptr) ::free(_ptr);
  }
  inline void writeParams(std::ostream &file) const{
    writePolicyName(file, "ALLOCPOLICY", "HostAllocPolicy");
  }
  inline void readParams(std::istream &file){
    checkPolicyName(file, "ALLOCPOLICY", "HostAllocPolicy");
  }

  inline AllocView _getAllocView(ViewMode mode) const{   
    return AllocView(_ptr);
  }

  inline void _move(HostAllocPolicy &into){
    into._ptr = _ptr;
    into._byte_size = _byte_size;
    _ptr = nullptr;
  }
 
public: 
  HostAllocPolicy(): _ptr(nullptr){}

  inline void deviceSetAdviseUVMreadOnly(const bool to) const{}

  inline void enqueuePrefetch(ViewMode mode) const{}

  static inline void startPrefetches(){ }

  static inline void waitPrefetches(){ }
  
  enum { UVMenabled = 0 }; //supports UVM
};

//This allocator maintains a device-resident copy of the data that is synchronized automatically when required
class ExplicitCopyAllocPolicy{
  hostDeviceMirroredContainer<char> *_con;

protected:
  struct AllocView{
    hostDeviceMirroredContainer<char>::View v;
    
    accelerator_inline void* operator()() const{ return (void*)v.data(); }

    AllocView() = default;
    AllocView(const AllocView &r) = default;
    AllocView(AllocView &&r) = default;
    AllocView(hostDeviceMirroredContainer<char>::View v): v(v){}

    inline void free(){ v.free(); }
  };

  ExplicitCopyAllocPolicy(): _con(nullptr){}

  inline void _alloc(const size_t byte_size, AllocLocationPref loc = AllocLocationPref::Host){ //copies are always maintained on both device and host so loc is ignored
    assert(!_con);
    _con = new hostDeviceMirroredContainer<char>(byte_size);
  }
  inline void _free(){
    if(_con) delete _con;
  }

  inline void _move(ExplicitCopyAllocPolicy &into){
    into._con = _con;
    _con = nullptr;
  }

  inline void writeParams(std::ostream &file) const{
    writePolicyName(file, "ALLOCPOLICY", "ExplicitCopyAllocPolicy");
  }
  inline void readParams(std::istream &file){
    checkPolicyName(file, "ALLOCPOLICY", "ExplicitCopyAllocPolicy");
  }

  inline AllocView _getAllocView(ViewMode mode) const{
    assert(_con);
    return AllocView(_con->view(mode));
  }
  
public: 
  inline void deviceSetAdviseUVMreadOnly(const bool to) const{ }

  inline void enqueuePrefetch(ViewMode mode) const{}

  static inline void startPrefetches(){ }

  static inline void waitPrefetches(){ }
  
  enum { UVMenabled = 0 }; //supports UVM
};


class NullAllocPolicy{
protected:
  struct AllocView{
    accelerator_inline void* operator()() const{ return nullptr; }
    void free(){}
  };

  inline void _alloc(const size_t byte_size, AllocLocationPref loc = AllocLocationPref::Host){ } //location pref irrelevant
  inline void _free(){ }
  inline void writeParams(std::ostream &file) const{
    writePolicyName(file, "ALLOCPOLICY", "NullAlocPolicy");
  }
  inline void readParams(std::istream &file){
    checkPolicyName(file, "ALLOCPOLICY", "NullAllocPolicy");
  }

  inline AllocView _getAllocView(ViewMode mode){ return AllocView(); }

  inline void _move(UVMallocPolicy &into){}

public: 

  inline void deviceSetAdviseUVMreadOnly(const bool to) const{ }

  inline void enqueuePrefetch(ViewMode mode) const{}

  static inline void startPrefetches(){ }

  static inline void waitPrefetches(){ }
  
  enum { UVMenabled = 1 }; //supports UVM
};

class ExplicitCopyPoolAllocPolicy{
  bool set;
  DeviceMemoryPoolManager::HandleIterator h;
protected:
  struct AllocView{
    void* ptr;
    DeviceMemoryPoolManager::HandleIterator h;
    
    accelerator_inline void* operator()() const{ return ptr; }

    AllocView() = default;
    AllocView(const AllocView &r) = default;
    AllocView(AllocView &&r) = default;
    AllocView(ViewMode mode, DeviceMemoryPoolManager::HandleIterator h): h(h), ptr(DeviceMemoryPoolManager::globalPool().openView(mode,h)){}

    void free(){
      DeviceMemoryPoolManager::globalPool().closeView(h);
    }
  };

  inline void _alloc(const size_t byte_size, AllocLocationPref loc = AllocLocationPref::Host){ //no option currently implemented to support starting on the device
    assert(!set);
    h = DeviceMemoryPoolManager::globalPool().allocate(byte_size);
    set = true;
  }
  inline void _free(){
    if(set){
      DeviceMemoryPoolManager::globalPool().free(h);
      set=false;
    }
  }
  inline void _move(ExplicitCopyPoolAllocPolicy &into){
    into.set = set;
    into.h = h;
    set = false;
  }

  inline void writeParams(std::ostream &file) const{
    writePolicyName(file, "ALLOCPOLICY", "ExplicitCopyPoolAllocPolicy");
  }
  inline void readParams(std::istream &file){
    checkPolicyName(file, "ALLOCPOLICY", "ExplicitCopyPoolAllocPolicy");
  }

  inline AllocView _getAllocView(ViewMode mode) const{
    assert(set);
    return AllocView(mode,h);
  }
  
public: 
  ExplicitCopyPoolAllocPolicy(): set(false){}

  inline void deviceSetAdviseUVMreadOnly(const bool to) const{}

  inline void enqueuePrefetch(ViewMode mode) const{
    assert(set);
    DeviceMemoryPoolManager::globalPool().enqueuePrefetch(mode,h);
  }

  //Start all queued prefetches
  static inline void startPrefetches(){ 
    DeviceMemoryPoolManager::globalPool().startPrefetches();
  }

  //Wait for all queued prefetches
  static inline void waitPrefetches(){ 
    DeviceMemoryPoolManager::globalPool().waitPrefetches();
  }
  
  enum { UVMenabled = 0 }; //supports UVM
};



class ExplicitCopyDiskBackedPoolAllocPolicy{
  bool set;
  HolisticMemoryPoolManager::HandleIterator h;
protected:
  struct AllocView{
    void* ptr;
    HolisticMemoryPoolManager::HandleIterator h;
    
    accelerator_inline void* operator()() const{ return ptr; }

    AllocView() = default;
    AllocView(const AllocView &r) = default;
    AllocView(AllocView &&r) = default;
    AllocView(ViewMode mode, HolisticMemoryPoolManager::HandleIterator h): h(h), ptr(HolisticMemoryPoolManager::globalPool().openView(mode,h)){}

    void free(){
      HolisticMemoryPoolManager::globalPool().closeView(h);
    }
  };

  inline void _alloc(const size_t byte_size, AllocLocationPref loc = AllocLocationPref::Host){ 
    assert(!set);
    h = HolisticMemoryPoolManager::globalPool().allocate(byte_size, loc == AllocLocationPref::Host ? HolisticMemoryPoolManager::HostPool : HolisticMemoryPoolManager::DevicePool); 
    set = true;
  }
  inline void _free(){
    if(set){
      HolisticMemoryPoolManager::globalPool().free(h);
      set=false;
    }
  }
  inline void _move(ExplicitCopyDiskBackedPoolAllocPolicy &into){
    into.set = set;
    into.h = h;
    set = false;
  }

  inline void writeParams(std::ostream &file) const{
    writePolicyName(file, "ALLOCPOLICY", "ExplicitCopyDiskBackedPoolAllocPolicy");
  }
  inline void readParams(std::istream &file){
    checkPolicyName(file, "ALLOCPOLICY", "ExplicitCopyDiskBackedPoolAllocPolicy");
  }

  inline AllocView _getAllocView(ViewMode mode) const{
    assert(set);
    return AllocView(mode,h);
  }
  
public: 
  ExplicitCopyDiskBackedPoolAllocPolicy(): set(false){}

  inline void deviceSetAdviseUVMreadOnly(const bool to) const{}

  inline void enqueuePrefetch(ViewMode mode) const{
    assert(set);
    HolisticMemoryPoolManager::globalPool().enqueuePrefetch(mode,h);
  }

  //Start all queued prefetches
  static inline void startPrefetches(){ 
    HolisticMemoryPoolManager::globalPool().startPrefetches();
  }

  //Wait for all queued prefetches
  static inline void waitPrefetches(){ 
    HolisticMemoryPoolManager::globalPool().waitPrefetches();
  }
  
  enum { UVMenabled = 0 }; //supports UVM
};


CPS_END_NAMESPACE
