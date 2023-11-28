#pragma once

CPS_START_NAMESPACE

//This class implements an array with view semantics
template<typename T, typename _AllocPolicy>
class VectorWithAview: public _AllocPolicy{
public:
  typedef _AllocPolicy AllocPolicy;
private:
  AllocLocationPref alloc_loc;
  size_t sz;  
  
  //Copy the data in the preferred location
  inline void copyDataInPrefLoc(const VectorWithAview & r){
    if(alloc_loc == AllocLocationPref::Host){
      CPSautoView(r_v, r, HostRead);
      CPSautoView(t_v, (*this), HostWrite);
      memcpy(t_v(), r_v(), sz * sizeof(T));
    }else{
      CPSautoView(r_v, r, DeviceRead);
      CPSautoView(t_v, (*this), DeviceWrite);
      copy_device_to_device(t_v(), r_v(), sz * sizeof(T));
    }
  }

public:

  VectorWithAview(size_t _sz = 0, AllocLocationPref _alloc_loc = AllocLocationPref::Host): sz(_sz), alloc_loc(_alloc_loc){
    this->_alloc(_sz * sizeof(T), alloc_loc);
  }
  VectorWithAview(VectorWithAview && r): sz(r.sz), alloc_loc(r.alloc_loc){
    r.sz = 0;
    r._move(*this);
  }
  VectorWithAview & operator=(VectorWithAview &&r){
    this->_free();
    sz = r.sz;
    alloc_loc = r.alloc_loc;
    r._move(*this);
    r.sz = 0;
    return *this;
  }

  VectorWithAview(const VectorWithAview & r): sz(r.sz), alloc_loc(r.alloc_loc){
    this->_alloc(sz * sizeof(T), alloc_loc);
    copyDataInPrefLoc(r);
  }

  VectorWithAview & operator=(const VectorWithAview &r){
    this->_free();
    sz = r.sz;
    alloc_loc = r.alloc_loc;
    this->_alloc(sz * sizeof(T), alloc_loc);
    copyDataInPrefLoc(r);
    return *this;
  }

  //If alloc is call again (eg because of a resize), where should the allocation be made?
  void setAllocLocation(AllocLocationPref to){ alloc_loc = to; }

  size_t size() const{ return sz; }
  
  void resize(size_t _sz){
    sz = _sz;
    this->_free();
    this->_alloc(_sz * sizeof(T), alloc_loc);
  }
    
  ~VectorWithAview(){
    this->_free();
  }

  class View: public AllocPolicy::AllocView{
    size_t sz;
  public:
    accelerator_inline T & operator[](const size_t i) const{ return *( (T*) this->operator()() + i ); }
    accelerator_inline size_t size() const{ return sz; }
    accelerator_inline T* data() const{ return (T*)this->operator()(); }

    View(typename AllocPolicy::AllocView aview, size_t s): AllocPolicy::AllocView(aview), sz(s){}
  };
  
  View view(ViewMode mode) const{ return View(this->_getAllocView(mode),this->sz); }
};

CPS_END_NAMESPACE
