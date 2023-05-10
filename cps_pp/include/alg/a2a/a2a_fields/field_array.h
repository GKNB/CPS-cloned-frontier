#ifndef _CPS_FIELD_ARRAY_H_
#define _CPS_FIELD_ARRAY_H_

#include<utility>
#include<vector>

#include<alg/a2a/utils.h>

CPS_START_NAMESPACE

//For the V and W fields we need to be able to contain arrays of PtrWrapper<CPSfield<...> > in such a way that their data can be accessed
//on GPUs. This class implements such an array type with a view construct that enables GPU access

template<typename FieldType>  //FieldType is assumed to have a view() method without an argument
class CPSfieldArray{
  std::vector<PtrWrapper<FieldType> > v;
public:
  typedef typename FieldType::View FieldView;

  CPSfieldArray(){}
  CPSfieldArray(const size_t n, const PtrWrapper<FieldType> &of = PtrWrapper<FieldType>()): v(n,of){}

  inline void resize(const size_t n, const PtrWrapper<FieldType> &of = PtrWrapper<FieldType>()){
    v.resize(n, of);
  }
  
  inline size_t size() const{ return v.size(); }
  inline PtrWrapper<FieldType> & operator[](size_t i){ return v[i]; }
  inline const PtrWrapper<FieldType> & operator[](size_t i) const{ return v[i]; }

  class View{
    FieldView *v;
    FieldView *host_v; //host-side duplicate of views used for freeing
    unsigned char* unset; //which views are not actually open. Should not be accessed on device
    size_t sz;

    enum DataLoc { Host, Device };
    DataLoc loc;
    
    void placeData(ViewMode mode, FieldView* host_views, size_t n){
      sz = n;
      size_t byte_size = n*sizeof(FieldView);
      host_v = host_views;
      
      if(mode == DeviceRead || mode == DeviceWrite || mode == DeviceReadWrite){
	v = (FieldView *)device_alloc_check(byte_size);
	copy_host_to_device(v, host_views, byte_size);
	loc = Device;
      }else{
	v = host_views;
	loc = Host;
      }
    } 
    
    void assign(ViewMode mode, const CPSfieldArray &a){
      size_t byte_size = a.size() * sizeof(FieldView);
      int n = a.size();
      
      unset = (unsigned char*)malloc(n*sizeof(unsigned char));
      memset(unset,0x0,n*sizeof(unsigned char));
      
      FieldView* _host_v = (FieldView*)malloc(byte_size);
      for(size_t i=0;i<n;i++){
  	assert(a[i].assigned());
  	new (_host_v+i) FieldView(a[i]->view(mode));
      }     
      placeData(mode,_host_v,n);
    }

    void assign(ViewMode mode, const CPSfieldArray &a, const std::vector<bool> &subset){
      size_t byte_size = a.size() * sizeof(FieldView);
      int n = a.size();

      unset = (unsigned char*)malloc(n*sizeof(unsigned char));
      memset(unset,0x0,n*sizeof(unsigned char));
      
      FieldView* _host_v = (FieldView*)malloc(byte_size);
      for(size_t i=0;i<n;i++){
	if(subset[i]){
	  assert(a[i].assigned());
	  new (_host_v+i) FieldView(a[i]->view(mode));
	}else{
	  unset[i] = (unsigned char)0x1;
	}	  
      }
      placeData(mode,_host_v,n);
    }

  public:
    size_t size() const{ return sz; }

    View(): v(nullptr), host_v(nullptr), sz(0), unset(nullptr){}
    View(ViewMode mode, const CPSfieldArray &a): View(){ assign(mode,a); }
    View(ViewMode mode, const CPSfieldArray &a, const std::vector<bool> &subset): View(){ assign(mode,a,subset); }    
    View(const View &r) = default;
    View(View &&r) = default;        
   
    //Deallocation must be either manually called or use CPSautoView
    void free(){
      if(v){
	if(loc == Device){
	  device_free(v);
	}//otherwise v = host_v and we should avoid freeing twice
	for(size_t i=0;i<sz;i++) if(unset[i]==(unsigned char)0x0) host_v[i].free(); 
	::free(host_v);
	::free(unset);
      }
    }

    accelerator_inline FieldView & operator[](const size_t i) const{ return v[i]; }
  };

  View view(ViewMode mode) const{ return View(mode, *this); }
  //Open a view only to some subset of elements. Undefined behavior if you access one that you are not supposed to!
  View view(ViewMode mode, const std::vector<bool> &subset) const{ return View(mode, *this, subset); }
  
  //Free all memory
  void free(){
    std::vector<PtrWrapper<FieldType> >().swap(v);
  }
}; 



CPS_END_NAMESPACE

#endif
