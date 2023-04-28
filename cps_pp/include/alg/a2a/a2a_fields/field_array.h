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
    size_t sz;

    enum DataLoc { Host, Device };
    DataLoc loc;
    
    void placeData(ViewMode mode, FieldView* host_views, size_t n){
      freeData();
      sz = n;
      size_t byte_size = n*sizeof(FieldView);
      host_v = host_views;
      
      if(mode == DeviceRead || mode == DeviceWrite){
	v = (FieldView *)device_alloc_check(byte_size);
	copy_host_to_device(v, host_views, byte_size);
	loc = Device;
      }else{
	v = host_views;
	loc = Host;
      }
    }
  
    void freeData(){
      if(v){
	if(loc == Device){
	  device_free(v);
	}
	for(size_t i=0;i<sz;i++) host_v[i].free(); 
	::free(host_v);
      }
      v=nullptr;
      host_v=nullptr;
      sz=0;
    }


  public:
    size_t size() const{ return sz; }

    View(): v(nullptr), sz(0){}
    View(ViewMode mode, const CPSfieldArray &a): View(){ assign(mode,a); }
    View(const View &r) = default;
    View(View &&r) = default;
        
    void assign(ViewMode mode, const CPSfieldArray &a){
      size_t byte_size = a.size() * sizeof(FieldView);
      int n = a.size();

      FieldView* _host_v = (FieldView*)malloc(byte_size);
      for(size_t i=0;i<n;i++){
  	assert(a[i].assigned());
  	new (_host_v+i) FieldView(a[i]->view(mode));
      }
      placeData(mode,_host_v,n);
    }

    //Deallocation must be either manually called or use CPSautoView
    void free(){
      freeData();
    }

    accelerator_inline FieldView & operator[](const size_t i) const{ return v[i]; }
  };

  View view(ViewMode mode) const{ return View(mode, *this); }

  //Free all memory
  void free(){
    std::vector<PtrWrapper<FieldType> >().swap(v);
  }
}; 



CPS_END_NAMESPACE

#endif
