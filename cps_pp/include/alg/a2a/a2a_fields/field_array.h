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
    size_t sz;
  public:
    size_t size() const{ return sz; }

    View(): v(nullptr), sz(0){}
    View(const CPSfieldArray &a): View(){ assign(a); }
    View(const View &r) = default;
    View(View &&r) = default;
        
    void assign(const CPSfieldArray &a){
      if(v != nullptr) device_free(v); //could be reused
      
      size_t byte_size = a.size() * sizeof(FieldView);
      v = (FieldView*)device_alloc_check(byte_size);
      sz = a.size();

      FieldView* tmpv = (FieldView*)malloc(byte_size);
      for(size_t i=0;i<sz;i++){
  	assert(a[i].assigned());
  	new (tmpv+i) FieldView(a[i]->view());
      }
      copy_host_to_device(v, tmpv, byte_size);

      for(size_t i=0;i<sz;i++)
  	tmpv[i].~FieldView();
      ::free(tmpv);
    }

    //Deallocation must be either manually called or use CPSautoView
    void free(){
      if(v) device_free(v);
    }

    accelerator_inline FieldView & operator[](const size_t i) const{ return v[i]; }
  };

  View view() const{ return View(*this); }

  //Free all memory
  void free(){
    std::vector<PtrWrapper<FieldType> >().swap(v);
  }
}; 



CPS_END_NAMESPACE

#endif