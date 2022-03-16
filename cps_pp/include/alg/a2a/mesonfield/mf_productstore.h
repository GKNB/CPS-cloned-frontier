#ifndef  _MF_PRODUCT_STORE_H
#define  _MF_PRODUCT_STORE_H

#include "mesonfield.h"

CPS_START_NAMESPACE

//Try to avoid recomputing products of meson fields by re-using wherever possible
template<typename mf_Policies>
class MesonFieldProductStore{
  typedef A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> MfType;
  typedef A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> const* MfPtr;
  typedef std::pair<MfPtr,MfPtr> KeyType;
  typedef std::map<KeyType,MfType> MapType;
  int products_reused; //number of products for which we were able to reuse data
  
  MapType products; //compute the product for each time separation independently

  bool store_all; //if true, all products computed will be stored for reused (default)
  std::set<KeyType> store_allowed; //if !store_all, only the products in this set will be stored for reuse
  
  MfType compute(MfPtr a, MfPtr b, const bool node_local){
    KeyType key(a,b);
    int Lt = GJP.TnodeSites()*GJP.Tnodes();
    MfType into;
    mult(into,*a,*b,node_local);
    if(store_all || store_allowed.count(key)) products[key] = into;    
    return into;
  }
public:
  MesonFieldProductStore(bool store_all = true): products_reused(0), store_all(store_all){}

  //By default we keep all products, but with this method you can precompute which stores you want to keep and disable all others
  void addAllowedStore(const MfType &a, const MfType &b){
    store_all = false;
    KeyType key(&a,&b);
    store_allowed.insert(key);
  }
  
  //Product 'compute' is multi-node so please don't use this inside a node-specific piece of code
  MfType getProduct(const MfType &a, const MfType &b, const bool node_local = false){
    KeyType key(&a,&b);
    typename MapType::const_iterator pp = products.find(key);
    if(pp != products.end()){ ++products_reused; return pp->second; }
    else return compute(&a,&b,node_local);
  }
  //Const version fails if product has not been recomputed
  const MfType & getPrecomputedProduct(const MfType &a, const MfType &b){
    KeyType key(&a,&b);
    typename MapType::const_iterator pp = products.find(key);
    if(pp != products.end()){ ++products_reused; return pp->second; }
    ERR.General("MesonFieldProductStore","getProduct (const version)","Product not pre-computed\n");
  } 
  
  int size() const{ return products.size(); }

  //Return total size in bytes of all stored meson fields
  size_t byte_size() const{
    size_t size = 0;
    for(typename MapType::const_iterator it = products.begin(); it != products.end(); it++){
      size += it->second.byte_size();
    }
    return size;
  }

  //Return the number of times we were able to reuse a product
  int productsReused() const{ return products_reused; }

};

//Allow for the predetermination of which products will be able to be reused
template<typename mf_Policies>
class MesonFieldProductStoreComputeReuse{
  typedef A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> MfType;
  typedef A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> const* MfPtr;
  typedef std::pair<MfPtr,MfPtr> KeyType;
 
  std::map<KeyType, int> used_count;
public:
  void addStore(const MfType &a, const MfType &b){
    KeyType key(&a,&b);
    auto it = used_count.find(key);
    if(it == used_count.end()) used_count[key] = 1;
    else ++it->second;
  }

  void addAllowedStores(MesonFieldProductStore<mf_Policies> &to) const{
    for(auto const &k : used_count){
      //std::cout << "Product " << k.first.first << " " << k.first.second << " reuse " << k.second << " allow store " << (k.second > 1) << std::endl;
      
      if(k.second > 1) to.addAllowedStore(*k.first.first,*k.first.second);
    }
  }
};


CPS_END_NAMESPACE

#endif
