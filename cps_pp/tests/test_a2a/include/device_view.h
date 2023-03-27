#pragma once

CPS_START_NAMESPACE

#ifdef USE_GRID

#ifdef __SYCL_DEVICE_ONLY__
  #define CONSTANT __attribute__((opencl_constant))
#else
  #define CONSTANT
#endif


template<typename T>
class ArrayArray{
public:
  typedef T FermionFieldType;
  typedef typename FermionFieldType::FieldSiteType FieldSiteType;
  typedef typename FermionFieldType::InputParamType FieldInputParamType;

  CPSfieldArray<FermionFieldType> a;
  CPSfieldArray<FermionFieldType> b; //these have been diluted in spin/color but not the other indices, hence there are nhit * 12 fields here (spin/color index changes fastest in mapping)

  ArrayArray(int na, int nb, const FieldInputParamType &simd_dims){
#if 0
    a.resize(na); for(int i=0;i<na;i++) a[i].emplace(simd_dims);
    b.resize(nb); for(int i=0;i<nb;i++) b[i].emplace(simd_dims);
#else
    a.resize(na); for(int i=0;i<na;i++) a[i].set(new FermionFieldType(simd_dims)); //  emplace(simd_dims);
    b.resize(nb); for(int i=0;i<nb;i++) b[i].set(new FermionFieldType(simd_dims)); //  emplace(simd_dims);
#endif
    
  }

  inline PtrWrapper<FermionFieldType> & getA(size_t i){ return a[i]; }
  inline PtrWrapper<FermionFieldType> & getB(size_t i){ return b[i]; }

};


  


template<typename GridA2Apolicies>
void testCPSfieldArray(){
  std::cout << "Starting testCPSfieldArray" << std::endl;
  typedef typename GridA2Apolicies::ComplexType ComplexType;
  size_t nsimd = ComplexType::Nsimd();
  typename SIMDpolicyBase<4>::ParamType simd_dims;
  SIMDpolicyBase<4>::SIMDdefaultLayout(simd_dims,nsimd,2); //only divide over spatial directions

  typedef typename GridA2Apolicies::FermionFieldType FermionFieldType;

  std::cout << "Generating random field" << std::endl;
  FermionFieldType field(simd_dims);
  field.testRandom();

  std::cout  << "Seting up field array" << std::endl;
  CPSfieldArray<FermionFieldType> farray(1);
  farray[0].set(new FermionFieldType(field));

  ComplexType* into = (ComplexType*)managed_alloc_check(sizeof(ComplexType));
  ComplexType expect = *field.site_ptr(size_t(0));

  std::cout << "Getting view" << std::endl;
  CPSautoView(av, farray); //auto destruct memory alloced
   
  using Grid::acceleratorThreads;

  typedef SIMT<ComplexType> ACC;

#ifdef GRID_CUDA
  using Grid::acceleratorAbortOnGpuError;	//FIXME: need to check if this logic here is correct! (Also 4 more places below!) I think Grid only implement that for CUDA
#elif defined(GRID_SYCL)
  using Grid::theGridAccelerator;
#endif

#if defined(GRID_HIP) || defined(GRID_CUDA)
  using Grid::LambdaApply;
  #if defined(GRID_HIP)
  using Grid::LambdaApply64;  //This is only defined for hip in Grid currently
  #endif
#endif


  std::cout << "Starting kernel" << std::endl;

  ComplexType* expect_p = farray[0]->fsite_ptr(size_t(0));
  std::cout << "Site 0 ptr " << expect_p << std::endl;

  {
    using namespace Grid;
    accelerator_for(x, farray[0]->size(), nsimd,
		    {
		      if(x == 0){
			ComplexType* site_ptr = av[0].fsite_ptr(x);
			auto v = ACC::read(*site_ptr);
			ACC::write(*into, v);
		      }
		    });
  }

  std::cout << "Got " << *into << " expect " << expect << std::endl;
  
  assert( Reduce(expect == *into) );
  
  managed_free(into);

  //Test copy
  std::cout << "Testing copy assignment" << std::endl;
  CPSfieldArray<FermionFieldType> farray2= farray;
  assert(farray2.size() == farray.size());
  for(int i=0;i<farray2.size();i++){
    assert( farray2[i]->equals( *farray[i] ) );
  }

  std::cout << "Testing copy" << std::endl;
  CPSfieldArray<FermionFieldType> farray3;
  farray3 = farray;
  assert(farray2.size() == farray.size());
  for(int i=0;i<farray3.size();i++){
    assert( farray3[i]->equals( *farray[i] ) );
  }

  //Test ptrwrapper
  PtrWrapper<FermionFieldType> p; p.set(new FermionFieldType(field));
  PtrWrapper<FermionFieldType> p2; p2.set(new FermionFieldType(field));

  std::cout << "Testing PtrWrapper copy" << std::endl;
  p2 = p;
  assert(p->equals(*p2));
  
  //Test array of arrays   
  std::cout << "Testing array of arrays mode copy" << std::endl;
  ArrayArray<FermionFieldType> a2(1,1,simd_dims);
 
  a2.getA(0) = p; //.set(new FermionFieldType(field)); //farray[0];
  a2.getB(0) = p; //.set(new FermionFieldType(field)); //farray[0]; 

  std::cout << "Testing array of arrays copy" << std::endl;
  ArrayArray<FermionFieldType> a2_cp(0,0,simd_dims);
  a2_cp = a2;

  assert( a2_cp.getA(0)->equals(field) );
  assert( a2_cp.getB(0)->equals(field) );

  
  std::cout << "Passed testCPSfieldArray" << std::endl;
}


template<typename GridA2Apolicies>
void testA2AfieldAccess(){
  std::cout << "Starting testA2AfieldAccess" << std::endl;
  typedef typename GridA2Apolicies::ComplexType ComplexType;
  size_t nsimd = ComplexType::Nsimd();
  typename SIMDpolicyBase<4>::ParamType simd_dims;
  SIMDpolicyBase<4>::SIMDdefaultLayout(simd_dims,nsimd,2); //only divide over spatial directions

  typedef typename GridA2Apolicies::FermionFieldType FermionFieldType;

  std::cout << "Generating random field" << std::endl;
  FermionFieldType field(simd_dims);
  field.testRandom();

  ComplexType* into = (ComplexType*)managed_alloc_check(sizeof(ComplexType));
  ComplexType expect = *field.site_ptr(size_t(0));
  
  A2AArg a2a_arg;
  a2a_arg.nl = 1;
  a2a_arg.nhits = 1;
  a2a_arg.rand_type = UONE;
  a2a_arg.src_width = 1;

  A2AvectorV<GridA2Apolicies> v(a2a_arg,simd_dims);
  v.getMode(0) = field;
  
  memset(into, 0, sizeof(ComplexType));

  using Grid::acceleratorThreads;

  typedef SIMT<ComplexType> ACC;

#ifdef GRID_CUDA
  using Grid::acceleratorAbortOnGpuError;
#elif defined(GRID_SYCL)
  using Grid::theGridAccelerator;
#endif
#if defined(GRID_HIP) || defined(GRID_CUDA)
  using Grid::LambdaApply;
  #if defined(GRID_HIP)
  using Grid::LambdaApply64;  //This is only defined for hip in Grid currently
  #endif
#endif

  std::cout << "Generating views" << std::endl;
  CPSautoView(vv, v);
   
  size_t fsize = field.size();
 
  std::cout << "Starting kernel, fsize " << fsize << std::endl;
  {
    using namespace Grid;
    accelerator_for(x, fsize, nsimd,
		    {
		      if(x==0){
			const ComplexType &val_vec = *vv.getMode(0).fsite_ptr(x);
			auto val_lane = ACC::read(val_vec);
			ACC::write(*into, val_lane);
		      }
		    });
  }

  std::cout << "Got " << *into << " expect " << expect << std::endl;
  
  assert( Reduce(expect == *into) );  

  managed_free(into);
 
  std::cout << "Passed testA2AfieldAccess" << std::endl;
}


struct autoViewTest1{
  double v;
  bool free_called;
  autoViewTest1(): free_called(false){}
  
  struct View{    
    double v;
    View(const autoViewTest1 &p): v(p.v){}
  };
  View view() const{ return View(*this); }
};

struct autoViewTest2{
  double v;
  bool free_called;    
  
  struct View{
    double v;
    bool* free_called;
    
    View(autoViewTest2 &p): v(p.v), free_called(&p.free_called){}
    void free(){ *free_called = true; }
  };
  View view(){ return View(*this); }
};

void testAutoView(){ 
  autoViewTest1 t1;
  t1.v = 3.14;
  
  {  
    CPSautoView(t1_v, t1);

    assert(t1_v.v == t1.v);
  }
  assert( t1.free_called == false );

  autoViewTest2 t2;
  t2.v = 6.28;
  
  {  
    CPSautoView(t2_v, t2);

    assert(t2_v.v == t2.v);
  }
  assert( t2.free_called == true );
  
}

void testViewArray(){
  //Test for a type that doesn't have a free method in its view
  std::vector<autoViewTest1> t1(2);
  t1[0].v = 3.14;
  t1[1].v = 6.28;
  
  std::vector<autoViewTest1*> t1_p = { &t1[0], &t1[1] };
  ViewArray<typename autoViewTest1::View> t1_v(t1_p);

  double* into = (double*)managed_alloc_check(2*sizeof(double));

  using Grid::acceleratorThreads;

#ifdef GRID_CUDA
  using Grid::acceleratorAbortOnGpuError;
#elif defined(GRID_SYCL)
  using Grid::theGridAccelerator;
#endif
#if defined(GRID_HIP) || defined(GRID_CUDA)
  using Grid::LambdaApply;
  #if defined(GRID_HIP)
  using Grid::LambdaApply64;  //This is only defined for hip in Grid currently
  #endif
#endif

  {
    using namespace Grid;
    accelerator_for(x, 100, 1,
		    {
		      if(x==0 || x==1){
			into[x] = t1_v[x].v;
		      }
		    });
  }
  assert(into[0] == 3.14);
  assert(into[1] == 6.28);


  //Test for a type that does have a free method in its view
  std::vector<autoViewTest2> t2(2);
  t2[0].v = 31.4;
  t2[1].v = 62.8;
  
  std::vector<autoViewTest2*> t2_p = { &t2[0], &t2[1] };
  ViewArray<typename autoViewTest2::View> t2_v(t2_p);

  {
    using namespace Grid;
    accelerator_for(x, 100, 1,
		    {
		      if(x==0 || x==1){
			into[x] = t2_v[x].v;
		      }
		    });
  }
  assert(into[0] == 31.4);
  assert(into[1] == 62.8);

  t2_v.free();
  
  assert(t2[0].free_called);
  assert(t2[1].free_called); 

  
  managed_free(into);
  std::cout << "testViewArray passed" << std::endl;
}





template<typename GridA2Apolicies>
void testCPSfieldDeviceCopy(){
#ifdef GPU_VEC

  typedef typename GridA2Apolicies::ComplexType ComplexType;
  size_t nsimd = ComplexType::Nsimd();
  typename SIMDpolicyBase<4>::ParamType simd_dims;
  SIMDpolicyBase<4>::SIMDdefaultLayout(simd_dims,nsimd,2); //only divide over spatial directions

  typedef typename GridA2Apolicies::FermionFieldType FermionFieldType;
  
  //Test a host-allocated CPSfield
  FermionFieldType field(simd_dims);
  field.testRandom();

  ComplexType* into = (ComplexType*)managed_alloc_check(sizeof(ComplexType));
  typedef SIMT<ComplexType> ACC;

  ComplexType expect = *field.site_ptr(size_t(0));

  using Grid::acceleratorThreads;

#ifdef GRID_CUDA
  using Grid::acceleratorAbortOnGpuError;
#elif defined(GRID_SYCL)
  using Grid::theGridAccelerator;
#endif
#if defined(GRID_HIP) || defined(GRID_CUDA)
  using Grid::LambdaApply;
  #if defined(GRID_HIP)
  using Grid::LambdaApply64;  //This is only defined for hip in Grid currently
  #endif
#endif
 
  auto field_v = field.view();

  {
    using namespace Grid;
    accelerator_for(x, 1, nsimd,
		    {
		      auto v = ACC::read(*field_v.site_ptr(x));
		      ACC::write(*into, v);
		    });
  }

  std::cout << "Got " << *into << " expect " << expect << std::endl;
  
  assert( Reduce(expect == *into) );

  //Test a view that is allocated in shared memory; basically a 1-element array
  ManagedPtrWrapper<typename FermionFieldType::View> wrp(field.view());
  
  memset(into, 0, sizeof(ComplexType));

  auto wrp_v = wrp.view();
  {
    using namespace Grid;
    accelerator_for(x, 1, nsimd,
		    {
		      auto v = ACC::read(*wrp_v->site_ptr(x));
		      ACC::write(*into, v);
		    });
  }

  std::cout << "Got " << *into << " expect " << expect << std::endl;
  
  assert( Reduce(expect == *into) );  

  managed_free(into);

#endif
}





template<typename GridA2Apolicies>
void testMultiSourceDeviceCopy(){
#ifdef GPU_VEC

  typedef typename GridA2Apolicies::ComplexType ComplexType;
  size_t nsimd = ComplexType::Nsimd();

  typedef typename GridA2Apolicies::SourceFieldType SourceFieldType;
  typename SourceFieldType::InputParamType simd_dims_3d;
  setupFieldParams<SourceFieldType>(simd_dims_3d);

  typedef typename GridA2Apolicies::SourcePolicies SourcePolicies;

  typedef A2AflavorProjectedExpSource<SourcePolicies> ExpSrcType;
  typedef A2AflavorProjectedHydrogenSource<SourcePolicies> HydSrcType;
  typedef Elem<ExpSrcType, Elem<HydSrcType,ListEnd > > SrcList;
  typedef A2AmultiSource<SrcList> MultiSrcType;

  MultiSrcType src;

  double rad = 2.0;
  int pbase[3] = {1,1,1};
  src.template getSource<0>().setup(rad,pbase, simd_dims_3d); //1s
  src.template getSource<1>().setup(2,0,0,rad,pbase, simd_dims_3d); //2s

  auto src_v = src.view();

  ComplexType expect1 = src.template getSource<0>().siteComplex(size_t(0));
  ComplexType expect2 = src.template getSource<1>().siteComplex(size_t(0));

  using Grid::acceleratorThreads;

#ifdef GRID_CUDA
  using Grid::acceleratorAbortOnGpuError;
#elif defined(GRID_SYCL)
  using Grid::theGridAccelerator;
#endif
#if defined(GRID_HIP) || defined(GRID_CUDA)
  using Grid::LambdaApply;
  #if defined(GRID_HIP)
  using Grid::LambdaApply64;  //This is only defined for hip in Grid currently
  #endif
#endif
 
  ComplexType* into = (ComplexType*)managed_alloc_check(2*sizeof(ComplexType));
  typedef SIMT<ComplexType> ACC;

  {
    using namespace Grid;
    accelerator_for(x, 100, nsimd,
		    {
		      if(x==0){
			auto v1 = ACC::read(src_v.template getSource<0>().siteComplex(x));
			ACC::write(into[0], v1);
			
			auto v2 = ACC::read(src_v.template getSource<1>().siteComplex(x));
			ACC::write(into[1], v2);
		      }
		    });
  }

  std::cout << "Got " << into[0] << " expect " << expect1 << std::endl;
  assert( Reduce(expect1 == into[0]) );
  std::cout << "Got " << into[1] << " expect " << expect2 << std::endl;
  assert( Reduce(expect2 == into[1]) );
		  
  managed_free(into);
  src_v.free();
  std::cout << "Passed testMultiSourceDeviceCopy" << std::endl;
#endif
}



template<typename GridA2Apolicies>
void testFlavorProjectedSourceView(){
#ifdef GPU_VEC  
  std::cout << "Starting testFlavorProjectedSourceView" << std::endl;
  
  typedef typename GridA2Apolicies::ComplexType ComplexType;
  size_t nsimd = ComplexType::Nsimd();

  typedef typename GridA2Apolicies::SourceFieldType SourceFieldType;
  typename SourceFieldType::InputParamType simd_dims_3d;
  setupFieldParams<SourceFieldType>(simd_dims_3d);

  typedef typename GridA2Apolicies::SourcePolicies SourcePolicies;

  int pbase[3] = {1,1,1};
  
  typedef A2AflavorProjectedExpSource<SourcePolicies> SrcType;
  SrcType src(2.0, pbase, simd_dims_3d);

  typedef typename ComplexType::scalar_type ScalarComplexType;
  typedef FlavorMatrixGeneral<ComplexType> vFmatType;
  typedef FlavorMatrixGeneral<ScalarComplexType> sFmatType;
  
  vFmatType* into = (vFmatType*)managed_alloc_check(sizeof(vFmatType));

  auto src_v = src.view();
  
  typedef SIMT<ComplexType> ACC;

  //Check we can access the source sites
  {
    std::cout << "Checking site access" << std::endl;
    using namespace Grid;
    ComplexType *into_c = (ComplexType*)into;
   
    accelerator_for(x, 100, nsimd,
		  {
		    if(x==0){
		      typename ACC::value_type tmp = ACC::read(src_v.siteComplex(0));
		      ACC::write(*into_c,tmp);  
		    }
		  });
    ComplexType expect = src.siteComplex(0);

    ScalarComplexType got_c = Reduce(*into_c);
    ScalarComplexType expect_c = Reduce(expect);
    assert(got_c == expect_c);
    std::cout << "Checked site access" << std::endl;
  }
    
   
  {
    std::cout << "Checking Fmat access" << std::endl;
    using namespace Grid;
 
    accelerator_for(x, 100, nsimd,
		  {
		    if(x==0){
		      FlavorMatrixGeneral<typename ACC::value_type> tmp;
		      //sFmatType tmp;
		      src_v.siteFmat(tmp, 0);
		      for(int f1=0;f1<2;f1++)
			for(int f2=0;f2<2;f2++)
			  ACC::write( (*into)(f1,f2), tmp(f1,f2) );
		    }
		  });

    
    vFmatType expect;
    src.siteFmat(expect,0);
    
    assert(equals(*into,expect,1e-12));
  }

  managed_free(into);
  std::cout << "Passed testFlavorProjectedSourceView" << std::endl;
#endif
}


#endif //USE_GRID

CPS_END_NAMESPACE
