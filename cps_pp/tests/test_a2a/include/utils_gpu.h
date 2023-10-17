#pragma once

CPS_START_NAMESPACE

void testhostDeviceMirroredContainer(){
  std::cout << "Starting testhostDeviceMirroredContainer" << std::endl;
  size_t bs = 100*sizeof(double);
  hostDeviceMirroredContainer<double> buf(100);
  double* tmp_gpu = (double*)device_alloc_check(bs);
  double* tmp_host = (double*)malloc(bs);
  device_memset(tmp_gpu,0,bs);
  memset(tmp_host,0,bs);
  
  //Write some data on host
  {
    double* p = buf.getHostWritePtr();
    for(int i=0;i<100;i++) *p++ = i;
  }
  {
    double const* pr = buf.getDeviceReadPtr();
    using namespace Grid;
    //Copy it to temp buffer on device
    accelerator_for(i, 100, 1, {
	tmp_gpu[i] = pr[i];
      });    	
    //Overwrite with new values
    double* pw = buf.getDeviceWritePtr();
    accelerator_for(i, 100, 1, {
	pw[i] = 100 + i;
      });    	
  }
  {
    //Host check
    bool fail = false;
    copy_device_to_host(tmp_host,tmp_gpu,bs);
    for(int i=0;i<100;i++)
      if(tmp_host[i] != i){ std::cout << "FAIL host check 1 got "<< tmp_host[i] << " expect " << i << std::endl; fail = true; }
    
    double const* p = buf.getHostReadPtr();
    for(int i=0;i<100;i++)
      if(p[i] != 100+i){ std::cout << "FAIL host check 2 got " << p[i] << " expect " << 100+i << std::endl; fail = true; }

    if(fail) ERR.General("","testhostDeviceMirroredContainer","Blocking copy test failed");
  }
  
  //Check asynchronous copies
  buf.reset();
  {
    double* p = buf.getHostWritePtr();
    for(int i=0;i<100;i++) *p++ = 3*i;
  } 
  buf.asyncHostDeviceSync();
  device_synchronize_all();
  if(!buf.deviceInSync()) ERR.General("","testhostDeviceMirroredContainer","Expect device in sync");

  hostDeviceMirroredContainer<double> buf2(100);
  
  {
    {
      using namespace Grid;
      double const* pr = buf.getDeviceReadPtr();
      //Copy it to temp buffer on device
      accelerator_for(i, 100, 1, {
	  tmp_gpu[i] = pr[i];
	});
    }
    {
      using namespace Grid;    
      //Overwrite with new values
      double* pw = buf2.getDeviceWritePtr();
      accelerator_for(i, 100, 1, {
	  pw[i] = 100 + 3*i;
	});
    }
  }
  {
    //Host check
    bool fail = false;
    copy_device_to_host(tmp_host,tmp_gpu,bs);
    for(int i=0;i<100;i++)
      if(tmp_host[i] != 3*i){ std::cout << "FAIL async host check 1 got "<< tmp_host[i] << " expect " << 3*i << std::endl; fail = true; }

    buf2.asyncHostDeviceSync();
    device_synchronize_all();
    if(!buf2.hostInSync()) ERR.General("","testhostDeviceMirroredContainer","Expect host in sync");

#ifdef GRID_HIP
    hipError_t e = hipStreamQuery(Grid::copyStream);
    hipError_t f = hipStreamQuery(Grid::computeStream);
    if(e == hipSuccess){ std::cout << "HIP runtime reports all copyStream operations are complete" << std::endl; }
    else{ std::cout << "!HIP runtime reports all copyStream operations are NOT complete" << std::endl; }
    if(f == hipSuccess){ std::cout << "HIP runtime reports all computeStream operations are complete" << std::endl; }
    else{ std::cout << "!HIP runtime reports all computeStream operations are NOT complete" << std::endl; }
#endif
    double const* p = buf2.getHostReadPtr();
    for(int i=0;i<100;i++)
      if(p[i] != 100+3*i){ std::cout << "FAIL async host check 2 got " << p[i] << " expect " << 100+3*i << std::endl; fail = true; }

    if(fail) ERR.General("","testhostDeviceMirroredContainer","Async copy test failed");
  }

  device_free(tmp_gpu);
  free(tmp_host);
  
  std::cout << "testhostDeviceMirroredContainer passed" << std::endl;
}



void testAsyncTransferManager(){
  double *vfrom = (double*)device_alloc_check(128,1000*sizeof(double));
  double *vto = (double*)device_alloc_check(128,1000*sizeof(double));  
  
  size_t MB=1024*1024; 
  void *ah = memalign(128,30*MB);
  void *bh = memalign(128,40*MB);
  void* ch = memalign(128,50*MB);

  void* ad = device_alloc_check(128,30*MB);
  void* bd = device_alloc_check(128,40*MB);
  void* cd = device_alloc_check(128,50*MB);

  asyncTransferManager man;
  man.setVerbose(true);
  for(int i=0;i<100;i++){  
    man.enqueue(ad,ah,30*MB);
    man.enqueue(bd,bh,40*MB);
    man.enqueue(cd,ch,50*MB);
  }
  man.start();
  std::cout << Grid::GridLogMessage << "Kernel launch" << std::endl;
  {
    using namespace Grid;
    accelerator_for(i, 1000, 1,{
	size_t n = 8000000;
	double v = vfrom[i];
	for(size_t a=0;a<n;a++)
	  v = 3.141*v+v;
	vto[i] = v;
      });
  }
  std::cout << Grid::GridLogMessage << "Kernel end" << std::endl;
  man.wait();
  std::cout << Grid::GridLogMessage << "Wait complete" << std::endl;
  
  free(ah);
  free(bh);
  free(ch);
  device_free(ad);
  device_free(bd);
  device_free(cd);
  device_free(vfrom);
  device_free(vto);
}

void testVectorWithAview(){
  std::cout << "Starting testVectorWithAview" << std::endl;
  {
    //Test default constructor    
    VectorWithAview<double, ExplicitCopyDiskBackedPoolAllocPolicy> v;
    assert(v.size() == 0);
    
    //Test resize
    v.resize(100);
    assert(v.size() == 100);

    //Test ability to write to full array on host
    {
      CPSautoView(v_v,v,HostWrite);
      for(int i=0;i<100;i++) v_v[i] = (double)i;
    }
    {
      CPSautoView(v_v,v,HostRead);
      for(int i=0;i<100;i++) assert(v_v[i] == (double)i);
    }

    //Test ability to write to full array on device
    {
      CPSautoView(v_v,v,DeviceWrite);
      using namespace Grid;
      accelerator_for(i, 100, 1, {
	  v_v[i] = (double)(2*i);
	});
    }
    {
      CPSautoView(v_v,v,HostRead);
      for(int i=0;i<100;i++) assert(v_v[i] == (double)(2*i));
    }
  }

  {
    //Test size initialization
    VectorWithAview<double, ExplicitCopyDiskBackedPoolAllocPolicy> v(100);
    assert(v.size() == 100);

    {
      CPSautoView(v_v,v,HostWrite);
      for(int i=0;i<100;i++) v_v[i] = (double)(3*i);
    }
    //Test copy construction
    VectorWithAview<double, ExplicitCopyDiskBackedPoolAllocPolicy> w(v);

    {
      CPSautoView(w_v,v,HostRead);
      for(int i=0;i<100;i++) assert(w_v[i] == (double)(3*i));
    }
    
    //Test move construction
    void* pw;
    {
      CPSautoView(w_v,w,HostRead);
      pw = w_v();
    }

    VectorWithAview<double, ExplicitCopyDiskBackedPoolAllocPolicy> x(std::move(w));
    assert(x.size() == 100);
    assert(w.size() == 0);
    
    {
      CPSautoView(x_v,x,HostRead);
      for(int i=0;i<100;i++) assert(x_v[i] == (double)(3*i));
    }
    
    void *px;
    {
      CPSautoView(x_v,x,HostRead);
      px = x_v();
    }
    assert(px == pw);
  }

  {
    VectorWithAview<double, ExplicitCopyDiskBackedPoolAllocPolicy> v(100);
    {
      CPSautoView(v_v,v,HostWrite);
      for(int i=0;i<100;i++) v_v[i] = (double)(3*i);
    }
    //Test copy
    VectorWithAview<double, ExplicitCopyDiskBackedPoolAllocPolicy> w;
    w = v;

    {
      CPSautoView(w_v,v,HostRead);
      for(int i=0;i<100;i++) assert(w_v[i] == (double)(3*i));
    }
    
    //Test move
    void* pw;
    {
      CPSautoView(w_v,w,HostRead);
      pw = w_v();
    }

    VectorWithAview<double, ExplicitCopyDiskBackedPoolAllocPolicy> x;
    x = std::move(w);
    assert(x.size() == 100);
    assert(w.size() == 0);
    
    {
      CPSautoView(x_v,x,HostRead);
      for(int i=0;i<100;i++) assert(x_v[i] == (double)(3*i));
    }
    
    void *px;
    {
      CPSautoView(x_v,x,HostRead);
      px = x_v();
    }
    assert(px == pw);
  }
  std::cout << "Passed testVectorWithAview" << std::endl;
}


void test_mmap_alloc(){
  void* p = mmap_alloc_check(128, 1024);
  assert(isAligned(p,128));
  mmap_free(p);

  p = mmap_alloc_check(32, 1024);
  assert(isAligned(p,32));
  mmap_free(p);

  p = mmap_alloc_check(1, 1024);
  mmap_free(p);
}

void test_write_data_bypass_cache(){
  //Try with data size smaller than 4kB
  {
    int nd = 128;
    size_t sz = nd*sizeof(double);
    double* p = (double*)malloc_check(sz);
    for(int i=0;i<nd;i++) p[i] = double(i);
    write_data_bypass_cache("test.dat", (char*)p, sz);

    double* q = (double*)malloc_check(sz);
    read_data_bypass_cache("test.dat", (char*)q, sz);  

    for(int i=0;i<nd;i++){
      assert(p[i] == q[i]);
      assert(q[i] == double(i));
    }
    free(p);
    free(q);
  }

  //Try with data size smaller than 10MB
  {
    int nd = 600;
    size_t sz = nd*sizeof(double);
    double* p = (double*)malloc_check(sz);
    for(int i=0;i<nd;i++) p[i] = double(i);
    write_data_bypass_cache("test.dat", (char*)p, sz);

    double* q = (double*)malloc_check(sz);
    read_data_bypass_cache("test.dat", (char*)q, sz);  

    for(int i=0;i<nd;i++){
      assert(p[i] == q[i]);
      assert(q[i] == double(i));
    }
    free(p);
    free(q);
  }


  {
    int nd = 20*1024*1024/8;
    size_t sz = nd*sizeof(double);
    double* p = (double*)malloc_check(sz);
    for(int i=0;i<nd;i++) p[i] = double(i);
    write_data_bypass_cache("test.dat", (char*)p, sz);

    double* q = (double*)malloc_check(sz);
    read_data_bypass_cache("test.dat", (char*)q, sz);  

    for(int i=0;i<nd;i++){
      assert(p[i] == q[i]);
      assert(q[i] == double(i));
    }
    free(p);
    free(q);
  }



}
  


CPS_END_NAMESPACE
