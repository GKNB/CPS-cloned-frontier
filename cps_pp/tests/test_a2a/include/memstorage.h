#pragma once

CPS_START_NAMESPACE

void testMemoryStorageBase(){
  std::cout << "Starting testMemoryStorageBase" << std::endl;
  MemoryStorageBase store;
  assert(!store.isInitialized());

  store.alloc(128, 10*sizeof(double));
  assert(store.isInitialized());

  {
    CPSautoView(store_v,store,HostWrite);    
    double* ptr = (double*)store_v();
    for(int i=0;i<10;i++) ptr[i] = i;
  }

  //Test copy constructor
  {
    MemoryStorageBase store2(store);
    assert(store2.isInitialized());
    CPSautoView(store2_v,store2,HostRead);    
    double* ptr2 = (double*)store2_v();    
    for(int i=0;i<10;i++) assert(ptr2[i] == double(i));
  }

  //Test copy
  {
    MemoryStorageBase store2;
    assert(!store2.isInitialized());
    store2 = store;
    CPSautoView(store2_v,store2,HostRead);
    double* ptr2 = (double*)store2_v();    
    for(int i=0;i<10;i++) assert(ptr2[i] == double(i));
  }

  //Test move
  {
    MemoryStorageBase store2(store);
    assert(store2.isInitialized());

    void* ptr2;
    {
      CPSautoView(store2_v,store2,HostRead);
      ptr2 = store2_v();
    }
    MemoryStorageBase store3;
    store3.move(store2);

    assert(!store2.isInitialized());
    assert(store3.isInitialized());
    CPSautoView(store3_v,store3,HostRead);
    assert(store3_v() == ptr2);
  }    

  //Test free
  store.freeMem();
  assert(!store.isInitialized());
  
  //Test use of external buffer
  void* ext = memalign(128,10*sizeof(double));
  store.enableExternalBuffer(ext, 10*sizeof(double), 128);
  
  store.alloc(128, 10*sizeof(double));
  {
    CPSautoView(store_v,store,HostRead);  
    assert(store_v() == ext);
  }
  store.freeMem();
 
  //Test disable
  store.disableExternalBuffer();
  
  store.alloc(128, 10*sizeof(double));
  {
    CPSautoView(store_v,store,HostRead);  
    assert(store_v() != ext);
  }
  std::cout << "testMemoryStorageBase passed" << std::endl;
}

class BurstBufferMemoryStorageTest: public BurstBufferMemoryStorage{
 public:
  bool isOnDisk() const{ return ondisk; }
  unsigned int getChecksum() const{ return ondisk_checksum; }
  const std::string &getFile() const{ return file; }
};


void testBurstBufferMemoryStorage(){
  std::cout << "Starting testBurstBufferMemoryStorage" << std::endl;
  BurstBufferMemoryStorage store;
  assert(!store.isInitialized());

  store.alloc(128, 10*sizeof(double));
  assert(store.isInitialized());

  {
    CPSautoView(store_v,store,HostWrite);
    double* ptr = (double*)store_v();
    for(int i=0;i<10;i++) ptr[i] = i;
  }

  store.distribute();

  assert(!store.isInitialized());
  assert( (( BurstBufferMemoryStorageTest& )store).isOnDisk() == true );
  assert( (( BurstBufferMemoryStorageTest& )store).getChecksum() != 0 );

  store.gather(true);
  assert(store.isInitialized() );

  {
    CPSautoView(store_v,store,HostRead);
    double* ptr = (double*)store_v();    
    for(int i=0;i<10;i++) assert(ptr[i] == double(i));
  }
  
  //Test copy uses different file
  BurstBufferMemoryStorage store2(store);
  assert( (( BurstBufferMemoryStorageTest& )store2).getFile() != (( BurstBufferMemoryStorageTest& )store).getFile() );
  {
    CPSautoView(store2_v,store2,HostRead);
    CPSautoView(store_v,store,HostRead);
    assert(store2_v() != store_v());
    
    double* ptr2 = (double*)store2_v();
    for(int i=0;i<10;i++) assert(ptr2[i] == double(i));
  }
  
  //Test move
  std::string f2 = (( BurstBufferMemoryStorageTest& )store2).getFile();
  void* d2;
  {
    CPSautoView(store2_v,store2,HostRead);
    d2 = store2_v();
  }
  BurstBufferMemoryStorage store3;
  store3.move(store2);
  assert( (( BurstBufferMemoryStorageTest& )store3).getFile() == f2 );
  assert( !store2.isInitialized());
  assert( (( BurstBufferMemoryStorageTest& )store2).isOnDisk() == false );

  CPSautoView(store3_v,store3,HostRead);  
  assert( store3_v() == d2 );

  double* ptr3 = (double*)store3_v();
  for(int i=0;i<10;i++) assert(ptr3[i] == double(i));

  std::cout << "testBurstBufferMemoryStorage passed" << std::endl;
}

void testDistributedStorage(){
  std::cout << "Starting testDistributedStorage" << std::endl;
  int nodes = 1;
  for(int i=0;i<4;i++) nodes *= GJP.Nodes(i);
  if(nodes > 1){
    DistributedMemoryStorage store;
    assert(!store.isInitialized() );

    store.alloc(128, 10*sizeof(double));
    assert(store.isInitialized() );

    {
      CPSautoView(store_v,store,HostWrite);
      double* ptr = (double*)store_v();
      for(int i=0;i<10;i++) ptr[i] = i;
    }

    store.distribute();

    assert(store.masterUID() != -1);
    if(store.masterUID() == UniqueID()) assert(store.isInitialized() );
    else  assert(!store.isInitialized() );

    store.gather(true);
    assert(store.isInitialized());

    void* store_p;
    {
      CPSautoView(store_v,store,HostRead);
      double* ptr = (double*)store_v();    
      for(int i=0;i<10;i++) assert(ptr[i] == double(i));
      store_p = store_v();
    }
  
    //Test copy 
    DistributedMemoryStorage store2(store);
    assert(store2.masterUID() == store.masterUID());
    {
      CPSautoView(store2_v,store2,HostRead);
      assert(store2_v() != store_p);
      double* ptr2 = (double*)store2_v();
      for(int i=0;i<10;i++) assert(ptr2[i] == double(i));
    }
  
    //Test move
    DistributedMemoryStorage store3;
    store3.move(store2);
    assert( !store2.isInitialized() );

    {
      CPSautoView(store3_v,store3,HostRead);
      double* ptr3 = (double*)store3_v();
      for(int i=0;i<10;i++) assert(ptr3[i] == double(i));
    }
  
    std::cout << "testDistributedStorage passed" << std::endl;
  }else{
    std::cout << "Requires >1 node" << std::endl;
  }
}

void testDistributedStorageOneSided(){
  std::cout << "Starting testDistributedStorageOneSided" << std::endl;
  int nodes = 1;
  for(int i=0;i<4;i++) nodes *= GJP.Nodes(i);
  if(nodes > 1){
    DistributedMemoryStorageOneSided store;
    assert(!store.isInitialized() );

    store.alloc(128, 10*sizeof(double));
    assert(store.isInitialized());

    {
      CPSautoView(store_v,store,HostWrite);
      double* ptr = (double*)store_v();
      for(int i=0;i<10;i++) ptr[i] = i;
    }

    store.distribute();

    assert(store.masterUID() != -1);
    if(store.masterUID() == UniqueID()) assert(store.isInitialized() );
    else  assert(!store.isInitialized() );

    store.gather(true);
    assert(store.isInitialized() );

    void *store_p;
    {
      CPSautoView(store_v,store,HostRead);
      double* ptr = (double*)store_v();
      for(int i=0;i<10;i++) assert(ptr[i] == double(i));
      store_p = store_v();
    }
  
    //Test copy 
    DistributedMemoryStorageOneSided store2(store);
    assert(store2.masterUID() == store.masterUID());

    {
      CPSautoView(store2_v,store2,HostRead);
      assert(store2_v() != store_p);
      double* ptr2 = (double*)store2_v();
      for(int i=0;i<10;i++) assert(ptr2[i] == double(i));
    }
    
    //Test move
    DistributedMemoryStorageOneSided store3;
    store3.move(store2);
    assert( !store2.isInitialized() );

    CPSautoView(store3_v,store3,HostRead);
    double* ptr3 = (double*)store3_v();
    for(int i=0;i<10;i++) assert(ptr3[i] == double(i));
  
    std::cout << "testDistributedStorageOneSided passed" << std::endl;
  }else{
    std::cout << "Requires >1 node" << std::endl;
  }
}



void testMmapMemoryStorage(){
  std::cout << "Starting testMmapMemoryStorage" << std::endl;
  MmapMemoryStorage store;
  assert(!store.isInitialized() );

  store.alloc(128, 10*sizeof(double));
  assert(store.isInitialized() );

  {
    CPSautoView(store_v,store,HostWrite);
    double* ptr = (double*)store_v();
    for(int i=0;i<10;i++) ptr[i] = i;
  }
    
  store.distribute();
  
  //Can still read here, distribute just advises the OS that the pages are not expected to be needed soon
  assert(store.isInitialized() );

  void* store_p ;
  {
    CPSautoView(store_v,store,HostRead);
    double* ptr = (double*)store_v();
    for(int i=0;i<10;i++) assert(ptr[i] == double(i));
    store_p = store_v();
  }

  store.gather(true);
  assert(store.isInitialized() );
  
   
  //Test copy uses different
  MmapMemoryStorage store2(store);
  assert( store2.getFilename() != store.getFilename() );
  {
    CPSautoView(store2_v,store2,HostRead);
    assert(store2_v() != store_p);
    double* ptr2 = (double*)store2_v();
    for(int i=0;i<10;i++) assert(ptr2[i] == double(i));
  }
  
  //Test move
  MmapMemoryStorage store3;
  store3.move(store2);
  assert( store2.getFilename() == "" );
  assert( !store2.isInitialized() );

  CPSautoView(store3_v,store3,HostRead);
  double* ptr3 = (double*)store3_v();
  for(int i=0;i<10;i++) assert(ptr3[i] == double(i));
  

  std::cout << "testMmapMemoryStorage passed" << std::endl;
}

class DeviceMemoryPoolManagerTest: public DeviceMemoryPoolManager{
public:
  DeviceMemoryPoolManagerTest(): DeviceMemoryPoolManager(){}
  DeviceMemoryPoolManagerTest(size_t max_size): DeviceMemoryPoolManager(max_size){}
};

void testPoolAllocator(){
  std::cout << "Starting testPoolAllocator" << std::endl;
  size_t MB = 1024*1024;
  size_t max_size = 3*MB;

  //Test the device entry logic
  {  
    DeviceMemoryPoolManagerTest pool(max_size);
    pool.setVerbose(true);
    auto h1 = pool.allocate(1*MB);
    assert(h1->valid);
    assert(pool.getAllocated() == 1*MB);

    auto h2 = pool.allocate(2*MB);
    assert(h2->valid);
    assert(h1->valid);
    assert(pool.getAllocated() == 3*MB);

    std::cout << ">>>>testPoolAllocator test check eviction for untouched data" << std::endl;
    //should steal from h1
    void* expect_ptr = h1->entry->ptr;
    auto h3 = pool.allocate(1*MB); 
    assert(h2->valid);
    assert(h3->valid);
    assert(!h1->valid);
    assert(pool.getAllocated() == 3*MB);
    assert(h3->entry->ptr == expect_ptr);

    std::cout << ">>>>testPoolAllocator test check eviction for untouched data 2" << std::endl;    
    //if we allocate another 1MB it should now evict h2
    //however h2's size is 2MB rather than 1MB so it should do a new allocation
    void* not_expect_ptr = h2->entry->ptr;
    auto h4 = pool.allocate(1*MB);
    assert(h3->valid);
    assert(h4->valid);
    assert(!h2->valid);
    //assert(h4->entry->ptr != not_expect_ptr); //it can happen that the allocation occurs in the newly freed segment!
    assert(pool.getAllocated() == 2*MB);

    //now we should have enough room for another full allocation
    auto h5 = pool.allocate(1*MB);
    assert(h3->valid);
    assert(h4->valid);
    assert(h5->valid);
    

    std::cout << ">>>>testPoolAllocator test eviction for touched data" << std::endl;
    //if we touch h3, h4 should become the next eviction target
    pool.openView(DeviceRead,h3);
    pool.closeView(h3);
    expect_ptr = h4->entry->ptr;
    auto h6 = pool.allocate(1*MB);
    assert(h3->valid);
    assert(!h4->valid);
    assert(h6->valid);
    assert(h6->entry->ptr == expect_ptr);
    assert(pool.getAllocated() == 3*MB);

    std::cout << ">>>>testPoolAllocator test view open for previously evicted data" << std::endl;
    assert(!h1->valid);    
    pool.openView(DeviceRead,h1);
    assert(h1->valid);
    pool.closeView(h1);

    std::cout << ">>>>testPoolAllocator test free operation" << std::endl;
    assert(pool.getAllocated() == 3*MB);
    expect_ptr = h1->entry->ptr;
    pool.free(h1);
    assert(pool.getAllocated() == 3*MB); //doesn't free memory!
    auto h7 = pool.allocate(1*MB);
    assert(h7->entry->ptr == expect_ptr); //should reuse the new free block

    pool.free(h2); pool.free(h3); pool.free(h4); pool.free(h5); pool.free(h6); pool.free(h7);
    
    std::cout << ">>>>testPoolAllocator test subset teardown" << std::endl;
  }

  {
    std::cout << ">>>>testPoolAllocator test free pool operation" << std::endl;
    DeviceMemoryPoolManagerTest pool(max_size);
    pool.setVerbose(true);
    auto h1 = pool.allocate(1*MB);
    auto h2 = pool.allocate(1*MB);

    //no room for a 2MB allocation
    //put two entries into free pool
    pool.free(h1);
    pool.free(h2);

    //it should free up both of them to make room
    auto h3 = pool.allocate(2*MB);
    assert(h3->valid);
    pool.free(h3);
    std::cout << ">>>>testPoolAllocator test subset teardown" << std::endl;    
  }

  {
    std::cout << ">>>>testPoolAllocator testing views" << std::endl;
    size_t bs = 100*sizeof(double);
    DeviceMemoryPoolManagerTest pool(bs);
    pool.setVerbose(true);

    double* tmp_gpu = (double*)device_alloc_check(bs);
    double* tmp_host = (double*)malloc(bs);
    device_memset(tmp_gpu,0,bs);
    memset(tmp_host,0,bs);
  
    auto h = pool.allocate(bs);

    //Write some data on host
    {
      double* p = (double*)pool.openView(HostWrite,h);
      for(int i=0;i<100;i++) *p++ = i;
      pool.closeView(h);
    }
    {
      double const* pr = (double const*)pool.openView(DeviceRead,h);
      using namespace Grid;
      //Copy it to temp buffer on device
      accelerator_for(i, 100, 1, {
	  tmp_gpu[i] = pr[i];
	});    	
      pool.closeView(h);
      //Overwrite with new values
      double* pw = (double*)pool.openView(DeviceWrite,h);
      accelerator_for(i, 100, 1, {
	  pw[i] = 100 + i;
	});    	
      pool.closeView(h);
    }
    {
      //Host check
      bool fail = false;
      copy_device_to_host(tmp_host,tmp_gpu,bs);
      for(int i=0;i<100;i++)
	if(tmp_host[i] != i){ std::cout << "FAIL host check 1 got "<< tmp_host[i] << " expect " << i << std::endl; fail = true; }
          
      if(fail) ERR.General("","testPoolAllocator","View test failed");

      double const* p = (double const*)pool.openView(HostRead,h);
      for(int i=0;i<100;i++)
	if(p[i] != 100+i){ std::cout << "FAIL host check 2 got " << p[i] << " expect " << 100+i << std::endl; fail = true; }
      pool.closeView(h);

      if(fail) ERR.General("","testPoolAllocator","View test 2 failed");
    }

    //Write some data on the device
    {
      //Overwrite with new values
      double* pw = (double*)pool.openView(DeviceWrite,h);
      using namespace Grid;
      accelerator_for(i, 100, 1, {
	  pw[i] = 314 + 9*i;
	});    	
      pool.closeView(h);
    }

    //Now force that entry to be evicted
    auto h2 = pool.allocate(bs);
    assert(!h->valid);
    //Overwrite the data to be sure
    {
      //Overwrite with new values
      double* pw = (double*)pool.openView(DeviceWrite,h2);
      using namespace Grid;
      accelerator_for(i, 100, 1, {
	  pw[i] = 0;
	});    	
      pool.closeView(h2);
    }

    //Check the data was properly copied to the host when the eviction occurred
    {
      bool fail = false;
      double const* p = (double const*)pool.openView(HostRead,h);
      for(int i=0;i<100;i++)
	if(p[i] != 314+9*i){ std::cout << "FAIL host check 3 got " << p[i] << " expect " << 314+9*i << std::endl; fail = true; }
      pool.closeView(h);
      if(fail) ERR.General("","testPoolAllocator","View test 3 failed");
    }
    
    //Open a device read view, it should evict h2
    {
      bool fail = false;
      double const* pr = (double const*)pool.openView(DeviceRead,h);
      assert(!h2->valid);
      using namespace Grid;
      //Copy it to temp buffer on device
      accelerator_for(i, 100, 1, {
	  tmp_gpu[i] = pr[i];
	});    	
      pool.closeView(h);
    
      //Host check
      copy_device_to_host(tmp_host,tmp_gpu,bs);
      for(int i=0;i<100;i++)
	if(tmp_host[i] != 314+9*i){ std::cout << "FAIL host check 4 got "<< tmp_host[i] << " expect " << 314+9*i << std::endl; fail = true; }
      if(fail) ERR.General("","testPoolAllocator","View test 4 failed");
    }

    pool.free(h);
    pool.free(h2);
    device_free(tmp_gpu);
    free(tmp_host);
  }

  {
    std::cout << ">>>>testPoolAllocator testing prefetch logic" << std::endl;
    asyncTransferManager::globalInstance().setVerbose(true);
    
    DeviceMemoryPoolManagerTest pool(max_size);
    pool.setVerbose(true);
    auto h1 = pool.allocate(1*MB);
    auto h2 = pool.allocate(1*MB);
    
    //Desynchronize device side    
    double* v1 = (double*)pool.openView(HostWrite,h1);
    double* v2 = (double*)pool.openView(HostWrite,h2);

    v1[0] = 3.142;
    v2[0] = 6.284;
    
    pool.closeView(h1);
    pool.closeView(h2);
    assert(!h1->device_in_sync);
    assert(!h1->lock_entry);
    
    //Enqueue prefetches
    pool.enqueuePrefetch(DeviceRead,h1);
    pool.enqueuePrefetch(DeviceRead,h2);

    //Lock should be engaged, and it should claim to be in sync but is not
    assert(h1->device_in_sync);
    assert(h1->lock_entry);

    pool.startPrefetches();
    pool.waitPrefetches();

    //Lock should be disengaged and data on device
    assert(!h1->lock_entry);
    assert(h1->device_in_sync);
    
    double r1, r2;
    copy_device_to_host(&r1,pool.openView(DeviceRead,h1),sizeof(double));
    copy_device_to_host(&r2,pool.openView(DeviceRead,h2),sizeof(double));
    assert(r1 == 3.142);
    assert(r2 == 6.284);
    pool.closeView(h1);
    pool.closeView(h2);

    pool.free(h1); pool.free(h2);
  }

  asyncTransferManager::globalInstance().setVerbose(false);
  std::cout << "testPoolAllocator passed" << std::endl;
}


void testHolisticPoolAllocator(){
  std::cout << "Starting testHolisticPoolAllocator" << std::endl;
  size_t MB = 1024*1024;
  size_t max_size = 3*MB;

  {
    std::cout << "TEST: Test device pool eviction to host" << std::endl;
    HolisticMemoryPoolManager pool(max_size, max_size);  
    pool.setVerbose(true);

    //Allocate full size on device
    auto h1 = pool.allocate(max_size, HolisticMemoryPoolManager::DevicePool);
    assert(h1->device_valid);
    assert(pool.getAllocated(HolisticMemoryPoolManager::DevicePool) == max_size);

    //Touch the device data to mark device side in sync
    {
      void* p = pool.openView(DeviceWrite,h1);
      device_memset(p,0x1A,max_size);
      pool.closeView(h1);
    }

    //Try to allocate again on device, it should evict the current entry to host
    auto h2 = pool.allocate(max_size, HolisticMemoryPoolManager::DevicePool);
    assert(h2->device_valid);
    assert(pool.getAllocated(HolisticMemoryPoolManager::DevicePool) == max_size);

    assert(!h1->device_valid);
    assert(h1->host_valid);
    assert(pool.getAllocated(HolisticMemoryPoolManager::HostPool) == max_size);
    {
      char* p = (char*)pool.openView(HostRead,h1);
      assert(*p == 0x1A);
      pool.closeView(h1);
    }
    
  }

  {
    std::cout << "TEST: Test host pool eviction to disk" << std::endl;
    HolisticMemoryPoolManager pool(max_size, max_size);  
    pool.setVerbose(true);

    //Allocate full size on host
    auto h1 = pool.allocate(max_size, HolisticMemoryPoolManager::HostPool);
    assert(h1->host_valid);
    assert(pool.getAllocated(HolisticMemoryPoolManager::HostPool) == max_size);

    //Touch the host data to mark host side in sync
    {
      void* p = pool.openView(HostWrite,h1);
      memset(p,0x1A,max_size);
      pool.closeView(h1);
    }

    //Try to allocate again on host, it should evict the current entry to disk
    auto h2 = pool.allocate(max_size, HolisticMemoryPoolManager::HostPool);
    assert(h2->host_valid);
    assert(pool.getAllocated(HolisticMemoryPoolManager::HostPool) == max_size);

    assert(!h1->host_valid);
    assert(h1->disk_in_sync);
    {
      char* p = (char*)pool.openView(HostRead,h1); //pull back from disk, should also evict h2
      assert(*p == 0x1A);
      pool.closeView(h1);
    }
    assert(!h2->host_valid);
    //assert(h2->disk_in_sync); //actually no, because data was never written to h2 there is no need to store it when evicted

    std::cout << "TEST: Check (manually) that the files are deleted:" << std::endl;
    pool.free(h1);
    //pool.free(h2);
    
    
  }
  
  std::cout << "testHolisticPoolAllocator passed" << std::endl;
}


CPS_END_NAMESPACE
