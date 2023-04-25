#pragma once

CPS_START_NAMESPACE

void testMemoryStorageBase(){
  std::cout << "Starting testMemoryStorageBase" << std::endl;
  MemoryStorageBase store;
  assert(store.data() == NULL);

  store.alloc(128, 10*sizeof(double));
  assert(store.data() != NULL);
  
  double* ptr = (double*)store.data();
  for(int i=0;i<10;i++) ptr[i] = i;

  //Test copy constructor
  {
    MemoryStorageBase store2(store);
    assert(store2.data() != NULL);
    double* ptr2 = (double*)store2.data();
    
    for(int i=0;i<10;i++) assert(ptr2[i] == double(i));
  }

  //Test copy
  {
    MemoryStorageBase store2;
    assert(store2.data() == NULL);
    store2 = store;

    double* ptr2 = (double*)store2.data();
    
    for(int i=0;i<10;i++) assert(ptr2[i] == double(i));
  }

  //Test move
  {
    MemoryStorageBase store2(store);
    assert(store2.data() != NULL);
    
    void* ptr2 = store2.data();

    MemoryStorageBase store3;
    store3.move(store2);

    assert(store2.data() == NULL);
    assert(store3.data() == ptr2);
  }    

  //Test free
  store.freeMem();
  assert(store.data() == NULL);
  
  //Test use of external buffer
  void* ext = memalign(128,10*sizeof(double));
  store.enableExternalBuffer(ext, 10*sizeof(double), 128);
  
  store.alloc(128, 10*sizeof(double));
  assert(store.data() == ext);
  store.freeMem();
 
  //Test disable
  store.disableExternalBuffer();
  
  store.alloc(128, 10*sizeof(double));
  assert(store.data() != ext);
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
  assert(store.data() == NULL);

  store.alloc(128, 10*sizeof(double));
  assert(store.data() != NULL);
  
  double* ptr = (double*)store.data();
  for(int i=0;i<10;i++) ptr[i] = i;

  store.distribute();

  assert(store.data() == NULL);
  assert( (( BurstBufferMemoryStorageTest& )store).isOnDisk() == true );
  assert( (( BurstBufferMemoryStorageTest& )store).getChecksum() != 0 );

  store.gather(true);
  assert(store.data() != NULL);
  
  ptr = (double*)store.data();
    
  for(int i=0;i<10;i++) assert(ptr[i] == double(i));
  
  //Test copy uses different file
  BurstBufferMemoryStorage store2(store);
  assert( (( BurstBufferMemoryStorageTest& )store2).getFile() != (( BurstBufferMemoryStorageTest& )store).getFile() );
  assert(store2.data() != store.data());

  double* ptr2 = (double*)store2.data();
  for(int i=0;i<10;i++) assert(ptr2[i] == double(i));
  
  //Test move
  std::string f2 = (( BurstBufferMemoryStorageTest& )store2).getFile();
  void* d2 = store2.data();
  BurstBufferMemoryStorage store3;
  store3.move(store2);
  assert( (( BurstBufferMemoryStorageTest& )store3).getFile() == f2 );
  assert( store2.data() == NULL );
  assert( (( BurstBufferMemoryStorageTest& )store2).isOnDisk() == false );
  assert( store3.data() == d2 );

  double* ptr3 = (double*)store3.data();
  for(int i=0;i<10;i++) assert(ptr3[i] == double(i));

  std::cout << "testBurstBufferMemoryStorage passed" << std::endl;
}

void testDistributedStorage(){
  std::cout << "Starting testDistributedStorage" << std::endl;
  int nodes = 1;
  for(int i=0;i<4;i++) nodes *= GJP.Nodes(i);
  if(nodes > 1){
    DistributedMemoryStorage store;
    assert(store.data() == NULL);

    store.alloc(128, 10*sizeof(double));
    assert(store.data() != NULL);
  
    double* ptr = (double*)store.data();
    for(int i=0;i<10;i++) ptr[i] = i;

    store.distribute();

    assert(store.masterUID() != -1);
    if(store.masterUID() == UniqueID()) assert(store.data() != NULL);
    else  assert(store.data() == NULL);

    store.gather(true);
    assert(store.data() != NULL);
  
    ptr = (double*)store.data();
    
    for(int i=0;i<10;i++) assert(ptr[i] == double(i));
  
    //Test copy 
    DistributedMemoryStorage store2(store);
    assert(store2.data() != store.data());
    assert(store2.masterUID() == store.masterUID());

    double* ptr2 = (double*)store2.data();
    for(int i=0;i<10;i++) assert(ptr2[i] == double(i));
  
    //Test move
    DistributedMemoryStorage store3;
    store3.move(store2);
    assert( store2.data() == NULL );

    double* ptr3 = (double*)store3.data();
    for(int i=0;i<10;i++) assert(ptr3[i] == double(i));
  
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
    assert(store.data() == NULL);

    store.alloc(128, 10*sizeof(double));
    assert(store.data() != NULL);
  
    double* ptr = (double*)store.data();
    for(int i=0;i<10;i++) ptr[i] = i;

    store.distribute();

    assert(store.masterUID() != -1);
    if(store.masterUID() == UniqueID()) assert(store.data() != NULL);
    else  assert(store.data() == NULL);

    store.gather(true);
    assert(store.data() != NULL);
  
    ptr = (double*)store.data();
    
    for(int i=0;i<10;i++) assert(ptr[i] == double(i));
  
    //Test copy 
    DistributedMemoryStorageOneSided store2(store);
    assert(store2.data() != store.data());
    assert(store2.masterUID() == store.masterUID());

    double* ptr2 = (double*)store2.data();
    for(int i=0;i<10;i++) assert(ptr2[i] == double(i));
  
    //Test move
    DistributedMemoryStorageOneSided store3;
    store3.move(store2);
    assert( store2.data() == NULL );

    double* ptr3 = (double*)store3.data();
    for(int i=0;i<10;i++) assert(ptr3[i] == double(i));
  
    std::cout << "testDistributedStorageOneSided passed" << std::endl;
  }else{
    std::cout << "Requires >1 node" << std::endl;
  }
}



void testMmapMemoryStorage(){
  std::cout << "Starting testMmapMemoryStorage" << std::endl;
  MmapMemoryStorage store;
  assert(store.data() == NULL);

  store.alloc(128, 10*sizeof(double));
  assert(store.data() != NULL);
  
  double* ptr = (double*)store.data();
  for(int i=0;i<10;i++) ptr[i] = i;

  store.distribute();
  
  //Can still read here, distribute just advises the OS that the pages are not expected to be needed soon
  assert(store.data() != NULL);
  for(int i=0;i<10;i++) assert(ptr[i] == double(i));

  store.gather(true);
  assert(store.data() != NULL);
  
   
  //Test copy uses different
  MmapMemoryStorage store2(store);
  assert( store2.getFilename() != store.getFilename() );
  assert(store2.data() != store.data());

  double* ptr2 = (double*)store2.data();
  for(int i=0;i<10;i++) assert(ptr2[i] == double(i));
  
  //Test move
  MmapMemoryStorage store3;
  store3.move(store2);
  assert( store2.getFilename() == "" );
  assert( store2.data() == NULL );
  double* ptr3 = (double*)store3.data();
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
    assert(h4->entry->ptr != not_expect_ptr);
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


  }

  std::cout << "testPoolAllocator passed" << std::endl;
}

CPS_END_NAMESPACE
