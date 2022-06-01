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
  
  //Test copy uses same file
  BurstBufferMemoryStorage store2(store);
  assert( (( BurstBufferMemoryStorageTest& )store2).getFile() == (( BurstBufferMemoryStorageTest& )store).getFile() );
  assert(store2.data() != store.data());

  double* ptr2 = (double*)store2.data();
  for(int i=0;i<10;i++) assert(ptr2[i] == double(i));
  
  //Test move
  BurstBufferMemoryStorage store3;
  store3.move(store2);
  assert( (( BurstBufferMemoryStorageTest& )store3).getFile() == (( BurstBufferMemoryStorageTest& )store).getFile() );
  assert( store2.data() == NULL );
  assert( (( BurstBufferMemoryStorageTest& )store2).isOnDisk() == false );
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



CPS_END_NAMESPACE
