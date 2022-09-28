#ifndef _UTILS_MEMSTORAGE_H_
#define _UTILS_MEMSTORAGE_H_

//"Storage" classes for managing memory

#include<util/time_cps.h>
#include "utils_malloc.h"
#include "utils_parallel.h"
#include "reuse_block_allocator.h"

CPS_START_NAMESPACE

struct nodeDistributeCounter{
  //Get a rank idx that cycles between 0... nodes-1
  static int getNext(){
    static int cur = 0;
    static int nodes = -1;
    if(nodes == -1){
      nodes = 1; for(int i=0;i<5;i++) nodes *= GJP.Nodes(i);
    }
    int out = cur;
    cur = (cur + 1) % nodes;
    return out;
  }

  //Keep tally of the number of MF uniquely stored on this node
  static int incrOnNodeCount(const int by){
    static int i = 0;
    i += by;
    return i;
  }
  static int onNodeCount(){
    return incrOnNodeCount(0);
  }

};

class MemoryStorageBase{
protected:
  void *ptr;
  int _alignment;
  size_t _size;

  //Functionality to use an externally provided memory location
  bool use_external_buffer;
  void *external_buffer;
  size_t external_buffer_size;
  int external_buffer_alignment;

public:

  MemoryStorageBase(): ptr(NULL), use_external_buffer(false), _size(0), _alignment(0){}

  MemoryStorageBase(const MemoryStorageBase &r): ptr(NULL), use_external_buffer(false), _size(0), _alignment(0){
    if(r.ptr != NULL){    
      alloc(r._alignment, r._size);
      memcpy(ptr, r.ptr, r._size);
    }else{
      _size = r._size;
      _alignment = r._alignment;
    }      
  }
  
  MemoryStorageBase & operator=(const MemoryStorageBase &r){
    if(r.ptr != NULL){
      if(_size != r._size || _alignment != r._alignment){
	freeMem();
	alloc(r._alignment, r._size);
      }
      memcpy(ptr, r.ptr, r._size);      
    }else{
      freeMem();
      ptr = NULL;
      _size = r._size;
      _alignment = r._alignment;
    }

    return *this;
  }

  ~MemoryStorageBase(){ freeMem(); }

  inline bool isOnNode() const{ return ptr != NULL; }

  //Future memory allocations will use this external buffer
  void enableExternalBuffer(void* p, size_t sz, int align){
    use_external_buffer = true;
    external_buffer = p;
    external_buffer_size = sz;
    external_buffer_alignment = align;
  }

  void disableExternalBuffer(){
    if(!use_external_buffer) return;
    if(ptr == external_buffer) ERR.General("MemoryStorageBase","disableExternalBuffer","Cannot disable external buffer if it is being used to store data!");
    use_external_buffer = false;
    external_buffer = NULL;
    external_buffer_size = 0;
    external_buffer_alignment = 0;
  }

#ifdef DISTRIBUTED_MEMORY_STORAGE_REUSE_MEMORY
  inline static ReuseBlockAllocatorsAligned & block_allocator(){ static ReuseBlockAllocatorsAligned r; return r; }
#endif

  void alloc(int alignment, size_t size){
    if(ptr != NULL){
      if(alignment == _alignment && size == _size) return;
      else{ 
	if(use_external_buffer) ERR.General("DistributedMemoryStorage","alloc","Currently using external buffer but size and alignment of alloc call does not match that of buffer");
#ifdef DISTRIBUTED_MEMORY_STORAGE_REUSE_MEMORY
	block_allocator().free(ptr); 
#else
	free(ptr);
#endif
	ptr = NULL; 
      }
    }
    assert(ptr == NULL);
    
    if(use_external_buffer){
      if(external_buffer_size != size || external_buffer_alignment != alignment)
	ERR.General("DistributedMemoryStorage","alloc","External buffer size and alignment of alloc call does not match that of buffer");
      ptr = external_buffer;
    }else{
#ifdef DISTRIBUTED_MEMORY_STORAGE_REUSE_MEMORY
      ptr = block_allocator().alloc(alignment,size);
#else
      ptr = memalign_check(alignment, size);
#endif
    }
    assert(ptr != NULL);
    
    _size = size;
    _alignment = alignment;
  }

  void freeMem(){
    if(ptr != NULL){
      if(!use_external_buffer){
#ifdef DISTRIBUTED_MEMORY_STORAGE_REUSE_MEMORY
	block_allocator().free(ptr);
#else
	free(ptr);
#endif
      }
      ptr = NULL;
    }
  }

  void move(MemoryStorageBase &r){
    ptr = r.ptr;
    _alignment = r._alignment;
    _size = r._size;

    use_external_buffer = r.use_external_buffer;
    external_buffer = r.external_buffer;
    external_buffer_size = r.external_buffer_size;
    external_buffer_alignment = r.external_buffer_alignment;

    r._size = 0;
    r.ptr = NULL;
  }


  inline void* data(){ return ptr; }
  inline void const* data() const{ return ptr; }
};

//A version that keeps all meson fields in each node's memory. Gather and distribute have no effect.
class InMemMemoryStorage: public MemoryStorageBase{
public:
  InMemMemoryStorage() = default;
  
  InMemMemoryStorage(const InMemMemoryStorage &r)= default;

  InMemMemoryStorage & operator=(const InMemMemoryStorage &r){ this->MemoryStorageBase::operator=(r); return *this; }
  
  inline void gather(bool require){ }

  void distribute(){ }
};


  

class DistributedMemoryStorage: public MemoryStorageBase{
  int _master_uid;
  int _master_mpirank;

  void initMaster(){
    if(_master_uid == -1){
      int master_uid = nodeDistributeCounter::getNext(); //round-robin assignment
      
      int nodes = 1;
      for(int i=0;i<5;i++) nodes *= GJP.Nodes(i);
#ifndef USE_MPI
      _master_uid = master_uid;
      if(nodes > 1) ERR.General("DistributedMemoryStorage","initMaster","Implementation requires MPI\n");
      else assert(master_uid == 0); //this should always be the case but is worth checking anyway!
#else

      //# define DO_MASTER_CHECK
      
# ifdef DO_MASTER_CHECK
      //Check all nodes are decided on the same master node
      int* masters = (int*)malloc_check(nodes *  sizeof(int));
      memset(masters, 0, nodes *  sizeof(int));
      masters[UniqueID()] = master_uid;

      int* masters_all = (int*)malloc_check(nodes *  sizeof(int));
      assert( MPI_Allreduce(masters, masters_all, nodes, MPI_INT, MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS );

      for(int i=0;i<nodes;i++) assert(masters_all[i] == master_uid);

      free(masters); free(masters_all);
# endif

      _master_uid = master_uid;
      _master_mpirank = MPI_UniqueID_map::UidToMPIrank(master_uid);
#endif
    }
  }

public:
  struct GatherPerf{
    size_t check_calls;
    double check_time;
    size_t alloc_calls;
    double alloc_time;
    size_t gather_calls;
    double gather_time;
    size_t bytes;
    size_t free_calls;
    double free_time;
    
    void reset(){
      check_calls=alloc_calls=gather_calls=bytes=free_calls=0;
      check_time=alloc_time=gather_time=free_time=0;
    }
    GatherPerf(){
      reset();
    }
    void print(){
      if(!UniqueID()){
	double avg_check_time = check_time / double(check_calls);
	double avg_alloc_time = alloc_time / double(alloc_calls);
	double avg_gather_time = gather_time / double(gather_calls);
	double avg_free_time = free_time / double(free_calls);
	double avg_bandwidth = double(bytes)/gather_time/(1024*1024); //MB/s
	std::ostringstream os; 
	os << "DistributedMemoryStorage::GatherPerf avg check time " << avg_check_time << "s, avg alloc time " << avg_alloc_time << "s, avg gather time " << avg_gather_time << "s, gather bandwidth " << avg_bandwidth << "MB/s, avg free time " << avg_free_time << "s\n";
	printf(os.str().c_str()); fflush(stdout);
      }
    }
  };
  static GatherPerf & perf(){ static GatherPerf p; return p; }

  DistributedMemoryStorage(): _master_uid(-1), MemoryStorageBase(){}

  DistributedMemoryStorage(const DistributedMemoryStorage &r): _master_uid(r._master_uid), _master_mpirank(r._master_mpirank),  MemoryStorageBase(r){ }

  DistributedMemoryStorage & operator=(const DistributedMemoryStorage &r){
    _master_uid = r._master_uid;
    _master_mpirank = r._master_mpirank;
    this->MemoryStorageBase::operator=(r);
    return *this;
  }

  inline int masterUID() const{ return _master_uid; }

  void move(DistributedMemoryStorage &r){
    _master_uid = r._master_uid;
    _master_mpirank = r._master_mpirank;
    this->MemoryStorageBase::move(r);
  }
 
  //Every node performs gather but if not required and not master, data is not kept  
  void gather_bcast_full(bool require){
#ifndef USE_MPI
    if(require && ptr == NULL) ERR.General("DistributedMemoryStorage","gather_bcast_full","Implementation requires MPI\n");
#else
    double time = dclock();
    int do_gather_node = (require && ptr == NULL);

# define ENABLE_GATHER_PRECHECK
# ifdef ENABLE_GATHER_PRECHECK
    //Check to see if a gather is actually necessary
    int do_gather_any = 0;
    assert( MPI_Allreduce(&do_gather_node, &do_gather_any, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS );

    perf().check_calls++;
    perf().check_time += dclock() - time;
# else
    int do_gather_any = 1;
# endif
# undef ENABLE_GATHER_PRECHECK

    //Do the gather. All nodes need memory space, albeit temporarily
    if(do_gather_any){
      bool did_alloc = false;

      if(UniqueID() != _master_uid && ptr == NULL){
	did_alloc = true;
	time = dclock();
	alloc(_alignment, _size);      
	perf().alloc_calls++;
	perf().alloc_time += dclock() - time;
      }
      time = dclock();
      assert( MPI_Bcast(ptr, _size, MPI_BYTE, _master_mpirank, MPI_COMM_WORLD) == MPI_SUCCESS );
      perf().gather_calls++;
      perf().gather_time += dclock() - time;
      perf().bytes += _size;

      //Non-master copies safe to throw away data if not required. 
      //If data was already present we don't throw away because it may have been pulled by a different call to gather. 
      //did_alloc is only true if this is not the master node and the mf was not allocated at the start of the call
      if(did_alloc && !require){
	time = dclock();
	freeMem();
	perf().free_calls++;
	perf().free_time += dclock() - time;
      }      
    }
#endif
  }
  
  //A smarter implementation of gather that uses an MPI sub-communicator to only send to nodes that need it - courtesy of D.Hoying
  void gather_bcast_subcomm(bool require){
#ifdef USE_MPI
    double time = dclock();
    int do_gather_node = (require && ptr == NULL);

    //#define ENABLE_GATHER_PRECHECK
# ifdef ENABLE_GATHER_PRECHECK
    //Check to see if a gather is actually necessary
    int do_gather_any = 0;
    assert( MPI_Allreduce(&do_gather_node, &do_gather_any, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS );
    perf().check_calls++;
    perf().check_time += dclock() - time;
# else
    int do_gather_any = 1;
# endif
# undef ENABLE_GATHER_PRECHECK

    if (do_gather_any){
      //only bcast to subset of nodes which require the meson field; this should scale with nodes much better than bcast to call
      MPI_Comm req_comm;
      int world_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
      int color = do_gather_node || UniqueID() == _master_uid ? 1 : 0;       //color 1 participates, color 0 does not
      //_master_mpirank is given rank 0 in the new communicator, all others get rank 1 (mpi resolves ties by reordering according to their old rank order)  
      // this ensures we know the rank of the sender which is all we care about
      assert(MPI_SUCCESS==MPI_Comm_split(MPI_COMM_WORLD, color, world_rank==_master_mpirank ? 0 : 1, &req_comm));

      if(color){
	if(UniqueID() != _master_uid && ptr == NULL){
	  time = dclock();
	  alloc(_alignment, _size);      
	  perf().alloc_calls++;
	  perf().alloc_time += dclock() - time;
	}
	time = -dclock();
	assert(MPI_Bcast(ptr, _size, MPI_BYTE, 0, req_comm) == MPI_SUCCESS); 
	time += dclock();
	
	perf().gather_time += time;
	perf().gather_calls++;
	perf().bytes += _size;
	int commsize;
	MPI_Comm_size(req_comm, &commsize);
      }
      MPI_Comm_free(&req_comm);
    }
#else
    if(require && ptr == NULL) ERR.General("DistributedMemoryStorage","gather_bcast_subcomm","Implementation requires MPI\n"); 
#endif //USE_MPI
  }

  inline void gather(bool require){
#ifdef GATHER_BCAST_SUBCOMM
    gather_bcast_subcomm(require);
#else
    gather_bcast_full(require);
#endif
  }

  void distribute(){
    initMaster();
    if(UniqueID() != _master_uid) freeMem();
  }
  
  //Change the master uid. THIS MUST BE THE SAME FOR ALL NODES!
  void reassign(const int to_uid){
    assert(_master_uid != -1);
    gather(UniqueID() == to_uid);
    
    //All nodes change master
    _master_uid = to_uid;
#ifdef USE_MPI
    _master_mpirank = MPI_UniqueID_map::UidToMPIrank(to_uid);
#else
    _master_mpirank = 0;
#endif


    distribute();
  }

  static void rebalance(std::vector<DistributedMemoryStorage*> &blocks){
    if(!UniqueID()){ printf("DistributedMemoryStorage: Performing rebalance of %d blocks\n",blocks.size()); fflush(stdout); }
    int nodes = 1;
    for(int i=0;i<5;i++) nodes *= GJP.Nodes(i);
    
    int init_count[nodes]; for(int i=0;i<nodes;i++) init_count[i] = 0;  

    const int my_uid = UniqueID();

    std::vector< std::vector<DistributedMemoryStorage*> > init_block_mapping(nodes);

    for(int i=0;i<blocks.size();i++){
      int master_uid = blocks[i]->masterUID();
      if(master_uid != -1){
	++init_count[master_uid];
	init_block_mapping[master_uid].push_back(blocks[i]);
      }
    }
    
    //Balance the number of blocks over the nodes, with the remainder assigned one each to nodes in ascending order
    int nblock = blocks.size();
    int nblock_bal_base = nblock/nodes;
    int nrem = nblock - nblock_bal_base * nodes;

    int count[nodes]; 
    for(int n=0;n<nodes;n++)
      count[n] = n < nrem ? nblock_bal_base + 1 : nblock_bal_base;

    if(!UniqueID()){
      printf("node:old:new\n");
      for(int n=0;n<nodes;n++) printf("%d:%d:%d ",n,init_count[n],count[n]);
      printf("\n");
      fflush(stdout);
    }

    //All the nodes that are relinquishing blocks first put their pointers in a pool, then all that are assuming blocks take from the pool
    std::list<DistributedMemoryStorage*> pool;
    for(int n=0;n<nodes;n++){
      const int delta = count[n] - init_count[n];
      if(delta < 0){
	const int npool = -delta;
	for(int i=0;i<npool;i++){
	  pool.push_back(init_block_mapping[n][ init_count[n]-1-i ]); //take from back
	}
      }
    }

    for(int n=0;n<nodes;n++){
      const int delta = count[n] - init_count[n];
      if(delta > 0){
	assert(pool.size() >= delta);
	for(int i=0;i<delta;i++){
	  pool.front()->reassign(n);
	  pool.pop_front();
	}
      }
    }

  }
  
};


//A similar class to the above but which stores to disk rather than distributing over the memory systems of a multi-node machine. This version is designed to work on systems with burst buffers that are on a shared filesystem such that only the head node writes to disk
class BurstBufferMemoryStorage: public MemoryStorageBase{
protected:
  std::string file;
  bool ondisk;
  unsigned int ondisk_checksum; //so we can tell if the data has changed since we wrote it previously

  unsigned int checksum() const{
    static bool init = false;
    static FPConv fp;    
    if(!init){
      fp.setFileFormat(FP_AUTOMATIC);
      init = true;
    }
    assert(sizeof(Float) == sizeof(double));
    assert(_size % sizeof(double) == 0);   
    return fp.checksum((char*)ptr, _size/sizeof(double), FP_AUTOMATIC);
  }  
  
public:
  struct GatherPerf{ //Note performance is only recorded for gathers that required communications from target node to origin
    double alloc_time;
    size_t gather_calls;
    double gather_time;
    double checksum_time;
    size_t bytes;
    
    void reset(){
      gather_calls=bytes=0;
      alloc_time=gather_time=checksum_time=0;
    }
    GatherPerf(){
      reset();
    }
    void print(int uid = 0){
      if(UniqueID() == uid){
	double avg_alloc_time = alloc_time / double(gather_calls);
	double avg_gather_time = gather_time / double(gather_calls);
	double avg_checksum_time = checksum_time / double(gather_calls);
	double avg_bandwidth = double(bytes)/gather_time/(1024*1024); //MB/s
	std::ostringstream os; 
	os << "BurstBufferMemoryStorage::GatherPerf over " << gather_calls << " calls,  avg alloc time " << avg_alloc_time << "s, avg gather time " << avg_gather_time << "s, avg checksum time " << avg_checksum_time << ", gather bandwidth " << avg_bandwidth << " MB/s\n";
	printf(os.str().c_str()); fflush(stdout);
      }
    }
  };
  static GatherPerf & perf(){ static GatherPerf p; return p; }

  BurstBufferMemoryStorage(): ondisk(false), MemoryStorageBase(){}

  BurstBufferMemoryStorage(const BurstBufferMemoryStorage &r): ondisk(false), MemoryStorageBase(r){ 
    if(r.ptr == NULL && r.ondisk) ERR.General("BurstBufferMemoryStorage","copy-constructor","Cannot copy without gathering first");
  }

  BurstBufferMemoryStorage & operator=(const BurstBufferMemoryStorage &r){
    if(r.ptr == NULL && r.ondisk) ERR.General("BurstBufferMemoryStorage","operator=","Cannot copy without gathering first");
    //Copies are independent so we cannot back with the same file
    file = "";
    ondisk = false;
    this->MemoryStorageBase::operator=(r);
    return *this;
  }

  //Set the path to the scratch directories using this static function. Path must exist
  static std::string & filestub(){ static std::string b = "burststore"; return b; } //change this to the base filename (an _<IDX>.dat is appended)
  
  void move(BurstBufferMemoryStorage &r){
    file = r.file;
    ondisk = r.ondisk;
    ondisk_checksum = r.ondisk_checksum;
    r.ondisk = false;
    r.file = "";
    this->MemoryStorageBase::move(r);
  }

#define DISTRIBUTE_FLUSH_MEMBUF
  
  //Load from disk (optional)
  void gather(bool require){
    int do_gather_node = (require && ptr == NULL);

    if(do_gather_node){
      if(!ondisk) ERR.General("BurstBufferMemoryStorage","gather(bool)","Cannot gather data as no copy has been written to disk");

      perf().alloc_time -= dclock();
      alloc(_alignment,_size);
      perf().alloc_time += dclock();

      perf().gather_time -= dclock();
      std::fstream f(file.c_str(), std::ios::in | std::ios::binary);
      if(!f.good()) ERR.General("BurstBufferMemoryStorage","gather(bool)","Failed to open file %s for read\n",file.c_str());
      size_t rd_size; f.read((char*)&rd_size,sizeof(size_t));
      if(rd_size != _size) ERR.General("BurstBufferMemoryStorage","gather(bool)","Data size %lu in file %s different from expected %lu\n", (unsigned long)rd_size, file.c_str(), (unsigned long)_size );
      unsigned int cksum_rd; f.read((char*)&cksum_rd,sizeof(unsigned int));
      f.read((char*)ptr,_size);
      if(!f.good()) ERR.General("BurstBufferMemoryStorage","gather(bool)","Read error in file %s\n",file.c_str());      
      f.close();
      perf().gather_time += dclock();

      perf().checksum_time -= dclock();
      unsigned int cksum_calc = checksum();
      if(cksum_calc != cksum_rd) ERR.General("BurstBufferMemoryStorage","gather(bool)","Checksum error on reading file %s, expected %u, got %u\n",file.c_str(),cksum_rd,cksum_calc);
      perf().checksum_time += dclock();

      perf().gather_calls++;
      perf().bytes += _size;
    }
    cps::sync();
  }
  
  void distribute(){
    if(ptr == NULL) return;

    unsigned int cksum = checksum();
    if(!ondisk || (ondisk && ondisk_checksum != cksum) ){ //it is assumed here that node 0 has a copy
      static int fidx = 0;
      if(!ondisk){
	int nodes = 1; for(int i=0;i<4;i++) nodes *= GJP.Nodes(i);
	int nodes_participating = 1;
	MPI_Allreduce(MPI_IN_PLACE, &nodes_participating, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	if(nodes_participating != nodes) ERR.General("BurstBufferMemoryStorage","distribute","All nodes must participate in a distribute operation");
	int fidx_use = fidx++;
	MPI_Bcast(&fidx_use, 1, MPI_INT, 0, MPI_COMM_WORLD);
	std::ostringstream os; os << filestub() << "_" << fidx_use << ".dat";
	file = os.str();
      }
	
      if(!UniqueID()){
	assert(ptr != NULL);
	std::fstream f(file.c_str(), std::ios::out | std::ios::binary);
	if(!f.good()) ERR.General("BurstBufferMemoryStorage","distribute","Failed to open file %s for write\n",file.c_str());
	f.write((char*)&_size,sizeof(size_t));
	f.write((char*)&cksum,sizeof(unsigned int));
	f.write((char*)ptr,_size);
	if(!f.good()) ERR.General("BurstBufferMemoryStorage","distribute","Write error in file %s\n",file.c_str());
#ifdef DISTRIBUTE_FLUSH_MEMBUF
	f.flush(); //should ensure data is written to disk immediately and not kept around in some memory buffer, but may slow things down
#endif	
      }
      ondisk = true;
      ondisk_checksum = cksum;
    }
    cps::sync();
    freeMem();
  }
    
};






//This works similarly to the above only each node writes its own unique copy rather than just the head node. This is intended for systems where nodes have independent scratch disks
class IndependentDiskWriteStorage: public BurstBufferMemoryStorage{
public:
  IndependentDiskWriteStorage(): BurstBufferMemoryStorage(){}

  IndependentDiskWriteStorage(const IndependentDiskWriteStorage &r): BurstBufferMemoryStorage(r){}

  inline IndependentDiskWriteStorage & operator=(const IndependentDiskWriteStorage &r){
    static_cast<BurstBufferMemoryStorage&>(*this) = r;
    return *this;
  }

  void distribute(){
    if(ptr == NULL) return;
    unsigned int cksum = checksum();
    if(!ondisk || (ondisk && ondisk_checksum != cksum) ){
      static int fidx = 0;
      if(!ondisk){
	std::ostringstream os; os << filestub() << "_" << fidx++ << "_node" << UniqueID() << ".dat";
	file = os.str();
      }
      assert(ptr != NULL);
      std::fstream f(file.c_str(), std::ios::out | std::ios::binary);
      if(!f.good()) ERR.General("IndependentDiskWriteStorage","gather(bool)","Failed to open file %s for write\n",file.c_str());
      f.write((char*)&_size,sizeof(size_t));
      f.write((char*)&cksum,sizeof(unsigned int));
      f.write((char*)ptr,_size);
      if(!f.good()) ERR.General("IndependentDiskWriteStorage","gather(bool)","Write error in file %s\n",file.c_str());
#ifdef DISTRIBUTE_FLUSH_MEMBUF
      f.flush(); //should ensure data is written to disk immediately and not kept around in some memory buffer, but may slow things down
#endif
      ondisk = true;
      ondisk_checksum = cksum;      
    }
    freeMem();
  }
};

#ifdef USE_MPI

//Use MPI one-sided communication to manage the gather
class DistributedMemoryStorageOneSided: public MemoryStorageBase{
  int _master_uid;
  int _master_mpirank;

  MPI_Win window;

  //Assumes _master_uid has been set
  void initWindow(){
    //Only master exposes memory
    if(UniqueID() == _master_uid){ 
      assert(MPI_Win_create(ptr, _size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &window) == MPI_SUCCESS);
    }else{
      assert(MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &window) == MPI_SUCCESS);
    }
  }

  void initMaster(){
    if(_master_uid == -1){
      int master_uid = nodeDistributeCounter::getNext(); //round-robin assignment
      
      int nodes = 1;
      for(int i=0;i<5;i++) nodes *= GJP.Nodes(i);
      _master_uid = master_uid;
      _master_mpirank = MPI_UniqueID_map::UidToMPIrank(master_uid);
      initWindow();
    }
  }

public:
  struct GatherPerf{ //Note performance is only recorded for gathers that required communications from target node to origin
    double alloc_time;
    size_t gather_calls;
    double gather_time;
    size_t bytes;
    
    void reset(){
      gather_calls=bytes=0;
      alloc_time=gather_time=0;
    }
    GatherPerf(){
      reset();
    }
    void print(int uid = 0){
      if(UniqueID() == uid){
	double avg_alloc_time = alloc_time / double(gather_calls);
	double avg_gather_time = gather_time / double(gather_calls);
	double avg_bandwidth = double(bytes)/gather_time/(1024*1024); //MB/s
	std::ostringstream os; 
	os << "DistributedMemoryStorageOneSided::GatherPerf over " << gather_calls << " calls,  avg alloc time " << avg_alloc_time << "s, avg gather time " << avg_gather_time << "s, gather bandwidth " << avg_bandwidth << " MB/s\n";
	printf(os.str().c_str()); fflush(stdout);
      }
    }
  };
  static GatherPerf & perf(){ static GatherPerf p; return p; }

  DistributedMemoryStorageOneSided(): _master_uid(-1), window(MPI_WIN_NULL), MemoryStorageBase(){}

  DistributedMemoryStorageOneSided(const DistributedMemoryStorageOneSided &r): window(MPI_WIN_NULL), MemoryStorageBase(r){
    _master_uid = r._master_uid;
    _master_mpirank = r._master_mpirank;
    if(r.window != MPI_WIN_NULL) initWindow();
  }

  DistributedMemoryStorageOneSided & operator=(const DistributedMemoryStorageOneSided &r){
    if(window != MPI_WIN_NULL) assert(MPI_Win_free(&window)==MPI_SUCCESS);
    this->MemoryStorageBase::operator=(r);
    _master_uid = r._master_uid;
    _master_mpirank = r._master_mpirank;
    if(r.window != MPI_WIN_NULL) initWindow();
    return *this;
  }

  inline int masterUID() const{ return _master_uid; }

  void move(DistributedMemoryStorageOneSided &r){
    _master_uid = r._master_uid;
    _master_mpirank = r._master_mpirank;
    window = r.window;
    r.window = MPI_WIN_NULL;
    this->MemoryStorageBase::move(r);
  }

  inline void gather(bool require){
    if(require && UniqueID() != _master_uid && _size != 0 && ptr == NULL){
      perf().alloc_time -= dclock();
      alloc(_alignment,_size);
      perf().alloc_time += dclock();

      perf().gather_calls++;
      perf().bytes += _size;
      perf().gather_time -= dclock();
      assert(MPI_Win_lock(MPI_LOCK_SHARED, _master_mpirank, MPI_MODE_NOCHECK, window)==MPI_SUCCESS);
      assert(MPI_Get(ptr, _size, MPI_BYTE, _master_mpirank, 0, _size, MPI_BYTE, window)==MPI_SUCCESS);
      assert(MPI_Win_unlock(_master_mpirank, window)==MPI_SUCCESS);
      perf().gather_time += dclock();      
    }
  }

  void distribute(){
    initMaster();
    if(UniqueID() != _master_uid) freeMem();
  }
  
  //Change the master uid. THIS MUST BE THE SAME FOR ALL NODES!
  void reassign(const int to_uid){
    assert(_master_uid != -1);
    gather(UniqueID() == to_uid);
    
    //All nodes change master
    assert(MPI_Win_free(&window) == MPI_SUCCESS);
    _master_uid = to_uid;
#ifdef USE_MPI
    _master_mpirank = MPI_UniqueID_map::UidToMPIrank(to_uid);
#else
    _master_mpirank = 0;
#endif
    initWindow();
    distribute();
  }

  static void rebalance(std::vector<DistributedMemoryStorageOneSided*> &blocks){
    if(!UniqueID()){ printf("DistributedMemoryStorageOneSided: Performing rebalance of %d blocks\n",blocks.size()); fflush(stdout); }
    int nodes = 1;
    for(int i=0;i<5;i++) nodes *= GJP.Nodes(i);
    
    int init_count[nodes]; for(int i=0;i<nodes;i++) init_count[i] = 0;  

    const int my_uid = UniqueID();

    std::vector< std::vector<DistributedMemoryStorageOneSided*> > init_block_mapping(nodes);

    for(int i=0;i<blocks.size();i++){
      int master_uid = blocks[i]->masterUID();
      if(master_uid != -1){
	++init_count[master_uid];
	init_block_mapping[master_uid].push_back(blocks[i]);
      }
    }
    
    //Balance the number of blocks over the nodes, with the remainder assigned one each to nodes in ascending order
    int nblock = blocks.size();
    int nblock_bal_base = nblock/nodes;
    int nrem = nblock - nblock_bal_base * nodes;

    int count[nodes]; 
    for(int n=0;n<nodes;n++)
      count[n] = n < nrem ? nblock_bal_base + 1 : nblock_bal_base;

    if(!UniqueID()){
      printf("node:old:new\n");
      for(int n=0;n<nodes;n++) printf("%d:%d:%d ",n,init_count[n],count[n]);
      printf("\n");
      fflush(stdout);
    }

    //All the nodes that are relinquishing blocks first put their pointers in a pool, then all that are assuming blocks take from the pool
    std::list<DistributedMemoryStorageOneSided*> pool;
    for(int n=0;n<nodes;n++){
      const int delta = count[n] - init_count[n];
      if(delta < 0){
	const int npool = -delta;
	for(int i=0;i<npool;i++){
	  pool.push_back(init_block_mapping[n][ init_count[n]-1-i ]); //take from back
	}
      }
    }

    for(int n=0;n<nodes;n++){
      const int delta = count[n] - init_count[n];
      if(delta > 0){
	assert(pool.size() >= delta);
	for(int i=0;i<delta;i++){
	  pool.front()->reassign(n);
	  pool.pop_front();
	}
      }
    }

  }

  ~DistributedMemoryStorageOneSided(){    
    /* if(window != MPI_WIN_NULL){ */
    /*   printf("Node %d fence\n", UniqueID()); fflush(stdout); */
    /*   MPI_Win_fence(0,window); */
    /* } */

    MPI_Barrier(MPI_COMM_WORLD); //ensure everything is complete before destroying
    //printf("Node %d free\n", UniqueID()); fflush(stdout);
    freeMem();
    if(window != MPI_WIN_NULL){
      //printf("Node %d win free\n", UniqueID()); fflush(stdout);
      assert(MPI_Win_free(&window)==MPI_SUCCESS);
    }
  }
  
};


//A memory storage that uses a mmap disk memory region. Gather calls a prefetch operation
class MmapMemoryStorage{
protected:
  void *ptr;
  size_t _size;
  std::string filename;
public:

  const std::string & getFilename() const{ return filename; }

  MmapMemoryStorage(): ptr(NULL), _size(0), filename(""){}

  MmapMemoryStorage(const MmapMemoryStorage &r): ptr(NULL), _size(0), filename(""){
    if(r.ptr != NULL){    
      alloc(0, r._size);
      memcpy(ptr, r.ptr, r._size);
    }else{
      _size = r._size;
    }      
  }
  
  MmapMemoryStorage & operator=(const MmapMemoryStorage &r){
    if(r.ptr != NULL){
      if(_size != r._size){
	freeMem();
	alloc(0, r._size);
      }
      memcpy(ptr, r.ptr, r._size);      
    }else{
      freeMem();
      _size = r._size;
    }

    return *this;
  }

  ~MmapMemoryStorage(){ freeMem(); }

  inline bool isOnNode() const{ return ptr != NULL; }

  //does nothing
  void enableExternalBuffer(void* p, size_t sz, int align){ }
  void disableExternalBuffer(){ }

  void flush(){ if(ptr != NULL) msync(ptr,_size,MS_SYNC); }
  
  void alloc(int alignment, size_t size){
    static int fidx = 0;

    if(filename == ""){
      std::ostringstream os; os << BurstBufferMemoryStorage::filestub() << "_" << (fidx++) << "_uid" << UniqueID();
      filename = os.str();
    }

    if(ptr != NULL){
      if(size == _size) return;
      else{
	freeMem();
      }
    }

    _size = size;

    int fd = open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t)0666);
    if( fd == -1 ){
      std::cout << "ERROR open: " << strerror(errno) << std::endl;
      assert(0);
    }
    if( posix_fallocate(fd, 0, _size) != 0 ){
      std::cout << "ERROR posix_fallocate: " << strerror(errno) << std::endl;
      assert(0);
    }    

    ptr = mmap(NULL, size, PROT_READ | PROT_WRITE , MAP_FILE|MAP_SHARED, fd, 0);
    if(ptr == MAP_FAILED){
      std::cout << "ERROR mmap: " << strerror(errno) << std::endl;
      assert(0);
    }
    close(fd);
  }

  void freeMem(){
    if(ptr != NULL){
      if( munmap(ptr, _size) != 0 ){
	std::cout << "ERROR munmap: " << strerror(errno) << std::endl;
	assert(0);
      }
      if( unlink(filename.c_str()) != 0 ){
	std::cout << "ERROR unlink: " << strerror(errno) << std::endl;
	assert(0);
      }
      ptr = NULL; 
    }
  }

  void move(MmapMemoryStorage &r){
    ptr = r.ptr;
    _size = r._size;
    filename = r.filename;

    r._size = 0;
    r.ptr = NULL;
    r.filename = "";
  }

  inline void* data(){ return ptr; }
  inline void const* data() const{ return ptr; }

  void gather(bool require){
    if(ptr != NULL){
      madvise(ptr, _size, MADV_SEQUENTIAL); //hopefully this will cause it to read ahead
      char volatile *pPage = (char *)ptr;
      size_t page_size = sysconf(_SC_PAGESIZE);
      size_t npage = _size / page_size;
      for(int i=0;i<npage;i++){
	(void)*pPage; //touch the page, should force a load (cf https://stackoverflow.com/questions/18929011/how-do-i-touch-a-memory-page-on-purpose)
	pPage += page_size; 
      }
      madvise(ptr, _size, MADV_WILLNEED);
    }else assert(0);
  }
  void distribute(){
    if(ptr != NULL){
      madvise(ptr, _size, MADV_DONTNEED);
    }else assert(0);  
  }
};






#endif


CPS_END_NAMESPACE

#endif
