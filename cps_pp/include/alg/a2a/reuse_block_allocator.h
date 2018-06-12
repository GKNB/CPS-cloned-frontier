#ifndef _REUSE_BLOCK_ALLOCATOR_MALLOC_H__
#define _REUSE_BLOCK_ALLOCATOR_MALLOC_H__

#include<list>
#include<map>
#include<set>

CPS_START_NAMESPACE

//A class that maintains and distributes blocks of memory facilitating reuse

struct ReuseBlockAllocatorMalloc{
  inline void* allocMem(const size_t sz){ return malloc(sz); }
};

class ReuseBlockAllocatorMemalign{
  size_t _alignment;
public:
  inline ReuseBlockAllocatorMemalign(): _alignment(128){}

  inline size_t & alignment(){ return _alignment; }
  
  inline void* allocMem(const size_t sz){ 
    void *p;
    assert( posix_memalign(&p, _alignment, sz) == 0);
    return p;
  }
};

template<typename AllocPolicy>
class ReuseBlockAllocator: public AllocPolicy{
  struct Block{
    void* ptr;
    size_t size;
    Block(void *p, size_t s): ptr(p), size(s){}
  };
  typedef std::list<Block> blockList;
  blockList blocks;

  typedef std::map<void*, typename blockList::iterator> blockPtrMapType;
  blockPtrMapType ptr_map;

  typedef std::list<typename blockList::iterator> blockIteratorList;  
  typedef std::map<size_t, blockIteratorList > allFreeMapType;
  allFreeMapType all_free; //All free blocks including memaligned blocks. (size)

public:
  void stats(std::ostream &os){
    size_t totsz = 0;
    for(typename blockList::const_iterator it = blocks.begin(); it != blocks.end(); it++) totsz += it->size;

    size_t unused = 0;
    os << "ReuseBlockAllocator " << double(totsz)/1024./1024. << " MB over " << blocks.size() << " blocks\n";
    for(typename allFreeMapType::iterator mit = all_free.begin(); mit != all_free.end(); mit++){
      if(mit->second.size() > 0 ){
	unused += mit->second.size() * mit->first;
	os << "ReuseBlockAllocator " << mit->second.size() << " blocks of size " << double(mit->first)/1024./1024. << " MB are awaiting reuse\n";
      }
    }
    os << "ReuseBlockAllocator " << double(unused)/1024./1024. << " MB unused space\n";
    os.flush();
  }

  inline void trim(){ //free any unused blocks
    std::vector<void*> torm;

    for(typename allFreeMapType::iterator mit = all_free.begin(); mit != all_free.end(); mit++){
      for(typename blockIteratorList::iterator fit = mit->second.begin(); fit != mit->second.end(); fit++){
	typename blockList::iterator bit = *fit;
	::free(bit->ptr);
	torm.push_back(bit->ptr);
	blocks.erase(bit);
      }
    }
    all_free.clear();
    
    for(int i=0;i<torm.size();i++) ptr_map.erase(torm[i]);
  }


  inline void clear(){ //free all blocks
    for(typename blockList::iterator it = blocks.begin(); it != blocks.end(); it++) ::free(it->ptr); 
    blocks.clear();
  }
  inline ~ReuseBlockAllocator(){ clear(); }
  
  inline void* alloc(const size_t bytes){
    typename allFreeMapType::iterator mit = all_free.find(bytes); //check if we have (or had) blocks of this size
    if(mit != all_free.end()){
      blockIteratorList &bit = mit->second;
      if(bit.size() != 0){
	Block &block = *bit.back();
	bit.pop_back(); //remove from free list
	return block.ptr;
      }
    }

    Block b( this->allocMem(bytes), bytes );
    blocks.push_back(b);
    ptr_map[b.ptr] = --blocks.end();
    return b.ptr;
  }

  inline void free(void* ptr){
    typename blockPtrMapType::iterator pit = ptr_map.find(ptr);
    if(pit == ptr_map.end()) ERR.General("ReuseBlockAllocator","free ptr","%p is not in ptr_map!\n",ptr);

    typename blockList::iterator bit = pit->second;
    size_t size = bit->size;
    all_free[size].push_back(bit);
  }
};


//We should maintain different allocators for different alignments 
class ReuseBlockAllocatorsAligned{
  typedef ReuseBlockAllocator<ReuseBlockAllocatorMemalign> Allocator;
  typedef std::map<size_t, Allocator* > AlignmentMap;
  AlignmentMap allocators;

  typedef std::map<void*, Allocator*> PtrAllocatorMap;
  PtrAllocatorMap ptr_map;
public:
  ~ReuseBlockAllocatorsAligned(){ 
    std::set<Allocator*> torm; //prevent duplicates
    for(AlignmentMap::iterator it = allocators.begin(); it != allocators.end(); it++) torm.insert(it->second);
    for(std::set<Allocator*>::iterator it = torm.begin(); it != torm.end(); it++) delete *it;
  }
  Allocator & getAllocator(size_t alignment, bool allow_new = false){
    //See if the allocator exists
    AlignmentMap::iterator mit = allocators.find(alignment);
    if(mit != allocators.end()) return *mit->second;
    
    if(!allow_new) ERR.General("ReuseBlockAllocatorsAligned","getAllocator","No allocator with alignment %d", (int)alignment);
    
    //See if we can reuse an existing allocator with a larger alignment that is divisible by 'alignment'
    for(AlignmentMap::iterator mit = allocators.begin(); mit != allocators.end(); mit++)
      if(mit->first % alignment == 0){
	allocators[alignment] = mit->second;
	return *mit->second;
      }
    
    //Add new allocator
    Allocator* p = new Allocator;
    p->alignment() = alignment;
    allocators[alignment] = p;
    return *p;
  }

  void stats(std::ostream &os){
    std::map<Allocator*, std::vector<size_t> > almap;
    for(AlignmentMap::iterator mit = allocators.begin(); mit != allocators.end(); mit++){
      almap[mit->second].push_back(mit->first);
    }
    os<< "ReuseBlockAllocatorsAligned has " << almap.size() << " allocators\n";
    for(std::map<Allocator*, std::vector<size_t> >::iterator it = almap.begin(); it != almap.end(); it++){
      os << "ReuseBlockAllocatorsAligned allocator with ptr " << it->first << " services alignments (" << it->second.front();
      for(int i=1;i<it->second.size();i++) os << ", " << it->second[i];
      os << ") and has stats:\n";
      it->first->stats(os);
    }
    os.flush();
  }

  inline void* alloc(const size_t alignment, const size_t bytes){
    Allocator &allocator = getAllocator(alignment,true);
    void* p = allocator.alloc(bytes);
    ptr_map[p] = &allocator;
    return p;
  }
  inline void free(void *ptr){
    PtrAllocatorMap::iterator it = ptr_map.find(ptr);
    if(it == ptr_map.end()) ERR.General("ReuseBlockAllocatorsAligned","free","Could not find allocator associated with ptr %p\n",ptr);
    it->second->free(ptr);
    ptr_map.erase(it);
  }

  inline void trim(){
    for(AlignmentMap::iterator mit = allocators.begin(); mit != allocators.end(); mit++){
      mit->second->trim();
    }
  }
};


CPS_END_NAMESPACE
#endif
