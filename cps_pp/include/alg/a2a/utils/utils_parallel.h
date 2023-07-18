#ifndef _UTILS_PARALLEL_H_
#define _UTILS_PARALLEL_H_

#include <util/gjp.h>
#ifdef USE_GRID
#include <Grid/Grid.h>
#endif
//QMP does not guarantee MPI
//#ifdef USE_QMP
//#define USE_MPI
//#endif
#ifdef USE_MPI
#include <mpi.h>
#endif

#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <limits>

#include "utils_malloc.h"
#include "utils_parallel_globalsum.h"

CPS_START_NAMESPACE

#ifdef USE_MPI

struct _MPI_UniqueID_map{
  std::map<int,int> mpi_rank_to_uid;
  std::map<int,int> uid_to_mpi_rank;
  
  void setup(){
    int nodes = 1;
    for(int i=0;i<5;i++) nodes *= GJP.Nodes(i);

    int* mpi_ranks = (int*)malloc_check(nodes *  sizeof(int));
    memset(mpi_ranks, 0, nodes *  sizeof(int));
    assert( MPI_Comm_rank(MPI_COMM_WORLD, mpi_ranks + UniqueID() ) == MPI_SUCCESS );

    int* mpi_ranks_all = (int*)malloc_check(nodes *  sizeof(int));
    assert( MPI_Allreduce(mpi_ranks, mpi_ranks_all, nodes, MPI_INT, MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS );

    for(int i=0;i<nodes;i++){
      int uid = i;
      int rank = mpi_ranks_all[i];
      
      mpi_rank_to_uid[rank] = uid;
      uid_to_mpi_rank[uid] = rank;
    }
    
    free(mpi_ranks);
    free(mpi_ranks_all);
  }
};

class MPI_UniqueID_map{
  static _MPI_UniqueID_map *getMap(){
    static _MPI_UniqueID_map* mp = NULL;
    if(mp == NULL){
      mp = new _MPI_UniqueID_map;
      mp->setup();
    }
    return mp;
  }
public:
  
  static int MPIrankToUid(const int rank){ return getMap()->mpi_rank_to_uid[rank]; }
  static int UidToMPIrank(const int uid){ return getMap()->uid_to_mpi_rank[uid]; }
};

#endif


//Divide work over nodes 
inline void getNodeWork(const int work, int &node_work, int &node_off, bool &do_work, const bool node_local = false){
  if(node_local){ node_work = work; node_off = 0; do_work = true; return; } //node does all the work

  int nodes = 1; for(int i=0;i<5;i++) nodes *= GJP.Nodes(i);
  int me = UniqueID();

  //Stolen from BFM :)
  int basework = work/nodes;
  int backfill = nodes-(work % nodes);
  node_work = (work+me)/nodes;
  node_off  = basework * me;
  if ( me > backfill ) 
    node_off+= (me-backfill);
  if(node_work > 0) do_work = true;
}

//Divide work over threads
//team is the number of threads in the team, me is the current thread index
inline void thread_work(size_t &my_work, size_t &my_offset, const size_t total_work, const int me, const int team){
  my_work = total_work/team;
  my_offset = me * my_work;
  
  size_t rem = total_work - my_work * team;
  if(me < rem){
    ++my_work; //first rem threads mop up the remaining work
    my_offset += me; //each thread before me has gained one extra unit of work
  }else my_offset += rem; //after the first rem threads, the offset shift is uniform
}

//MPI utilities
#ifdef USE_MPI
//get MPI rank of this node
inline int getMyMPIrank(){
  int my_mpi_rank;
  int ret = MPI_Comm_rank(MPI_COMM_WORLD, &my_mpi_rank);
  if(ret != MPI_SUCCESS) ERR.General("A2AmesonField","read","Comm_rank failed\n");
  return my_mpi_rank;
}
//get the MPI rank of the node with UniqueID() == 0
inline int getHeadMPIrank(){
  int head_mpi_rank;
  int rank_tmp = UniqueID() == 0 ? getMyMPIrank() : 0;
  int ret = MPI_Allreduce(&rank_tmp,&head_mpi_rank, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD); //node is now the MPI rank corresponding to UniqueID == _node
  if(ret != MPI_SUCCESS) ERR.General("A2AmesonField","read","Reduce failed\n");
  return head_mpi_rank;
}

//Lexicographic mapping of node coordinate to integer
inline int node_lex(const int* coor, const int ndir){
  int out = 0;
  for(int i=ndir-1;i>=0;i--){
    out *= GJP.Nodes(i);
    out += coor[i];
  }
  return out;  
}

//Generate map to convert lexicographic node index from GJP to an MPI rank in MPI_COMM_WORLD
inline void getMPIrankMap(std::vector<int> &map){
  int nodes = 1;
  int my_node_coor[5];
  for(int i=0;i<5;i++){
    nodes*= GJP.Nodes(i);
    my_node_coor[i] = GJP.NodeCoor(i);
  }
  const int my_node_lex = node_lex( my_node_coor, 5 );
  const int my_mpi_rank = getMyMPIrank();

  int *node_map_send = (int*)malloc_check(nodes*sizeof(int));
  memset(node_map_send,0,nodes*sizeof(int));
  node_map_send[my_node_lex] = my_mpi_rank;

  map.resize(nodes);
  int ret = MPI_Allreduce(node_map_send, &map[0], nodes, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  assert(ret == MPI_SUCCESS);
  free(node_map_send);
}


template<typename mf_Float>
struct getMPIdataType{
};
template<>
struct getMPIdataType<double>{
  static MPI_Datatype doit(){ return MPI_DOUBLE; }
};
template<>
struct getMPIdataType<float>{
  static MPI_Datatype doit(){ return MPI_FLOAT; }
};


#endif //USE_MPI

//Check each node can write to disk
inline void checkWriteable(const std::string &dir,const int conf){
  std::string file;
  {
    std::ostringstream os; os << dir << "/writeTest.node" << UniqueID() << ".conf" << conf;
    file = os.str();
  }
  std::ofstream of(file.c_str());
  double fail = 0;
  if(!of.good()){ std::cout << "checkWriteable failed to open file for write: " << file << std::endl; std::cout.flush(); fail = 1; }

  of << "Test\n";
  if(!of.good()){ std::cout << "checkWriteable failed to write to file: " << file << std::endl; std::cout.flush(); fail = 1; }

  glb_sum_five(&fail);

  if(fail != 0.){
    if(!UniqueID()){ printf("Disk write check failed\n");  fflush(stdout); }
    exit(-1);
  }else{
    if(!UniqueID()){ printf("Disk write check passed\n"); fflush(stdout); }
  }
}



inline bool checkDirExists(const std::string& dir){
  DIR* dirp = opendir(dir.c_str());
  if(dirp){
    closedir(dirp);
    return true;
  }else if(errno == ENOENT){
    return false;
  }else{
    ERR.General("","checkDirExists failed with error %s searching for path %s. cf https://linux.die.net/man/3/opendir for error descriptions.\n",strerror(errno),dir.c_str());
  }
  return false; //never reached
}


inline void makedir(const std::string& dir, const mode_t mode = 0775){
  if(!UniqueID()){
    int ret = mkdir(dir.c_str(), mode);
    if(ret == 0 || (ret == -1 && errno == EEXIST) ){
      //all is well!
    }else ERR.General("","makedir","Creation of directory %s failed with error %s\n",dir.c_str(),strerror(errno));
  }
  cps::sync();
  assert(checkDirExists(dir));
}

inline void printTimeStats(const std::string &descr, double time){
  int nodes = GJP.TotalNodes();
  std::vector<double> t(nodes, 0);
  t[UniqueID()] = time;
  globalSum(t.data(), nodes);

  double avg = 0, var = 0, max = 0, min = std::numeric_limits<double>::max();
  int maxnode=0, minnode=0;
  
  for(int i=0;i<nodes;i++){
    avg += t[i];
    var += t[i]*t[i];
    if(t[i] > max){
      max = t[i];
      maxnode = i;
    }
    if(t[i] < min){
      min = t[i];
      minnode = i;
    }    
  }
  avg /= nodes;
  var /= nodes;
  var -= avg*avg;
  
  if(!UniqueID()){
    std::cout << descr << ": avg=" << avg << "s, std.dev=" << sqrt(var) << "s, max=" << max << "s (" << maxnode << "), min=" << min << "s (" << minnode << ")" << std::endl;
  }
}

//Return true if this node has timeslices that lie between tstart and tstart + tsep  (mod Lt) . Range is inclusive
inline bool onNodeTimeslicesInRange(const int tstart, const int tsep){
  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  int toff = GJP.TnodeCoor()*GJP.TnodeSites();
  std::cout << "Checking for on-node timeslices tstart=" << tstart << " tsep=" << tsep << std::endl;
  for(int tlin=tstart;tlin<=tstart+tsep;tlin++){
    int tprd = tlin % Lt;
    int tlcl = tprd - toff;
    std::cout << "tlin=" << tlin << " tprd=" << tprd << " tlcl=" << tlcl << " on-node ? " << (tlcl >= 0 && tlcl < GJP.TnodeSites()) << std::endl;
    
    if(tlcl >= 0 && tlcl < GJP.TnodeSites()) return true;
  }
  return false;
}

CPS_END_NAMESPACE


#endif
