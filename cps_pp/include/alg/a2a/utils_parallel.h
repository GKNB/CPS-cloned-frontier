#ifndef _UTILS_PARALLEL_H_
#define _UTILS_PARALLEL_H_

#include <util/gjp.h>
#ifdef USE_GRID
#include <Grid/Grid.h>
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif

CPS_START_NAMESPACE

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
inline void thread_work(int &my_work, int &my_offset, const int total_work, const int me, const int team){
  my_work = total_work/team;
  my_offset = me * my_work;
  
  int rem = total_work - my_work * team;
  if(me < rem){
    ++my_work; //first rem threads mop up the remaining work
    my_offset += me; //each thread before me has gained one extra unit of work
  }else my_offset += rem; //after the first rem threads, the offset shift is uniform
}


//Functions for performing global and timeslice sums of single or double precision quantities. Daiqian had to implement these himself as CPS can only do this with the Float=double type
template <typename T>
void QMP_sum_array(T *result, int len){
#ifdef USE_QMP
  if(sizeof(T) == sizeof(double)) {
    QMP_sum_double_array((double*)result, len);
  } else if(sizeof(T) == sizeof(float)) {
    QMP_sum_float_array((float*)result, len);
  } else {
    QMP_error("QMP_sum_array::data type not supported!\n");
  }
#else
  //CK: This only works for single-node code
  int nodes = 1; for(int i=0;i<4;i++) nodes *= cps::GJP.Nodes(i);
  if(nodes != 1){
    cps::ERR.General("","QMP_sum_array(T *result, int len)","Only implemented for QMP on parallel machines");
  }
  //do nothing!
#endif
}

#ifndef USE_QMP
inline void QMP_sum_double_array(double *result, int len){
  //CK: This only works for single-node code
  int nodes = 1; for(int i=0;i<4;i++) nodes *= cps::GJP.Nodes(i);
  if(nodes != 1){
    cps::ERR.General("","QMP_sum_double_array fake definition","Not implemented on parallel machines: use QMP!");
  }
}
inline void QMP_sum_float_array(float *result, int len){
  //CK: This only works for single-node code
  int nodes = 1; for(int i=0;i<4;i++) nodes *= cps::GJP.Nodes(i);
  if(nodes != 1){
    cps::ERR.General("","QMP_sum_float_array fake definition","Not implemented on parallel machines: use QMP!");
  }
}
#endif


//Wrapper for global sums including Grid vectorized types
template<typename T>
void globalSumComplex(std::complex<T>* v, const int n){
  QMP_sum_array( (T*)v,2*n);
}
#ifdef USE_GRID
template<typename T>
struct _globalSumComplexGrid{
  static inline void doit(T *v, const int n){
    typedef typename T::scalar_type scalar_type; //an std::complex type
    typedef typename scalar_type::value_type floatType;    
    int vmult = sizeof(T)/sizeof(scalar_type);
    floatType * ptr = (floatType *)v; 
    QMP_sum_array(ptr,2*n*vmult);
  }
};

void globalSumComplex(Grid::vComplexD* v, const int n){
  _globalSumComplexGrid<Grid::vComplexD>::doit(v,n);
}
void globalSumComplex(Grid::vComplexF* v, const int n){
  _globalSumComplexGrid<Grid::vComplexF>::doit(v,n);
}
#endif



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

  int *node_map_send = (int*)malloc(nodes*sizeof(int));
  memset(node_map_send,0,nodes*sizeof(int));
  node_map_send[my_node_lex] = my_mpi_rank;

  map.resize(nodes);
  int ret = MPI_Allreduce(node_map_send, &map[0], nodes, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  assert(ret == MPI_SUCCESS);
  free(node_map_send);
}
#endif 


CPS_END_NAMESPACE


#endif
