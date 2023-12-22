#ifndef _UTILS_PARALLEL_GLOBALSUM_H_
#define _UTILS_PARALLEL_GLOBALSUM_H_

#include <util/gjp.h>
#ifdef USE_GRID
#include <Grid/Grid.h>
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif

#include "utils_parallel.h"
#include "utils_logging.h"

#include <condition_variable>
#include <mutex>
#include <thread>

CPS_START_NAMESPACE

inline void globalSum(double *result, size_t len = 1){
#ifdef A2A_GLOBALSUM_DISK
  LOGA2A << "Performing reduction of " << byte_to_MB(len*sizeof(double)) << " MB through disk" << std::endl;
  disk_reduce(result,len);
#elif defined(A2A_GLOBALSUM_MAX_ELEM) //if this is defined, the global sum will be broken up into many of this size
  size_t b = A2A_GLOBALSUM_MAX_ELEM;
  LOGA2A << "Performing reduction of " << len << " doubles in " << (len+b-1) / b << " blocks of size " << b << std::endl;
  size_t lencp = len;
  for(size_t off = 0; off < lencp; off += b){
    size_t rlen = std::min(len, b);
    MPI_Allreduce(MPI_IN_PLACE, result, rlen, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    len -= rlen;
    result += rlen;
  }
#elif defined(USE_QMP)
  QMP_sum_double_array(result, len);
#elif defined(USE_MPI)
  MPI_Allreduce(MPI_IN_PLACE, result, len, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
  if(GJP.TotalNodes()!=1) ERR.General("","globalSum(double *result, int len)","Only implemented for QMP/MPI on parallel machines");
#endif
}

inline void globalSum(float *result, size_t len = 1){
#ifdef A2A_GLOBALSUM_DISK
  LOGA2A << "Performing reduction of " << byte_to_MB(len*sizeof(float)) << " MB through disk" << std::endl;
  disk_reduce(result,len);
#elif defined(A2A_GLOBALSUM_MAX_ELEM) //if this is defined, the global sum will be broken up into many of this size
  size_t b = A2A_GLOBALSUM_MAX_ELEM;
  LOGA2A << "Performing reduction of " << len << " floats in " << (len+b-1) / b << " blocks of size " << b << std::endl;
  size_t lencp = len;
  for(size_t off = 0; off < lencp; off += b){
    size_t rlen = std::min(len, b);
    MPI_Allreduce(MPI_IN_PLACE, result, rlen, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    len -= rlen;
    result += rlen;
  }
#elif defined(USE_QMP)
  QMP_sum_float_array(result, len);
#elif defined(USE_MPI)
  MPI_Allreduce(MPI_IN_PLACE, result, len, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#else
  if(GJP.TotalNodes()!=1) ERR.General("","globalSum(float *result, int len)","Only implemented for QMP/MPI on parallel machines");
#endif
}

inline void globalSum(std::complex<double> *result, size_t len = 1){
  globalSum( (double*)result, 2*len );
}
inline void globalSum(std::complex<float> *result, size_t len = 1){
  globalSum( (float*)result, 2*len );
}



#ifdef USE_GRID

#if defined(GRID_CUDA) || defined(GRID_HIP)
inline void globalSum(thrust::complex<double>* v, const size_t n = 1){
  globalSum( (double*)v,2*n);
}
inline void globalSum(thrust::complex<float>* v, const size_t n = 1){
  globalSum( (float*)v,2*n);
}


#endif //GRID_CUDA || GRID_HIP

template<typename T>
struct _globalSumComplexGrid{
  static inline void doit(T *v, const int n){
    typedef typename T::scalar_type scalar_type; //an std::complex type
    typedef typename scalar_type::value_type floatType;    
    int vmult = sizeof(T)/sizeof(scalar_type);
    floatType * ptr = (floatType *)v; 
    globalSum(ptr,2*n*vmult);
  }
};

inline void globalSum(Grid::vComplexD* v, const size_t n = 1){
  _globalSumComplexGrid<Grid::vComplexD>::doit(v,n);
}
inline void globalSum(Grid::vComplexF* v, const size_t n = 1){
  _globalSumComplexGrid<Grid::vComplexF>::doit(v,n);
}

#endif //GRID


class MPIallReduceQueued{
public:
  class Request;
  typedef std::list<Request>::iterator handleType;

  class Request{
    friend class MPIallReduceQueued;
    std::condition_variable cv;
    std::mutex* mutex_p;
    bool complete;

    void* data;
    size_t size;
    MPI_Datatype datatype;

    handleType handle;
    std::list<Request>* tasks;

    void setHandle(handleType h){ handle = h; }

  public:
    Request(void* data, size_t size, MPI_Datatype datatype, std::mutex &m, std::list<Request> &tasks): data(data), size(size), datatype(datatype), mutex_p(&m), tasks(&tasks), complete(false){}   
    Request(Request &&r): mutex_p(r.mutex_p), complete(r.complete), data(r.data), size(r.size), datatype(r.datatype), handle(r.handle), tasks(r.tasks){};

    void signalComplete(){
      std::unique_lock lk(*mutex_p);
      complete = true;
      lk.unlock(); //cf https://en.cppreference.com/w/cpp/thread/condition_variable
      cv.notify_one();
    }

    void wait(){
      std::unique_lock lk(*mutex_p);
      cv.wait(lk, [&c=complete]{ return c; });

      //Once the wait has completed, the request should be removed from the queue
      //Do this under the mutex lock!
      if(!lk.owns_lock()) lk.lock();
      tasks->erase(handle);
    }
  };

private:
  mutable std::mutex m_mutex;
  std::list<Request> m_tasks;
  std::thread* m_thr;
  bool m_stop; //stop signal for thread
  bool m_have_work; //signal if work is available
  std::condition_variable m_work_cv; //condition variable to check for work
  std::list<handleType> m_queue;
  bool m_verbose;
  
  inline static bool checkStop(bool & stop, std::mutex& mutex){
    std::lock_guard _(mutex); return stop;
  }
  inline static bool getWork(handleType &into, std::list<handleType> &queue, bool &have_work, std::mutex& mutex){
    std::lock_guard _(mutex);
    if(queue.size()){ //FIFO queue
      into=queue.front();
      queue.pop_front();
      return true;
    }else{
      have_work = false;
      return false;
    }
  }

public:
  MPIallReduceQueued(): m_thr(nullptr), m_stop(false), m_have_work(false), m_verbose(false){

    m_thr = new std::thread([&m_stop=m_stop,&m_queue=m_queue,&m_mutex=m_mutex, &m_have_work=m_have_work, &m_work_cv=m_work_cv, &m_verbose=m_verbose]{
	while(!checkStop(m_stop, m_mutex)){ //outer loop
	  //wait until there is some work to be done
	  if(m_verbose) std::cout << "MPIAllReduceQueued waiting for work" << std::endl;
	  {
	    std::unique_lock lk(m_mutex);
	    m_work_cv.wait(lk, [&m_have_work=m_have_work, &m_stop=m_stop]{ return m_have_work || m_stop; }); //allow it to break out if told to stop
	  }
	  if(m_verbose) std::cout << "MPIAllReduceQueued work is available" << std::endl;
	  handleType h;
	  while(getWork(h,m_queue,m_have_work,m_mutex)){
	    if(m_verbose) std::cout << "MPIAllReduceQueued reducing " << h->data << " size " << h->size << std::endl;
	    assert( MPI_Allreduce(MPI_IN_PLACE, h->data, h->size, h->datatype, MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS );
	    if(m_verbose) std::cout << "MPIAllReduceQueued completed reduction, signaling completion" << std::endl;
	    h->signalComplete();
	  }
	}
      });
    
  }

  void setVerbose(bool to){ m_verbose=to; }
  
  handleType enqueue(void *data, size_t size, MPI_Datatype type){
    std::unique_lock lk(m_mutex);
    auto h = m_tasks.insert(m_tasks.end(), Request(data,size,type,m_mutex,m_tasks));
    h->setHandle(h);
    m_queue.push_back(h);
    m_have_work = true;
    lk.unlock();
    m_work_cv.notify_one();
    return h;
  }

  ~MPIallReduceQueued(){
    if(m_thr){
      {
	std::unique_lock lk(m_mutex);
	m_stop = true;
	lk.unlock();
	m_work_cv.notify_one();
      }
      m_thr->join();
    }
  }

  static inline MPIallReduceQueued & globalInstance(){ static MPIallReduceQueued q; return q; }
};


CPS_END_NAMESPACE

#endif
