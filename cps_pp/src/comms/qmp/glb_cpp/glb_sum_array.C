#include<config.h>
#include<util/qcdio.h>
//These are routines not provided for by 
//#ifndef USE_QMP
//#define USE_MPI
//#endif
#ifdef USE_MPI
#include<mpi.h>
#endif
//-------------------------------------------------------------------
/*!\file
  \brief Definition of glb_sum routine.
*/
#include<comms/glb.h>
#include<comms/scu.h>
#include<util/gjp.h>
#include<comms/double64.h>
#include <comms/sysfunc_cps.h>
#include <comms/glb_sum_internal.h>
CPS_START_NAMESPACE
int glb_sum (long *send, const long n_elem)
{
  int ret = n_elem;
#ifdef USE_MPI
  long recv[n_elem];
  ret = MPI_Allreduce ((long *) send, recv, n_elem, MPI_LONG, MPI_SUM,
                       MPI_COMM_WORLD);
  memcpy (send, recv, n_elem * sizeof (long));
#else
  if (UniqueID ())
    ERR.General ("", "glb_sum(ld,ld)", "Needs MPI");
#endif
  return ret;
}

int glb_sum (uint32_t * send, const long n_elem)
{
  int ret = n_elem;
#ifdef USE_MPI
  uint32_t recv[n_elem];
  ret =
    MPI_Allreduce ((uint32_t *) send, recv, n_elem, MPI_UNSIGNED, MPI_SUM,
                   MPI_COMM_WORLD);
  memcpy (send, recv, n_elem * sizeof (uint32_t));
#else
  double recv[n_elem];
  for (int i = 0; i < n_elem; i++) {
    recv[i] = send[i];
  }
  QMP_sum_double_array (recv, n_elem);
  for (int i = 0; i < n_elem; i++) {
    send[i] = recv[i];
  }
#endif
  return ret;
}

int glb_sum (int *send, const long n_elem)
{
  int ret = n_elem;
  int recv[n_elem];
#ifdef USE_MPI
  ret = MPI_Allreduce ((int *) send, recv, n_elem, MPI_INT, MPI_SUM,
                       MPI_COMM_WORLD);
  memcpy (send, recv, n_elem * sizeof (int));
#else
  for (int i = 0; i < n_elem; i++)
    QMP_sum_int (send + i);
#endif
  return ret;
}

int glb_sum (double *send, const long n_elem)
{
  int ret = n_elem;
#ifdef USE_MPI
  double recv[n_elem];
  ret = MPI_Allreduce ((double *) send, recv, n_elem, MPI_DOUBLE, MPI_SUM,
                       MPI_COMM_WORLD);
  memcpy (send, recv, n_elem * sizeof (double));
#else
  QMP_sum_double_array (send, n_elem);
#endif
  return ret;
}

CPS_END_NAMESPACE
