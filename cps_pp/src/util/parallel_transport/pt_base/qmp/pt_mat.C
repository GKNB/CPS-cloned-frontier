#ifdef USE_QMP
#include <assert.h>
#include <util/omp_wrapper.h>
/*! \file
  \brief  Definition of parallel transport definitions for QCDOC.
  
*/
//--------------------------------------------------------------------
//
//
//--------------------------------------------------------------------
#include <string.h>
#include "asq_data_types.h"
#include "pt_int.h"
#include "pt_qcdoc.h"
#include <util/gjp.h>

//CK: parallel transport  psi(x-mu) = U_mu(x-mu)psi(x)
//                        psi(x+mu) = U_mu^dag(x)psi(x)
//
// or 'backwards' psi(x) = U_mu(x)psi(x+mu), requires backwards comms of psi only
//    'forwards'  psi(x) = U_mu^dag(x-mu)psi(x-mu), requires forwards comms of psi and link

//Parallel transport of a matrix defined on one half of the
//checkerboaded lattice
//
//Parameters
//
//n - The number of direction in which to perform the parallel transport
//mout - Result of the parallel transport, on sites with opposite parity of min
//min - Initial field, defined on sites with only one parity
//dir - a list of the n directions in which the field will be transported
//cb - Checkerboard parity of the vector min

#undef PROFILE
void PT::mat_cb(int n, IFloat **mout, IFloat **min, const int *dir, int
parity, IFloat * new_gauge_field)
{
  mat_cb_norm(n,mout,min,dir,parity,new_gauge_field);
}

#undef PROFILE
void PT::mat_cb(int n, IFloat **mout, IFloat **min, const int *dir, int
parity)
{
  mat_cb_norm(n,mout,min,dir,parity,gauge_field_addr);
}

static const int MAX_DIR=10;

#undef PROFILE
void PT::mat_cb_norm(int n, IFloat **mout, IFloat **min, const int *dir, int
parity, IFloat * gauge)
{
  //List of the different directions
  int wire[MAX_DIR];
  int i;
//  printf("PT::mat_cb_norm\n");

  QMP_msgmem_t *msg_mem_p = (QMP_msgmem_t *)Alloc("","vec_cb_norm", "msg_mem_p", 2*non_local_dirs*sizeof(QMP_msgmem_t));
  QMP_msghandle_t* msg_handle_p = (QMP_msghandle_t *)Alloc("","vec_cb_norm", "msg_handle_p", 2*non_local_dirs*sizeof(QMP_msghandle_t));
  QMP_msghandle_t multiple;
  static int call_num = 0;
  int vlen = VECT_LEN;
  int vlen2 = VECT_LEN;

  call_num++;
  
  //Name our function
  char *fname="pt_mat_cb()";
  //  VRB.Func("",fname);
  
  //Set the transfer directions
  //If wire[i] is even, then we have communication in the negative direction
  //If wire[i] is odd, then we have communication in the positive direction
  for(i=0;i<n;i++)
    wire[i]=dir[i];

#ifdef PROFILE
  Float dtime  = - dclock();
#endif
  int non_local_dir=0;

//#pragma omp parallel default(shared)
{

  //If wire[i] is odd, then we have parallel transport in the
  //positive direction.  In this case, multiplication by the link matrix is
  //done before the field is transferred over to the adjacent node
  //
  //If we have transfer in the negative T direction (wire[i] = 6), then
  //we have to copy the appropriate fields to a send buffer
//#pragma omp for
  for(i=0;i<n;i++)
    {
      if(!local[wire[i]/2])
      {
	if(wire[i]%2)
	  {
	    if(conjugated)
	      pt_cmm_cpp(non_local_chi_cb[wire[i]],(long)uc_nl_cb_pre[parity][wire[i]/2],(long)min[i],(long)snd_buf_cb[wire[i]/2],(long)gauge);
	    else
	      pt_cmm_dag_cpp(non_local_chi_cb[wire[i]],(long)uc_nl_cb_pre[parity][wire[i]/2],(long)min[i],(long)snd_buf_cb[wire[i]/2],(long)gauge);
	  }
	else if((wire[i] == 6))
	  {
	    for(int j = 0; j < non_local_chi_cb[6];j++)
	      memcpy(snd_buf_t_cb + j*GAUGE_LEN,min[i] + 3 * *(Toffset[parity]+j)*3,GAUGE_LEN*sizeof(IFloat));
	  }
      }
    }

//#pragma omp barrier
//#pragma omp master 
{
  for(i=0;i<n;i++)
    if(!local[wire[i]/2])
    {
      //Calculate the starting address for the data to be sent
      IFloat *addr = min[i] + GAUGE_LEN * offset_cb[wire[i]];

      msg_mem_p[2*non_local_dir] = QMP_declare_msgmem((void *)rcv_buf[wire[i]], 3*non_local_chi_cb[wire[i]]*VECT_LEN*sizeof(IFloat));
      
      //Initialize the msg_mem for sends
      if(wire[i]%2) 
	msg_mem_p[2*non_local_dir+1] = QMP_declare_msgmem((void *)snd_buf_cb[wire[i]/2], 3*non_local_chi_cb[wire[i]]*VECT_LEN*sizeof(IFloat));
      else if(wire[i] == 6)
	msg_mem_p[2*non_local_dir+1] = QMP_declare_msgmem((void *)snd_buf_t_cb, 3*non_local_chi_cb[wire[i]]*VECT_LEN*sizeof(IFloat));
      else
	msg_mem_p[2*non_local_dir+1] = QMP_declare_strided_msgmem((void *)addr, (size_t)(3*blklen_cb[wire[i]]), numblk_cb[wire[i]], (ptrdiff_t)(3*stride_cb[wire[i]]+3*blklen_cb[wire[i]]));
      
      msg_handle_p[2*non_local_dir] = QMP_declare_receive_relative(msg_mem_p[2*non_local_dir], wire[i]/2, 1-2*(wire[i]%2), 0);
      msg_handle_p[2*non_local_dir+1] = QMP_declare_send_relative(msg_mem_p[2*non_local_dir+1], wire[i]/2, 2*(wire[i]%2)-1, 0);

      non_local_dir++;

    }

  if(non_local_dir) {
    multiple = QMP_declare_multiple(msg_handle_p, 2*non_local_dir);
    QMP_start(multiple);
  }
} //#pragma omp master {

  //Do local calculations
//#pragma omp for
  for(i=0;i<n;i++)
    {
      if((wire[i]%2 && conjugated) || ((wire[i]%2 == 0) && (conjugated == 0)))
	pt_cmm_cpp(local_chi_cb[wire[i]],(long)uc_l_cb[parity][wire[i]],(long)min[i],(long)mout[i],(long)gauge);
      else
	pt_cmm_dag_cpp(local_chi_cb[wire[i]],(long)uc_l_cb[parity][wire[i]],(long)min[i],(long)mout[i],(long)gauge);
    }

//#pragma omp barrier
//#pragma omp master 
{
  if(non_local_dir) {
    QMP_status_t qmp_complete_status = QMP_wait(multiple);
    if (qmp_complete_status != QMP_SUCCESS)
          QMP_error("Send failed in vec_cb_norm: %s\n", QMP_error_string(qmp_complete_status));
    QMP_free_msghandle(multiple);
    for(int i = 0; i < 2*non_local_dir; i++)
      QMP_free_msgmem(msg_mem_p[i]);
    Free(msg_handle_p);
    Free(msg_mem_p);
  }
} //#pragma omp master {

  //If wire[i] is even, then we have transport in the negative direction
  //In this case, the vector field is multiplied by the SU(3) link matrix
  //after all communication is complete
  IFloat *fp0,*fp1;
//#pragma omp for
  for(i=0;i<n;i++)
    {
      if(!local[wire[i]/2])
      	{
	  if(!(wire[i]%2))
	    {
	      if(conjugated)
		pt_cmm_dag_cpp(non_local_chi_cb[wire[i]],(long)uc_nl_cb[parity][wire[i]],(long)rcv_buf[wire[i]],(long)mout[i],(long)gauge);
	      else
		pt_cmm_cpp(non_local_chi_cb[wire[i]],(long)uc_nl_cb[parity][wire[i]],(long)rcv_buf[wire[i]],(long)mout[i],(long)gauge);
	    }
	  //Otherwise we have parallel transport in the positive direction.
	  //In this case, the received data has already been pre-multiplied
	  //All we need to do is to put the transported field in the correct place
	  else
	    {
	      //int destination, source;
	      //Place the data in the receive buffer into the result vector
	      for(int s=0;s<non_local_chi_cb[wire[i]];s++)
		{
		  //source = uc_nl_cb[parity][wire[i]][s].src;
		  fp0 = (IFloat *)((long)rcv_buf[wire[i]]+3*uc_nl_cb[parity][wire[i]][s].src);
		  //destination = uc_nl_cb[parity][wire[i]][s].dest;
		  fp1 = (IFloat *)(mout[i]+3*uc_nl_cb[parity][wire[i]][s].dest);
		  memcpy(fp1,fp0,GAUGE_LEN*sizeof(IFloat));
		}
	    }
	}
    }

} //#pragma omp parallel
#ifdef PROFILE
  dtime +=dclock();
  print_flops("",fname,99*vol*n,dtime);
#endif
//  ParTrans::PTflops +=99*n*vol;
}

//-----------------------------------------------------------------------------

static void UnLexVector(int v_str_ord, int *into, int from, int vlen, int *lattsize){
  if(v_str_ord != PT_XYZT){ printf("UnLexVector: Cannot unlex vector coords that aren't in XYZT ordering"); exit(-1); }
  //int result = x[0] + size[0]*(x[1]+size[1]*(x[2]+size[2]*x[3]));
  from /= vlen;
  into[0] = from % lattsize[0];
  from /= lattsize[0];
  into[1] = from % lattsize[1];
  from /= lattsize[1];
  into[2] = from % lattsize[2];
  from /= lattsize[2];
  into[3] = from % lattsize[3];
}

static void printMat(PTmatrix* m, int x,int y,int z,int t){
  int idx = x + cps::GJP.XnodeSites()*(y+cps::GJP.YnodeSites()*(z+cps::GJP.ZnodeSites()*t));
  IFloat *mat = (IFloat*)(m+idx);
  int off =0 ;
  for(int i=0;i<3;i++){
    for(int j=0;j<3;j++){
      printf("(%f,%f) ",mat[off],mat[off+1]);
      off+=2;
    }
    printf("\n");
  }
}
//Parallel transport of a matrix. through one hop.
//The matrix min is parallel transported and the result is placed in mout
#if 1
#define PROFILE
void PT::mat(int n, PTmatrix **mout, PTmatrix **min, const int *dir){
  int wire[MAX_DIR];
  int i;
  QMP_msgmem_t msg_mem_p[2*MAX_DIR];
  QMP_msghandle_t msg_handle_p[2*MAX_DIR];
  QMP_msghandle_t multiple;
  static double setup=0.,qmp=0.,localt=0.,nonlocal=0.;
  static int call_num = 0;

  call_num++;
  char *fname="pt_mat()";
//  VRB.Func("",fname);
//  if (call_num%100==1) printf("PT:mat()\n");

  if (!QMP_get_node_number() ) printf("PT::mat with omp_get_max_threads() = %d\n", omp_get_max_threads() );

  for(i=0;i<n;i++) wire[i] = dir[i]; 
#ifdef PROFILE
  Float dtime2  = - dclock();
#endif

  double dtime = -dclock();

  int non_local_dir=0;
  
  for(i=0;i<n;i++)
  if (!local[wire[i]/2]) {
    //Calculate the address for transfer in a particular direction
    Float * addr = ((Float *)min[i]+GAUGE_LEN*offset[wire[i]]);
    msg_mem_p[2*non_local_dir] = QMP_declare_msgmem((void *)rcv_buf[wire[i]], 3*non_local_chi[wire[i]]*VECT_LEN*sizeof(IFloat));
    msg_mem_p[2*non_local_dir+1] = QMP_declare_strided_msgmem((void *)addr, (size_t)(3*blklen[wire[i]]), numblk[wire[i]], (ptrdiff_t)(3*stride[wire[i]]+3*blklen[wire[i]]));
    
    //direction mu = wire[i]/2, 
    //'sign' is relative direction of the node that we are communicating with
    //for the receive from the node in front, sign is +1, opposite for receive from node behind
    //for the send to the node behind, sign is -1 , opposite for send to node in front
    //if wire[i] is even, we receive from the node in front (backwards comms)
    //if wire[i] is odd, we receive from the node behind (forwards comms)

    msg_handle_p[2*non_local_dir] = QMP_declare_receive_relative(msg_mem_p[2*non_local_dir], wire[i]/2, 1-2*(wire[i]%2), 0);
    msg_handle_p[2*non_local_dir+1] = QMP_declare_send_relative(msg_mem_p[2*non_local_dir+1], wire[i]/2, 2*(wire[i]%2)-1, 0);

    non_local_dir++;
  }
  //for forward comms (forward transport), both U^dag(x-mu) and psi(x-mu) are communicated. 
  //for backward comms (backwards transport), just psi(x+mu) is communicated. 
  //G-parity complex conjugation at boundary for link is already done
  //This code is for local sites, but this can include a G-parity boundary if the number of nodes in that direction is 1

  if(cps::GJP.Gparity()){
    int vlen = VECT_LEN*sizeof(IFloat); //size of incoming vector

    int* gp_l_orig_offsets[n];
    PTmatrix* gp_l_conj_buf[n];

    //For G-parity, fermionic boundary conditions have a + sign between flavours 1 and 2, and a - sign between flavours 2 and 1.
    //These are implemented on the gauge fields:  *-->[U_0 U_1 U_2 U_3]-->[U*_0 U*_1 U*_2 -U*_3]-->* cycling between flavours 1 and 2
    //so any matrix M created as products of gauge links does not obey M^(2)(x) = [M^(1)(x)]* when the fermion BCs are active
    //there can be a relative sign between them that is not known without knowing exactly what links make up M

    //There are two solutions: 
    //1) switch off the fermion G-parity minus sign - this has to be done in ParTransGauge constructor (or switch them off and on every time pt_mat is called). Then we can just calculate M^(1)(x) and know M^2(x) automatically
    //   the code will then only work for parallel transport of gauge links in the gauge force calculation where there are no fermionic aspects. Not sure if correct for other uses of ParTransGauge
    //2) maintain 2 independent flavour fields that swap at the boundary - more general but requires 2x the amount of computational effort for PT of gauge links than version 1
    
    //Implement version 1 for optimal speed

    for(int wireidx=0;wireidx<n;wireidx++){
      int wire_dir = wire[wireidx];
      int sites = local_chi_gp[wire_dir];
      if(sites==0) continue;
      
      gp_l_conj_buf[wireidx] = (PTmatrix *)FastAlloc("PT","mat","gp_l_conj_buf",sizeof(PTmatrix)*sites);
      gp_l_orig_offsets[wireidx] = (int *)FastAlloc("PT","mat","gp_l_orig_offsets",sizeof(int)*sites);

      for(int s=0; s<sites; s++){
	IFloat *srcmat = (IFloat*)(  (long)min[wireidx] + 3*uc_l_gpbound[wire_dir][s].src  );
	IFloat *destmat = (IFloat*)(gp_l_conj_buf[wireidx]+s);

	int pos[4]; 
	UnLexVector(v_str_ord,pos,uc_l_gpbound[wire_dir][s].src, vlen, size);
	int posy[4];
	UnLexVector(v_str_ord,posy,uc_l_gpbound[wire_dir][s].dest, vlen, size);

	//printf("wire %d, star-copying m_in for PT from x=(%d,%d,%d,%d) to y=(%d,%d,%d,%d)  [site %d in buffer]\n",wire_dir,pos[0],pos[1],pos[2],pos[3],posy[0],posy[1],posy[2],posy[3],s);
	StarCopy(destmat,srcmat); //copy-conjugate matrix to buffer

	//Copy(destmat,srcmat); //copy-conjugate matrix to buffer
	gp_l_orig_offsets[wireidx][s] = uc_l_gpbound[wire_dir][s].src; //to restore at end of function
	uc_l_gpbound[wire_dir][s].src = s*vlen;
      }
    }

    //do the local pt on the G-parity boundary
    if(omp_get_max_threads()<n){
      //omp_set_num_threads(n); THIS IS A BAD IDEA

      //Do it unthreaded
      for(int n_t =0; n_t < n; n_t++){ //local stuff
	int w_t = wire[n_t];
	int ipoints = local_chi_gp[w_t]/2;
	int offset = 0;
	partrans_cmm_agg((uc_l_gpbound[w_t]+offset*2),gp_l_conj_buf[n_t],mout[n_t],ipoints);
      }

    }else{

      //assume nt > n!
      static char *cname="mat()";
#pragma omp parallel default(shared)
      {
	int iam,nt,ipoints,istart,offset;
	iam = omp_get_thread_num(); //current thread number
	nt = omp_get_num_threads(); //number of threads
	int nt_dir = nt/n; //split number of matrices (= number of directions in direction vector) to be transported across the threads. nt_dir is number of threads per matrix
	int n_t = iam/nt_dir; //matrix number of this thread
	int i_t = iam%nt_dir; //index within set of threads operating on this matrix
	if (n_t >= n ){  n_t = n-1;
	  i_t = iam - (n-1)*nt_dir;
	  nt_dir = nt -(n-1)*nt_dir;
	}
	int w_t = wire[n_t]; //direction of comms for this matrix
	ipoints = (local_chi_gp[w_t]/2)/nt_dir; //number of local sites for this matrix divide by number of threads working on this matrix (why divide by 2?)
	offset = ipoints*i_t;
	if (i_t == (nt_dir-1)) ipoints = (local_chi_gp[w_t]/2)-offset;
	partrans_cmm_agg((uc_l_gpbound[w_t]+offset*2),gp_l_conj_buf[n_t],mout[n_t],ipoints);
      }
    }
    
    //restore original site offsets
    for(int wireidx=0;wireidx<n;wireidx++){
      int wire_dir = wire[wireidx];
      int sites = local_chi_gp[wire_dir];
      if(sites==0) continue;
      for(int s=0; s<sites; s++){
	uc_l_gpbound[wire_dir][s].src = gp_l_orig_offsets[wireidx][s];
      }
      Free(gp_l_conj_buf[wireidx]);
      Free(gp_l_orig_offsets[wireidx]);
    }

  }//end of G-parity local

  if (call_num==1 && !QMP_get_node_number())
	printf("non_local_dir=%d\n",non_local_dir);

  if(non_local_dir) {
    multiple = QMP_declare_multiple(msg_handle_p, 2*non_local_dir);
    QMP_start(multiple);
  }

  dtime += dclock();
  setup +=dtime;
  dtime = -dclock();
  int if_print = 0;
  if ( (call_num%10000==1) && (!QMP_get_node_number()) ) if_print=1;

#undef USE_TEST2
#ifdef USE_TEST2

  if(omp_get_max_threads()<n){
    //omp_set_num_threads(n); THIS IS A BAD IDEA

    //Do it unthreaded
    for(int n_t =0; n_t < n; n_t++){ //local stuff
      int w_t = wire[n_t];
      int ipoints = local_chi[w_t]/2;
      int offset = 0;
      partrans_cmm_agg((uc_l[w_t]+offset*2),min[n_t],mout[n_t],ipoints);
    }

  }else{
  
    //assume nt > n!
    static char *cname="mat()";
#pragma omp parallel default(shared)
    {  
      int iam,nt,ipoints,istart,offset;
      iam = omp_get_thread_num(); //current thread number
      nt = omp_get_num_threads(); //number of threads
      int nt_dir = nt/n; //split number of matrices (= number of directions in direction vector) to be transported across the threads. nt_dir is number of threads per matrix
  assert(nt_dir>0);
      int i_t = iam%nt_dir; //index within set of threads operating on this matrix
      if (n_t >= n ){  n_t = n-1;
	i_t = iam - (n-1)*nt_dir;
	nt_dir = nt -(n-1)*nt_dir;
      }
      int w_t = wire[n_t]; //direction of comms for this matrix
      ipoints = (local_chi[w_t]/2)/nt_dir; //number of local sites for this matrix divide by number of threads working on this matrix (why divide by 2?)
  if(((local_chi[w_t]/2)%nt_dir)>0) ipoints += 1;
      offset = ipoints*i_t;
      if (i_t == (nt_dir-1)) ipoints = (local_chi[w_t]/2)-offset;
      if ( if_print )
      printf("thread %d of %d local_chi/2 nt_dir n_t i_t ipoints offset= %d %d %d %d %d %d\n",iam,nt,local_chi[w_t]/2,nt_dir,n_t,i_t,ipoints,offset);
      //Interleaving of local computation of matrix multiplication
      //if wire[i] is even, we receive from the node in front (backwards comms)
      //if wire[i] is odd, we receive from the node behind (forwards comms)

      partrans_cmm_agg((uc_l[w_t]+offset*2),min[n_t],mout[n_t],ipoints);

      if ( if_print )
	printf("thread %d of %d done\n",iam,nt);
    }
  
  }

#else
  {
    //Interleaving of local computation of matrix multiplication
#pragma omp parallel for default(shared)
    for(i=0;i<n;i++){
  int iam,nt;
  iam = omp_get_thread_num();
  nt = omp_get_num_threads();
    if ( if_print )
      printf("thread %d of %d i=%d\n",iam,nt,i);
      partrans_cmm_agg(uc_l[wire[i]],min[i],mout[i],local_chi[wire[i]]/2);
    }
  }
#endif

  dtime += dclock();
  localt +=dtime;
  dtime = -dclock();
//#pragma omp barrier
//#pragma omp master 
  {
    if(non_local_dir) {
      QMP_status_t qmp_complete_status = QMP_wait(multiple);
      if (qmp_complete_status != QMP_SUCCESS)
	QMP_error("Send failed in vec_cb_norm: %s\n", QMP_error_string(qmp_complete_status));
      QMP_free_msghandle(multiple);
      for(int i = 0; i < 2*non_local_dir; i++)
	QMP_free_msgmem(msg_mem_p[i]);
      //    Free(msg_handle_p);
      //    Free(msg_mem_p);
    }
  } //#pragma omp master {
  dtime += dclock();
  qmp +=dtime;
  dtime = -dclock();

  //Do non-local computations
#ifdef USE_TEST2
  if(omp_get_max_threads()<n){
    //omp_set_num_threads(n); //THIS IS A BAD IDEA
    
    //Do it unthreaded
    for(int n_t =0; n_t < n; n_t++){
      int w_t = wire[n_t];
      int ipoints = non_local_chi[w_t]/2;
      int offset = 0;
      if (ipoints>0){
	int mu = w_t/2;
	if(cps::GJP.Bc(mu) == cps::BND_CND_GPARITY && 
	   ( (cps::GJP.NodeCoor(mu)==0 && w_t%2==1) || (cps::GJP.NodeCoor(mu)==cps::GJP.Nodes(mu)-1 && w_t%2==0)  ) ){
	  gauge_agg *agg = uc_nl[w_t]+offset*2;
	  for(int m=0;m<2*ipoints;m++){
	    IFloat* mat = (IFloat *)((long long)rcv_buf[w_t] + 3*agg[m].src);
	    for(int c=1;c<18;c+=2) mat[c]*=-1; //complex conjugate all the matrices in the receive buffer
	  }
	}
	partrans_cmm_agg((uc_nl[w_t]+offset*2),(PTmatrix *)rcv_buf[w_t],mout[n_t],ipoints);
      }
    }

  }else{
  
    //assume nt > n!
#pragma omp parallel default(shared)
    {
      int iam,nt,ipoints,istart,offset;
      iam = omp_get_thread_num();
      nt = omp_get_num_threads();
      int nt_dir = nt/n;
      int n_t = iam/nt_dir;
      int i_t = iam%nt_dir;
      if (n_t >= n ){  n_t = n-1;
	i_t = iam - (n-1)*nt_dir;
	nt_dir = nt -(n-1)*nt_dir;
      }
      int w_t = wire[n_t];
      ipoints = (non_local_chi[w_t]/2)/nt_dir;
  if(((non_local_chi[w_t]/2)%nt_dir)>0) ipoints += 1;
      offset = ipoints*i_t;
      if (i_t == (nt_dir-1)) ipoints = (non_local_chi[w_t]/2)-offset;
      if ( if_print )
      printf("thread %d of %d local_chi/2 nt_dir n_t i_t ipoints offset= %d %d %d %d %d %d\n",iam,nt,non_local_chi[w_t]/2,nt_dir,n_t,i_t,ipoints,offset);
      //Non-local computation
      if (ipoints>0){
	int mu = w_t/2;
	if(cps::GJP.Bc(mu) == cps::BND_CND_GPARITY && 
	   ( (cps::GJP.NodeCoor(mu)==0 && w_t%2==1) || (cps::GJP.NodeCoor(mu)==cps::GJP.Nodes(mu)-1 && w_t%2==0)  ) ){
	  gauge_agg *agg = uc_nl[w_t]+offset*2;
	  for(int m=0;m<2*ipoints;m++){
	    IFloat* mat = (IFloat *)((long long)rcv_buf[w_t] + 3*agg[m].src);
	    for(int c=1;c<18;c+=2) mat[c]*=-1; //complex conjugate all the matrices in the receive buffer
	  }
	}
	partrans_cmm_agg((uc_nl[w_t]+offset*2),(PTmatrix *)rcv_buf[w_t],mout[n_t],ipoints);
      }
      if ( if_print )
	printf("thread %d of %d done\n",iam,nt);
    }

  }


#else
  {
#pragma omp parallel for
    for(i=0;i<n;i++) 
      if (!local[wire[i]/2]) {
#ifdef USE_OMP
	if (call_num%10000==1 && !QMP_get_node_number() ) 
	  printf("thread %d of %d i=%d\n",omp_get_thread_num(),omp_get_num_threads(),i);
#endif
	int mu = wire[i]/2;
	//GPARITY CODE BELOW IS BROKEN (ALSO NOT USED)
	if(cps::GJP.Bc(mu) == cps::BND_CND_GPARITY)
	  for(int m=1;m<non_local_chi[wire[i]]*VECT_LEN*3;m+=2) rcv_buf[wire[i]][m]*=-1; //complex conjugate all the matrices in the receive buffer
	partrans_cmm_agg(uc_nl[wire[i]],(PTmatrix *)rcv_buf[wire[i]],mout[i],non_local_chi[wire[i]]/2);
      }

  }//#pragma omp parallel
#endif

  dtime += dclock();
  nonlocal +=dtime;

  if (call_num%100==0){
    static char *cname="mat()";
    if (!QMP_get_node_number() ) {
    print_flops("mat():local*100",0,localt);
    print_flops("mat():nonlocal*100",0,nonlocal);
    print_flops("mat():qmp*100",0,qmp);
    print_flops("mat():setup*100",0,setup);
    }
    localt=nonlocal=qmp=setup=0.;
  }

//#ifdef PROFILE
#if 0
  if (call_num%100==0){
  dtime2 +=dclock();
  print_flops(fname,198*vol*n,dtime2);
  }
#endif
//  ParTrans::PTflops +=198*n*vol;
//
  if(call_num>=10000) call_num=0;
}
#endif
#undef PROFILE

#endif






