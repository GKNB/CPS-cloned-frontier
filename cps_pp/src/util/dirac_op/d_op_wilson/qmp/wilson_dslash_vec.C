#include <config.h>
#include <stdio.h>
#include <math.h>
#include <qmp.h>
#include <util/dirac_op.h>
#include <util/omp_wrapper.h>
#include <util/timer.h>

#warning "Using vectorised wilson dslash"

CPS_START_NAMESPACE
/*! \file
  \brief  Routine used internally in the DiracOpWilson class.

*/
/***************************************************************************/
/*                                                                         */
/* wilson_dslash: It calculates chi = Dslash * psi, or                     */
/*                                chi = DslashDag * psi, where Dslassh is  */
/* the Wilson fermion Dslash matrix. Dslash is a function of the gauge     */
/* fields u.                                                               */
/* cb = 0/1 denotes input on even/odd checkerboards i.e.                   */
/* cb = 1 --> Dslash_EO, acts on an odd column vector and produces an      */
/* even column vector                                                      */
/* cb = 0 --> Dslash_OE, acts on an even column vector and produces an     */
/* odd column vector,                                                      */
/*                                                                         */
/* dag = 0/1  results into calculating Dslash or DslashDagger.             */
/* lx,ly,lz,lt is the lattice size.                                        */
/*                                                                         */
/* This routine is to be used with scalar machines.                        */
/*                                                                         */
/* WARNING:                                                                 */
/*                                                                          */
/* This set of routines will work only if the node sublattices have         */
/* even number of sites in each direction.                                  */
/*                                                                          */
/***************************************************************************/
  CPS_END_NAMESPACE
#include <util/data_types.h>
#include <util/vector.h>
#include <util/gjp.h>
#include <util/wilson.h>
#include <util/error.h>
#include <util/verbose.h>
#include <util/smalloc.h>
#include <util/time_cps.h>
#include <comms/sysfunc_cps.h>
  CPS_START_NAMESPACE
#include "wilson_op.h"
static int Printf (char *format, ...)
{
}

//#define Printf if ( QMP_is_primary_node() ) printf
//#define Printf printf

//! Access to the elements of the \e SU(3) matrix
/*!
  Gets the element of the \e SU(3) matrix \e u with row \e row,
  column \e col and complex component \e d
*/
#define UELEM(u,r,row,col,d) *(u+(r+2*(row+3*(col+3*d))))
//! Access to the elements of a spinor vector.
/*!
  Gets the element of the spinor \e f with spin \e s,
  colour \e c and complex component \e r
*/
#define FERM(f,r,c,s) *(f +(r+2*(c+3*s)))

static unsigned long called = 0;
//static int initted=0;
static int init_len = 0;
static double setup = 0;
static double local = 0;
static double nonlocal = 0;
static double qmp = 0;
//keep track of whether initializations were performed in the G-parity framework. If they were and we switch out of G-parity
//mode, we want to reset the initialization status
static int gparity_init_status = 0;

static inline void MOVE_VEC (IFloat * buf, IFloat * psi, int vec_len,
                             unsigned long vec_offset)
{
  for (int i = 0; i < vec_len; i++) {
    moveMem (buf, psi, SPINOR_SIZE * sizeof (IFloat) / sizeof (char));
    buf += SPINOR_SIZE;
    psi += vec_offset;
  }
}

static inline void MOVE_VEC2 (IFloat * buf, IFloat * psi, int vec_len,
                              unsigned long vec_offset)
{
  for (int i = 0; i < vec_len; i++) {
    moveMem (buf, psi, SPINOR_SIZE * sizeof (IFloat) / sizeof (char));
    buf += vec_offset;
    psi += vec_offset;
  }
}



void wilson_dslash_vec (IFloat * chi_p_f,
                        IFloat * u_p_f,
                        IFloat * psi_p_f,
                        int cb,
                        int dag,
                        Wilson * wilson_p,
                        int vec_len, unsigned long vec_offset)
{
  //if(!UniqueID()){ printf("Running vectorised wilson dslash\n"); fflush(stdout); }


  char *cname = "";
  char *fname = "wilson_dslash_vec";
  static Timer timer_setup (fname,"setup()");
  static Timer timer_local (fname,"local()");
  static Timer timer_nl (fname,"nonlocal()");
  static Timer timer_qmp (fname,"qmp()");

  int nflavor = GJP.Gparity()+1;

  int lx, ly, lz, lt;
  int vol;

  int temp_size = SPINOR_SIZE;

  Float fbuf[temp_size];

  timer_setup.start();
  for (int i = 0; i < temp_size; i++)
    fbuf[i] = 0.;

  /*--------------------------------------------------------------------------*/
  /* Initializations                                                          */
  /*--------------------------------------------------------------------------*/
  int sdag;                     /* = +/-1 if dag = 0/1 */
  int cbn;
  Float *chi_p = (Float *) chi_p_f;
  Float *u_p = (Float *) u_p_f;
  Float *psi_p = (Float *) psi_p_f;


  lx = wilson_p->ptr[0];
  ly = wilson_p->ptr[1];
  lz = wilson_p->ptr[2];
  lt = wilson_p->ptr[3];
  vol = wilson_p->vol[0];

  int lattsz[4] = { lx, ly, lz, lt };

  static int local_comm[4];

  if (dag == 0)
    sdag = 1;
  else if (dag == 1)
    sdag = -1;
  else {
    ERR.General (" ", fname, "dag must be 0 or 1");
  }

  //cb is the fermion parity, cbn is the gauge field parity
  if (cb == 0)
    cbn = 1;
  else if (cb == 1)
    cbn = 0;
  else {
    ERR.General (" ", fname, "cb must be 0 or 1");
  }

  Float *ind_buf[8];
  unsigned long ind_nl[8];

  static unsigned long num_nl[8];       //number non-local sites
  static unsigned long *u_ind[8];
  static unsigned long *f_ind[8];
  static unsigned long *t_ind[8];
  static Float *Send_buf[8];
  static Float *Recv_buf[8];
  QMP_msgmem_t Send_mem[8];
  QMP_msgmem_t Recv_mem[8];
  QMP_msghandle_t multiple_p[16];
  QMP_msghandle_t multiple;
  int n_dir=0;


  //reset setup variables if G-parity status changes
  if(init_len != 0 && ( gparity_init_status == 1 && !GJP.Gparity() ||  gparity_init_status == 0 && GJP.Gparity() ) ) init_len = -1;

  if (vec_len != init_len) {
    //These are only initialized once unless the vec_len changes
    //For G-parity tests where we change the volume, I have added a reset for when GJP.Gparity changes above
    //coupled with the initialization status below
    gparity_init_status = GJP.Gparity() ? 1 : 0;

    VRB.Result (cname, fname, "init_len(%d)!=vec_len(%d)\n", init_len, vec_len);
    if (init_len != 0)
      for (int i = 0; i < 8; i++) {
        if (u_ind[i]) sfree (u_ind[i]);
        if (f_ind[i]) sfree (f_ind[i]);
        if (Send_buf[i]) sfree (Send_buf[i]);
        if (Recv_buf[i]) sfree (Recv_buf[i]);
      }

    num_nl[0] = num_nl[4] = vol / lx;
    num_nl[1] = num_nl[5] = vol / ly;
    num_nl[2] = num_nl[6] = vol / lz;
    num_nl[3] = num_nl[7] = vol / lt;

    if (!UniqueID ()) {
      printf
        ("Lattice is %d x %d x %d x %d, volume is %d, cb volume %d. Input 'vol' is %d.\n",
         GJP.XnodeSites (), GJP.YnodeSites (), GJP.ZnodeSites (),
         GJP.TnodeSites (), GJP.VolNodeSites (), GJP.VolNodeSites () / 2, vol);
    }

    for (int i = 0; i < 4; i++) {
      if (GJP.Nodes (i) > 1) {
        local_comm[i] = 0;
        if (!UniqueID ()) {
          printf ("Non-local in direction %d, sites on wall %d\n", i,
                  num_nl[i]);
          fflush (stdout);
        }
      }

      else
        local_comm[i] = 1;
    }
    for (int i = 0; i < 8; i++) {
      int mu = i % 4;
      int sign = 1 - 2 * (i / 4);       //1,1,1,1,-1,-1,-1,-1
      if (local_comm[mu]) {
        num_nl[i] = 0;
        u_ind[i] = NULL;
        f_ind[i] = NULL;
        Send_buf[i] = NULL;
        Recv_buf[i] = NULL;
      } else {
        u_ind[i] = (unsigned long *) smalloc (cname, fname, "u_ind[i]", num_nl[i] * sizeof (unsigned long));
        f_ind[i] = (unsigned long *) smalloc (cname, fname, "f_ind[i]", num_nl[i] * sizeof (unsigned long));
        t_ind[i] = (unsigned long *) smalloc (cname, fname, "f_ind[i]", num_nl[i] * sizeof (unsigned long));
        size_t buf_size = num_nl[i] * SPINOR_SIZE * sizeof (Float) * vec_len * nflavor;
        Send_buf[i] = (Float *) smalloc (cname, fname, "Send_buf[i]", buf_size);
        Recv_buf[i] = (Float *) smalloc (cname, fname, "Recv_buf[i]", buf_size);
      }
    }
    VRB.Flow (cname,fname,"initted\n");
    init_len = vec_len;
  }
  n_dir=0;
  for (int i = 0; i < 8; i++) {
      int mu = i % 4;
      int sign = 1 - 2 * (i / 4);       //1,1,1,1,-1,-1,-1,-1
      if (!local_comm[i%4]) {
        size_t buf_size = num_nl[i] * SPINOR_SIZE * sizeof (Float) * vec_len * nflavor;
        Send_mem[n_dir] = QMP_declare_msgmem (Send_buf[i], buf_size);
        multiple_p[n_dir*2] = QMP_declare_send_relative (Send_mem[n_dir], mu, -sign, 0);
        Recv_mem[n_dir] = QMP_declare_msgmem (Recv_buf[i], buf_size);
        multiple_p[n_dir*2+1] = QMP_declare_receive_relative (Recv_mem[n_dir], mu, sign, 0);
	n_dir ++;
      }
  }
  if(n_dir>0) multiple = QMP_declare_multiple(multiple_p,2*n_dir);

  //printf("Node %d, local_comm=%d %d %d %d\n",UniqueID(),local_comm[0],local_comm[1],local_comm[2],local_comm[3]); fflush(stdout);

  if (called % 100000 == 0) {
    VRB.Result (cname, fname, "local_comm=%d %d %d %d\n", local_comm[0],
                local_comm[1], local_comm[2], local_comm[3]);
    if (called > 0)
      called = 0;
  }

  int u_cboff = vol * nflavor;            //vol is the checkerboard volume, i.e. half the 4d volume
  //With G-parity layout is [(f0 odd)(f1 odd)(f0 even)(f1 even)]  each bracket is one cbvolume

  //
  //  non-local send
  //
  //
  GJP.SetNthreads ();

#pragma omp parallel for default(shared)
  for (int dir = 0; dir < 8; dir++) {
    int x, y, z, t;
    int r, c, s;
    int xyzt;
    int parity;
    int mu;
    Float *chi;
    Float *u;
    Float *psi;

    Float tmp[temp_size];
    Float tmp1[temp_size];
    Float tmp2[temp_size];
    Float tmp3[temp_size];
    Float tmp4[temp_size];
    Float tmp5[temp_size];
    Float tmp6[temp_size];
    Float tmp7[temp_size];
    Float tmp8[temp_size];

    Float *temps[4][2] =
      { {tmp1, tmp5}, {tmp2, tmp6}, {tmp3, tmp7}, {tmp4, tmp8} };
    
    for(int f = 0; f < nflavor; f++){
      ind_nl[dir] = 0;
      ind_buf[dir] = Send_buf[dir];

      for (x = 0; x < lx; x++) {
	for (y = 0; y < ly; y++) {
	  for (z = 0; z < lz; z++) {
	    for (t = 0; t < lt; t++) {
	      parity = x + y + z + t;
	      parity = parity % 2;
	      if (parity == cbn) {
		//for Dslash_xy\psi_y, y is opposite parity to that of sites at which \psi is taken from 
		//printf("Node %d, non-local send: %d %d %d %d\n",UniqueID(),x,y,z,t); fflush(stdout); 

		/* x,y,z,t addressing of cbn checkerboard */
		xyzt = (x / 2) + (lx / 2) * (y + ly * (z + lz * t));
		int pos[4] = { x, y, z, t };

		chi = chi_p + SPINOR_SIZE * (xyzt + f*vol);
		int psi_bufoff = num_nl[dir] * SPINOR_SIZE * vec_len;     //offset for G-parity f1 pointers in buffer

		if (dir < 4) {    //backwards send. Fermion site is on the left-most boundary. Send \psi{x+\mu} to x
		  mu = dir;
		  //printf("Node %d, dir %d, mu %d, 1, local_comm[mu] = %d\n",UniqueID(),dir,mu,local_comm[mu]); fflush(stdout); 
		  int posp[4] = { x, y, z, t };
		  posp[mu] = (posp[mu] + 1) % lattsz[mu];
		  int posp_xyzt =
		    (posp[0] / 2) + (lx / 2) * (posp[1] +
						ly * (posp[2] + lz * posp[3]));
		  //printf("Node %d, dir %d, mu %d, 2, local_comm[mu] = %d\n",UniqueID(),dir,mu,local_comm[mu]); fflush(stdout); 

		  if ((pos[mu] == lattsz[mu] - 1) && !local_comm[mu]) {   //posp[mu]==0
		    //place fermions in buffer to prepare for backwards send
		    int buf_off = 0;
		    
		    if (GJP.Gparity ()){
		      int psi_f0_bufoff = 0;
		      int psi_f1_bufoff = psi_bufoff;
		      if (GJP.Bc (mu) == BND_CND_GPARITY && GJP.NodeCoor (mu) == 0) {
			//implement G-parity twist at boundary. Note this data is sent in the *minus* direction
			psi_f0_bufoff = psi_bufoff;
			psi_f1_bufoff = 0;
		      }
		      buf_off = f == 0 ? psi_f0_bufoff : psi_f1_bufoff;
		    }
		    psi = psi_p + SPINOR_SIZE * (posp_xyzt + f*vol);
		    MOVE_VEC (ind_buf[dir] + buf_off, psi, vec_len, vec_offset);

		    *(u_ind[dir] + ind_nl[dir]) = xyzt + u_cboff * cbn;
		    *(f_ind[dir] + ind_nl[dir]) = posp_xyzt;
		    *(t_ind[dir] + ind_nl[dir]) = xyzt;
		    ind_buf[dir] += SPINOR_SIZE * vec_len;
		    ind_nl[dir]++;		
		  }

		} else {          //Forwards send, fermion is on right-most boundary. Send P_- U^\dagger_{x-\mu} \psi_{x-\mu} to x
		  /* 1+gamma_mu */
		  /*-----------*/
		  mu = dir - 4;

		  int posm[4] = { x, y, z, t };
		  posm[mu] =
		    posm[mu] - 1 +
		    ((lattsz[mu] - posm[mu]) / lattsz[mu]) * lattsz[mu];
		  int posm_xyzt =
		    (posm[0] / 2) + (lx / 2) * (posm[1] +
						ly * (posm[2] + lz * posm[3]));

		  u = u_p + GAUGE_SIZE * (posm_xyzt + u_cboff * cb + vol*f);
		  psi = psi_p + SPINOR_SIZE * (posm_xyzt + vol*f);


		  if ((pos[mu] == 0) && !local_comm[mu]) {        //posm on right-most boundary
		    Printf
		      ("getMinusData((IFloat *)fbuf, (IFloat *)tmp%d, SPINOR_SIZE, %d);\n",
		       dir + 1, mu);
		    *(u_ind[dir] + ind_nl[dir]) = posm_xyzt + u_cboff * cb;
		    *(f_ind[dir] + ind_nl[dir]) = posm_xyzt;
		    *(t_ind[dir] + ind_nl[dir]) = xyzt;		  

		    for (int vec_ind = 0; vec_ind < vec_len; vec_ind++) {
		      PLUSMU (mu, u, tmp, temps[mu][1], sdag, psi);
		      int buf_off = 0;

		      if (GJP.Gparity ()){
			//both psi and u are at the same site. G-parity twist should swap which component of chi gets each contribution
			int psi_f0_bufoff = 0;
			int psi_f1_bufoff = psi_bufoff;
			if (GJP.Bc (mu) == BND_CND_GPARITY
			    && GJP.NodeCoor (mu) == GJP.Nodes (mu) - 1) {
			  //implement G-parity twist at boundary. Note this data is sent in the *plus* direction
			  psi_f0_bufoff = psi_bufoff;
			  psi_f1_bufoff = 0;
			}
			buf_off = f == 0 ? psi_f0_bufoff : psi_f1_bufoff;
		      }
		      moveMem ((IFloat *) ind_buf[dir] + buf_off, (IFloat *) temps[mu][1],
			       SPINOR_SIZE * sizeof (Float) / sizeof (char));
		      psi = psi + vec_offset;

		      ind_buf[dir] += SPINOR_SIZE;
		    }
		    ind_nl[dir]++;
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }

  for (int i = 0; i < 8; i++) {
    if (ind_nl[i] != num_nl[i])
      VRB.Result (cname, fname, "ind_nl[%d](%d)!=num_nl[%d](%d)\n", i,
                  ind_nl[i], i, num_nl[i]);
#if 0
    if (!local_comm[i % 4]) {
      Printf ("QMP_start(Recv[%d])(%p)\n", i, Recv[i]);
      QMP_start (Recv[i]);
      Printf ("QMP_start(Send[%d])(%p)\n", i, Send[i]);
      QMP_start (Send[i]);
    }
#endif
  }
  if(n_dir>0) QMP_start (multiple);

  timer_setup.stop();
  timer_local.start();

  GJP.SetNthreads ();

  /*--------------------------------------------------------------------------*/
  /* Loop over sites                                                          */
  /*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/
/* Loop over sites                                                          */
/*--------------------------------------------------------------------------*/
  for (int i = 0; i < SPINOR_SIZE; i++)
    fbuf[i] = 0.;
  int index = 0;

#pragma omp parallel for default(shared)
  for (index = 0; index < vol * 2; index++) {
    //  Printf("wilson_dslash: %d %d %d %d\n",x,y,z,t);
    //  if ((called%10000==0) &&(!UniqueID())){
    //          printf("wilson_dslash: index=%d thread %d of %d\n",index,omp_get_thread_num(),omp_get_num_threads());
    //  }
    int r, c, s;
    int x, y, z, t;
    int xyzt;
    int parity;
    int mu;
    Float *chi;
    Float *u;
    Float *psi;

    Float tmp[temp_size];
    Float tmp1[temp_size];
    Float tmp2[temp_size];
    Float tmp3[temp_size];
    Float tmp4[temp_size];
    Float tmp5[temp_size];
    Float tmp6[temp_size];
    Float tmp7[temp_size];
    Float tmp8[temp_size];

    Float *temps[4][2] =
      { {tmp1, tmp5}, {tmp2, tmp6}, {tmp3, tmp7}, {tmp4, tmp8} };

    int temp = index;
    x = temp % lx;
    temp = temp / lx;
    y = temp % ly;
    temp = temp / ly;
    z = temp % lz;
    temp = temp / lz;
    t = temp % lt;
    temp = temp / lt;

    if (0)
      if ((called % 1000000 == 0) && (!UniqueID ())) {
        printf ("wilson_dslash: %d %d %d %d %d: thread %d of %d tmp=%p \n",
                index, x, y, z, t, omp_get_thread_num (),
                omp_get_num_threads (), tmp);
      }


    parity = x + y + z + t;
    parity = parity % 2;
    if (parity == cbn) {

      /* x,y,z,t addressing of cbn checkerboard */
      xyzt = (x / 2) + (lx / 2) * (y + ly * (z + lz * t));
      int pos[4] = { x, y, z, t };
      //        VRB.Result(fname,"local", "%d %d %d %d\n",x,y,z,t);

      for(int f=0;f<nflavor;f++){
	chi = chi_p + SPINOR_SIZE * (xyzt + f*vol); 

	for (int vec_ind = 0; vec_ind < vec_len; vec_ind++) {
	  for (s = 0; s < 4; s++)
	    for (c = 0; c < 3; c++)
	      for (r = 0; r < 2; r++)
		FERM (chi, r, c, s) = 0.;
	  chi += vec_offset;
	}
      }
	
      for (mu = 0; mu < 4; mu++) {
        //1-gamma_mu
        int posp[4] = { x, y, z, t };
        posp[mu] = (posp[mu] + 1) % lattsz[mu];
        int posp_xyzt =
          (posp[0] / 2) + (lx / 2) * (posp[1] + ly * (posp[2] + lz * posp[3]));

	for(int f=0;f<nflavor;f++){
	  chi = chi_p + SPINOR_SIZE * (xyzt + f*vol);
	  u = u_p + GAUGE_SIZE * (xyzt + u_cboff * cbn + f*vol);
	  psi = psi_p + SPINOR_SIZE * (posp_xyzt + f*vol);

	  if ((pos[mu] == lattsz[mu] - 1) && !local_comm[mu]) {
	    //not sure if this actually *does* anything!
	    psi = fbuf + f*SPINOR_SIZE;
	  } else {
	    //internal site or local comms for edge sites
	    
	    for (int vec_ind = 0; vec_ind < vec_len; vec_ind++) {
	      Float* psi_use = psi;

	      if (GJP.Gparity ()){ 
		Float *psi_f0_use = f==0 ? psi : psi - vol*SPINOR_SIZE;
		Float *psi_f1_use = psi_f0_use + vol*SPINOR_SIZE;

		if (GJP.Bc (mu) == BND_CND_GPARITY && pos[mu] == lattsz[mu] - 1) {
		  //psi crosses G-parity boundary, do flavour twist
		  Float *tmp = psi_f0_use;
		  psi_f0_use = psi_f1_use;
		  psi_f1_use = tmp;
		}
		psi_use = f==0 ? psi_f0_use : psi_f1_use;
	      }//Gparity
		
	      MINUSMU (mu, u, tmp, temps[mu][0], sdag, psi_use);

	      for (s = 0; s < 4; s++)
		for (c = 0; c < 3; c++)
		  for (r = 0; r < 2; r++)
		    if (mu == 0)
		      FERM (chi, r, c, s) = FERM (temps[mu][0], r, c, s);
		    else
		      FERM (chi, r, c, s) += FERM (temps[mu][0], r, c, s);			      

	      psi += vec_offset;
	      chi += vec_offset;
	    }                   //vec_ind loop
          }                     //local comms loop
        }                       //flavor  
      }                         //mu loop

      for (mu = 0; mu < 4; mu++) {
        /* 1+gamma_mu */
        /*-----------*/
        int posm[4] = { x, y, z, t };
        posm[mu] =
          posm[mu] - 1 + ((lattsz[mu] - posm[mu]) / lattsz[mu]) * lattsz[mu];
        int posm_xyzt =
          (posm[0] / 2) + (lx / 2) * (posm[1] + ly * (posm[2] + lz * posm[3]));

	for(int f=0;f<nflavor;f++){
	  chi = chi_p + SPINOR_SIZE * (xyzt + f*vol);
	  u = u_p + GAUGE_SIZE * (posm_xyzt + u_cboff * cb + f*vol);      //note opposite parity
	  psi = psi_p + SPINOR_SIZE * (posm_xyzt + f*vol);	  
	  
	  if (pos[mu] == 0 && (!local_comm[mu])) {
	    moveMem ((IFloat *) temps[mu][1] + f*SPINOR_SIZE, (IFloat *) (fbuf + f*SPINOR_SIZE),
		     SPINOR_SIZE * sizeof (Float) / sizeof (char));
	  } else {
	    for (int vec_ind = 0; vec_ind < vec_len; vec_ind++) {
	      Float *chi_use = chi;

	      PLUSMU (mu, u, tmp, temps[mu][1], sdag, psi);

	      if (GJP.Gparity ()) {
		//here the gauge fields and psi are drawn from the same site but placed into chi at the next site over
		//hence for the G-parity twist we need to swap over the parts contributing to psi_f0 and psi_f1
		
		Float *chi_f0_use = f==0 ? chi : chi - vol*SPINOR_SIZE;
		Float *chi_f1_use = chi_f0_use + vol*SPINOR_SIZE;

		if (GJP.Bc (mu) == BND_CND_GPARITY && pos[mu] == 0) {
		  //contribution to chi crosses G-parity boundary, do flavour twist
		  Float* tmp = chi_f0_use;
		  chi_f0_use = chi_f1_use;
		  chi_f1_use = tmp;
		}
		chi_use = f == 0 ? chi_f0_use : chi_f1_use;
	      } //G-parity

	      for (s = 0; s < 4; s++)
		for (c = 0; c < 3; c++)
		  for (r = 0; r < 2; r++)
		    FERM (chi_use, r, c, s) += FERM (temps[mu][1], r, c, s);			      

	      psi += vec_offset;
	      chi += vec_offset;		
	    }//vec_ind loop
	  }//local comms	 
	}//flavor
      }//mu
    }                     //parity==cbn
  }//index

  timer_local.stop();
//  dtime += dclock (true);
//  local += dtime;

  timer_qmp.start();
//  dtime = -dclock ();

  for (int i = 0; i < 8; i++) {
#if 0
    if (!local_comm[i % 4]) {
      QMP_status_t send_status = QMP_wait (Send[i]);
      if (send_status != QMP_SUCCESS)
        QMP_error ("Send failed in wilson_dslash: %s\n",
                   QMP_error_string (send_status));
      QMP_status_t rcv_status = QMP_wait (Recv[i]);
      if (rcv_status != QMP_SUCCESS)
        QMP_error ("Receive failed in wilson_dslash: %s\n",
                   QMP_error_string (rcv_status));
    }
#endif
    ind_nl[i] = 0;
    ind_buf[i] = Recv_buf[i];
  }
  if(n_dir>0){
  QMP_status_t send_status = QMP_wait (multiple);
      if (send_status != QMP_SUCCESS)
        QMP_error ("QMP_multiple failed in wilson_dslash: %s\n", QMP_error_string (send_status));
  }
  timer_qmp.stop();
  timer_nl.start();

  //
  // non-local
  //

#define USE_TEST2
#ifdef USE_TEST2
  index = 0;
  int nl_offset[8];
  for (int i = 0; i < 8; i++) {
    index += num_nl[i];
    nl_offset[i] = index;
  }

#undef NL_OMP
  {
    int i_mu;

    for (i_mu = 0; i_mu < 4; i_mu++) {
#pragma omp parallel for default(shared)
      for (int i = 0; i < num_nl[i_mu]; i++) {
	for(int f=0;f<nflavor;f++){
	  Float tmp[temp_size];
	  Float tmp1[temp_size];
	  
	  Float *chi;
	  Float *u;
	  Float *psi;

	  chi = chi_p + SPINOR_SIZE * (*(t_ind[i_mu] + i) + f*vol);
	  u = u_p + GAUGE_SIZE * (*(u_ind[i_mu] + i) + f*vol);
	  psi = ind_buf[i_mu] + (i + f*num_nl[i_mu]) * SPINOR_SIZE * vec_len;
	  
	  int r, c, s;
	  for (int vec_ind = 0; vec_ind < vec_len; vec_ind++) {
	    MINUSMU (i_mu, u, tmp, tmp1, sdag, psi);
	    for (s = 0; s < 4; s++)
	      for (c = 0; c < 3; c++)
		for (r = 0; r < 2; r++)
		  FERM (chi, r, c, s) += FERM (tmp1, r, c, s);
	    psi += SPINOR_SIZE;
	    chi += vec_offset;
	  }
	  
	  chi = chi_p + SPINOR_SIZE * (*(t_ind[i_mu + 4] + i) + f*vol);
	  u = u_p + GAUGE_SIZE * (*(u_ind[i_mu + 4] + i) + f*vol);
	  psi = ind_buf[i_mu + 4] + (i + f*num_nl[i_mu + 4]) * SPINOR_SIZE * vec_len;

	  for (int vec_ind = 0; vec_ind < vec_len; vec_ind++) {
	    for (s = 0; s < 4; s++)
	      for (c = 0; c < 3; c++)
		for (r = 0; r < 2; r++)
		  FERM (chi, r, c, s) += FERM (psi, r, c, s);
	    psi += SPINOR_SIZE;
	    chi += vec_offset;
	  }
	}
      }
    }

  }

#endif

  timer_nl.stop();

  called++;
#if 0
  if (called % 100 == 0) {
    print_flops ("wilson_dslash_vec()", "local*1000", 0, local);
    print_flops ("wilson_dslash_vec()", "nonlocal*1000", 0, nonlocal);
    print_flops ("wilson_dslash_vec()", "qmp*1000", 0, qmp);
    print_flops ("wilson_dslash_vec()", "setup*1000", 0, setup);
    local = nonlocal = qmp = setup = 0.;
  }
#endif
  DiracOp::CGflops += 1320 * vol * vec_len * nflavor;

  for (int i = 0; i < n_dir; i++) {
    QMP_free_msgmem (Send_mem[i]);
    QMP_free_msgmem (Recv_mem[i]);
  }
  if(n_dir>0) QMP_free_msghandle (multiple);
}

CPS_END_NAMESPACE
