#ifndef INCLUDED_EIG_IO_H
#define INCLUDED_EIG_IO_H

#define _FILE_OFFSET_BITS 64
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <complex>
#include <vector>
#include <memory.h>
#include <iostream>

#include <sys/stat.h>
#include <comms/sysfunc_cps.h>
#include <util/sumarray.h>
#include <util/time_cps.h>
//#include <util/eigen_container.h>
#include <util/gjp.h>
#include <util/verbose.h>
#include <unistd.h>

//inline double dclock() {
//      struct timeval tv; 
//      gettimeofday(&tv,NULL);
//      return (1.0*tv.tv_usec + 1.0e6*tv.tv_sec) / 1.0e6;
//}
//
#if 0
extern MPI_Comm QMP_COMM_WORLD;
#else
#define QMP_COMM_WORLD MPI_COMM_WORLD
#endif



uint32_t crc32_fast (const void *data, size_t length, uint32_t previousCrc32);

namespace cps
{
#if 1
  struct _evc_meta_
  {
    int s[5];
    int b[5];
    int nkeep;
    int nkeep_single;

    // derived
    int nb[5];
    int blocks;

    int neig;

    int index;

    uint32_t crc32;

    int FP16_COEF_EXP_SHARE_FLOATS;
//    bool big_endian;
  };

 class EigenCache ; //forward declaration


  class EvecReader
  {
  private:
    static const char *cname;

  public:
      bool machine_is_little_endian;
    int bigendian;		// read/write data back in big endian
    _evc_meta_ args;
      EvecReader ():bigendian (0)	// fixed to little endian
    {
      machine_is_little_endian = machine_endian ();
    };
     ~EvecReader ()
    {
    };
    typedef float OPT;
    int nthreads;

    static const char *header;

    int vol4d, vol5d;
    int f_size, f_size_block, f_size_coef_block, nkeep_fp16;

    char *raw_in;


    std::vector < std::vector < OPT > >block_data;
    std::vector < std::vector < OPT > >block_data_ortho;
    std::vector < std::vector < OPT > >block_coef;

//float* ord_in; // in non-bfm ordering:  co fastest, then x,y,z,t,s
    int machine_endian ()
    {
      char endian_c[4] = { 1, 0, 0, 0 };
      uint32_t *endian_i = (uint32_t *) endian_c;
      if (*endian_i == 0x1)
//        printf("little endian\n");
	return 1;
      else
//        printf("big endian\n");
	return 0;

    }


    inline int sumArray (long *recv, const long *send, const long n_elem)
    {
#ifdef USE_QMP
      return MPI_Allreduce ((long *) send, recv, n_elem, MPI_LONG, MPI_SUM,
			    QMP_COMM_WORLD);
#else
      memmove (recv, send, n_elem * sizeof (long));
      return 0;
#endif
    }

    inline int sumArray (uint32_t * recv, const uint32_t * send,
			 const long n_elem)
    {
#ifdef USE_QMP
      return MPI_Allreduce ((uint32_t *) send, recv, n_elem, MPI_UNSIGNED,
			    MPI_SUM, QMP_COMM_WORLD);
#else
      memmove (recv, send, n_elem * sizeof (uint32_t));
      return 0;
#endif
    }

    inline int sumArray (int *recv, const int *send, const long n_elem)
    {
#ifdef USE_QMP
      return MPI_Allreduce ((int *) send, recv, n_elem, MPI_INT, MPI_SUM,
			    QMP_COMM_WORLD);
#else
      memmove (recv, send, n_elem * sizeof (int));
      return 0;
#endif
    }

    inline int sumArray (double *recv, const double *send, const long n_elem)
    {
#ifdef USE_QMP
      return MPI_Allreduce ((double *) send, recv, n_elem, MPI_DOUBLE, MPI_SUM,
			    QMP_COMM_WORLD);
#else
      memmove (recv, send, n_elem * sizeof (double));
      return 0;
#endif
    }

    template < class M > int sumArray (M * vs, const long n_elem)
    {
      // M can be double or long
      int status = 0;
#ifdef USE_QMP
      M tmp[n_elem];
      status = sumArray (tmp, vs, n_elem);
      memcpy (vs, tmp, n_elem * sizeof (M));
#endif
      return status;
    }

    void fix_short_endian (unsigned short *dest, int nshorts)
    {
      if ((bigendian && machine_is_little_endian) ||	// written for readability
	  (!bigendian && !machine_is_little_endian))
	for (int i = 0; i < nshorts; i++) {
	  char *c1 = (char *) &dest[i];
	  char tmp = c1[0];
	  c1[0] = c1[1];
	  c1[1] = tmp;
	}
    }

    void fix_float_endian (float *dest, int nfloats)
    {
      int n_endian_test = 1;
      //bool machine_is_little_endian = *(char *)&n_endian_test == 1;

      if ((bigendian && machine_is_little_endian) ||	// written for readability
	  (!bigendian && !machine_is_little_endian)) {
	int i;
	for (i = 0; i < nfloats; i++) {
	  float before = dest[i];
	  char *c = (char *) &dest[i];
	  char tmp;
	  int j;
	  for (j = 0; j < 2; j++) {
	    tmp = c[j];
	    c[j] = c[3 - j];
	    c[3 - j] = tmp;
	  }
	  float after = dest[i];
	  printf ("fix_float_endian: %g ->%g\n", before, after);
	}
      }

    }


    int get_bfm_index (int *pos, int co)
    {

      int ls = args.s[4];
      int vol_4d_oo = vol4d / 2;
      int vol_5d = vol_4d_oo * ls;

      int NtHalf = args.s[3] / 2;
      int simd_coor = pos[3] / NtHalf;
      int regu_coor =
	(pos[0] +
	 args.s[0] * (pos[1] +
		      args.s[1] * (pos[2] +
				   args.s[2] * (pos[3] % NtHalf)))) / 2;
//      int regu_vol = vol_4d_oo / 2;

      return +regu_coor * ls * 48 + pos[4] * 48 + co * 4 + simd_coor * 2;
    }
    int get_cps_index (int *pos, int co)
    {

      int ls = args.s[4];
      int vol_4d_oo = vol4d / 2;
      int vol_5d = vol_4d_oo * ls;

//      int SimdT = 1;
//      int NtHalf = args.s[3];
//      int simd_coor = pos[3] / NtHalf;
//      assert (simd_coor == 0);
//      int regu_vol = vol_4d_oo / SimdT;
      int regu_coor = (pos[0] + args.s[0] *
		       (pos[1] + args.s[1] *
			(pos[2] + args.s[2] *
			 (pos[3] + args.s[3] * pos[4])))) / 2;

      return ((regu_coor) * 12 + co) * 2;
//      return regu_coor * ls * 48 + pos[4] * 48 + co * 4 + simd_coor * 2;
    }

    void index_to_pos (int i, int *pos, int *latt)
    {
      int d;
      for (d = 0; d < 5; d++) {
	pos[d] = i % latt[d];
	i /= latt[d];
      }
    }

    int pos_to_index (int *pos, int *latt)
    {
      return pos[0] + latt[0] * (pos[1] +
				 latt[1] * (pos[2] +
					    latt[2] * (pos[3] +
						       latt[3] * pos[4])));
    }


    void pos_to_blocked_pos (int *pos, int *pos_in_block, int *block_coor)
    {
      int d;
      for (d = 0; d < 5; d++) {
	block_coor[d] = pos[d] / args.b[d];
	pos_in_block[d] = pos[d] - block_coor[d] * args.b[d];
      }
    }

    template < class T >
      void caxpy_single (T * res, std::complex < T > ca, T * x, T * y,
			 int f_size)
    {
      std::complex < T > *cx = (std::complex < T > *)x;
      std::complex < T > *cy = (std::complex < T > *)y;
      std::complex < T > *cres = (std::complex < T > *)res;
      int c_size = f_size / 2;

      for (int i = 0; i < c_size; i++)
	cres[i] = ca * cx[i] + cy[i];
    }

    template < class T >
      void caxpy_threaded (T * res, std::complex < T > ca, T * x, T * y,
			   int f_size)
    {
      std::complex < T > *cx = (std::complex < T > *)x;
      std::complex < T > *cy = (std::complex < T > *)y;
      std::complex < T > *cres = (std::complex < T > *)res;
      int c_size = f_size / 2;

#pragma omp for
      for (int i = 0; i < c_size; i++)
	cres[i] = ca * cx[i] + cy[i];
    }

    template < class T > void scale_single (T * res, T s, int f_size)
    {
      for (int i = 0; i < f_size; i++)
	res[i] *= s;
    }

    template < class T >
      void caxpy (T * res, std::complex < T > ca, T * x, T * y, int f_size)
    {
      std::complex < T > *cx = (std::complex < T > *)x;
      std::complex < T > *cy = (std::complex < T > *)y;
      std::complex < T > *cres = (std::complex < T > *)res;
      int c_size = f_size / 2;

#pragma omp parallel for
      for (int i = 0; i < c_size; i++)
	cres[i] = ca * cx[i] + cy[i];
    }

    template < class T > std::complex < T > sp_single (T * a, T * b, int f_size)
    {
      std::complex < T > *ca = (std::complex < T > *)a;
      std::complex < T > *cb = (std::complex < T > *)b;
      int c_size = f_size / 2;

      int i;
      std::complex < T > ret = 0.0;
      for (i = 0; i < c_size; i++)
	ret += conj (ca[i]) * cb[i];

      return ret;
    }

    template < class T > std::complex < T > sp (T * a, T * b, int f_size) {
      std::complex < T > *ca = (std::complex < T > *)a;
      std::complex < T > *cb = (std::complex < T > *)b;
      int c_size = f_size / 2;

      std::complex < T > res = 0.0;
#pragma omp parallel shared(res)
      {
	std::complex < T > resl = 0.0;
#pragma omp for
	for (int i = 0; i < c_size; i++)
	  resl += conj (ca[i]) * cb[i];

#pragma omp critical
	{
	  res += resl;
	}
      }
      return res;
    }

    template < class T > T norm_of_evec (std::vector < std::vector < T > >&v,
					 int j) {
      T gg = 0.0;
#pragma omp parallel shared(gg)
      {
	T ggl = 0.0;
#pragma omp for
	for (int nb = 0; nb < args.blocks; nb++) {
	  T *res = &v[nb][(int64_t) f_size_block * j];
	  ggl += sp_single (res, res, f_size_block).real ();
	}

#pragma omp critical
	{
	  gg += ggl;
	}
      }
      return gg;
    }

    void write_bytes (void *buf, int64_t s, FILE * f, uint32_t & crc)
    {
      static double data_counter = 0.0;

      // checksum
      crc = crc32_fast (buf, s, crc);

      double t0 = dclock ();
      if (fwrite (buf, s, 1, f) != 1) {
	fprintf (stderr, "Write failed!\n");
	exit (2);
      }
      double t1 = dclock ();

      data_counter += (double) s;
      if (data_counter > 1024. * 1024. * 256) {
	printf ("Writing at %g GB/s\n",
		(double) s / 1024. / 1024. / 1024. / (t1 - t0));
	data_counter = 0.0;
      }
    }

    void write_floats (FILE * f, uint32_t & crc, OPT * in, int64_t n)
    {
      float *buf = (float *) malloc (sizeof (float) * n);
      if (!buf) {
	fprintf (stderr, "Out of mem\n");
	exit (1);
      }
      // convert to float if needed
#pragma omp parallel for
      for (int64_t i = 0; i < n; i++)
	buf[i] = in[i];

      fix_float_endian (buf, n);

      write_bytes (buf, n * sizeof (float), f, crc);

      free (buf);
    }

    void read_floats (char *&ptr, OPT * out, int64_t n)
    {
      float *in = (float *) ptr;
      ptr += 4 * n;

      for (int64_t i = 0; i < n; i++)
	out[i] = in[i];
      fix_float_endian (out, n);
//      std::cout << "out[0]= " << out[0] << std::endl;
      if (std::isnan (out[0]))
	std::cout << "read_floats out[0]= " << out[0] << std::endl;
    }

    int fp_map (float in, float min, float max, int N)
    {
      // Idea:
      //
      // min=-6
      // max=6
      //
      // N=1
      // [-6,0] -> 0, [0,6] -> 1;  reconstruct 0 -> -3, 1-> 3
      //
      // N=2
      // [-6,-2] -> 0, [-2,2] -> 1, [2,6] -> 2;  reconstruct 0 -> -4, 1->0, 2->4
      int ret = (int) ((float) (N + 1) * ((in - min) / (max - min)));
      if (ret == N + 1) {
	ret = N;
      }
      return ret;
    }

    float fp_unmap (unsigned short val, float min, float max, int N)
    {
      unsigned short tmp = val;
      fix_short_endian (&val, 1);
      if ((float) ((int) val + 0.5) != (float) (val + 0.5))
	std::cout << tmp << "after fix \t" << val << std::endl;
      return min + (float) ((int) val + 0.5) * (max - min) / (float) (N + 1);
    }

#define SHRT_UMAX 65535
#define BASE 1.4142135623730950488

    float unmap_fp16_exp (unsigned short e)
    {
      float de = (float) ((int) e - SHRT_UMAX / 2);
      return pow (BASE, de);
    }

    void read_floats_fp16 (char *&ptr, OPT * out, int64_t n, int nsc)
    {


      int64_t nsites = n / nsc;
      if (n % nsc) {
	fprintf (stderr, "Invalid size in write_floats_fp16\n");
	exit (4);
      }

      unsigned short *in = (unsigned short *) ptr;
      ptr += 2 * (n + nsites);

#define assert(exp)  { if ( !(exp) ) { fprintf(stderr,"Assert " #exp " failed\n"); exit(84); } }

      // do for each site
      for (int64_t site = 0; site < nsites; site++) {

	OPT *ev = &out[site * nsc];

	unsigned short *bptr = &in[site * (nsc + 1)];

	unsigned short exp = *bptr++;
	fix_short_endian (&exp, 1);
	OPT max = unmap_fp16_exp (exp);
	OPT min = -max;

	for (int i = 0; i < nsc; i++) {
	  ev[i] = fp_unmap (*bptr++, min, max, SHRT_UMAX);
	}
      if(std::isnan(ev[0]))
	std::cout << "read_floats_fp16 ev[0]= " << ev[0] << std::endl;

      }

    }

//added by bzy. read post-processed metadata
//    int read_metadata (const char *root, _evc_meta_ & args)
    int read_metadata (const char *root)
    {
      char buf[1024];
      sprintf (buf, "%s/metadata.txt", root);
      uint32_t nprocessors = 1;
      FILE *f = NULL;
      uint32_t status = 0;
      if (UniqueID () == 0) {
	printf ("node 0, before fopen \n");
	f = fopen (buf, "r");
	status = f ? 1 : 0;
      }
      sumArray (&status, 1);
//      _grid->GlobalSum(status);
      if (!status) {
	printf ("failed to open %s \n", buf);
	return false;
      }
      for (int i = 0; i < 5; i++) {
	args.s[i] = 0;
	args.b[i] = 0;
	args.nb[i] = 0;
      }
      args.neig = 0;
      args.nkeep = 0;
      args.nkeep_single = 0;
      args.blocks = 0;
      args.FP16_COEF_EXP_SHARE_FLOATS = 0;

#define _IRL_READ_INT(buf,p) if (f) { assert(fscanf(f,buf,p)==1); } else { *(p) = 0; }
      if (UniqueID () == 0) {
	printf ("node 0, before reading metadata\n");
	for (int i = 0; i < 5; i++) {
	  sprintf (buf, "s[%d] = %%d\n", i);
	  _IRL_READ_INT (buf, &args.s[i]);
	}
	for (int i = 0; i < 5; i++) {
	  sprintf (buf, "b[%d] = %%d\n", i);
	  _IRL_READ_INT (buf, &args.b[i]);
	}
	for (int i = 0; i < 5; i++) {
	  sprintf (buf, "nb[%d] = %%d\n", i);
	  _IRL_READ_INT (buf, &args.nb[i]);
	}
	_IRL_READ_INT ("neig = %d\n", &args.neig);
	_IRL_READ_INT ("nkeep = %d\n", &args.nkeep);
	_IRL_READ_INT ("nkeep_single = %d\n", &args.nkeep_single);
	_IRL_READ_INT ("blocks = %d\n", &args.blocks);
//      _IRL_READ_INT ("big_endian = %d\n", &args.big_endian);
	_IRL_READ_INT ("FP16_COEF_EXP_SHARE_FLOATS = %d\n",
		       &args.FP16_COEF_EXP_SHARE_FLOATS);
	printf ("node 0, after reading metadata \n");
      }
      //     bigendian = args.big_endian; //currently fixed to be little endian
      sumArray (args.s, 5);
      sumArray (args.b, 5);
      sumArray (args.nb, 5);
      sumArray (&args.neig, 1);
      sumArray (&args.nkeep, 1);
      sumArray (&args.nkeep_single, 1);
      sumArray (&args.blocks, 1);
      sumArray (&args.FP16_COEF_EXP_SHARE_FLOATS, 1);

//we do not divide the fifth dimension
      int nn[5];
      nn[4] = 1;
      for (int i = 0; i < 4; i++) {
	//     assert(GJP.Sites(i) % args.s[i] == 0);
	//    nn[i] = GJP.Sites(i)/ args.s[i];
	nn[i] = GJP.Nodes (i);
	nprocessors *= nn[i];
      }
      double barrier = 0;
      sumArray (&barrier, 1);
      //std::cout << GridLogMessage << "Reading data that was generated on node-layout " << nn << std::endl;

      std::vector < uint32_t > crc32_arr (nprocessors, 0);
      if (UniqueID () == 0) {
	printf ("node 0, before reading crc32\n");
	for (int i = 0; i < nprocessors; i++) {
	  sprintf (buf, "crc32[%d] = %%X\n", i);
	  _IRL_READ_INT (buf, &crc32_arr[i]);
	  printf ("crc32[%d] = %X\n", i, crc32_arr[i]);
	}
	printf ("node 0, after reading crc32\n");
      }
      //printf("node %d, before sumarray crc32\n", UniqueID());
      sumArray (&crc32_arr[0], nprocessors);
      args.crc32 = crc32_arr[UniqueID ()];

#undef _IRL_READ_INT

      if (f)
	fclose (f);
      return 1;
    }

//    int read_meta (const char *root, _evc_meta_ & args)
    int read_meta (const char *root)
    {

      char buf[1024];
      char val[1024];
      char line[1024];


      // read meta data
      sprintf (buf, "%s/metadata.txt", root);
      FILE *f = fopen (buf, "r");
      if (!f) {
	fprintf (stderr, "Could not open %s\n", buf);
	return 3;
      }

      while (!feof (f)) {
	int i;

	if (!fgets (line, sizeof (line), f))
	  break;

	if (sscanf (line, "%s = %s\n", buf, val) == 2) {

	  char *r = strchr (buf, '[');
	  if (r) {
	    *r = '\0';
	    i = atoi (r + 1);

#define PARSE_ARRAY(n) \
	    if (!strcmp (buf, #n)) {			\
					args.n[i] = atoi(val);		\
				}
	    PARSE_ARRAY (s)
	      else
	    PARSE_ARRAY (b)
	      else
	    PARSE_ARRAY (nb)
	      else
	    {
	      fprintf (stderr, "Unknown array '%s' in %s.meta\n", buf, root);
	      return 4;
	    }

	  } else {

#define PARSE_INT(n)				\
				if (!strcmp(buf,#n)) {			\
					args.n = atoi(val);			\
				}
#define PARSE_HEX(n)				\
				if (!strcmp(buf,#n)) {			\
					sscanf(val,"%X",&args.n);		\
				}

	    PARSE_INT (neig)
	      else
	    PARSE_INT (nkeep)
	      else
	    PARSE_INT (nkeep_single)
	      else
	    PARSE_INT (blocks)
	      else
	    PARSE_INT (FP16_COEF_EXP_SHARE_FLOATS)
	      else
	    PARSE_INT (index)
	      else
	    PARSE_HEX (crc32)
	      else
	    {
	      fprintf (stderr, "Unknown parameter '%s' in %s.meta\n",
		       buf, root);
	      return 4;
	    }


	  }

	} else {
	  printf ("Improper format: %s\n", line);	// double nl is OK
	}

      }

      fclose (f);
      return 1;
    }

    int decompress (const char *root_, std::vector < OPT * >&dest_all);



  };

#endif
  void alcf_evecs_save(char* dest,EigenCache* ec,int nkeep);
  void movefloattoFloat (Float * out, float *in, int f_size);

}
#endif
