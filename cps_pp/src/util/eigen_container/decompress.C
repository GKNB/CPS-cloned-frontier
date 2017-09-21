#define _FILE_OFFSET_BITS 64
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
//#include "sumarray.h"
#include <util/time_cps.h>
#include <util/eigen_container.h>
#include <util/gjp.h>
#include <util/verbose.h>
#include <util/eig_io.h>
#include <unistd.h>

using namespace cps;
using namespace std;



namespace cps
{


  void movefloattoFloat (Float * out, float *in, int f_size)
  {

    float flt;
    for (int i = 0; i < f_size; i++)
    {
      flt = in[i];
      out[i] = (Float) flt;
    }
  };


  int EvecReader::globalToLocalCanonicalBlock (int slot,
					       const std::vector <
					       int >&src_nodes, int nb)
  {
    // processor coordinate
    int _nd = (int) src_nodes.size ();
    std::vector < int >_src_nodes = src_nodes;
    std::vector < int >pco (_nd);
    Lexicographic::CoorFromIndex (pco, slot, _src_nodes);
    std::vector < int >cpco (pco);

    // get local block

    std::vector < int >_nb;
    _nb.resize (5);
    _nb[4] = GJP.SnodeSites () / args.b[4];	// b is the block size
    _nb[0] = GJP.XnodeSites () / args.b[0];
    _nb[1] = GJP.YnodeSites () / args.b[1];
    _nb[2] = GJP.ZnodeSites () / args.b[2];
    _nb[3] = GJP.TnodeSites () / args.b[3];
    std::vector < int >_nbc (_nb);
    assert (_nd == 5);
    std::vector < int >c_src_local_blocks (_nd);
    for (int i = 0; i < _nd; i++) {
      assert (GJP.Sites (i) % (src_nodes[i] * args.b[i]) == 0);
      c_src_local_blocks[i] = GJP.Sites (i) / src_nodes[i] / args.b[i];
    }
    std::vector < int >cbcoor (_nd);	// coordinate of block in slot in canonical form
    Lexicographic::CoorFromIndex (cbcoor, nb, c_src_local_blocks);

    // cpco, cbcoor
    std::vector < int >clbcoor (_nd);
    for (int i = 0; i < _nd; i++) {
      int cgcoor = cpco[i] * c_src_local_blocks[i] + cbcoor[i];	// global block coordinate
      int pcoor = cgcoor / _nbc[i];	// processor coordinate in my Grid
      int tpcoor = GJP.NodeCoor (i);
      if (pcoor != tpcoor)
	return -1;
      clbcoor[i] = cgcoor - tpcoor * _nbc[i];	// canonical local block coordinate for canonical dimension i
    }

    int lnb;
    Lexicographic::IndexFromCoor (clbcoor, lnb, _nbc);
    //std::cout << "Mapped slot = " << slot << " nb = " << nb << " to " << lnb << std::endl;
    return lnb;
  }


  void EvecReader::get_read_geometry (const std::vector < int >&cnodes,
				      std::map < int,
				      std::vector < int > >&slots,
				      std::vector < int >&slot_lvol,
				      std::vector < int >&lvol,
				      int64_t & slot_lsites, int &ntotal)
  {

    int _nd = (int) cnodes.size ();
    std::vector < int >nodes = cnodes;

    int glb_i[5];
    glb_i[0] = GJP.XnodeSites () * GJP.Xnodes ();
    glb_i[1] = GJP.YnodeSites () * GJP.Ynodes ();
    glb_i[2] = GJP.ZnodeSites () * GJP.Znodes ();
    glb_i[3] = GJP.TnodeSites () * GJP.Tnodes ();
    glb_i[4] = GJP.SnodeSites () * GJP.Snodes ();

    slots.clear ();
    slot_lvol.clear ();
    lvol.clear ();

    int i;
    ntotal = 1;
    int64_t lsites = 1;
    slot_lsites = 1;
    for (i = 0; i < _nd; i++) {
      assert (glb_i[i] % nodes[i] == 0);
      slot_lvol.push_back (glb_i[i] / nodes[i]);
      lvol.push_back (glb_i[i] / GJP.Nodes (i));
      lsites *= lvol.back ();
      slot_lsites *= slot_lvol.back ();
      ntotal *= nodes[i];
    }

    std::vector < int >lcoor, gcoor, scoor;
    lcoor.resize (_nd);
    gcoor.resize (_nd);
    scoor.resize (_nd);

    // create mapping of indices to slots
    for (int lidx = 0; lidx < lsites; lidx++) {
      Lexicographic::CoorFromIndex (lcoor, lidx, lvol);
      for (int i = 0; i < _nd; i++) {
	gcoor[i] = lcoor[i] + GJP.NodeCoor (i) * lvol[i];
	scoor[i] = gcoor[i] / slot_lvol[i];
      }
      int slot;
      Lexicographic::IndexFromCoor (scoor, slot, nodes);
      std::map < int, std::vector < int >>::iterator sl = slots.find (slot);
      if (sl == slots.end ())
	slots[slot] = std::vector < int >();
      slots[slot].push_back (lidx);
    }
  }



  int EvecReader::decompress (const char *root_, std::vector < OPT * >&dest_all)
  {

    const char *fname = "decompress()";
    printf ("dest_all.size() %d  args.neig %d\n", dest_all.size (), args.neig);
    assert (dest_all.size () <= args.neig);
    int dest_total = dest_all.size ();

    vector < vector < OPT > >block_data;
    vector < vector < OPT > >block_data_ortho;
    vector < vector < OPT > >block_coef;

#pragma omp parallel
    {
#pragma omp single
      {
	nthreads = omp_get_num_threads ();
      }
    }

    if (UniqueID () == 0)
      printf ("%s\n%d threads\n\n", header, nthreads);
    double barrier = 0;
    sumArray (&barrier, 1);

    const char *root = root_;
    {
      int i;
      if (!read_metadata (root))
	return -1;

      double barrier = 0;
      sumArray (&barrier, 1);
      if (UniqueID () == 0)
	printf ("after read metadata\n");

      vol4d = args.s[0] * args.s[1] * args.s[2] * args.s[3];
      vol5d = vol4d * args.s[4];
      f_size = vol5d / 2 * 24;

      if (UniqueID () == 1) {
	printf ("Parameters:\n");
	for (i = 0; i < 5; i++)
	  printf ("s[%d] = %d\n", i, args.s[i]);
	for (i = 0; i < 5; i++)
	  printf ("b[%d] = %d\n", i, args.b[i]);
	printf ("nkeep = %d\n", args.nkeep);
	printf ("nkeep_single = %d\n", args.nkeep_single);

	printf ("f_size = %d\n", f_size);
	printf ("FP16_COEF_EXP_SHARE_FLOATS = %d\n",
		args.FP16_COEF_EXP_SHARE_FLOATS);
//      printf ("crc32 = %X\n", args.crc32);
	printf ("\n");
      }
      // sanity check
      args.blocks = 1;
      for (i = 0; i < 5; i++) {
	if (args.s[i] % args.b[i]) {
	  fprintf (stderr, "Invalid blocking in dimension %d\n", i);
	  return 72;
	}

	args.nb[i] = args.s[i] / args.b[i];
	args.blocks *= args.nb[i];	//local 5D volume
      }

      f_size_block = f_size / args.blocks;

      if (UniqueID () == 0) {
	printf ("number of blocks = %d\n", args.blocks);
	printf ("f_size_block = %d\n", f_size_block);

	printf ("Internally using sizeof(OPT) = %d\n", sizeof (OPT));

	printf ("\n");
      }

      nkeep_fp16 = args.nkeep - args.nkeep_single;
      if (nkeep_fp16 < 0)
	nkeep_fp16 = 0;

      f_size_coef_block = args.neig * 2 * args.nkeep;
    }

//    int n_cycle = 32;
    int slot = GJP.NodeCoor (0);
    int ntotal = 1;
    for (int i = 0; i < 3; i++) {
      slot *= GJP.Nodes (i);
      slot += GJP.NodeCoor (i + 1);
      ntotal *= GJP.Nodes (i);
    }
    ntotal *= GJP.Nodes (3);
    int nperdir = ntotal / n_cycle;
    if (nperdir < 1)
      nperdir = 1;

    int dir = slot / nperdir;
    if (!UniqueID ())
      std::cout << "nperdir= " << nperdir << " dir= " << dir << std::endl;

//  std::stringstream node_path;
    char node_path[512];
    sprintf (node_path, "%s/%2.2d/%10.10d", root_, dir, slot / nperdir, slot);
    if (!UniqueID ())
      printf ("node_path=%s\n", node_path);

    for (int cycle = 0; cycle < n_cycle; cycle++) {
      if (UniqueID () % n_cycle == cycle) {
	char buf[1024];
	off_t size;

	sprintf (buf, "%s.compressed", node_path);
	FILE *f = fopen (buf, "rb");
	if (!f) {
	  fprintf (stderr, "Could not open %s\n", buf);
	  //return 3;
	  sleep (2);
	  f = fopen (buf, "rb");
	  if (!f) {
	    fprintf (stderr, "Could not open %s again.\n", buf);
	    return 3;
	  }
	}

	fseeko (f, 0, SEEK_END);

	size = ftello (f);

	fseeko (f, 0, SEEK_SET);

	double size_in_gb = (double) size / 1024. / 1024. / 1024.;
	if (UniqueID () == cycle) {
	  printf ("Node %d, Compressed file is %g GB\n", UniqueID (),
		  size_in_gb);
	}

	raw_in = (char *) malloc (size);
	if (!raw_in) {
	  fprintf (stderr, "Out of mem\n");
	  return 5;
	}

	double t0 = dclock ();

	if (fread (raw_in, size, 1, f) != 1) {
	  fprintf (stderr, "Invalid fread\n");
	  return 6;
	}

	double t1 = dclock ();

	if (UniqueID () == cycle) {
	  printf ("Read %.4g GB in %.4g seconds at %.4g GB/s\n",
		  size_in_gb, t1 - t0, size_in_gb / (t1 - t0));
	}
//      uint32_t crc_comp = crc32_fast (raw_in, size, 0);
	double t2 = dclock ();
	uint32_t crc_comp2 = crc32 (0, (const Bytef *) raw_in, size);
	double t3 = dclock ();



	if (UniqueID () == cycle) {
//        printf ("Computed CRC32: %X   (in %.4g seconds)\n", crc_comp, t2 - t1);
	  printf ("Computed CRC32(zlib): %X   (in %.3g seconds)\n", crc_comp2,
		  t3 - t2);
	  printf ("Expected CRC32: %X\n", args.crc32_header[UniqueID ()]);
	}

	if (crc_comp2 != args.crc32_header[UniqueID ()]) {
	  fprintf (stderr, "Corrupted file!\n");
	  exit (-42);
	  return 9;
	}

	fclose (f);
      }
      double barrier = 0;
      sumArray (&barrier, 1);
    }

    {
      // allocate memory before decompressing
      double uncomp_opt_size = 0.0;
      block_data_ortho.resize (args.blocks);
      for (int i = 0; i < args.blocks; i++) {
	block_data_ortho[i].resize (f_size_block * args.nkeep);
	memset (&block_data_ortho[i][0], 0,
		f_size_block * args.nkeep * sizeof (OPT));
	uncomp_opt_size += (double) f_size_block *args.nkeep;
      }
      block_coef.resize (args.blocks);
      for (int i = 0; i < args.blocks; i++) {
	block_coef[i].resize (f_size_coef_block);
	memset (&block_coef[i][0], 0, f_size_coef_block * sizeof (OPT));
	uncomp_opt_size += (double) f_size_coef_block;
      }
      double t0 = dclock ();

      // read
#define FP_16_SIZE(a,b)  (( (a) + (a/b) )*2)
      int _t = (int64_t) f_size_block * (args.nkeep - nkeep_fp16);

      //char* ptr = raw_in;
#pragma omp parallel
      {
#pragma omp for
	for (int nb = 0; nb < args.blocks; nb++) {
	  char *ptr = raw_in + nb * _t * 4;
	  read_floats (ptr, &block_data_ortho[nb][0], _t);

	}
#pragma omp for
	for (int nb = 0; nb < args.blocks; nb++) {
	  char *ptr =
	    raw_in + args.blocks * _t * 4 +
	    FP_16_SIZE ((int64_t) f_size_block * nkeep_fp16, 24) * nb;
	  read_floats_fp16 (ptr, &block_data_ortho[nb][_t],
			    (int64_t) f_size_block * nkeep_fp16, 24);
	}
      }

      double t1 = dclock ();

      char *raw_in_coef =
	raw_in + args.blocks * _t * 4 +
	FP_16_SIZE ((int64_t) f_size_block * nkeep_fp16, 24) * args.blocks;
      int64_t sz1 = 2 * (args.nkeep - nkeep_fp16) * 4;
      int64_t sz2 =
	FP_16_SIZE (2 * nkeep_fp16, args.FP16_COEF_EXP_SHARE_FLOATS);
#pragma omp parallel for
      for (int nb = 0; nb < args.blocks; nb++) {
	for (int j = 0; j < args.neig; j++) {
	  char *ptr = raw_in_coef + (sz1 + sz2) * (j * args.blocks + nb);
	  read_floats (ptr, &block_coef[nb][2 * args.nkeep * j],
		       2 * (args.nkeep - nkeep_fp16));
	  read_floats_fp16 (ptr,
			    &block_coef[nb][2 * args.nkeep * j +
					    2 * (args.nkeep - nkeep_fp16)],
			    2 * nkeep_fp16, args.FP16_COEF_EXP_SHARE_FLOATS);
	}
	//if (nb == 0 && UniqueID() == 0) {
	//      for (int i = 0; i < args.neig; i++)
	//              for (int nk = 0; nk < args.nkeep; nk++)
	//                      printf("block_coef[0][%d], %d th evec = %e %e\n", nk, i, block_coef[0][2*args.nkeep * i+nk],
	//                                      block_coef[0][2*args.nkeep * i+nk+1]);
	//}
      }

      double t2 = dclock ();

      if (UniqueID () == 0) {
	printf
	  ("Decompressing single/fp16 to OPT in %g seconds for evec and %g seconds for coefficients; %g GB uncompressed\n",
	   t1 - t0, t2 - t1,
	   uncomp_opt_size * sizeof (OPT) / 1024. / 1024. / 1024.);
      }

    }

    {
      char buf[1024];
      off_t size;
      uint32_t crc32 = 0x0;


      // now loop through eigenvectors and decompress them
#if 0
      //OPT* dest_all = (OPT*)malloc(f_size * sizeof(OPT) * args.neig);
      //dest_all = (OPT*)malloc(f_size * sizeof(OPT) * args.neig);

      if (!dest_all) {
	fprintf (stderr, "Out of mem\n");
	return 33;
      }
#endif
      VRB.Result (cname, fname, "f_size=%d args.neig=%d\n", f_size, args.neig);
//              memset(dest_all, 0, f_size * sizeof(OPT) * args.neig);
//      for(int i=0;i<args.neig;i++)
      for (int i = 0; i < dest_total; i++)
	memset (dest_all[i], 0, f_size * sizeof (OPT));


      block_data.resize (args.blocks);
      for (int i = 0; i < args.blocks; i++) {
	block_data[i].resize (f_size_block);
	memset (&block_data[i][0], 0, sizeof (OPT) * f_size_block);
      }


      //if(UniqueID() == 0) {
      //      for (int i = 0; i < 10; i++) {
      //              printf("block_data_ortho[0][%d] = %e\n", i, block_data_ortho[0][i]);
      //              printf("block_coef[0][%d] = %e\n", i, block_coef[0][i]);
      //      }
      //}
      double t0 = dclock ();
#pragma omp parallel
      {
	for (int j = 0; j < dest_total; j++) {

//                              OPT* dest = &dest_all[ (int64_t)f_size * j ];
	  OPT *dest = dest_all[j];

	  double ta, tb;
	  int tid = omp_get_thread_num ();

	  if (!tid)
	    ta = dclock ();

#if 1

#pragma omp for
	  for (int nb = 0; nb < args.blocks; nb++) {

	    OPT *dest_block = &block_data[nb][0];

#if 1
	    {
	      // do reconstruction of this block
	      memset (dest_block, 0, sizeof (OPT) * f_size_block);
	      for (int i = 0; i < args.nkeep; i++) {
		OPT *ev_i = &block_data_ortho[nb][(int64_t) f_size_block * i];
		OPT *coef = &block_coef[nb][2 * (i + args.nkeep * j)];
		caxpy_single (dest_block, *(complex < OPT > *)coef, ev_i,
			      dest_block, f_size_block);
	      }
	    }
#else
	    {
	      complex < OPT > *res = (complex < OPT > *)dest_block;
	      for (int l = 0; l < f_size_block / 2; l++) {
		complex < OPT > r = 0.0;
		for (int i = 0; i < args.nkeep; i++) {
		  complex < OPT > *ev_i =
		    (complex < OPT >
		     *) & block_data_ortho[nb][(int64_t) f_size_block * i];
		  complex < OPT > *coef =
		    (complex < OPT >
		     *) & block_coef[nb][2 * (i + args.nkeep * j)];
		  r += coef[i] * ev_i[l];
		}
		res[l] = r;
	      }
	    }
#endif
	  }
#else

	  for (int nb = 0; nb < args.blocks; nb++) {

	    OPT *dest_block = &block_data[nb][0];
	    // do reconstruction of this block
#pragma omp for
	    for (int ll = 0; ll < f_size_block; ll++)
	      dest_block[ll] = 0;

	    for (int i = 0; i < args.nkeep; i++) {
	      OPT *ev_i = &block_data_ortho[nb][(int64_t) f_size_block * i];
	      OPT *coef = &block_coef[nb][2 * (i + args.nkeep * j)];
	      caxpy_threaded (dest_block, *(complex < OPT > *)coef, ev_i,
			      dest_block, f_size_block);
	    }
	  }



#endif
	  if (!tid && UniqueID () == 0) {
	    tb = dclock ();
	    if (j % 100 == 0)
	      printf ("%d - %g seconds\n", j, tb - ta);
	  }
#pragma omp for
	  for (int idx = 0; idx < vol4d; idx++) {
	    int pos[5], pos_in_block[5], block_coor[5];
	    index_to_pos (idx, pos, args.s);

	    int parity = (pos[0] + pos[1] + pos[2] + pos[3]) % 2;
	    if (parity == 1) {

	      for (pos[4] = 0; pos[4] < args.s[4]; pos[4]++) {
		pos_to_blocked_pos (pos, pos_in_block, block_coor);

		int bid = pos_to_index (block_coor, args.nb);
		int ii = pos_to_index (pos_in_block, args.b) / 2;
		OPT *dst = &block_data[bid][ii * 24];

		int co;
		for (co = 0; co < 12; co++) {
		  OPT *out = &dest[get_cps_index (pos, co)];
		  out[0] = dst[2 * co + 0];
		  out[1] = dst[2 * co + 1];
		}
	      }
	    }
	  }
	  //if (tid == 0 && UniqueID() == 0) {
	  //      for (int ii = 0; ii < 10; ii++)
	  //              printf("dest[%d]=%f", ii, dest[ii]);
	  //}
	}
      }

      double t1 = dclock ();

      sumArray (&barrier, 1);
      if (UniqueID () == 0) {
	printf ("Reconstruct eigenvectors in %g seconds\n", t1 - t0);
      }

    }
    if (UniqueID () == 0) {
      for (int i = 0; i < 10; i++) {
	std::cout << dest_all[i] << std::endl;
      }
      std::cout << "end decompressing" << endl;
    }
    sumArray (&barrier, 1);


#undef FP_16_SIZE
    free (raw_in);
    return 0;
  }


  int EvecReader::read_compressed_vectors (const char *root,
					   const char *cdir,
					   std::vector < OPT * >&dest_all,
					   int start, int end,
					   float time_out)
  {
    std::string path (root);
    std::string fname ("read_compressed_vectors()");

    int dest_total = dest_all.size ();
    if( start<0 ) start=0;
    if( end<0 ) end=dest_total;
    VRB.Result(cname,fname.c_str(),"start=%d end=%d\n",start,end);
    
//    int start = 0;
//    int end = dest_total;
//      const int N_threads_old=bfmarg::threads;
//      const int N_threads = 64;
//      omp_set_num_threads(N_threads);

    char *cname = "read_compressed_vectors";
    vector < vector < OPT > >block_data;
    vector < vector < OPT > >block_data_ortho;
    vector < vector < OPT > >block_coef;

    int glb_i[5];
    glb_i[0] = GJP.XnodeSites () * GJP.Xnodes ();
    glb_i[1] = GJP.YnodeSites () * GJP.Ynodes ();
    glb_i[2] = GJP.ZnodeSites () * GJP.Znodes ();
    glb_i[3] = GJP.TnodeSites () * GJP.Tnodes ();
    glb_i[4] = GJP.SnodeSites () * GJP.Snodes ();

    vol4d =
      GJP.NodeSites (0) * GJP.NodeSites (1) * GJP.NodeSites (2) *
      GJP.NodeSites (3);
    vol5d = vol4d * GJP.SnodeSites ();

    //.........................Reading eigenvalues ...........................
    long nvec = 0;
    if (!UniqueID ()) {
      const std::string filename = path + "/eigen-values.txt";
      FILE *file = fopen (filename.c_str (), "r");
      fscanf (file, "%ld\n", &nvec);
      fclose (file);
    }
    sumArray (&nvec, 1);
    printf ("dest_all.size() %d  args.neig %d\n", dest_all.size (), args.neig);
    assert (dest_all.size () <= args.neig);
    assert (nvec == args.neig);


#if 0
    double vals[nvec];
    memset (vals, 0, sizeof (vals));
    if (!UniqueID ()) {
      std::cout << "Reading eigenvalues \n";
      const std::string filename = path + "/eigen-values.txt";
      FILE *file = fopen (filename.c_str (), "r");
      fscanf (file, "%ld\n", &nvec);
      for (int i = 0; i < end; i++) {
	fscanf (file, "%lE\n", &vals[i]);
	std::cout << sqrt (vals[i]) << std::endl;
      }
      fclose (file);
    }
    sumArray (vals, dest_total);
    evals.resize (dest_total);
    for (int k = 0; k < dest_total; k++) {
//              eig->bl[k] = vals[k];
      evals[k] = vals[k];
    }
    if (UniqueID () == 0)
      std::cout << "End Reading eigenvalues\n";
#endif

    //.......................Reading metadata........................

    char hostname[1024];
    sprintf (hostname, "%d", UniqueID ());

    //      evc_meta args;
    char buf[1024];
    sprintf (buf, "%s/metadata.txt", path.c_str ());
    FILE *f = NULL;
    //std::string buf = path + "/metadata.txt";
#if 0
//    uint32_t nprocessors = 1;
//    uint32_t nprocessors_real = 1;
    uint32_t status = 0;
    if (UniqueID () == 0)
      std::cout << "here" << std::endl;
    if (UniqueID () == 0) {
      printf ("node 0, before fopen metadata. \n");
      f = fopen (buf, "r");
      status = f ? 1 : 0;
    }
    sumArray (&status, 1);
    if (!status) {
      printf ("fopen fail \n");
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
      _IRL_READ_INT ("FP16_COEF_EXP_SHARE_FLOATS = %d\n",
		     &args.FP16_COEF_EXP_SHARE_FLOATS);
      printf ("node 0, after reading metadata\n");
    }
    sumArray (args.s, 5);
    sumArray (args.b, 5);
    sumArray (args.nb, 5);
    sumArray (&args.neig, 1);
    sumArray (&args.nkeep, 1);
    sumArray (&args.nkeep_single, 1);
    sumArray (&args.blocks, 1);
    sumArray (&args.FP16_COEF_EXP_SHARE_FLOATS, 1);

    vector < int >nn (5, 1);
    for (int i = 0; i < 5; i++) {
      nprocessors_real *= GJP.Nodes (i);;
    }
    for (int i = 0; i < 5; i++) {
      assert (glb_i[i] % args.s[i] == 0);
      nn[i] = glb_i[i] / args.s[i];
      nprocessors *= nn[i];
    }
    sync ();

    vector < uint32_t > crc32_arr (nprocessors, 0);
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
    //args.crc32 = crc32_arr[UniqueID()];

#undef _IRL_READ_INT

    if (f)
      fclose (f);
    if (UniqueID () == 0)
      printf ("after read metadata\n");
#endif

    cps::sync ();

    if (UniqueID () == 1) {
      printf ("Parameters:\n");
      for (int i = 0; i < 5; i++)
	printf ("s[%d] = %d\n", i, args.s[i]);
      for (int i = 0; i < 5; i++)
	printf ("b[%d] = %d\n", i, args.b[i]);
      printf ("nkeep = %d\n", args.nkeep);
      printf ("nkeep_single = %d\n", args.nkeep_single);

      printf ("FP16_COEF_EXP_SHARE_FLOATS = %d\n",
	      args.FP16_COEF_EXP_SHARE_FLOATS);
      printf ("\n");
    }




    if (nn[4] != 1) {
      std::cout << "nn[4] != 1. forget Grid -> GJP conversion? \n";
    }
    assert (nn[4] == 1);


    if (UniqueID () == 0)
      printf
	("Reading data that was generated on node-layout: %d %d %d %d %d\n ",
	 nn[0], nn[1], nn[2], nn[3], nn[4]);

    int nb_per_node = 1;	//# of blocks per node
    for (int dir = 0; dir < 5; dir++) {
      nb_per_node *= GJP.NodeSites (dir) / args.b[dir];
    }
    sync ();

    int f_size_coef_block = args.neig * 2 * args.nkeep;
    //int vol5d = GJP.XnodeSites() * GJP.YnodeSites() *GJP.ZnodeSites() *GJP.TnodeSites() *GJP.SnodeSites();
    int64_t f_size = vol5d / 2 * 24;
    int f_size_block = f_size / nb_per_node;

    block_coef.resize (nb_per_node);
    for (int i = 0; i < nb_per_node; i++) {
      block_coef[i].resize (f_size_coef_block);
      memset (&block_coef[i][0], 0, f_size_coef_block * sizeof (float));
    }
    block_data_ortho.resize (nb_per_node);
    for (int i = 0; i < nb_per_node; i++) {
      block_data_ortho[i].resize (f_size_block * args.nkeep);
      memset (&block_data_ortho[i][0], 0,
	      f_size_block * args.nkeep * sizeof (float));
    }
    block_data.resize (nb_per_node);
    for (int i = 0; i < nb_per_node; i++) {
      block_data[i].resize (f_size_block);
      memset (&block_data[i][0], 0, sizeof (float) * f_size_block);
    }


    // now get read geometry
    std::map < int, std::vector < int >>slots;
    std::vector < int >slot_lvol, lvol;
    int64_t slot_lsites;
    int ntotal;
    std::vector < int >_nn (nn.begin (), nn.end ());
    get_read_geometry (_nn, slots, slot_lvol, lvol, slot_lsites, ntotal);
    if (UniqueID () == 0)
      std::cout << "After get_read_geometry()" << endl;
    sync ();
    int _nd = (int) lvol.size ();

    // types
    //typedef typename Field::scalar_type Coeff_t;
    //typedef typename CoarseField::scalar_type CoeffCoarse_t;
    typedef double RealD;

    // slot layout
    int num_dir = 32;
    int nperdir = ntotal / num_dir;
    if (nperdir < 1)
      nperdir = 1;

    // load all necessary slots and store them appropriately
    for (std::map < int, std::vector < int > >::iterator sl = slots.begin ();
	 sl != slots.end (); sl++) {
      std::vector < int >&idx = sl->second;
      int slot = sl->first;
      std::vector < float >rdata;
      vector < uint32_t > slot_checksums (args.blocks * 2 + 1);

      if ((!crc32_checked) && cdir) {
	char checksum_filename[1024];
	sprintf (checksum_filename, "%s/%d_checksum.txt", cdir, slot);
	f = fopen (checksum_filename, "r");
	if (!f) {
	  printf ("Reading %s failed.\n", checksum_filename);
	  assert (f);
	}
	for (int line = 0; line < 2 * args.blocks + 1; line++) {
	  fscanf (f, "%X\n", &slot_checksums[line]);
	}
	fclose (f);
      }

      char buf[4096];

      // load one slot vector
      sprintf (buf, "%s/%2.2d/%10.10d.compressed", path.c_str (),
	       slot / nperdir, slot);
      f = fopen (buf, "rb");
      if (!f) {
	fprintf (stderr, "Node %s cannot read %s\n", hostname, buf);
	fflush (stderr);
	return false;
      }

      uint32_t crc = 0x0;
      off_t size;

      //GridStopWatch gsw;
      //gsw.Start();
      //double t0 = -dclock();

//      fseeko (f, 0, SEEK_END);
//      size = ftello (f);
      fseeko (f, 0, SEEK_SET);


      double t1 = -dclock ();
      //{
      int nsingleCap = args.nkeep_single;

      int64_t _cf_block_size = slot_lsites * 12 / 2 / args.blocks;

#define FP_16_SIZE(a,b)  (( (a) + (a/b) )*2)

      // first read single precision basis vectors
      for (int nb = 0; nb < args.blocks; nb++) {

	int mnb = globalToLocalCanonicalBlock (slot, _nn, nb);
	if (mnb != -1) {
	  //read now
	  int64_t read_size = _cf_block_size * 2 * nsingleCap * 4;
	  fseeko (f, read_size * nb, SEEK_SET);
	  std::vector < char >raw_in (read_size);
	  assert (fread (&raw_in[0], read_size, 1, f) == 1);
//    uint32_t crc_comp = crc32_fast(&raw_in[0],read_size,0);
	  uint32_t crc_comp = crc32 (0, (const Bytef *) &raw_in[0], read_size);
	  if ((!crc32_checked) && cdir)	// turns off if checksum directory is not specified
	    if (crc_comp != slot_checksums[nb]) {
	      printf ("nb = %d, crc_compute = %X, crc_read[nb] = %X\n", nb,
		      crc_comp, slot_checksums[nb]);
//                                                      assert(crc_comp == slot_checksums[nb]);
	    }
	  char *ptr = &raw_in[0];

#pragma omp parallel
	  {
	    std::vector < float >buff (_cf_block_size * 2, 0);
#pragma omp for
	    for (int i = 0; i < nsingleCap; i++) {
	      char *lptr = ptr + buff.size () * (i) * 4;
	      read_floats (lptr, &buff[0], buff.size ());
	      memcpy (&block_data_ortho[mnb][i * buff.size ()], &buff[0],
		      buff.size () * sizeof (float));
	    }
	  }
	}
      }

//#pragma omp barrier
//#pragma omp single
      sync ();
      if (UniqueID () == 0)
	std::
	  cout << "Finished reading block_data_ortho 0-nsingleCap" << std::endl;

      //ptr = ptr + _cf_block_size * 2*nsingleCap*args.blocks*4;



      // TODO: at this point I should add a checksum test for block_sp(nb,v,v) for all blocks, then I would know that the mapping
      // to blocks is OK at this point; after that ...

      // then read fixed precision basis vectors
//#pragma omp parallel

      for (int nb = 0; nb < args.blocks; nb++) {
	int mnb = globalToLocalCanonicalBlock (slot, _nn, nb);
	if (mnb != -1) {
	  int64_t read_size =
	    FP_16_SIZE (2 * _cf_block_size, 24) * (args.nkeep - nsingleCap);
	  int64_t seek_size =
	    _cf_block_size * 2 * nsingleCap * args.blocks * 4 + (args.nkeep -
								 nsingleCap) *
	    nb * FP_16_SIZE (2 * _cf_block_size, 24);
	  fseeko (f, seek_size, SEEK_SET);
	  std::vector < char >raw_in (read_size);
	  assert (fread (&raw_in[0], read_size, 1, f) == 1);
//        uint32_t crc_comp = crc32_fast (&raw_in[0], read_size, 0);
	  uint32_t crc_comp = crc32 (0, (const Bytef *) &raw_in[0], read_size);
	  if (cdir)
	    if (crc_comp != slot_checksums[nb + args.blocks]) {
	      printf ("nb = %d, crc_compute = %X, crc_read[nb+blocks] = %X\n",
		      nb, crc_comp, slot_checksums[nb + args.blocks]);
	      assert (crc_comp == slot_checksums[nb + args.blocks]);
	    }
	  char *ptr = &raw_in[0];

#pragma omp parallel
	  {
	    std::vector < float >buff (_cf_block_size * 2, 0);
#pragma omp for
	    for (int i = nsingleCap; i < args.nkeep; i++) {
	      char *lptr =
		ptr + FP_16_SIZE (buff.size (), 24) * (i - nsingleCap);
	      read_floats_fp16 (lptr, &buff[0], buff.size (), 24);
	      memcpy (&block_data_ortho[mnb][i * buff.size ()], &buff[0],
		      buff.size () * sizeof (float));
	    }
	  }
	}
      }


      //      ptr = ptr + FP_16_SIZE( _cf_block_size*2*(args.nkeep - nsingleCap)*args.blocks, 24 );
      if (UniqueID () == 0)
	std::cout << "Finished reading block_data_ortho nsingleCap-nkeep " <<
	  std::endl;
      sync ();

      int64_t seek_size = _cf_block_size * 2 * nsingleCap * args.blocks * 4 +
	FP_16_SIZE (_cf_block_size * 2 * (args.nkeep - nsingleCap) *
		    args.blocks, 24);
      fseeko (f, seek_size, SEEK_SET);
      int64_t read_size =
	(4 * args.nkeep_single * 2 +
	 FP_16_SIZE ((args.nkeep - args.nkeep_single) * 2,
		     args.FP16_COEF_EXP_SHARE_FLOATS))
	* (args.neig * args.blocks);

      std::vector < char >raw_in (read_size);
      assert (fread (&raw_in[0], read_size, 1, f) == 1);
//      uint32_t crc_comp = crc32_fast (&raw_in[0], read_size, 0);
      uint32_t crc_comp = crc32 (0, (const Bytef *) &raw_in[0], read_size);
      if (cdir)
	if (crc_comp != slot_checksums[2 * args.blocks]) {
	  printf
	    ("Reading block_coef crc_compute = %X, crc_read[2*blocks] = %X\n",
	     crc_comp, slot_checksums[2 * args.blocks]);
	  assert (crc_comp == slot_checksums[2 * args.blocks]);
	}

      char *ptr = &raw_in[0];

#pragma omp parallel
      {
	std::vector < float >buf1 (args.nkeep_single * 2);
	std::vector < float >buf2 ((args.nkeep - args.nkeep_single) * 2);

#pragma omp for
	for (int j = 0; j < args.neig; j++)
	  for (int nb = 0; nb < args.blocks; nb++) {
	    int ii, oi;
	    int mnb = globalToLocalCanonicalBlock (slot, _nn, nb);
	    //int mnb = (nb < nb_per_node) ? nb : -1;
	    if (mnb != -1) {

	      char *lptr = ptr + (4 * buf1.size () + FP_16_SIZE (buf2.size (),
								 args.
								 FP16_COEF_EXP_SHARE_FLOATS))
		* (nb + j * args.blocks);
	      int l;
	      read_floats (lptr, &buf1[0], buf1.size ());
	      //automatically increase lptr
	      memcpy (&block_coef[mnb][j * (buf1.size () + buf2.size ())],
		      &buf1[0], buf1.size () * sizeof (float));
	      //for (l=0;l<nkeep_single;l++) {
	      //      ((CoeffCoarse_t*)&coef._v[j]._odata[oi]._internal._internal[l])[ii] = CoeffCoarse_t(buf1[2*l+0],buf1[2*l+1]);
	      //}
	      read_floats_fp16 (lptr, &buf2[0], buf2.size (),
				args.FP16_COEF_EXP_SHARE_FLOATS);
	      memcpy (&block_coef[mnb]
		      [j * (buf1.size () + buf2.size ()) + buf1.size ()],
		      &buf2[0], buf2.size () * sizeof (float));
	      //      for (l=nkeep_single;l<nkeep;l++) {
	      //              ((CoeffCoarse_t*)&coef._v[j]._odata[oi]._internal._internal[l])[ii] = CoeffCoarse_t(buf2[2*(l-nkeep_single)+0],buf2[2*(l-nkeep_single)+1]);
	      //}

	    }
	  }

      }
      fclose (f);
      t1 += dclock ();
      //std::cout << "Processed " << totalGB << " GB of compressed data at " << totalGB/t1<< " GB/s" << std::endl;
      if (UniqueID () == 0)
	std::
	  cout << "Processed compressed data in " << t1 << " sec " << std::endl;
      //}
    }
    sync ();
    for (int i = start; i < end; i++)
      memset (dest_all[i-start], 0, f_size * sizeof (OPT));

    double t0 = dclock ();
    //Now change the args to the local one.
    args.blocks = nb_per_node;
    args.s[0] = GJP.NodeSites (0);
    args.s[1] = GJP.NodeSites (1);
    args.s[2] = GJP.NodeSites (2);
    args.s[3] = GJP.NodeSites (3);
    args.s[4] = GJP.NodeSites (4);

    args.nb[0] = args.s[0] / args.b[0];
    args.nb[1] = args.s[1] / args.b[1];
    args.nb[2] = args.s[2] / args.b[2];
    args.nb[3] = args.s[3] / args.b[3];
    args.nb[4] = args.s[4] / args.b[4];

#pragma omp parallel
    {
      for (int j = start; j < end; j++) {
//      if (UniqueID() == 0) std::cout << "j = " << j << std::endl;

//                      float* dest = &dest_all[ (int64_t)f_size * j ];
	OPT *dest = dest_all[j-start];

	double ta, tb;
	int tid = omp_get_thread_num ();

	//if (!tid)
	ta = dclock ();

//#if 1

#pragma omp for
	for (int nb = 0; nb < nb_per_node; nb++) {

	  float *dest_block = &block_data[nb][0];

	  {
	    // do reconstruction of this block
	    memset (dest_block, 0, sizeof (float) * f_size_block);
	    for (int i = 0; i < args.nkeep; i++) {
	      float *ev_i = &block_data_ortho[nb][(int64_t) f_size_block * i];
	      float *coef = &block_coef[nb][2 * (i + args.nkeep * j)];
	      caxpy_single (dest_block, *(complex < float >*) coef, ev_i,
			    dest_block, f_size_block);
	    }
	  }
	}

	if (!tid && UniqueID () == 0) {
	  tb = dclock ();
	  if (j % 100 == 0)
	    printf ("%d - %g seconds\n", j, tb - ta);
	}
	//int[5] loc_s = {GJP.NodeSites(0), GJP.NodeSites(1), GJP.NodeSites(2), GJP.NodeSites(3),GJP.NodeSites(4)};

#pragma omp for
	for (int idx = 0; idx < vol4d; idx++) {
	  int pos[5], pos_in_block[5], block_coor[5];
	  index_to_pos (idx, pos, args.s);

	  int parity = (pos[0] + pos[1] + pos[2] + pos[3]) % 2;
	  if (parity == 1) {

	    for (pos[4] = 0; pos[4] < args.s[4]; pos[4]++) {
	      pos_to_blocked_pos (pos, pos_in_block, block_coor);

	      int bid = pos_to_index (block_coor, args.nb);
	      int ii = pos_to_index (pos_in_block, args.b) / 2;
	      float *dst = &block_data[bid][ii * 24];

	      int co;
	      for (co = 0; co < 12; co++) {
//                                                      float* out=&dest[ get_bfm_index(pos,co) ];
		float *out = &dest[get_cps_index (pos, co)];
		out[0] = dst[2 * co + 0];
		out[1] = dst[2 * co + 1];
	      }
	    }
	  }
	}

	//if (tid == 0 && UniqueID() == 0) {
	//      for (int ii = 0; ii < 10; ii++)
	//              printf("dest[%d]=%f", ii, dest[ii]);
	//}
      }
    }

    double t1 = dclock ();

    sync ();
    if (UniqueID () == 0) {
      printf ("Reconstruct eigenvectors in %g seconds\n", t1 - t0);
    }

    if (UniqueID () == 0) {
      for (int i = 0; i < 10; i++) {
	std::cout << dest_all[i] << std::endl;
      }
      std::cout << "end decompressing" << endl;
    }
    sync ();
    int size_5d_prec = GJP.VolNodeSites () * GJP.SnodeSites () * 24 / 2;
    sync ();
#undef FP_16_SIZE
    return true;
  }


}
