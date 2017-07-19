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



//int main(int argc, char* argv[]) {
//int decompress(const char* root_ , const char* node_path, OPT* &dest_all) {

namespace cps{


#if 0
const char *EvecReader::cname = "EvecReader";
const char* EvecReader::header =
"QCD eigenvector decompressor\n"
"Authors: Christoph Lehner\n"
"Date: 2017\n";
#endif


int EvecReader::decompress(const char* root_ , std::vector < OPT *> &dest_all) {

   const char *fname="decompress()";
   printf("dest_all.size() %d  args.neig %d\n",dest_all.size(),args.neig);
   assert(dest_all.size() <= args.neig);
   int dest_total = dest_all.size();

#pragma omp parallel
	{
#pragma omp single
		{
			nthreads = omp_get_num_threads();
		}
	}

	if (UniqueID() == 0) 
		printf("%s\n%d threads\n\n",header,nthreads);
	double barrier =0;
	sumArray(&barrier, 1);

	//  if (argc < 1+1) {
	//    fprintf(stderr,"Arguments: fileroot\n");
	//    return 1;
	//  }

	//  char* root = argv[1];
	const char* root = root_;
	{
		int i;
		if (!read_metadata(root))
			return -1;

		double barrier =0;
		sumArray(&barrier, 1);
		if (UniqueID() == 0)
			printf("after read metadata\n");

		vol4d = args.s[0] * args.s[1] * args.s[2] * args.s[3];
		vol5d = vol4d * args.s[4];
		f_size = vol5d / 2 * 24;

		if (UniqueID() == 1) {
		printf("Parameters:\n");
		for (i=0;i<5;i++)
			printf("s[%d] = %d\n",i,args.s[i]);
		for (i=0;i<5;i++)
			printf("b[%d] = %d\n",i,args.b[i]);
		printf("nkeep = %d\n",args.nkeep);
		printf("nkeep_single = %d\n",args.nkeep_single);

		printf("f_size = %d\n",f_size);
		printf("FP16_COEF_EXP_SHARE_FLOATS = %d\n",args.FP16_COEF_EXP_SHARE_FLOATS);
		printf("crc32 = %X\n",args.crc32);
		printf("\n");
		}

		// sanity check
		args.blocks = 1;
		for (i=0;i<5;i++) {
			if (args.s[i] % args.b[i]) {
				fprintf(stderr,"Invalid blocking in dimension %d\n",i);
				return 72;
			}

			args.nb[i] = args.s[i] / args.b[i];
			args.blocks *= args.nb[i]; //local 5D volume
		}

		f_size_block = f_size / args.blocks; 

		if (UniqueID() == 0){
		printf("number of blocks = %d\n",args.blocks);
		printf("f_size_block = %d\n",f_size_block);

		printf("Internally using sizeof(OPT) = %d\n",sizeof(OPT));

		printf("\n");
		}

		nkeep_fp16 = args.nkeep - args.nkeep_single;
		if (nkeep_fp16 < 0)
			nkeep_fp16 = 0;

		f_size_coef_block = args.neig * 2 * args.nkeep;
	}

	int n_cycle = 32;
	int slot = GJP.NodeCoor(0);
	int ntotal = 1;
	for(int i=0;i<3;i++){
		slot *= GJP.Nodes(i);
		slot += GJP.NodeCoor(i+1);
		ntotal *= GJP.Nodes(i);
	}
		ntotal *= GJP.Nodes(3);
    int nperdir = ntotal / n_cycle;
    if (nperdir<1) nperdir=1;

	int dir = slot / nperdir;
 	if(!UniqueID()) 
	std::cout <<"nperdir= "<<nperdir<<" dir= "<<dir<<std::endl;

//	std::stringstream node_path;
    char node_path[512];
	sprintf(node_path,"%s/%2.2d/%10.10d",root_,dir,slot/nperdir,slot);
 	if(!UniqueID()) printf("node_path=%s\n", node_path);

	for (int cycle = 0; cycle < n_cycle; cycle++)
	{
		if (UniqueID() % n_cycle == cycle) {
		char buf[1024];
		off_t size;

		sprintf(buf,"%s.compressed",node_path);
		printf("Opening %s\n",buf);
		FILE* f = fopen(buf,"rb");
		if (!f) {
			fprintf(stderr,"Could not open %s\n",buf);
			//return 3;
			sleep(2);
			f = fopen(buf,"rb");
			if (!f) {
				fprintf(stderr,"Could not open %s again.\n",buf);
				return 3;
			}
		}

		fseeko(f,0,SEEK_END);

		size = ftello(f);

		fseeko(f,0,SEEK_SET);

		double size_in_gb = (double)size / 1024. / 1024. / 1024.;
		if (UniqueID() == cycle) {
			printf("Node %d, Compressed file is %g GB\n",UniqueID(), size_in_gb);
		}

		raw_in = (char*)malloc( size );
		if (!raw_in) {
			fprintf(stderr,"Out of mem\n");
			return 5;
		}

		double t0 = dclock();

		if (fread(raw_in,size,1,f) != 1) {
			fprintf(stderr,"Invalid fread\n");
			return 6;
		}

		double t1 = dclock();

		if (UniqueID() == cycle) {
		printf("Read %.4g GB in %.4g seconds at %.4g GB/s\n",
				size_in_gb, t1-t0,size_in_gb / (t1-t0) );
		}

		uint32_t crc_comp = crc32_fast(raw_in,size,0);

		double t2 = dclock();

		if (UniqueID() == cycle) {
		printf("Computed CRC32: %X   (in %.4g seconds)\n",crc_comp,t2-t1);
		printf("Expected CRC32: %X\n",args.crc32);
		}

		if (crc_comp != args.crc32) {
			fprintf(stderr,"Corrupted file!\n");
			return 9;
		}

		fclose(f);
		}
		double barrier =0;
		sumArray(&barrier, 1);
	}

	{
		// allocate memory before decompressing
		double uncomp_opt_size = 0.0;
		block_data_ortho.resize(args.blocks);
		for (int i=0;i<args.blocks;i++) {
			block_data_ortho[i].resize(f_size_block * args.nkeep);    
			memset(&block_data_ortho[i][0], 0, f_size_block * args.nkeep* sizeof(OPT));
			uncomp_opt_size += (double)f_size_block * args.nkeep;
		}
		block_coef.resize(args.blocks);
		for (int i=0;i<args.blocks;i++) {
			block_coef[i].resize(f_size_coef_block);    
			memset(&block_coef[i][0], 0, f_size_coef_block * sizeof(OPT));
			uncomp_opt_size += (double)f_size_coef_block;
		}
		double t0 = dclock();

		// read
#define FP_16_SIZE(a,b)  (( (a) + (a/b) )*2)
		int _t = (int64_t)f_size_block * (args.nkeep - nkeep_fp16);

		//char* ptr = raw_in;
#pragma omp parallel
		{
#pragma omp for  
			for (int nb=0;nb<args.blocks;nb++) {
				char* ptr = raw_in + nb*_t*4;
				read_floats(ptr,  &block_data_ortho[nb][0], _t );

			}
#pragma omp for
			for (int nb=0;nb<args.blocks;nb++) {
				char* ptr = raw_in + args.blocks*_t*4 + FP_16_SIZE( (int64_t)f_size_block * nkeep_fp16 , 24 ) * nb;
				read_floats_fp16(ptr,  &block_data_ortho[nb][ _t ], (int64_t)f_size_block * nkeep_fp16, 24 );
			}
		}

		double t1 = dclock();

		char* raw_in_coef = raw_in + args.blocks*_t*4 + FP_16_SIZE( (int64_t)f_size_block * nkeep_fp16 , 24 ) * args.blocks;
		int64_t sz1 = 2*(args.nkeep - nkeep_fp16)*4;
		int64_t sz2 = FP_16_SIZE( 2*nkeep_fp16, args.FP16_COEF_EXP_SHARE_FLOATS);
#pragma omp parallel for
		for (int nb=0;nb<args.blocks;nb++) {
			for (int j=0;j<args.neig;j++) {
				char* ptr = raw_in_coef + (sz1+sz2)*(j * args.blocks + nb);
				read_floats(ptr,  &block_coef[nb][2*args.nkeep*j], 2*(args.nkeep - nkeep_fp16) );
				read_floats_fp16(ptr,  &block_coef[nb][2*args.nkeep*j + 2*(args.nkeep - nkeep_fp16) ], 2*nkeep_fp16 , args.FP16_COEF_EXP_SHARE_FLOATS);
			}
			//if (nb == 0 && UniqueID() == 0) {
			//	for (int i = 0; i < args.neig; i++)
			//		for (int nk = 0; nk < args.nkeep; nk++)
			//			printf("block_coef[0][%d], %d th evec = %e %e\n", nk, i, block_coef[0][2*args.nkeep * i+nk],
			//					block_coef[0][2*args.nkeep * i+nk+1]);
			//}
		}

		double t2 = dclock();

		if (UniqueID() == 0){
		printf("Decompressing single/fp16 to OPT in %g seconds for evec and %g seconds for coefficients; %g GB uncompressed\n",t1-t0,t2-t1,
				uncomp_opt_size * sizeof(OPT) / 1024./1024./1024.);
		}

	}

	{
		char buf[1024];
		off_t size;
		uint32_t crc32 = 0x0;


		// now loop through eigenvectors and decompress them
#if 0
		//OPT* dest_all = (OPT*)malloc(f_size * sizeof(OPT) * args.neig);
//		dest_all = (OPT*)malloc(f_size * sizeof(OPT) * args.neig);
		
		if (!dest_all) {
			fprintf(stderr,"Out of mem\n");
			return 33;
		}
#endif
		VRB.Result(cname,fname,"f_size=%d args.neig=%d\n",f_size,args.neig);
//		for(int i=0;i<args.neig;i++)
		for(int i=0;i<dest_total;i++)
		memset(dest_all[i], 0, f_size * sizeof(OPT) );

		block_data.resize(args.blocks);
		for (int i=0;i<args.blocks;i++){
			block_data[i].resize(f_size_block);    
			memset(&block_data[i][0], 0, sizeof(OPT) * f_size_block);
		}
		

		//if(UniqueID() == 0) {
		//	for (int i = 0; i < 10; i++) {
		//		printf("block_data_ortho[0][%d] = %e\n", i, block_data_ortho[0][i]);
		//		printf("block_coef[0][%d] = %e\n", i, block_coef[0][i]);
		//	}
		//}
		double t0 = dclock();
#pragma omp parallel
		{
			for(int j=0;j<dest_total;j++){

//				OPT* dest = dest_all[ (int64_t)f_size * j ];
				OPT* dest = dest_all[  j ];

				double ta,tb;
				int tid = omp_get_thread_num();

				if (!tid)
					ta = dclock();

#if 1

#pragma omp for
				for (int nb=0;nb<args.blocks;nb++) {

					OPT* dest_block = &block_data[nb][0];

#if 1
					{
						// do reconstruction of this block
						memset(dest_block,0,sizeof(OPT)*f_size_block);
						for (int i=0;i<args.nkeep;i++) {
							OPT* ev_i = &block_data_ortho[nb][ (int64_t)f_size_block * i ];
							OPT* coef = &block_coef[nb][ 2*( i + args.nkeep*j ) ];
							caxpy_single(dest_block, *(complex<OPT>*)coef, ev_i, dest_block, f_size_block);
						}
					}
#else
					{
						complex<OPT>* res = (complex<OPT>*)dest_block;
						for (int l=0;l<f_size_block/2;l++) {
							complex<OPT> r = 0.0;
							for (int i=0;i<args.nkeep;i++) {
								complex<OPT>* ev_i = (complex<OPT>*)&block_data_ortho[nb][ (int64_t)f_size_block * i ];
								complex<OPT>* coef = (complex<OPT>*)&block_coef[nb][ 2*( i + args.nkeep*j ) ];
								r += coef[i] * ev_i[l];
							}
							res[l] = r;
						}
					}
#endif
				}
#else

				for (int nb=0;nb<args.blocks;nb++) {

					OPT* dest_block = &block_data[nb][0];
					// do reconstruction of this block
#pragma omp for
					for (int ll=0;ll<f_size_block;ll++)
						dest_block[ll] = 0;

					for (int i=0;i<args.nkeep;i++) {
						OPT* ev_i = &block_data_ortho[nb][ (int64_t)f_size_block * i ];
						OPT* coef = &block_coef[nb][ 2*( i + args.nkeep*j ) ];
						caxpy_threaded(dest_block, *(complex<OPT>*)coef, ev_i, dest_block, f_size_block);
					}
				}



#endif
				if (!tid && UniqueID() == 0) {
					tb = dclock();
					if (j % 100 == 0)
						printf("%d - %g seconds\n",j,tb-ta);
				}

#pragma omp for
				for (int idx=0;idx<vol4d;idx++) {
					int pos[5], pos_in_block[5], block_coor[5];
					index_to_pos(idx,pos,args.s);

					int parity = (pos[0] + pos[1] + pos[2] + pos[3]) % 2;
					if (parity == 1) {

						for (pos[4]=0;pos[4]<args.s[4];pos[4]++) {
							pos_to_blocked_pos(pos,pos_in_block,block_coor);

							int bid = pos_to_index(block_coor, args.nb);
							int ii = pos_to_index(pos_in_block, args.b) / 2;
							OPT* dst = &block_data[bid][ii*24];

							int co;
							for (co=0;co<12;co++) {
								OPT* out=&dest[ get_bfm_index(pos,co) ];
								out[0] = dst[2*co + 0];
								out[1] = dst[2*co + 1];
							}
						}
					}
				}
				//if (tid == 0 && UniqueID() == 0) {
				//	for (int ii = 0; ii < 10; ii++)
				//		printf("dest[%d]=%f", ii, dest[ii]);
				//}
			}
		}

		double t1 = dclock();

		sumArray(&barrier, 1);
		if (UniqueID() == 0) {
		printf("Reconstruct eigenvectors in %g seconds\n",t1-t0);
		}

	}
	if (UniqueID() == 0) {
	for (int i = 0; i < 10; i++) {
		std::cout << dest_all[i] << std::endl;
	}
	std::cout << "end decompressing" << endl;
	}
	sumArray(&barrier, 1);


	free(raw_in);
	return 0;
}


}
