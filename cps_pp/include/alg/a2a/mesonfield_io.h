#ifndef MESONFIELD_IO
#define MESONFIELD_IO

struct hostEndian{
  enum EndianType { BIG, LITTLE };
  inline static EndianType get(){ //copied from fpconv
    char end_check[4] = {1,0,0,0};
    uint32_t *lp = (uint32_t *)end_check;
    if ( *lp == 0x1 ) { 
      return LITTLE;
    } else {
      return BIG;
    }
  }
};

template<typename T>
struct FPformat{
  inline static FP_FORMAT get(){ //also taken from fpconv
    assert(sizeof(T) == 4 || sizeof(T) == 8);
    static const hostEndian::EndianType endian = hostEndian::get();
    
    if(sizeof(T) == 8){
      return endian == hostEndian::LITTLE ? FP_IEEE64LITTLE : FP_IEEE64BIG;
    }else {  // 32 bits
      union { 
	float pinum;
	char pichar[4];
      }cpspi;

      FP_FORMAT format;
      
      cpspi.pinum = FPConv_PI;
      if(endian == hostEndian::BIG) {
	format = FP_IEEE32BIG;
	for(int i=0;i<4;i++) {
	  if(cpspi.pichar[i] != FPConv_ieee32pi_big[i]) {
	    format = FP_TIDSP32;
	    break;
	  }
	}
      }
      else {
	format = FP_IEEE32LITTLE;
	for(int i=0;i<4;i++) {
	  if(cpspi.pichar[i] != FPConv_ieee32pi_big[3-i]) {
	    format = FP_TIDSP32;
	    break;
	  }
	}
      }
      return format;
    } // end of 32 bits
  }   
};

template<typename T>
struct FPformat<std::complex<T> >{
  inline static FP_FORMAT get(){ return FPformat<T>::get(); }
};


template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::write(const std::string &filename, FP_FORMAT fileformat) const{
  if(!UniqueID()) printf("Writing meson field of size %d kB to file %s\n",byte_size()/1024,filename.c_str());
  MPI_Barrier(MPI_COMM_WORLD);
  if(node_mpi_rank != -1){
    int my_rank;
    int ret = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(ret != MPI_SUCCESS) ERR.General("A2AmesonField","write","Comm_rank failed\n");
    
    if(!UniqueID() && node_mpi_rank != my_rank)
      ERR.General("A2AmesonField","write","Mesonfield must be present on head node\n");
  }

  if(!UniqueID()){
    FP_FORMAT dataformat = FPformat<ScalarComplexType>::get();
    FPConv conv;
    if(fileformat == FP_AUTOMATIC)
      fileformat = dataformat;
    else
      assert(conv.size(fileformat) == conv.size(dataformat));
    
    conv.setHostFormat(dataformat);
    conv.setFileFormat(fileformat);

    int dsize = conv.size(dataformat);
    unsigned int checksum = conv.checksum( (char*)mf, 2*fsize, dataformat);

  
    //Header
    std::ofstream file(filename.c_str(),std::ofstream::out);
    assert(!file.fail());
    file.exceptions ( std::ofstream::failbit | std::ofstream::badbit );

    file << "BEGIN_HEADER\n";
    file << "HDR_VERSION = 1\n";
    file << "DATA_FORMAT = " << conv.name(fileformat) << '\n';
    file << "CHECKSUM = " << checksum << "\n";
    file << "END_HEADER\n";
    
    //Parameters    
    char* la2aparams_buf = (char*)malloc(10000 * sizeof(char));
    char* ra2aparams_buf = (char*)malloc(10000 * sizeof(char));
    {
      VML vml;
      vml.Create(la2aparams_buf,10000,VML_ENCODE);
      A2AArg &a2a_args_l = const_cast<A2AArg &>(lindexdilution.getArgs());
      assert( a2a_args_l.Vml(&vml,"A2AARGS_L") );
      vml.Destroy();
    }
    {
      VML vml;
      vml.Create(ra2aparams_buf,10000,VML_ENCODE);
      A2AArg &a2a_args_r = const_cast<A2AArg &>(rindexdilution.getArgs());
      assert( a2a_args_r.Vml(&vml,"A2AARGS_R") );
      vml.Destroy();
    }		
      
    file << "BEGIN_PARAMS\n";
    //int nmodes_l, nmodes_r;  //derived from index dilutions
    file << "FSIZE = " << fsize << "\n";
    file << "TL = " << tl << "\n";
    file << "TR = " << tr << "\n";
    file << "LINDEXDILUTION = " << lindexdilution.name() << "\n";
    file << "RINDEXDILUTION = " << rindexdilution.name() << "\n";
    file << "STRLEN_A2AARGS_L = " << strlen(la2aparams_buf) << '\n';
    file << "STRLEN_A2AARGS_R = " << strlen(ra2aparams_buf) << '\n';
    file << la2aparams_buf << ra2aparams_buf;
    file << "END_PARAMS\n";      

    free(la2aparams_buf);
    free(ra2aparams_buf);

    //Data
    file << "BEGIN_DATA\n";
      
    static const int chunk = 32768; //32kb chunks
    assert(chunk % dsize == 0);
    int fdinchunk = chunk/dsize;
    char* wbuf = (char*)malloc(chunk * sizeof(char)); 
      
    char const* dptr = (char const*)mf;

    int off = 0;
    int nfd = 2*fsize;
    while(off < nfd){
      int grab = std::min(nfd-off, fdinchunk); //How many data elements to grab
      int grabchars = grab * dsize;
      conv.host2file(wbuf,dptr,grab);
      file.write(wbuf,grabchars);
      off += grab;
      dptr += grabchars;
    }
    file << "\nEND_DATA\n";
    free(wbuf);
    file.close();    
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::read(const std::string &filename){
  if(!UniqueID()) printf("Reading meson field from file %s\n",filename.c_str());

  //Get this node's mpi rank
  int my_mpi_rank;
  int ret = MPI_Comm_rank(MPI_COMM_WORLD, &my_mpi_rank);
  if(ret != MPI_SUCCESS) ERR.General("A2AmesonField","read","Comm_rank failed\n");
  
  //Broadcast to all nodes the mpi rank of the head node (UniqueID() == 0)
  MPI_Barrier(MPI_COMM_WORLD);
  int head_mpi_rank;
  int rank_tmp = (UniqueID() == 0 ? my_mpi_rank : 0);
  ret = MPI_Allreduce(&rank_tmp,&head_mpi_rank, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD); //node is now the MPI rank corresponding to UniqueID == _node
  if(ret != MPI_SUCCESS) ERR.General("A2AmesonField","read","Reduce failed\n");

  //Open file on head node
  std::ifstream file;

  int read_fsize;
  unsigned int checksum;

  int a2aparams_l_buflen, a2aparams_r_buflen;
  char *a2aparams_l_buf, *a2aparams_r_buf;
  char dformatbuf[256];
  
  if(UniqueID() == 0){
    file.open(filename.c_str());
    assert(!file.fail());
    file.exceptions ( std::ofstream::failbit | std::ofstream::badbit );

    std::string str;
    
    //Header
    getline(file,str); assert(str == "BEGIN_HEADER");
    getline(file,str); assert(str == "HDR_VERSION = 1");
    getline(file,str); assert( sscanf(str.c_str(),"DATA_FORMAT = %s",dformatbuf) == 1 );
    getline(file,str); assert( sscanf(str.c_str(),"CHECKSUM = %u",&checksum) == 1 );
    getline(file,str); assert(str == "END_HEADER");

    //Params
    getline(file,str); assert(str == "BEGIN_PARAMS");    
    getline(file,str); assert( sscanf(str.c_str(),"FSIZE = %d",&read_fsize) == 1 );
    getline(file,str); assert( sscanf(str.c_str(),"TL = %d",&tl) == 1 );
    getline(file,str); assert( sscanf(str.c_str(),"TR = %d",&tr) == 1 );

    char nmbuf[256];
    getline(file,str); assert( sscanf(str.c_str(),"LINDEXDILUTION = %s",nmbuf) == 1 );
    assert( std::string(nmbuf) == lindexdilution.name() );

    getline(file,str); assert( sscanf(str.c_str(),"RINDEXDILUTION = %s",nmbuf) == 1 );
    assert( std::string(nmbuf) == rindexdilution.name() );
    
    getline(file,str); assert( sscanf(str.c_str(),"STRLEN_A2AARGS_L = %d",&a2aparams_l_buflen) == 1 );
    getline(file,str); assert( sscanf(str.c_str(),"STRLEN_A2AARGS_R = %d",&a2aparams_r_buflen) == 1 );
    ++a2aparams_l_buflen; //leave room for null character!
    ++a2aparams_r_buflen;
    
    //Read the VMLs for the left and right A2A params and squirt out for each node to decode
    a2aparams_l_buf = (char*)malloc(a2aparams_l_buflen * sizeof(char));
    a2aparams_r_buf = (char*)malloc(a2aparams_r_buflen * sizeof(char));
    file.get(a2aparams_l_buf,a2aparams_l_buflen,EOF);
    file.get(a2aparams_r_buf,a2aparams_r_buflen,EOF);

    getline(file,str); assert(str == "END_PARAMS");
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  //Squirt A2Aparams and whatnot over to other nodes for data setup
  ret = MPI_Bcast(&checksum, 1, MPI_UNSIGNED, head_mpi_rank, MPI_COMM_WORLD);
  if(ret != MPI_SUCCESS) ERR.General("A2AmesonField","read","Squirt 1 fail\n");
  
  ret = MPI_Bcast(&read_fsize, 1, MPI_INT, head_mpi_rank, MPI_COMM_WORLD);
  if(ret != MPI_SUCCESS) ERR.General("A2AmesonField","read","Squirt 2 fail\n");
  
  ret = MPI_Bcast(&tl, 1, MPI_INT, head_mpi_rank, MPI_COMM_WORLD);
  if(ret != MPI_SUCCESS) ERR.General("A2AmesonField","read","Squirt 3 fail\n");

  ret = MPI_Bcast(&tr, 1, MPI_INT, head_mpi_rank, MPI_COMM_WORLD);
  if(ret != MPI_SUCCESS) ERR.General("A2AmesonField","read","Squirt 4 fail\n");
  
  ret = MPI_Bcast(&a2aparams_l_buflen, 1, MPI_INT, head_mpi_rank, MPI_COMM_WORLD);
  if(ret != MPI_SUCCESS) ERR.General("A2AmesonField","read","Squirt 5 fail\n");

  ret = MPI_Bcast(&a2aparams_r_buflen, 1, MPI_INT, head_mpi_rank, MPI_COMM_WORLD);
  if(ret != MPI_SUCCESS) ERR.General("A2AmesonField","read","Squirt 6 fail\n");

  if(UniqueID() != 0){
    //Other nodes create bufs for a2a params
    a2aparams_l_buf = (char*)malloc(a2aparams_l_buflen * sizeof(char));
    a2aparams_r_buf = (char*)malloc(a2aparams_r_buflen * sizeof(char));    
  }

  ret = MPI_Bcast(a2aparams_l_buf, a2aparams_l_buflen, MPI_CHAR, head_mpi_rank, MPI_COMM_WORLD);
  if(ret != MPI_SUCCESS) ERR.General("A2AmesonField","read","Squirt 7 fail\n");

  ret = MPI_Bcast(a2aparams_r_buf, a2aparams_r_buflen, MPI_CHAR, head_mpi_rank, MPI_COMM_WORLD);
  if(ret != MPI_SUCCESS) ERR.General("A2AmesonField","read","Squirt 8 fail\n");

  //Every node parse the params buffers
  A2AArg read_a2a_args_l, read_a2a_args_r;
  {
    VML vml;
    vml.Create(a2aparams_l_buf,a2aparams_l_buflen,VML_DECODE);
    assert( read_a2a_args_l.Vml(&vml,"A2AARGS_L") );
    vml.Destroy();
  }
  {
    VML vml;
    vml.Create(a2aparams_r_buf,a2aparams_r_buflen,VML_DECODE);
    assert( read_a2a_args_r.Vml(&vml,"A2AARGS_R") );
    vml.Destroy();
  }
  free(a2aparams_l_buf); free(a2aparams_r_buf);

  
  //Setup the data buffer
  this->setup(read_a2a_args_l,read_a2a_args_r,tl,tr);

  assert(read_fsize == fsize);

  if(!UniqueID()){
    //Node 0 finish reading data
    FP_FORMAT dataformat = FPformat<ScalarComplexType>::get();
    FPConv conv;
    
    conv.setHostFormat(dataformat);	
    FP_FORMAT fileformat = conv.setFileFormat(dformatbuf);    
    assert(conv.size(fileformat) == conv.size(dataformat));

    int dsize = conv.size(fileformat);

    std::string str;
    getline(file,str); assert(str == "BEGIN_DATA");

    std::streampos dstart = file.tellg();
    file.seekg(0,std::ios::end);
    std::streampos fend = file.tellg();
    file.seekg(dstart);
    
    static const int chunk = 32768; //32kb chunks
    assert(chunk % dsize == 0);
    int fdinchunk = chunk/dsize;
    char *rbuf = (char *)malloc(chunk * sizeof(char)); //leave room for auto null char
      
    char *dptr = (char *)mf;

    int off = 0;
    int nfd = 2*fsize;
    while(off < nfd){
      int grab = std::min(nfd-off, fdinchunk); //How many data elements to grab
      int grabchars = grab * dsize;

      file.read(rbuf,grabchars);
      int got = file.gcount();
      
      if(file.gcount() != grabchars)
	ERR.General("","","Only managed to read %d chars, needed %d\n",file.gcount(),grabchars);
      
      conv.file2host(dptr,rbuf,grab);

      off += grab;
      dptr += grabchars;
    }
    free(rbuf);

    file.ignore(1); //newline
    getline(file,str); assert(str == "END_DATA");
    file.close();
  }

  //Broadcast data
  ret = MPI_Bcast(mf, 2*fsize*sizeof(typename ScalarComplexType::value_type) , MPI_CHAR, head_mpi_rank, MPI_COMM_WORLD);
  if(ret != MPI_SUCCESS) ERR.General("A2AmesonField","read","Squirt data fail\n");

  //Every node do the checksum
  FPConv conv;
  FP_FORMAT dataformat = FPformat<ScalarComplexType>::get();
  assert( conv.checksum((char*)mf, 2*fsize, dataformat) == checksum );  
}


#endif
