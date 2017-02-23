//Parallel write
template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy>
void CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy>::writeParallel(std::ostream &file, FP_FORMAT fileformat) const{
  assert(!file.fail());
  file.exceptions ( std::ofstream::failbit | std::ofstream::badbit );
    
#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  
  FP_FORMAT dataformat = FPformat<SiteType>::get();
  FPConv conv;
  if(fileformat == FP_AUTOMATIC)
    fileformat = dataformat;
  else
    assert(conv.size(fileformat) == conv.size(dataformat));
  
  conv.setHostFormat(dataformat);
  conv.setFileFormat(fileformat);
  
  const int dsize = conv.size(dataformat); //underlying floating point data type
  assert(sizeof(SiteType) % dsize == 0);
  const int nd_in_sitetype = sizeof(SiteType) / dsize;
  
  unsigned int checksum = conv.checksum( (char*)f, nd_in_sitetype*fsize, dataformat); //assumes complex or real SiteType
  
  //Header
  file << "BEGIN_HEADER\n";
  file << "HDR_VERSION = 1\n";
  file << "DATA_FORMAT = " << conv.name(fileformat) << '\n';
  file << "BASE_FLOATBYTES = " << dsize << '\n';
  file << "SITETYPE_NFLOATS = " << nd_in_sitetype << '\n';
  file << "CHECKSUM = " << checksum << "\n";
  file << "NODE_ID = " << UniqueID() << "\n";
  file << "NODE_COOR = ";
  for(int i=0;i<4;i++) file << GJP.NodeCoor(i) << " ";
  file << GJP.NodeCoor(4) << "\n";  
  file << "NODE_SITES = ";
  for(int i=0;i<4;i++) file << GJP.NodeSites(i) << " ";
  file << GJP.NodeSites(4) << "\n";    
  file << "END_HEADER\n";
  
  //Parameters    
  file << "BEGIN_PARAMS\n";
  file << "SITES = " << this->nsites() << "\n";
  file << "FLAVORS = " << this->nflavors() << "\n";
  file << "FSITES = " << this->nfsites() << "\n";
  file << "SITESIZE = " << SiteSize << "\n";
  file << "FSIZE = " << fsize << "\n";
  this->MappingPolicy::writeParams(file);
  this->AllocPolicy::writeParams(file);
  file << "END_PARAMS\n";      

  //Data
  file << "BEGIN_DATA\n";
      
  static const int chunk = 32768; //32kb chunks
  assert(chunk % dsize == 0);
  int fdinchunk = chunk/dsize;
  char* wbuf = (char*)malloc(chunk * sizeof(char)); 
      
  char const* dptr = (char const*)f;

  int off = 0;
  int nfd = nd_in_sitetype*fsize;
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

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}


template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy>
void CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy>::writeParallel(const std::string &file_stub, FP_FORMAT fileformat) const{
  std::ostringstream os; os << file_stub << "." << UniqueID();
  std::ofstream of(os.str().c_str(),std::ofstream::out);
  writeParallel(of,fileformat);
  of.close();
}



//Parallel read
template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy>
void CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy>::readParallel(std::istream &file){
#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  
  assert(!file.fail());
  file.exceptions ( std::ofstream::failbit | std::ofstream::badbit );

  assert(!file.fail());
  
  std::string str;
  char dformatbuf[256];
  unsigned int checksum;
  int rd_dsize, rd_nd_in_sitetype;
  
  //Header
  getline(file,str); assert(str == "BEGIN_HEADER");
  getline(file,str); assert(str == "HDR_VERSION = 1");
  getline(file,str); assert( sscanf(str.c_str(),"DATA_FORMAT = %s",dformatbuf) == 1 );
  getline(file,str); assert( sscanf(str.c_str(),"BASE_FLOATBYTES = %d",&rd_dsize) == 1 ); 
  getline(file,str); assert( sscanf(str.c_str(),"SITETYPE_NFLOATS = %d",&rd_nd_in_sitetype) == 1 ); 
  getline(file,str); assert( sscanf(str.c_str(),"CHECKSUM = %u",&checksum) == 1 );

  int rd_node;
  getline(file,str); assert( sscanf(str.c_str(),"NODE_ID = %d",&rd_node) == 1 ); 
  assert(rd_node == UniqueID());

  int node_coor[5];
  getline(file,str); assert( sscanf(str.c_str(),"NODE_COOR = %d %d %d %d %d",&node_coor[0],&node_coor[1],&node_coor[2],&node_coor[3],&node_coor[4]) == 5 );  
  for(int i=0;i<5;i++) assert(node_coor[i] == GJP.NodeCoor(i));

  int node_sites[5];
  getline(file,str); assert( sscanf(str.c_str(),"NODE_SITES = %d %d %d %d %d",&node_sites[0],&node_sites[1],&node_sites[2],&node_sites[3],&node_sites[4]) == 5 ); 
  for(int i=0;i<5;i++) assert(node_sites[i] == GJP.NodeSites(i));
  
  getline(file,str); assert(str == "END_HEADER");

  
  //Parameters
  getline(file,str); assert(str == "BEGIN_PARAMS");

  int rd_sites, rd_flavors, rd_fsites, rd_sitesize, rd_fsize;
  getline(file,str); assert( sscanf(str.c_str(),"SITES = %d",&rd_sites) == 1 );
  getline(file,str); assert( sscanf(str.c_str(),"FLAVORS = %d",&rd_flavors) == 1 ); 
  getline(file,str); assert( sscanf(str.c_str(),"FSITES = %d",&rd_fsites) == 1 );
  getline(file,str); assert( sscanf(str.c_str(),"SITESIZE = %d",&rd_sitesize) == 1 );
  getline(file,str); assert( sscanf(str.c_str(),"FSIZE = %d",&rd_fsize) == 1 );

  assert(rd_sitesize == SiteSize);

  //overwrite policy parameters and check
  this->MappingPolicy::readParams(file);
  this->AllocPolicy::readParams(file);

  getline(file,str); assert(str == "END_PARAMS");

  int current_fsize = fsize;
  
  //Write over current field params
  assert(this->nflavors() == rd_flavors);
  assert(this->nsites() == rd_sites && this->nfsites() == rd_fsites);
  
  this->fsize = this->nfsites() * SiteSize;
  
  if(fsize != current_fsize){ //reallocate if wrong size
    freemem();  
    alloc();
  }

  FP_FORMAT dataformat = FPformat<SiteType>::get();
  FPConv conv;
  
  conv.setHostFormat(dataformat);	
  FP_FORMAT fileformat = conv.setFileFormat(dformatbuf);  //read file in whatever format specified in header
  assert(conv.size(fileformat) == conv.size(dataformat));
  
  int dsize = conv.size(fileformat);
  assert(sizeof(SiteType) % dsize == 0);
  int nd_in_sitetype = sizeof(SiteType)/dsize;

  assert(dsize == rd_dsize);
  assert(nd_in_sitetype == rd_nd_in_sitetype);

  getline(file,str); assert(str == "BEGIN_DATA");

  static const int chunk = 32768; //32kb chunks
  assert(chunk % dsize == 0);
  int fdinchunk = chunk/dsize;
  char *rbuf = (char *)malloc(chunk * sizeof(char)); //leave room for auto null char
      
  char *dptr = (char *)f;

  int off = 0;
  int nfd = nd_in_sitetype*fsize;
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

  //Checksum
  unsigned int calc_cksum = conv.checksum((char*)f, nd_in_sitetype*fsize, dataformat);

  assert( calc_cksum == checksum );

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}

template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy>
void CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy>::readParallel(const std::string &file_stub){
  std::ostringstream os; os << file_stub << "." << UniqueID();
  std::ifstream ifs(os.str().c_str(),std::ifstream::in);
  readParallel(ifs);
  ifs.close();
}
