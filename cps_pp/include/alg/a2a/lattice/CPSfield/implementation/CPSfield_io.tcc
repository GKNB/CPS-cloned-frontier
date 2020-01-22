//Parallel write
template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy>
void CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy>::writeParallel(std::ostream &file, FP_FORMAT fileformat, CPSfield_checksumType cksumtype) const{
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

  //Header
  file << "BEGIN_HEADER\n";

#define HEADER_VERSION 2
  
#if HEADER_VERSION == 2
  file << "HDR_VERSION = 2\n";
  file << "DATA_FORMAT = " << conv.name(fileformat) << '\n';
  file << "BASE_FLOATBYTES = " << dsize << '\n';
  file << "SITETYPE_NFLOATS = " << nd_in_sitetype << '\n';
  file << "CHECKSUM_TYPE = " << checksumTypeToString(cksumtype) << '\n';

  if(cksumtype == checksumBasic){
    unsigned int checksum = conv.checksum( (char*)f, nd_in_sitetype*fsize, dataformat); //assumes complex or real SiteType
    file << "CHECKSUM = " << checksum << "\n";
  }else{
    uint32_t checksum = conv.checksumCRC32( (char*)f, nd_in_sitetype*fsize, dataformat);
    file << "CHECKSUM = " << checksum << "\n";
  }
  
  file << "NODE_ID = " << UniqueID() << "\n";
  file << "NODE_COOR =";
  for(int i=0;i<5;i++) file << " " << GJP.NodeCoor(i);
  file << "\n";  
  file << "NODE_SITES =";
  for(int i=0;i<5;i++) file << " " << GJP.NodeSites(i);
  file << "\n";
  file << "NODE_GEOMETRY =";
  for(int i=0;i<5;i++) file << " " << GJP.Nodes(i);
  file << '\n';
#else

  unsigned int checksum = conv.checksum( (char*)f, nd_in_sitetype*fsize, dataformat); //assumes complex or real SiteType
  
  //Header
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

#endif

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
  char* wbuf = (char*)malloc_check(chunk * sizeof(char)); 
      
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
void CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy>::writeParallel(const std::string &file_stub, FP_FORMAT fileformat, CPSfield_checksumType cksumtype) const{
  std::ostringstream os; os << file_stub << "." << UniqueID();
  std::ofstream of(os.str().c_str(),std::ofstream::out);
  writeParallel(of,fileformat,cksumtype);
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
  uint32_t checksumcrc32;
  CPSfield_checksumType checksumtype;
  
  int rd_dsize, rd_nd_in_sitetype;
  
  //Header
  getline(file,str); assert(str == "BEGIN_HEADER");
  
  int header_version;
  getline(file,str); assert( sscanf(str.c_str(),"HDR_VERSION = %d", &header_version) == 1 );

  if(header_version == 1){
    getline(file,str); assert( sscanf(str.c_str(),"DATA_FORMAT = %s",dformatbuf) == 1 );
    getline(file,str); assert( sscanf(str.c_str(),"BASE_FLOATBYTES = %d",&rd_dsize) == 1 ); 
    getline(file,str); assert( sscanf(str.c_str(),"SITETYPE_NFLOATS = %d",&rd_nd_in_sitetype) == 1 ); 
    getline(file,str); assert( sscanf(str.c_str(),"CHECKSUM = %u",&checksum) == 1 );

    checksumtype = checksumBasic;
    
    int rd_node;
    getline(file,str); assert( sscanf(str.c_str(),"NODE_ID = %d",&rd_node) == 1 ); 
    assert(rd_node == UniqueID());

    int node_coor[5];
    getline(file,str); assert( sscanf(str.c_str(),"NODE_COOR = %d %d %d %d %d",&node_coor[0],&node_coor[1],&node_coor[2],&node_coor[3],&node_coor[4]) == 5 );  
    for(int i=0;i<5;i++) assert(node_coor[i] == GJP.NodeCoor(i));

    int node_sites[5];
    getline(file,str); assert( sscanf(str.c_str(),"NODE_SITES = %d %d %d %d %d",&node_sites[0],&node_sites[1],&node_sites[2],&node_sites[3],&node_sites[4]) == 5 ); 
    for(int i=0;i<5;i++) assert(node_sites[i] == GJP.NodeSites(i));
  
  }else if(header_version == 2){

    getline(file,str); assert( sscanf(str.c_str(),"DATA_FORMAT = %s",dformatbuf) == 1 );
    getline(file,str); assert( sscanf(str.c_str(),"BASE_FLOATBYTES = %d",&rd_dsize) == 1 ); 
    getline(file,str); assert( sscanf(str.c_str(),"SITETYPE_NFLOATS = %d",&rd_nd_in_sitetype) == 1 );

    char cksumtypebuf[256];
    getline(file,str); assert( sscanf(str.c_str(),"CHECKSUM_TYPE = %s",cksumtypebuf) == 1 );
    checksumtype = checksumTypeFromString(std::string(cksumtypebuf));

    getline(file,str);
    if(checksumtype == checksumBasic){
      assert( sscanf(str.c_str(),"CHECKSUM = %u",&checksum) == 1 );
    }else{
      //assert( sscanf(str.c_str(),"CHECKSUM = %" SCNu32,&checksumcrc32) == 1 );
      char cksumbuf[256];
      assert( sscanf(str.c_str(),"CHECKSUM = %s",cksumbuf) == 1 );
      std::stringstream ss; ss << cksumbuf; ss >> checksumcrc32;
    }
      
    int rd_node;
    getline(file,str); assert( sscanf(str.c_str(),"NODE_ID = %d",&rd_node) == 1 ); 
    assert(rd_node == UniqueID());

    int node_coor[5];
    getline(file,str); assert( sscanf(str.c_str(),"NODE_COOR = %d %d %d %d %d",&node_coor[0],&node_coor[1],&node_coor[2],&node_coor[3],&node_coor[4]) == 5 );  
    for(int i=0;i<5;i++) assert(node_coor[i] == GJP.NodeCoor(i));

    int node_sites[5];
    getline(file,str); assert( sscanf(str.c_str(),"NODE_SITES = %d %d %d %d %d",&node_sites[0],&node_sites[1],&node_sites[2],&node_sites[3],&node_sites[4]) == 5 ); 
    for(int i=0;i<5;i++) assert(node_sites[i] == GJP.NodeSites(i));

    int node_geom[5];
    getline(file,str); assert( sscanf(str.c_str(),"NODE_GEOMETRY = %d %d %d %d %d",&node_geom[0],&node_geom[1],&node_geom[2],&node_geom[3],&node_geom[4]) == 5 ); 
    for(int i=0;i<5;i++) assert(node_geom[i] == GJP.Nodes(i));
    
  }else{
    ERR.General("CPSfield","readParallel","Unknown header version %d\n",header_version);
  }
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
  char *rbuf = (char *)malloc_check(chunk * sizeof(char)); //leave room for auto null char
      
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
  if(checksumtype == checksumBasic){
    unsigned int calc_cksum = conv.checksum((char*)f, nd_in_sitetype*fsize, dataformat);
    assert( calc_cksum == checksum );
  }else{
    uint32_t calc_cksum = conv.checksumCRC32((char*)f, nd_in_sitetype*fsize, dataformat);
    assert( calc_cksum == checksumcrc32 );
  }

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



				  
struct nodeCoor{
  int uid;
  int coor[5];

  void set(){
    uid = UniqueID();
    for(int i=0;i<5;i++) coor[i] = GJP.NodeCoor(i);
  }
};				  

template< typename SiteType, int SiteSize, typename MappingPolicy>
class writeParallelSeparateMetadata{};


template< typename SiteType, int SiteSize>
class writeParallelSeparateMetadata<SiteType,SiteSize,FourDpolicy<DynamicFlavorPolicy> >{ //4d lexicographic with flavors set by GJP.Gparity()
  typedef FourDpolicy<DynamicFlavorPolicy> MappingPolicy;
  typedef FourDSIMDPolicy<DynamicFlavorPolicy> SIMDmappingPolicy;
  FP_FORMAT fileformat;
  FP_FORMAT dataformat;
  FPConv conv;

  int nodes;
  int float_bytes; //underlying floating point data type
  int floats_per_site; //size of site data in units of floating point type
public:
  writeParallelSeparateMetadata(const FP_FORMAT _fileformat): fileformat(_fileformat), dataformat(FPformat<SiteType>::get()){
    if(fileformat == FP_AUTOMATIC)
      fileformat = dataformat;
    else
      assert(conv.size(fileformat) == conv.size(dataformat));
    
    conv.setHostFormat(dataformat);
    conv.setFileFormat(fileformat);

    float_bytes = conv.size(dataformat); //underlying floating point data type
    assert(sizeof(SiteType) % float_bytes == 0);
    floats_per_site = SiteSize * sizeof(SiteType) / float_bytes;
    
    nodes = 1;
    for(int i=0;i<5;i++) nodes *= GJP.Nodes(i); 
  }

private:
  void writeGlobalMetadata(const std::string &filename) const{
    if(!UniqueID()){
      std::ofstream of(filename.c_str());
      assert(!of.fail());
      of.exceptions ( std::ofstream::failbit | std::ofstream::badbit );

      of << "NODES =";
      for(int i=0;i<5;i++) of << " " << GJP.Nodes(i);
      of << '\n';
    
      of << "NODE_SITES =";
      for(int i=0;i<5;i++) of << " " << GJP.NodeSites(i);
      of << "\n";

      of << "BASE_FLOATBYTES = " << float_bytes << '\n';
      of << "SITE_FLOATS = " << floats_per_site << '\n';
      of << "DATA_FORMAT = " << conv.name(fileformat) << '\n';
      of << "CHECKSUM_TYPE = checksumCRC32\n";
    }
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }

  //Gather CRC32 checksums *to head node only*
  template<typename AllocPolicy>
  void gatherNodeCRC32checksums(std::vector<uint32_t> &into, const CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy> &field) const{
#ifndef USE_MPI
    ERR.General("writeParallelSeparateMetadata","gatherNodeCRC32checksums","Requires MPI\n");
#else    
    int myrank = getMyMPIrank();
    int headrank = getHeadMPIrank();
    
    if(!UniqueID()) into.resize(nodes);

    struct nd{
      int uid;
      uint32_t checksum;

      nd(){
	uid = UniqueID();
      }
    };
    
    size_t fsites = field.nfsites(); //number of "sites" (including doubling due to second flavor if applicable)

    nd node_data;
    
    node_data.checksum = conv.checksumCRC32( (char*)field.ptr(), floats_per_site*fsites, dataformat);

    nd* recv_buf = myrank == headrank ? (nd*)malloc_check(nodes * sizeof(nd)) : NULL;
    
    int ret = MPI_Gather(&node_data, sizeof(nd), MPI_BYTE,
			 recv_buf, sizeof(nd), MPI_BYTE,
			 headrank, MPI_COMM_WORLD);
    if(ret != MPI_SUCCESS) ERR.General("writeParallelSeparateMetadata","gatherNodeCRC32checksums","Gather failed\n");

    //We have to allow for the possibility that the MPI_COMM_WORLD rank is not equal to UniqueID
    if(myrank == headrank){
      std::vector<int> uid_map(nodes);
      for(int i=0;i<nodes;i++) uid_map[ recv_buf[i].uid ] = i;
      for(int i=0;i<nodes;i++) into[i] = recv_buf[ uid_map[i] ].checksum;       
      free(recv_buf);
    }
#endif
  }
  
  void gatherNodeCoors(std::vector<nodeCoor> &into) const{
#ifndef USE_MPI
    ERR.General("writeParallelSeparateMetadata","gatherNodeCoors","Requires MPI\n");
#else
    int myrank = getMyMPIrank();
    int headrank = getHeadMPIrank();
    
    if(!UniqueID()) into.resize(nodes);
    
    nodeCoor mycoor;
    mycoor.set();

    nodeCoor* recv_buf = myrank == headrank ? (nodeCoor*)malloc_check(nodes * sizeof(nodeCoor)) : NULL;
        
    int ret = MPI_Gather(&mycoor, sizeof(nodeCoor), MPI_BYTE,
			 recv_buf, sizeof(nodeCoor), MPI_BYTE,
			 headrank, MPI_COMM_WORLD);
    if(ret != MPI_SUCCESS) ERR.General("writeParallelSeparateMetadata","gatherNodeCoors","Gather failed\n");

    if(myrank == headrank){
      //We have to allow for the possibility that the MPI_COMM_WORLD rank is not equal to UniqueID
      std::vector<int> uid_map(nodes);
      for(int i=0;i<nodes;i++) uid_map[ recv_buf[i].uid ] = i;
      for(int i=0;i<nodes;i++) into[i] = recv_buf[ uid_map[i] ]; 
      free(recv_buf);
    }
#endif
  }			  

  //Node index will be appended to file_stub
  template<typename AllocPolicy>
  void writeNodeData(const std::string &file_stub, const CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy> &field) const{
    std::ostringstream os; os << file_stub << '.' << UniqueID();
    
    std::ofstream file(os.str().c_str());
    assert(!file.fail());
    file.exceptions ( std::ofstream::failbit | std::ofstream::badbit );
    
    static const size_t chunk = 32768; //32kb chunks
    assert(chunk % float_bytes == 0);
    size_t fdinchunk = chunk/float_bytes;
    char* wbuf = (char*)malloc_check(chunk * sizeof(char)); 
      
    char const* dptr = (char const*)field.ptr();
    size_t fsites = field.nfsites();
    
    size_t off = 0;
    size_t nfd = size_t(floats_per_site)*size_t(fsites); //total number of floating point data in local volume including flavor doubling if appropriate
    while(off < nfd){ 
      size_t grab = std::min(nfd-off, fdinchunk); //How many data elements to grab
      size_t grabchars = grab * float_bytes;
      conv.host2file(wbuf,dptr,grab);
      file.write(wbuf,grabchars);
      off += grab;
      dptr += grabchars;
    }
    free(wbuf);
  }
  template<typename AllocPolicy>
  void writeNodeMetadataSingle(const std::string &filename, const CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy> &field) const{
    std::vector<nodeCoor> node_coors;
    gatherNodeCoors(node_coors);

    std::vector<uint32_t> node_checksums;
    gatherNodeCRC32checksums(node_checksums, field);

    if(!UniqueID()){
      std::ofstream of(filename.c_str());
      assert(!of.fail());
      of.exceptions ( std::ofstream::failbit | std::ofstream::badbit );
      
      for(int n=0;n<nodes;n++){
	// <node id> <xnodecoor> <ynodecoor> <znodecoor> <tnodecoor> <snodecoor> <checksum>
	of << node_coors[n].uid;
	for(int i=0;i<5;i++) of << " " << node_coors[n].coor[i];
	of << " " << node_checksums[n] << std::endl;	
      }
    }
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }

  void writeNodeMetadataMulti(const std::string &filename, const std::vector<std::vector<uint32_t> > node_checksums) const{
    std::vector<nodeCoor> node_coors;
    gatherNodeCoors(node_coors);

    int nfield = node_checksums.size();
    
    if(!UniqueID()){
      std::ofstream of(filename.c_str());
      assert(!of.fail());
      of.exceptions ( std::ofstream::failbit | std::ofstream::badbit );

      //number of fields is first entry in file
      of << nfield << std::endl;
      
      for(int n=0;n<nodes;n++){
	// <node id> <xnodecoor> <ynodecoor> <znodecoor> <tnodecoor> <snodecoor> <checksum 0> <checksum 1> ....
	of << node_coors[n].uid;
	for(int i=0;i<5;i++) of << " " << node_coors[n].coor[i];
	for(int f=0;f<nfield;f++) of << " " << node_checksums[f][n];
	of << std::endl;	
      }
    }
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }

  template<typename AllocPolicy>
  void writeNodeMetadataMulti(const std::string &filename, const std::vector<CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy> const*> &fields) const{
    int nfield = fields.size();
    
    std::vector<std::vector<uint32_t> > node_checksums(nfield);
    for(int f=0;f<nfield;f++)
      gatherNodeCRC32checksums(node_checksums[f], *fields[f]);

    writeNodeMetadataMulti(filename, node_checksums);
  }

    
public:
  template<typename AllocPolicy>
  void writeOneField(const std::string &path, const CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy> &field) const{
    //Create paths
    makedir(path);
    
    std::string checkpoint_path = path + "/checkpoint";
    makedir(checkpoint_path);

    writeGlobalMetadata(path + "/global_metadata.txt");
    writeNodeMetadataSingle(path + "/node_metadata.txt",field);
    writeNodeData(checkpoint_path + "/checkpoint",field);
  }
  template<typename AllocPolicy>
  void writeManyFields(const std::string &path, const std::vector<CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy> const*> &fields) const{
    //Create paths
    makedir(path);
    
    writeGlobalMetadata(path + "/global_metadata.txt");
    writeNodeMetadataMulti(path + "/node_metadata.txt",fields);

    for(int f=0;f<fields.size();f++){
      std::ostringstream checkpoint_path; checkpoint_path << path << "/checkpoint_" << f;
      makedir(checkpoint_path.str());
    
      writeNodeData(checkpoint_path.str() + "/checkpoint",*fields[f]);
    }
  }
  //version for SIMD fields. Here we convert to non-SIMD fields for IO
  template<typename SIMDsiteType,typename AllocPolicy>
  void writeManyFields(const std::string &path, const std::vector<CPSfield<SIMDsiteType,SiteSize,SIMDmappingPolicy,AllocPolicy> const*> &fields) const{
    //Create paths
    makedir(path);
    
    writeGlobalMetadata(path + "/global_metadata.txt");
    
    std::vector<std::vector<uint32_t> > node_checksums(fields.size());
    NullObject nul;
    CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy> nonsimd(nul);
    for(int f=0;f<fields.size();f++){
      nonsimd.importField(*fields[f]);
      gatherNodeCRC32checksums(node_checksums[f], nonsimd);
            
      std::ostringstream checkpoint_path; checkpoint_path << path << "/checkpoint_" << f;
      makedir(checkpoint_path.str());
    
      writeNodeData(checkpoint_path.str() + "/checkpoint",nonsimd);
    }
    writeNodeMetadataMulti(path + "/node_metadata.txt",node_checksums);    
  } 
  
};





template< typename SiteType, int SiteSize, typename MappingPolicy>
class readParallelSeparateMetadata{};


template< typename SiteType, int SiteSize>
class readParallelSeparateMetadata<SiteType,SiteSize,FourDpolicy<DynamicFlavorPolicy> >{ //4d lexicographic with flavors set by GJP.Gparity()
  typedef FourDpolicy<DynamicFlavorPolicy> MappingPolicy;
  typedef FourDSIMDPolicy<DynamicFlavorPolicy> SIMDmappingPolicy;
  
  FPConv conv;
  FP_FORMAT dataformat;
  FP_FORMAT fileformat;
  
  int nodes;
  int flavors;
  int float_bytes; //underlying floating point data type
  int floats_per_site; //size of site data in units of floating point type

  int nodes_in[5];
  int nodesites_in[5];
  int nodenum_in;
  int fsites_in; //number of sites * flavors in original local volume
public:
  readParallelSeparateMetadata(): dataformat(FPformat<SiteType>::get()){
    conv.setHostFormat(dataformat);
    
    float_bytes = conv.size(dataformat); //underlying floating point data type
    assert(sizeof(SiteType) % float_bytes == 0);
    floats_per_site = SiteSize * sizeof(SiteType) / float_bytes;
    
    nodes = 1;
    for(int i=0;i<5;i++) nodes *= GJP.Nodes(i);

    flavors = GJP.Gparity() + 1;
  }

private:
  void readGlobalMetadata(const std::string &filename){ //all nodes read this
    std::ifstream file(filename.c_str());
    assert(!file.fail());
    file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );

    std::string str;
    
    getline(file,str); assert( sscanf(str.c_str(),"NODES = %d %d %d %d %d",&nodes_in[0],&nodes_in[1],&nodes_in[2],&nodes_in[3],&nodes_in[4]) == 5 ); 
    getline(file,str); assert( sscanf(str.c_str(),"NODE_SITES = %d %d %d %d %d",&nodesites_in[0],&nodesites_in[1],&nodesites_in[2],&nodesites_in[3],&nodesites_in[4]) == 5 ); 

    nodenum_in = 1;
    for(int i=0;i<5;i++) nodenum_in *= nodes_in[i]; 
    
    fsites_in = flavors;
    for(int i=0;i<MappingPolicy::EuclideanDimension;i++) fsites_in *= nodesites_in[i];
    
    int rd_float_bytes;
    getline(file,str); assert( sscanf(str.c_str(),"BASE_FLOATBYTES = %d",&rd_float_bytes) == 1 );
    assert(rd_float_bytes == float_bytes);

    int rd_floats_per_site;
    getline(file,str); assert( sscanf(str.c_str(),"SITE_FLOATS = %d",&rd_floats_per_site) == 1 );
    assert(rd_floats_per_site == floats_per_site);

    char dformatbuf[256];
    getline(file,str); assert( sscanf(str.c_str(),"DATA_FORMAT = %s",dformatbuf) == 1 );
    fileformat = conv.setFileFormat(dformatbuf);  //read file in whatever format specified in header
    assert(conv.size(fileformat) == conv.size(dataformat));
    
    char cksumtypebuf[256];
    getline(file,str); assert( sscanf(str.c_str(),"CHECKSUM_TYPE = %s",cksumtypebuf) == 1 );
    assert(std::string(cksumtypebuf) == "checksumCRC32");
  }
  

  void readNodeMetadataSingle(std::vector<nodeCoor> &node_coors, std::vector<uint32_t> &checksums,  const std::string &filename) const{ //all nodes read this too
    std::ifstream file(filename.c_str());
    assert(!file.fail());
    file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );

    node_coors.resize(nodenum_in);
    checksums.resize(nodenum_in);
    
    for(int n=0;n<nodenum_in;n++){
      file >> node_coors[n].uid;
      for(int i=0;i<5;i++) file >> node_coors[n].coor[i];
      file >> checksums[n];
    }
  }
  //checksums indexed [field idx][node]
  //returns number of fields
  int readNodeMetadataMulti(std::vector<nodeCoor> &node_coors, std::vector<std::vector<uint32_t> > &checksums,  const std::string &filename) const{ //all nodes read this too
    std::ifstream file(filename.c_str());
    assert(!file.fail());
    file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );

    int nfield;
    file >> nfield;

    checksums.resize(nfield);
    for(int i=0;i<nfield;i++) checksums[i].resize(nodenum_in);
    
    node_coors.resize(nodenum_in);
    
    for(int n=0;n<nodenum_in;n++){
      file >> node_coors[n].uid;
      for(int i=0;i<5;i++) file >> node_coors[n].coor[i];
      for(int f=0;f<nfield;f++) file >> checksums[f][n];
    }
    return nfield;
  }
  
  std::pair<char*,size_t> readNodeData(const std::string &file_stub, const int node_idx_in, const uint32_t checksum) const{
    std::ostringstream os; os << file_stub << '.' << node_idx_in;
    
    std::ifstream file(os.str().c_str());
    assert(!file.fail());
    file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
    
    static const int chunk = 32768; //32kb chunks
    assert(chunk % float_bytes == 0);
    size_t fdinchunk = chunk/float_bytes;
    char *rbuf = (char *)malloc_check(chunk * sizeof(char)); //leave room for auto null char

    size_t fsize = size_t(fsites_in) * size_t(floats_per_site) * size_t(float_bytes);
    char *dptr = (char *)malloc_check(fsize * sizeof(char));
    
    size_t off = 0;
    size_t nfd = size_t(floats_per_site) * fsites_in;
		       
    char *dptr_rd = dptr;
    while(off < nfd){
      size_t grab = std::min(nfd-off, fdinchunk); //How many data elements to grab
      size_t grabchars = grab * float_bytes;

      file.read(rbuf,grabchars);
      assert(!file.bad());
      size_t got = file.gcount();
      
      if(file.gcount() != grabchars)
	ERR.General("","","Only managed to read %d chars, needed %d\n",file.gcount(),grabchars);

      conv.file2host(dptr_rd,rbuf,grab);
      
      off += grab;
      dptr_rd += grabchars;
    }
    free(rbuf);
    
    uint32_t calc_cksum = conv.checksumCRC32(dptr, nfd, dataformat);
    if(calc_cksum != checksum){
      std::ostringstream os; os << "Node " << UniqueID() << " computed checksum " << calc_cksum << " doesn't match stored value " << checksum << '\n';      
      ERR.General("readParallelSeparateMetadata","readNodeData","%s",os.str().c_str());
    }
    
    return std::pair<char*,size_t>(dptr,fsize);
  }

  bool sameLayout() const{
    if(nodenum_in != nodes) return false;
    for(int i=0;i<5;i++){
      if(nodes_in[i] != GJP.Nodes(i)) return false;
      if(nodesites_in[i] != GJP.NodeSites(i)) return false;
    }
    return true;
  }
  bool sameCoordinate(const nodeCoor &coor){
    if(UniqueID() != coor.uid) return false;
    for(int i=0;i<5;i++)
      if(coor.coor[i] != GJP.NodeCoor(i)) return false;
    return true;
  }
  
public:
  template<typename AllocPolicy>
  void readOneField(CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy> &field, const std::string &path){
    std::string checkpoint_path = path + "/checkpoint";
    readGlobalMetadata(path + "/global_metadata.txt");
    
    std::vector<nodeCoor> node_coors;
    std::vector<uint32_t> checksums;
    readNodeMetadataSingle(node_coors, checksums, path + "/node_metadata.txt");

    if(sameLayout() && sameCoordinate(node_coors[UniqueID()])){ //easiest if node layout same as in original files
      std::pair<char*,size_t> data = readNodeData(checkpoint_path + "/checkpoint", UniqueID(), checksums[UniqueID()]);
      memcpy(field.ptr(), data.first, data.second);
      free(data.first);
    }else{
      ERR.General("readParallelSeparateMetadata","readOneField","Not yet implemented for layouts different from original\n");
    }      
  }
  template<typename AllocPolicy>
  void readManyFields(std::vector<CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy>*> &fields, const std::string &path){
    readGlobalMetadata(path + "/global_metadata.txt");
    
    std::vector<nodeCoor> node_coors;
    std::vector<std::vector<uint32_t> > checksums; //[field idx][node]
    int nfield = readNodeMetadataMulti(node_coors, checksums, path + "/node_metadata.txt");
    if(nfield != fields.size()) ERR.General("readParallelSeparateMetadata","readOneField","Path %s contains %d fields, not equal to the number of output fields provided, %d\n",path.c_str(),nfield,fields.size());
    
    if(sameLayout() && sameCoordinate(node_coors[UniqueID()])){ //easiest if node layout same as in original files
      for(int f=0;f<fields.size();f++){
	std::ostringstream checkpoint_path; checkpoint_path << path << "/checkpoint_" << f;
      
	std::pair<char*,size_t> data = readNodeData(checkpoint_path.str() + "/checkpoint", UniqueID(), checksums[f][UniqueID()]);
	memcpy(fields[f]->ptr(), data.first, data.second);
	free(data.first);
      }	
    }else{
      ERR.General("readParallelSeparateMetadata","readOneField","Not yet implemented for layouts different from original\n");
    }      
  }

  //SIMD version. Here we do IO in non-SIMD format
  template<typename SIMDsiteType,typename AllocPolicy>
  void readManyFields(std::vector<CPSfield<SIMDsiteType,SiteSize,SIMDmappingPolicy,AllocPolicy>*> &fields, const std::string &path){
    readGlobalMetadata(path + "/global_metadata.txt");
    
    std::vector<nodeCoor> node_coors;
    std::vector<std::vector<uint32_t> > checksums; //[field idx][node]
    int nfield = readNodeMetadataMulti(node_coors, checksums, path + "/node_metadata.txt");
    if(nfield != fields.size()) ERR.General("readParallelSeparateMetadata","readOneField","Path %s contains %d fields, not equal to the number of output fields provided, %d\n",path.c_str(),nfield,fields.size()); 
    
    if(sameLayout() && sameCoordinate(node_coors[UniqueID()])){ //easiest if node layout same as in original files
      NullObject nul;
      CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy> nonsimd(nul);
      for(int f=0;f<fields.size();f++){
	std::ostringstream checkpoint_path; checkpoint_path << path << "/checkpoint_" << f;
      
	std::pair<char*,size_t> data = readNodeData(checkpoint_path.str() + "/checkpoint", UniqueID(), checksums[f][UniqueID()]);
	memcpy(nonsimd.ptr(), data.first, data.second);
	free(data.first);

	fields[f]->importField(nonsimd);
      }	
    }else{
      ERR.General("readParallelSeparateMetadata","readOneField","Not yet implemented for layouts different from original\n");
    }      
  }
  
  
};
  


template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy>
void CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy>::writeParallelSeparateMetadata(const std::string &path, FP_FORMAT fileformat) const{
  cps::writeParallelSeparateMetadata<SiteType,SiteSize,MappingPolicy> wrt(fileformat);
  wrt.writeOneField(path, *this);
}
template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy>
void CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy>::readParallelSeparateMetadata(const std::string &path){
  cps::readParallelSeparateMetadata<SiteType,SiteSize,MappingPolicy> rd;
  rd.readOneField(*this, path);
}
