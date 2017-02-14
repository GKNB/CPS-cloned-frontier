template<typename IOType, typename FieldType>
struct A2Avector_IOconvert{
  inline static void write(std::ostream &file, FP_FORMAT fileformat, const FieldType &f, IOType &tmp){
    tmp.importField(f);
    tmp.writeParallel(file,fileformat);
  }
  inline static void read(std::istream &file, FieldType &f, IOType &tmp){    
    tmp.readParallel(file);
    f.importField(tmp);    
  }
  
};
template<typename IOType>
struct A2Avector_IOconvert<IOType,IOType>{
  inline static void write(std::ostream &file, FP_FORMAT fileformat, const IOType &f, IOType &tmp){
    f.writeParallel(file,fileformat);
  }
  inline static void read(std::istream &file, IOType &f, IOType &tmp){
    f.readParallel(file);
  }
};



template< typename mf_Policies>
void A2AvectorV<mf_Policies>::writeParallel(const std::string &file_stub, FP_FORMAT fileformat) const{
  std::ostringstream os; os << file_stub << "." << UniqueID();
  std::ofstream file(os.str().c_str(),std::ofstream::out);

  file << "BEGIN_HEADER\n";
  file << "HDR_VERSION = 1\n";  
  file << "END_HEADER\n";

  file << "BEGIN_PARAMS\n";
  char* a2aparams_buf = (char*)malloc(10000 * sizeof(char));
  {
    VML vml;
    vml.Create(a2aparams_buf,10000,VML_ENCODE);
    A2AArg a2a_args = this->getArgs();
    assert( a2a_args.Vml(&vml,"A2AARGS") );
    vml.Destroy();
  }
  file << "STRLEN_A2AARGS = " << strlen(a2aparams_buf) << '\n';
  file << a2aparams_buf;

  free(a2aparams_buf);

  file << "END_PARAMS\n";

  file << "BEGIN_VFIELDS\n";

  typedef CPSfermion4D<typename mf_Policies::ScalarComplexType, FourDpolicy, DynamicFlavorPolicy, StandardAllocPolicy> IOFieldType; //always write in non-SIMD format for portability
  IOFieldType tmp;
  
  for(int i=0;i<nv;i++)
    A2Avector_IOconvert<IOFieldType, FermionFieldType>::write(file, fileformat, *v[i], tmp);

  file << "END_VFIELDS\n";

  file.close();
}

  
template< typename mf_Policies>
void A2AvectorV<mf_Policies>::readParallel(const std::string &file_stub){
  std::ostringstream os; os << file_stub << "." << UniqueID();
  std::ifstream file(os.str().c_str(),std::ifstream::in); std::string str;

  getline(file,str); assert(str == "BEGIN_HEADER");
  getline(file,str); assert(str == "HDR_VERSION = 1");
  getline(file,str); assert(str == "END_HEADER");
  getline(file,str); assert(str == "BEGIN_PARAMS");

  int a2aparams_buflen;
  getline(file,str); assert( sscanf(str.c_str(),"STRLEN_A2AARGS = %d",&a2aparams_buflen) == 1 );
  ++a2aparams_buflen; //leave room for null char

  char* a2aparams_buf = (char*)malloc(a2aparams_buflen * sizeof(char));
  file.get(a2aparams_buf,a2aparams_buflen,EOF);
  
  A2AArg read_a2a_args;
  {
    VML vml;
    vml.Create(a2aparams_buf,a2aparams_buflen,VML_DECODE);
    assert( read_a2a_args.Vml(&vml,"A2AARGS") );
    vml.Destroy();
  }
  free(a2aparams_buf);
  
  A2Aparams tmp_params(read_a2a_args);
  static_cast<A2Aparams&>(*this) = tmp_params; //this is ugly but yeah...

  assert(v.size() > 0); //should have been setup with an A2Aargs such that the existing v size is non-zero. We can then get the field SIMD params from an existing vector
  
  typename FermionFieldType::InputParamType fparams = v[0]->getDimPolParams(); //field params, will be overwritten when field read. Hopefully this has sensible default values!  
  if(v.size() != nv){    
    v.resize(nv);
    this->allocInitializeFields(v,fparams);
  }

  getline(file,str); assert(str == "END_PARAMS");
  getline(file,str); assert(str == "BEGIN_VFIELDS");

  typedef CPSfermion4D<typename mf_Policies::ScalarComplexType, FourDpolicy, DynamicFlavorPolicy, StandardAllocPolicy> IOFieldType; //always write in non-SIMD format for portability
  IOFieldType tmp;
  
  for(int i=0;i<nv;i++)
    A2Avector_IOconvert<IOFieldType, FermionFieldType>::read(file, *v[i], tmp);    
  
  getline(file,str); assert(str == "END_VFIELDS");  
}



template< typename mf_Policies>
void A2AvectorW<mf_Policies>::writeParallel(const std::string &file_stub, FP_FORMAT fileformat) const{
  std::ostringstream os; os << file_stub << "." << UniqueID();
  std::ofstream file(os.str().c_str(),std::ofstream::out);

  file << "BEGIN_HEADER\n";
  file << "HDR_VERSION = 1\n";  
  file << "END_HEADER\n";

  file << "BEGIN_PARAMS\n";
  char* a2aparams_buf = (char*)malloc(10000 * sizeof(char));
  {
    VML vml;
    vml.Create(a2aparams_buf,10000,VML_ENCODE);
    A2AArg a2a_args = this->getArgs();
    assert( a2a_args.Vml(&vml,"A2AARGS") );
    vml.Destroy();
  }
  file << "STRLEN_A2AARGS = " << strlen(a2aparams_buf) << '\n';
  file << a2aparams_buf;

  free(a2aparams_buf);

  file << "END_PARAMS\n";

  file << "BEGIN_WLFIELDS\n";

  typedef CPSfermion4D<typename mf_Policies::ScalarComplexType, FourDpolicy, DynamicFlavorPolicy, StandardAllocPolicy> WlIOFieldType; //always write in non-SIMD format for portability
  WlIOFieldType tmp;
  
  for(int i=0;i<nl;i++)
    A2Avector_IOconvert<WlIOFieldType, FermionFieldType>::write(file, fileformat, *wl[i], tmp);

  file << "END_WLFIELDS\n";

  file << "BEGIN_WHFIELDS\n";

  typedef CPScomplex4D<typename mf_Policies::ScalarComplexType, FourDpolicy, DynamicFlavorPolicy, StandardAllocPolicy> WhIOFieldType;
  WhIOFieldType tmp2;
  
  for(int i=0;i<nhits;i++){
    A2Avector_IOconvert<WhIOFieldType, ComplexFieldType>::write(file, fileformat, *wh[i], tmp2);
  }
  file << "END_WHFIELDS\n";
  
  file.close();
}

  
template< typename mf_Policies>
void A2AvectorW<mf_Policies>::readParallel(const std::string &file_stub){
  std::ostringstream os; os << file_stub << "." << UniqueID();
  std::ifstream file(os.str().c_str(),std::ifstream::in); std::string str;

  getline(file,str); assert(str == "BEGIN_HEADER");
  getline(file,str); assert(str == "HDR_VERSION = 1");
  getline(file,str); assert(str == "END_HEADER");
  getline(file,str); assert(str == "BEGIN_PARAMS");

  int a2aparams_buflen;
  getline(file,str); assert( sscanf(str.c_str(),"STRLEN_A2AARGS = %d",&a2aparams_buflen) == 1 );
  ++a2aparams_buflen; //leave room for null char

  char* a2aparams_buf = (char*)malloc(a2aparams_buflen * sizeof(char));
  file.get(a2aparams_buf,a2aparams_buflen,EOF);
  
  A2AArg read_a2a_args;
  {
    VML vml;
    vml.Create(a2aparams_buf,a2aparams_buflen,VML_DECODE);
    assert( read_a2a_args.Vml(&vml,"A2AARGS") );
    vml.Destroy();
  }
  free(a2aparams_buf);
  
  A2Aparams tmp_params(read_a2a_args);
  static_cast<A2Aparams&>(*this) = tmp_params; //this is ugly but yeah...

  assert(wl.size() > 0 || wh.size() > 0); //should have been setup with an A2Aargs such that the existing v size is non-zero. We can then get the field SIMD params from an existing vector

  typename FermionFieldType::InputParamType fparams = (wl.size() > 0 ? wl[0]->getDimPolParams() : wh[0]->getDimPolParams());
  if(wl.size() != nl){    
    wl.resize(nl);
    this->allocInitializeLowModeFields(wl,fparams);
  }
  if(wh.size() != nhits){    
    wh.resize(nhits);
    this->allocInitializeHighModeFields(wh,fparams);
  }
  
  getline(file,str); assert(str == "END_PARAMS");
  getline(file,str); assert(str == "BEGIN_WLFIELDS");

  typedef CPSfermion4D<typename mf_Policies::ScalarComplexType, FourDpolicy, DynamicFlavorPolicy, StandardAllocPolicy> WlIOFieldType;
  WlIOFieldType tmp;
  
  for(int i=0;i<nl;i++)
    A2Avector_IOconvert<WlIOFieldType, FermionFieldType>::read(file, *wl[i], tmp);    
  
  getline(file,str); assert(str == "END_WLFIELDS");

  getline(file,str); assert(str == "BEGIN_WHFIELDS");

  typedef CPScomplex4D<typename mf_Policies::ScalarComplexType, FourDpolicy, DynamicFlavorPolicy, StandardAllocPolicy> WhIOFieldType;
  WhIOFieldType tmp2;
  
  for(int i=0;i<nhits;i++)
    A2Avector_IOconvert<WhIOFieldType, ComplexFieldType>::read(file, *wh[i], tmp2);    
  
  getline(file,str); assert(str == "END_WHFIELDS");  
}
