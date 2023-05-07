template<typename IOType, typename FieldType>
struct A2Avector_IOconvert{
  inline static void write(std::ostream &file, FP_FORMAT fileformat, CPSfield_checksumType cksumtype, const FieldType &f, IOType &tmp){
    tmp.importField(f);
    tmp.writeParallel(file,fileformat,cksumtype);
  }
  inline static void read(std::istream &file, FieldType &f, IOType &tmp){    
    tmp.readParallel(file);
    f.importField(tmp);    
  }
  
};
template<typename IOType>
struct A2Avector_IOconvert<IOType,IOType>{
  inline static void write(std::ostream &file, FP_FORMAT fileformat, CPSfield_checksumType cksumtype, const IOType &f, IOType &tmp){
    f.writeParallel(file,fileformat,cksumtype);
  }
  inline static void read(std::istream &file, IOType &f, IOType &tmp){
    f.readParallel(file);
  }
};



template< typename mf_Policies>
void A2AvectorV<mf_Policies>::writeParallel(const std::string &file_stub, FP_FORMAT fileformat, CPSfield_checksumType cksumtype) const{
  std::ostringstream os; os << file_stub << "." << UniqueID();
  std::ofstream file(os.str().c_str(),std::ofstream::out);

  file << "BEGIN_HEADER\n";
  file << "HDR_VERSION = 1\n";  
  file << "END_HEADER\n";

  file << "BEGIN_PARAMS\n";
  char* a2aparams_buf = (char*)malloc_check(10000 * sizeof(char));
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

  typedef CPSfermion4D<typename mf_Policies::ScalarComplexType, FourDpolicy<DynamicFlavorPolicy>, typename mf_Policies::AllocPolicy> IOFieldType; //always write in non-SIMD format for portability
  IOFieldType tmp;
  
  for(int i=0;i<nv;i++)
    A2Avector_IOconvert<IOFieldType, FermionFieldType>::write(file, fileformat, cksumtype, *v[i], tmp);

  file << "END_VFIELDS\n";

  file.close();
}


template< typename mf_Policies>
void A2AvectorV<mf_Policies>::writeParallelWithGrid(const std::string &file_stub) const
{
#ifndef HAVE_LIME
  ERR.General("A2AvectorV","writeParallelWithGrid","Requires LIME");
#else 
  std::string info_file = file_stub + "_info.xml";
  std::string data_file = file_stub + "_data.scidac";

  if(!UniqueID())
  {
    Grid::XmlWriter WRx(info_file);
    write(WRx, "data_length", v.size());

    char* a2aparams_buf = (char*)malloc_check(10000 * sizeof(char));
    {
      VML vml;
      vml.Create(a2aparams_buf,10000,VML_ENCODE);
      A2AArg a2a_args = this->getArgs();
      assert( a2a_args.Vml(&vml,"A2AARGS") );
      vml.Destroy();
    }
    write(WRx, "a2a_args", std::string(a2aparams_buf));
    free(a2aparams_buf);
  }

  Grid::GridCartesian *UGrid = FgridBase::getUGrid();
  Grid::emptyUserRecord record;
  Grid::ScidacWriter WR(UGrid->IsBoss());
  WR.open(data_file);
  for(int k=0;k<v.size();k++)
  {
    typename mf_Policies::GridFermionField grid_rep(UGrid);
    v[k]->exportGridField(grid_rep);
    WR.writeScidacFieldRecord(grid_rep,record);	  
  }
  WR.close();
#endif
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

  char* a2aparams_buf = (char*)malloc_check(a2aparams_buflen * sizeof(char));
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

  typedef CPSfermion4D<typename mf_Policies::ScalarComplexType, FourDpolicy<DynamicFlavorPolicy>, typename mf_Policies::AllocPolicy> IOFieldType; //always write in non-SIMD format for portability
  IOFieldType tmp;
  
  for(int i=0;i<nv;i++)
    A2Avector_IOconvert<IOFieldType, FermionFieldType>::read(file, *v[i], tmp);    
  
  getline(file,str); assert(str == "END_VFIELDS");  
}


template<typename mf_Policies>
void A2AvectorV<mf_Policies>::readParallelWithGrid(const std::string &file_stub)
{
#ifndef HAVE_LIME
  ERR.General("A2AvectorV","readParallelWithGrid","Requires LIME");
#else 
  std::string info_file = file_stub + "_info.xml";
  std::string data_file = file_stub + "_data.scidac";

  Grid::XmlReader RDx(info_file);
  int data_len = -1;
  read(RDx,"data_length", data_len);

  std::string a2aparams_str;
  read(RDx, "a2a_args", a2aparams_str);

  if(!UniqueID())
  {
    char* a2aparams_buf_this = (char*)malloc_check(10000 * sizeof(char));
    {
      VML vml;
      vml.Create(a2aparams_buf_this,10000,VML_ENCODE);
      A2AArg a2a_args = this->getArgs();
      assert( a2a_args.Vml(&vml,"A2AARGS") );
      vml.Destroy();
    }
    assert(a2aparams_str == std::string(a2aparams_buf_this) && "Error: args saved on disk and args used for initialize V&W are different!");
    free(a2aparams_buf_this);
  }

  assert(v.size() == data_len && "Error: size suggested by a2a_arg is different from that in the saved file");

  Grid::GridCartesian *UGrid = FgridBase::getUGrid();
  Grid::emptyUserRecord record;
  Grid::ScidacReader RD;
  RD.open(data_file);
  for(int k=0;k<data_len;k++)
  {
    typename mf_Policies::GridFermionField grid_rep(UGrid);
    RD.readScidacFieldRecord(grid_rep,record);
    v[k]->importGridField(grid_rep);
  }
  RD.close();
#endif
}

template< typename mf_Policies>
void A2AvectorW<mf_Policies>::writeParallel(const std::string &file_stub, FP_FORMAT fileformat, CPSfield_checksumType cksumtype) const{
  std::ostringstream os; os << file_stub << "." << UniqueID();
  std::ofstream file(os.str().c_str(),std::ofstream::out);

  file << "BEGIN_HEADER\n";
  file << "HDR_VERSION = 1\n";  
  file << "END_HEADER\n";

  file << "BEGIN_PARAMS\n";
  char* a2aparams_buf = (char*)malloc_check(10000 * sizeof(char));
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

  typedef CPSfermion4D<typename mf_Policies::ScalarComplexType, FourDpolicy<DynamicFlavorPolicy>, typename mf_Policies::AllocPolicy> WlIOFieldType; //always write in non-SIMD format for portability
  WlIOFieldType tmp;
  
  for(int i=0;i<nl;i++)
    A2Avector_IOconvert<WlIOFieldType, FermionFieldType>::write(file, fileformat, cksumtype, *wl[i], tmp);

  file << "END_WLFIELDS\n";

  file << "BEGIN_WHFIELDS\n";

  typedef CPScomplex4D<typename mf_Policies::ScalarComplexType, FourDpolicy<DynamicFlavorPolicy>, typename mf_Policies::AllocPolicy> WhIOFieldType;
  WhIOFieldType tmp2;
  
  for(int i=0;i<nhits;i++){
    A2Avector_IOconvert<WhIOFieldType, ComplexFieldType>::write(file, fileformat, cksumtype, *wh[i], tmp2);
  }
  file << "END_WHFIELDS\n";
  
  file.close();
}

 
template< typename mf_Policies>
void A2AvectorW<mf_Policies>::writeParallelWithGrid(const std::string &file_stub) const
{
#ifndef HAVE_LIME
  ERR.General("A2AvectorW","writeParallelWithGrid","Requires LIME");
#else 
  std::string info_file = file_stub + "_info.xml";
  std::string data_file = file_stub + "_data.scidac";

  if(!UniqueID())
  {
    Grid::XmlWriter WRx(info_file);
    write(WRx, "low_mode_data_length", wl.size());
    write(WRx, "high_mode_data_length", wh.size());

    char* a2aparams_buf = (char*)malloc_check(10000 * sizeof(char));
    {
      VML vml;
      vml.Create(a2aparams_buf,10000,VML_ENCODE);
      A2AArg a2a_args = this->getArgs();
      assert( a2a_args.Vml(&vml,"A2AARGS") );
      vml.Destroy();
    }
    write(WRx, "a2a_args", std::string(a2aparams_buf));
    free(a2aparams_buf);
  }

  Grid::GridCartesian *UGrid = FgridBase::getUGrid();
  Grid::emptyUserRecord record;
  Grid::ScidacWriter WR(UGrid->IsBoss());
  WR.open(data_file);
  for(int k=0; k<wl.size();k++)
  {
    typename mf_Policies::GridFermionField grid_rep(UGrid);
    wl[k]->exportGridField(grid_rep);
    WR.writeScidacFieldRecord(grid_rep,record);	  
  }
  for(int k=0; k<wh.size(); k++)
  {
    typename mf_Policies::GridComplexField grid_rep(UGrid);
    wh[k]->exportGridField(grid_rep);
    WR.writeScidacFieldRecord(grid_rep,record);
  }
  WR.close();
#endif
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

  char* a2aparams_buf = (char*)malloc_check(a2aparams_buflen * sizeof(char));
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

  typedef CPSfermion4D<typename mf_Policies::ScalarComplexType, FourDpolicy<DynamicFlavorPolicy>, typename mf_Policies::AllocPolicy> WlIOFieldType;
  WlIOFieldType tmp;
  
  for(int i=0;i<nl;i++)
    A2Avector_IOconvert<WlIOFieldType, FermionFieldType>::read(file, *wl[i], tmp);    
  
  getline(file,str); assert(str == "END_WLFIELDS");

  getline(file,str); assert(str == "BEGIN_WHFIELDS");

  typedef CPScomplex4D<typename mf_Policies::ScalarComplexType, FourDpolicy<DynamicFlavorPolicy>, typename mf_Policies::AllocPolicy> WhIOFieldType;
  WhIOFieldType tmp2;
  
  for(int i=0;i<nhits;i++)
    A2Avector_IOconvert<WhIOFieldType, ComplexFieldType>::read(file, *wh[i], tmp2);    
  
  getline(file,str); assert(str == "END_WHFIELDS");  
}


template<typename mf_Policies>
void A2AvectorW<mf_Policies>::readParallelWithGrid(const std::string &file_stub)
{
#ifndef HAVE_LIME
  ERR.General("A2AvectorW","readParallelWithGrid","Requires LIME");
#else 
  std::string info_file = file_stub + "_info.xml";
  std::string data_file = file_stub + "_data.scidac";

  Grid::XmlReader RDx(info_file);
  int low_mode_data_len = -1;
  read(RDx,"low_mode_data_length", low_mode_data_len);
  int high_mode_data_len = -1;
  read(RDx,"high_mode_data_length", high_mode_data_len);

  std::string a2aparams_str;
  read(RDx, "a2a_args", a2aparams_str);

  if(!UniqueID())
  {
    char* a2aparams_buf_this = (char*)malloc_check(10000 * sizeof(char));
    {
      VML vml;
      vml.Create(a2aparams_buf_this,10000,VML_ENCODE);
      A2AArg a2a_args = this->getArgs();
      assert( a2a_args.Vml(&vml,"A2AARGS") );
      vml.Destroy();
    }
    assert(a2aparams_str == std::string(a2aparams_buf_this) && "Error: args saved on disk and args used for initialize V&W are different!");
    free(a2aparams_buf_this);
  }

  assert(wl.size() == low_mode_data_len && "Error: size suggested by a2a_arg for wl is different from that in the saved file");
  assert(wh.size() == high_mode_data_len && "Error: size suggested by a2a_arg for wh is different from that in the saved file");

  Grid::GridCartesian *UGrid = FgridBase::getUGrid();
  Grid::emptyUserRecord record;
  Grid::ScidacReader RD;
  RD.open(data_file);
  for(int k=0;k<low_mode_data_len;k++)
  {
    typename mf_Policies::GridFermionField grid_rep(UGrid);
    RD.readScidacFieldRecord(grid_rep,record);
    wl[k]->importGridField(grid_rep);
  }
  for(int k=0;k<high_mode_data_len;k++)
  {
    typename mf_Policies::GridComplexField grid_rep(UGrid);
    RD.readScidacFieldRecord(grid_rep,record);
    wh[k]->importGridField(grid_rep);
  }
  RD.close();
#endif
}



//Write V/W fields to a format with metadata and binary data separate. User provides a unique directory path. Directory is created if doesn't already exist
template< typename mf_Policies>
void A2AvectorV<mf_Policies>::writeParallelSeparateMetadata(const std::string &path, FP_FORMAT fileformat) const{
  typedef typename baseCPSfieldType<FermionFieldType>::type baseField;

  std::vector<baseField const*> ptrs_wr(v.size());
  for(int i=0;i<v.size();i++){
    assert(v[i].assigned());
    ptrs_wr[i] = v[i].ptr();
  }
  
  cps::writeParallelSeparateMetadata<typename mf_Policies::ScalarComplexType, baseField::FieldSiteSize,
				     typename getScalarMappingPolicy<typename baseField::FieldMappingPolicy>::type> wr(fileformat);

  wr.writeManyFields(path, ptrs_wr);
}

template< typename mf_Policies>
void A2AvectorV<mf_Policies>::readParallelSeparateMetadata(const std::string &path){
  typedef typename baseCPSfieldType<FermionFieldType>::type baseField;

  std::vector<baseField*> ptrs_rd(v.size());
  for(int i=0;i<v.size();i++){
    assert(v[i].assigned());
    ptrs_rd[i] = v[i].ptr();
  }
  
  cps::readParallelSeparateMetadata<typename mf_Policies::ScalarComplexType, baseField::FieldSiteSize,
				    typename getScalarMappingPolicy<typename baseField::FieldMappingPolicy>::type> rd;

  rd.readManyFields(ptrs_rd, path);
} 


template< typename mf_Policies>
void A2AvectorW<mf_Policies>::writeParallelSeparateMetadata(const std::string &path, FP_FORMAT fileformat) const{
  makedir(path);

  //Low and high mode fields not same type, so save in different subdirectories
  std::string Wlpath = path + "/wl";
  std::string Whpath = path + "/wh";
  
  typedef typename baseCPSfieldType<FermionFieldType>::type baseFieldL;
  typedef typename baseCPSfieldType<ComplexFieldType>::type baseFieldH;

  {
    std::vector<baseFieldL const*> ptrs_wr(wl.size());
    for(int i=0;i<wl.size();i++){
      assert(wl[i].assigned());
      ptrs_wr[i] = wl[i].ptr();
    }
  
    cps::writeParallelSeparateMetadata<typename mf_Policies::ScalarComplexType, baseFieldL::FieldSiteSize,
				       typename getScalarMappingPolicy<typename baseFieldL::FieldMappingPolicy>::type> wr(fileformat);

    wr.writeManyFields(Wlpath, ptrs_wr);
  }
  {
    std::vector<baseFieldH const*> ptrs_wr(wh.size());
    for(int i=0;i<wh.size();i++){
      assert(wh[i].assigned());
      ptrs_wr[i] = wh[i].ptr();
    }
  
    cps::writeParallelSeparateMetadata<typename mf_Policies::ScalarComplexType, baseFieldH::FieldSiteSize,
				       typename getScalarMappingPolicy<typename baseFieldH::FieldMappingPolicy>::type> wr(fileformat);

    wr.writeManyFields(Whpath, ptrs_wr);
  }
}

template< typename mf_Policies>
void A2AvectorW<mf_Policies>::readParallelSeparateMetadata(const std::string &path){
  std::string Wlpath = path + "/wl";
  std::string Whpath = path + "/wh";

  typedef typename baseCPSfieldType<FermionFieldType>::type baseFieldL;
  typedef typename baseCPSfieldType<ComplexFieldType>::type baseFieldH;

  {
    std::vector<baseFieldL*> ptrs_rd(wl.size());
    for(int i=0;i<wl.size();i++){
      assert(wl[i].assigned());
      ptrs_rd[i] = wl[i].ptr();
    }
  
    cps::readParallelSeparateMetadata<typename mf_Policies::ScalarComplexType, baseFieldL::FieldSiteSize,
				      typename getScalarMappingPolicy<typename baseFieldL::FieldMappingPolicy>::type> rd;

    rd.readManyFields(ptrs_rd, Wlpath);
  }
  {
    std::vector<baseFieldH*> ptrs_rd(wh.size());
    for(int i=0;i<wh.size();i++){
      assert(wh[i].assigned());
      ptrs_rd[i] = wh[i].ptr();
    }
  
    cps::readParallelSeparateMetadata<typename mf_Policies::ScalarComplexType, baseFieldH::FieldSiteSize,
				      typename getScalarMappingPolicy<typename baseFieldH::FieldMappingPolicy>::type> rd;

    rd.readManyFields(ptrs_rd, Whpath);
  }
} 
