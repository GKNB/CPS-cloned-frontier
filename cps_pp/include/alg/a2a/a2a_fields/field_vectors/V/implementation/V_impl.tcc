template< typename mf_Policies>
double A2AvectorV<mf_Policies>::Mbyte_size(const A2AArg &_args, const FieldInputParamType &field_setup_params){
  return VW_Mbyte_size<A2AvectorV<mf_Policies> >(_args,field_setup_params);
}

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



#ifdef USE_GRID

//Convert a V field to Grid format
template<typename GridFieldType, typename A2Apolicies>
void convertToGrid(std::vector<GridFieldType> &V_out, const A2AvectorV<A2Apolicies> &V_in, Grid::GridBase *grid){
  V_out.resize(V_in.getNv(), GridFieldType(grid));

  for(int i=0;i<V_in.getNv();i++)
    V_in.getMode(i).exportGridField(V_out[i]);
}

#endif
