template< typename mf_Policies>
double A2AvectorW<mf_Policies>::Mbyte_size(const A2AArg &_args, const FieldInputParamType &field_setup_params){
  FullyPackedIndexDilution dil(_args);
  double ffield_size = double(FermionFieldType::byte_size(field_setup_params))/(1024.*1024.);
  double cfield_size = double(ComplexFieldType::byte_size(field_setup_params))/(1024.*1024.);
  return dil.getNl() * ffield_size + dil.getNhits() * cfield_size;
}

template< typename mf_Policies>
void A2AvectorW<mf_Policies>::initialize(const FieldInputParamType &field_setup_params){
  checkSIMDparams<FieldInputParamType>::check(field_setup_params);
  wl.resize(nl); this->allocInitializeLowModeFields(wl,field_setup_params);
  wh.resize(nhits); this->allocInitializeHighModeFields(wh,field_setup_params);
}

template< typename mf_Policies>
A2AvectorW<mf_Policies>::A2AvectorW(const A2AArg &_args): FullyPackedIndexDilution(_args), wh_rand_performed(false){ initialize(NullObject()); }

template< typename mf_Policies>
A2AvectorW<mf_Policies>::A2AvectorW(const A2AArg &_args, const FieldInputParamType &field_setup_params): FullyPackedIndexDilution(_args), wh_rand_performed(false){
  initialize(field_setup_params); }

template< typename mf_Policies>
A2AvectorW<mf_Policies>::A2AvectorW(const A2Aparams &_args): FullyPackedIndexDilution(_args), wh_rand_performed(false){ initialize(NullObject()); }

template< typename mf_Policies>
A2AvectorW<mf_Policies>::A2AvectorW(const A2Aparams &_args, const FieldInputParamType &field_setup_params): FullyPackedIndexDilution(_args), wh_rand_performed(false){ initialize(field_setup_params); }

template< typename mf_Policies>
void A2AvectorW<mf_Policies>::setWh(const std::vector<ScalarComplexFieldType> &to){
  assert(to.size() == nhits);
  for(int i=0;i<nhits;i++) wh[i]->importField(to[i]);
  wh_rand_performed = true;
}


//Get the diluted source with index id.
//We allow for time dilution into Lt/src_width blocks of size src_width in the time direction
//Alongside the spin/color/flavor index upon which to place the random numbers, the index dil_id also contains the time block index
template< typename mf_Policies>
template<typename TargetFermionFieldType>
void A2AvectorW<mf_Policies>::getDilutedSource(TargetFermionFieldType &into, const int dil_id) const{
  typedef FieldSiteType mf_Complex;
  typedef typename TargetFermionFieldType::FieldSiteType TargetComplex;
  const char* fname = "getDilutedSource(...)";
  int hit, tblock, spin_color, flavor;
  StandardIndexDilution stdidx(getArgs());  
  stdidx.indexUnmap(dil_id,hit,tblock,spin_color,flavor);
  
  //Dimensions of 4d (possibly SIMD vectorized [spatial only]) complex field
  const int src_layout[4] = { wh[hit]->nodeSites(0), wh[hit]->nodeSites(1), wh[hit]->nodeSites(2), wh[hit]->nodeSites(3) };
  
  assert(src_layout[3] == GJP.TnodeSites()); //check no vectorization in t
  
  assert(GJP.Tnodes()*GJP.TnodeSites() % args.src_width == 0); //assumed an even number of time blocks fit into lattice

  VRB.Result("A2AvectorW", fname, "Generating random wall source %d = (%d, %d, %d, %d).\n    ", dil_id, hit, tblock, flavor, spin_color);
  const int tblock_origt = tblock * args.src_width; //origin of t block in global coordinates
  const int tblock_lessthant = tblock_origt + args.src_width; //where does it end?

  int tblock_origt_lcl = tblock_origt - GJP.TnodeCoor()*GJP.TnodeSites(); //same as above in local coords
  int tblock_lessthant_lcl = tblock_lessthant - GJP.TnodeCoor()*GJP.TnodeSites();
  
  into.zero();

  if(tblock_lessthant_lcl <= 0 || tblock_origt_lcl >= GJP.TnodeSites()){ //none of source is on this node
    VRB.Result("A2AvectorW", fname, "Not on node\n    ");
    return;
  }

  //Some of the source is on this node
  if(tblock_origt_lcl < 0) tblock_origt_lcl = 0; //beginning of source is before origin
  if(tblock_lessthant_lcl > GJP.TnodeSites()) tblock_lessthant_lcl = GJP.TnodeSites(); //end is after local time size

  const int lcl_src_twidth = tblock_lessthant_lcl - tblock_origt_lcl;
  
  const int src_size = src_layout[0]*src_layout[1]*src_layout[2]*lcl_src_twidth;  //size of source 3D*width slice in units of complex numbers  
  CPSautoView(into_v,into,HostWrite);
  CPSautoView(wh_v,(*wh[hit]),HostRead);
#pragma omp parallel for
  for(int i=0;i<src_size;i++){
    int x[4];
    int rem = i;
    x[0] = rem % src_layout[0]; rem /= src_layout[0];
    x[1] = rem % src_layout[1]; rem /= src_layout[1];
    x[2] = rem % src_layout[2]; rem /= src_layout[2];
    x[3] = tblock_origt_lcl + rem;

    TargetComplex *into_site = (TargetComplex*)(into_v.site_ptr(x,flavor) + spin_color);
    mf_Complex const* from_site = (mf_Complex*)wh_v.site_ptr(x,flavor); //note same random numbers for each spin/color!
    *into_site = *from_site;
  }
}

//When gauge fixing prior to taking the FFT it is necessary to uncompact the wh field in the spin-color index, as these indices are acted upon by the gauge fixing
//(I suppose technically only the color indices need uncompacting; this might be considered as a future improvement)
template< typename mf_Policies>
void A2AvectorW<mf_Policies>::getSpinColorDilutedSource(FermionFieldType &into, const int hit, const int sc_id) const{
  const char* fname = "getSpinColorDilutedSource(...)";
  
  into.zero();
  CPSautoView(into_v,into,HostReadWrite);
  CPSautoView(wh_v,(*wh[hit]),HostRead);
#pragma omp parallel for
  for(int i=0;i<wh[hit]->nfsites();i++){ //same mapping, different site_size
    FieldSiteType &into_site = *(into_v.fsite_ptr(i) + sc_id);
    const FieldSiteType &from_site = *(wh_v.fsite_ptr(i));
    into_site = from_site;
  }
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



template< typename mf_Policies>
void A2AvectorW<mf_Policies>::writeParallelByParts(const std::string &file_stub) const
{
  typedef typename mf_Policies::ScalarFermionFieldType ScalarFermionFieldType;
  typedef typename mf_Policies::ScalarComplexFieldType ScalarComplexFieldType;

  if(wl.size()){
    ScalarFermionFieldType tmp;
    cpsFieldPartIOwriter<ScalarFermionFieldType> wr(file_stub + "_lo");
    for(int k=0;k<wl.size();k++){
      tmp.importField(*wl[k]);
      wr.write(tmp);
    }
    wr.close();
  }
  if(wh.size()){
    ScalarComplexFieldType tmp;
    cpsFieldPartIOwriter<ScalarComplexFieldType> wr(file_stub + "_hi");
    for(int k=0;k<wh.size();k++){
      tmp.importField(*wh[k]);
      wr.write(tmp);
    }
    wr.close();
  }
}

template< typename mf_Policies>
void A2AvectorW<mf_Policies>::readParallelByParts(const std::string &file_stub)
{
  typedef typename mf_Policies::ScalarFermionFieldType ScalarFermionFieldType;
  typedef typename mf_Policies::ScalarComplexFieldType ScalarComplexFieldType;

  if(wl.size()){
    ScalarFermionFieldType tmp;
    cpsFieldPartIOreader<ScalarFermionFieldType> rd(file_stub + "_lo");
    for(int k=0;k<wl.size();k++){
      rd.read(tmp);
      wl[k]->importField(tmp);
    }
    rd.close();
  }
  if(wh.size()){
    ScalarComplexFieldType tmp;
    cpsFieldPartIOreader<ScalarComplexFieldType> rd(file_stub + "_hi");
    for(int k=0;k<wh.size();k++){
      rd.read(tmp);
      wh[k]->importField(tmp);
    }
    rd.close();
  }
}


#ifdef USE_GRID

//Convert a W field to Grid format
template<typename GridFieldType, typename A2Apolicies>
void convertToGrid(std::vector<GridFieldType> &W_out, const A2AvectorW<A2Apolicies> &W_in, Grid::GridBase *grid){
  W_out.resize(W_in.getNv(), GridFieldType(grid));

  for(int i=0;i<W_in.getNl();i++)
    W_in.getWl(i).exportGridField(W_out[i]);

  typename A2Apolicies::FermionFieldType tmp_ferm(W_in.getWh(0).getDimPolParams());
  for(int i=W_in.getNl();i<W_in.getNv();i++){
    W_in.getDilutedSource(tmp_ferm, i-W_in.getNl());
    tmp_ferm.exportGridField(W_out[i]);
  }
}

#endif
