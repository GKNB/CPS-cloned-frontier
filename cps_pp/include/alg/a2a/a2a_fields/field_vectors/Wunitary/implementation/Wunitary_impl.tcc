template< typename mf_Policies>
double A2AvectorWunitary<mf_Policies>::Mbyte_size(const A2AArg &_args, const FieldInputParamType &field_setup_params){
  FullyPackedIndexDilution dil(_args);
  double ffield_size = double(FermionFieldType::byte_size(field_setup_params))/(1024.*1024.);
  double cfield_size = double(ComplexFieldType::byte_size(field_setup_params))/(1024.*1024.);
  return dil.getNl() * ffield_size + dil.getNhighModes() * cfield_size;
}

template< typename mf_Policies>
void A2AvectorWunitary<mf_Policies>::initialize(const FieldInputParamType &field_setup_params){
  checkSIMDparams<FieldInputParamType>::check(field_setup_params);
  wl.resize(nl); this->allocInitializeLowModeFields(wl,field_setup_params);
  wh.resize(this->getNhighModes()); this->allocInitializeHighModeFields(wh,field_setup_params);
}

template< typename mf_Policies>
A2AvectorWunitary<mf_Policies>::A2AvectorWunitary(const A2AArg &_args): TimeSpinColorPackedIndexDilution(_args), wh_rand_performed(false){ initialize(NullObject()); }

template< typename mf_Policies>
A2AvectorWunitary<mf_Policies>::A2AvectorWunitary(const A2AArg &_args, const FieldInputParamType &field_setup_params): TimeSpinColorPackedIndexDilution(_args), wh_rand_performed(false){
  initialize(field_setup_params); }

template< typename mf_Policies>
A2AvectorWunitary<mf_Policies>::A2AvectorWunitary(const A2Aparams &_args): TimeSpinColorPackedIndexDilution(_args), wh_rand_performed(false){ initialize(NullObject()); }

template< typename mf_Policies>
A2AvectorWunitary<mf_Policies>::A2AvectorWunitary(const A2Aparams &_args, const FieldInputParamType &field_setup_params): TimeSpinColorPackedIndexDilution(_args), wh_rand_performed(false){ initialize(field_setup_params); }

template< typename mf_Policies>
void A2AvectorWunitary<mf_Policies>::setWh(const std::vector<ScalarComplexFieldType> &to){
  assert(to.size() == wh.size());
  for(int i=0;i<wh.size();i++) wh[i]->importField(to[i]);
  wh_rand_performed = true;
}


//Get the diluted source with index id.
//We allow for time dilution into Lt/src_width blocks of size src_width in the time direction
//Alongside the spin/color/flavor index upon which to place the random numbers, the index dil_id also contains the time block index
template< typename mf_Policies>
template<typename TargetFermionFieldType>
void A2AvectorWunitary<mf_Policies>::getDilutedSource(TargetFermionFieldType &into, const int dil_id) const{
  typedef FieldSiteType mf_Complex;
  typedef typename TargetFermionFieldType::FieldSiteType TargetComplex;
  const char* fname = "getDilutedSource(...)";
  int hit, tblock, spin_color, flavor;
  StandardIndexDilution stdidx(getArgs());  
  stdidx.indexUnmap(dil_id,hit,tblock,spin_color,flavor);
  
  int hidx = this->indexMap(hit,flavor);

  //Dimensions of 4d (possibly SIMD vectorized [spatial only]) complex field
  const int src_layout[4] = { wh[hidx]->nodeSites(0), wh[hidx]->nodeSites(1), wh[hidx]->nodeSites(2), wh[hidx]->nodeSites(3) };
  
  assert(src_layout[3] == GJP.TnodeSites()); //check no vectorization in t
  
  assert(GJP.Tnodes()*GJP.TnodeSites() % args.src_width == 0); //assumed an even number of time blocks fit into lattice

  VRB.Result("A2AvectorWunitary", fname, "Generating random wall source %d = (%d, %d, %d, %d).\n    ", dil_id, hit, tblock, flavor, spin_color);
  const int tblock_origt = tblock * args.src_width; //origin of t block in global coordinates
  const int tblock_lessthant = tblock_origt + args.src_width; //where does it end?

  int tblock_origt_lcl = tblock_origt - GJP.TnodeCoor()*GJP.TnodeSites(); //same as above in local coords
  int tblock_lessthant_lcl = tblock_lessthant - GJP.TnodeCoor()*GJP.TnodeSites();
  
  into.zero();

  if(tblock_lessthant_lcl <= 0 || tblock_origt_lcl >= GJP.TnodeSites()){ //none of source is on this node
    VRB.Result("A2AvectorWunitary", fname, "Not on node\n    ");
    return;
  }

  //Some of the source is on this node
  if(tblock_origt_lcl < 0) tblock_origt_lcl = 0; //beginning of source is before origin
  if(tblock_lessthant_lcl > GJP.TnodeSites()) tblock_lessthant_lcl = GJP.TnodeSites(); //end is after local time size

  const int lcl_src_twidth = tblock_lessthant_lcl - tblock_origt_lcl;
  
  const int src_size = src_layout[0]*src_layout[1]*src_layout[2]*lcl_src_twidth;  //size of source 3D*width slice in units of complex numbers  
  CPSautoView(into_v,into,HostWrite);
  CPSautoView(wh_v,(*wh[hidx]),HostRead);
#pragma omp parallel for
  for(int i=0;i<src_size;i++){
    int x[4];
    int rem = i;
    x[0] = rem % src_layout[0]; rem /= src_layout[0];
    x[1] = rem % src_layout[1]; rem /= src_layout[1];
    x[2] = rem % src_layout[2]; rem /= src_layout[2];
    x[3] = tblock_origt_lcl + rem;

    for(int f=0;f<nflavors;f++){ //not a unit matrix in flavor
      TargetComplex *into_site = (TargetComplex*)(into_v.site_ptr(x,f) + spin_color); //delta in spin_color
      mf_Complex const* from_site = (mf_Complex*)wh_v.site_ptr(x,f); //note same random numbers for each spin/color!
      *into_site = *from_site;
    }
  }
}

//When gauge fixing prior to taking the FFT it is necessary to uncompact the wh field in the spin-color index, as these indices are acted upon by the gauge fixing
//(I suppose technically only the color indices need uncompacting; this might be considered as a future improvement)
template< typename mf_Policies>
void A2AvectorWunitary<mf_Policies>::getSpinColorDilutedSource(FermionFieldType &into, const int high_mode_idx, const int sc_id) const{
  const char* fname = "getSpinColorDilutedSource(...)";
  
  into.zero();
  CPSautoView(into_v,into,HostReadWrite);
  CPSautoView(wh_v,(*wh[high_mode_idx]),HostRead);
#pragma omp parallel for
  for(int i=0;i<wh[high_mode_idx]->nfsites();i++){ //same mapping, different site_size
    FieldSiteType &into_site = *(into_v.fsite_ptr(i) + sc_id);
    const FieldSiteType &from_site = *(wh_v.fsite_ptr(i));
    into_site = from_site;
  }
}


template< typename mf_Policies>
void A2AvectorWunitary<mf_Policies>::writeParallelWithGrid(const std::string &file_stub) const
{
#ifndef HAVE_LIME
  ERR.General("A2AvectorWunitary","writeParallelWithGrid","Requires LIME");
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
void A2AvectorWunitary<mf_Policies>::readParallelWithGrid(const std::string &file_stub)
{
#ifndef HAVE_LIME
  ERR.General("A2AvectorWunitary","readParallelWithGrid","Requires LIME");
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

