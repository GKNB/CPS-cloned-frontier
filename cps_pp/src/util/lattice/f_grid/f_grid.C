#include<config.h>
#include<util/lattice/fgrid.h>

#ifdef USE_GRID

CPS_START_NAMESPACE

bool FgridBase::grid_initted=false;
bool FgridBase::grid_layouts_initted=false;
int FgridBase::Ls = -1;

Grid::GridCartesian * FgridBase::UGridD = NULL;
Grid::GridCartesian * FgridBase::UGridF = NULL;
Grid::GridRedBlackCartesian * FgridBase::UrbGridD = NULL;
Grid::GridRedBlackCartesian * FgridBase::UrbGridF = NULL;
Grid::GridCartesian * FgridBase::FGridD = NULL;
Grid::GridCartesian * FgridBase::FGridF = NULL;
Grid::GridRedBlackCartesian * FgridBase::FrbGridF = NULL;
Grid::GridRedBlackCartesian * FgridBase::FrbGridD = NULL;
std::vector < int > FgridBase::vol(4,1);
std::vector < int > FgridBase::nodes(4,1);


void FgridBase::initializeGrid(int *argc, char ***argv){
  if (!grid_initted){
    static const char* cname = "FgridBase";
    static const char* fname = "initializeGrid()";
    VRB.Func (cname, fname);
    if(argc == NULL || argv == NULL && !GJP.getArgSet())
      ERR.General(cname,fname,"GJP must have command line arguments set if these are not passed into the current function");
    
    if(argc == NULL) argc = GJP.argc_p();
    if(argv == NULL) argv = GJP.argv_p();

    //Initialize Grid
    Grid::Grid_init (argc,argv);
    VRB.Debug (cname, fname, "Grid initted\n");
    VRB.FuncEnd (cname, fname);    
    setGridInitted();
  }
}



void FgridBase::initializeGridLayouts(){
  static const char* cname = "FgridBase";
  static const char* fname = "initializeGridLayouts()";

  if(!grid_initted) ERR.General(cname, fname, "Grid must be initialized");
  
  if (!grid_layouts_initted){
    VRB.Func (cname, fname);
      
    //Set sizes
    vol.resize(4);
    for (int i = 0; i < 4; i++)
      vol[i] = GJP.NodeSites (i) * GJP.Nodes (i);;
    nodes.resize(4);
    for (int i = 0; i < 4; i++)
      nodes[i] = GJP.Nodes (i);
      
    if(Ls == -1){ //has not been set manually using SetLs
      assert(GJP.Snodes() == 1);
      Ls = GJP.SnodeSites();
    }

    //Create 4D grids
    VRB.Result(cname,fname,"vol nodes Nd=%d Grid::vComplexD::Nsimd()=%d\n",Nd,Grid::vComplexD::Nsimd());
    VRB.Result (cname, fname, "vol nodes Nd=%d Grid::vComplexD::Nsimd()=%d\n", Nd, Grid::vComplexD::Nsimd ());
    for (int i = 0; i < 4; i++)
      VRB.Debug (cname, fname, "%d %d \n", vol[i], nodes[i]);
    UGridD =
      Grid::SpaceTimeGrid::makeFourDimGrid (vol,
					    Grid::GridDefaultSimd (Nd, Grid:: vComplexD:: Nsimd ()),
					    nodes);
    UGridF =
      Grid::SpaceTimeGrid::makeFourDimGrid (vol,
					    Grid::GridDefaultSimd (Nd, Grid:: vComplexF:: Nsimd ()),
					    nodes);
    VRB.Debug (cname, fname, "UGridD=%p UGridF=%p\n", UGridD, UGridF);

    //Check CPS and Grid agree on node layout
    bool fail = false;
    for (int i = 0; i < 4; i++)
      if (GJP.NodeCoor (i) != UGridD->_processor_coor[i])
	fail = true;
    if (fail)
      for (int i = 0; i < 4; i++) {
	printf ("ERROR: Mismatch between Grid and CPS node coordinates CPS: %d  pos[%d]=%d Grid: %d pos[%d]=%d\n", UniqueID (), i,
		GJP.NodeCoor (i), UGridD->_processor, i,
		UGridD->_processor_coor[i]);
      }
      
    //Create the rest of the Grids
    VRB.Debug (cname, fname, "UGrid.lSites()=%d\n", UGridD->lSites ());
    UrbGridD = Grid::SpaceTimeGrid::makeFourDimRedBlackGrid (UGridD);
    UrbGridF = Grid::SpaceTimeGrid::makeFourDimRedBlackGrid (UGridF);
    VRB.Result (cname, fname, "UrbGridD=%p UrbGridF=%p\n", UrbGridD, UrbGridF);
    FGridD = Grid::SpaceTimeGrid::makeFiveDimGrid (Ls, UGridD);
    FGridF = Grid::SpaceTimeGrid::makeFiveDimGrid (Ls, UGridF);
    VRB.Result (cname, fname, "FGridD=%p FGridF=%p\n", FGridD, FGridF);
    VRB.Result (cname, fname, "FGridD.lSites()=%d\n", FGridD->lSites ());
    FrbGridD = Grid::SpaceTimeGrid::makeFiveDimRedBlackGrid (Ls, UGridD);
    FrbGridF = Grid::SpaceTimeGrid::makeFiveDimRedBlackGrid (Ls, UGridF);
    VRB.Result (cname, fname, "FrbGridD=%p FrbGridF=%p\n", FrbGridD, FrbGridF);

    VRB.Debug (cname, fname, "Grid layouts initted\n");
    VRB.FuncEnd (cname, fname);    
    setGridLayoutsInitted();
  }
}



int FgridBase::SetLs (int _Ls)
{
  Ls = _Ls;

  //If we have previously initialized we should recreate the 5D Grids, otherwise this Ls will be used in the constructor
  if(grid_layouts_initted){
    static const char* cname = "FgridBase";
    static const char* fname = "SetLs(int)";

    FGridD = Grid::SpaceTimeGrid::makeFiveDimGrid (Ls, UGridD);
    FGridF = Grid::SpaceTimeGrid::makeFiveDimGrid (Ls, UGridF);
    VRB.Result (cname, fname, "FGridD=%p FGridF=%p\n", FGridD, FGridF);
    VRB.Result (cname, fname, "FGridD.lSites()=%d\n", FGridD->lSites ());
    FrbGridD = Grid::SpaceTimeGrid::makeFiveDimRedBlackGrid (Ls, UGridD);
    FrbGridF = Grid::SpaceTimeGrid::makeFiveDimRedBlackGrid (Ls, UGridF);
    VRB.Result (cname, fname, "FrbGridD=%p FrbGridF=%p\n", FrbGridD, FrbGridF);
  }
  return Ls;
}


FgridBase::FgridBase(FgridParams & params):cname ("FgridBase"), mass (1.){
  //,epsilon(0.),

  const char *fname ("FgridBase()");
  VRB.Func (cname, fname);

  //Setup Grid and the Grid objects
  initializeGrid();
  initializeGridLayouts();

  *((FgridParams *) this) = params;
  eps = params.epsilon;
  omegas = params.omega;

  //              VRB.Debug(cname,fname,"mobius_scale=%g\n",mobius_scale);
  mob_b = 0.5 * (mobius_scale + mobius_bmc);
  mob_c = mob_b - mobius_bmc;

  if (!GJP.Gparity ()) {
    //              ERR.General(cname,fname,"Only implemented for Grid with Gparity at the moment\n");
    n_gp = 1;
  } else
    n_gp = 2;

  threads = Grid::GridThread::GetThreads ();

#ifdef HAVE_HANDOPT
  if (GJP.Gparity ())
    Grid::WilsonKernelsStatic::HandOpt = 0;	//Doesn't seem to be working with Gparity
  else
    Grid::WilsonKernelsStatic::HandOpt = 1;
#endif

  Umu = new Grid::LatticeGaugeFieldD (UGridD);
  //  Umu_f = new Grid::LatticeGaugeFieldF(UGrid_f);

  VRB.FuncEnd (cname, fname);
}


//------------------------------------------------------------------
/*!
  \param five The 5-dimensional field.
  \param four The 4-dimensional field.
  \param s_u The global 5th direction (s) coordinate where the
  upper two components (right chirality) of the 5-dim. field
  take the values of those of the 4-dim. field.
  \param s_l The global 5th direction (s) coordinate where the
  lower two components (left chirality) of the 5-dim. field
  take the values of those of the 4-dim. field.
  \post The 5-dim field is zero everywhere except where the global
  5th direction coordinate (s) is \a s_l or \a s_u, where it takes the values
  explained above.
*/
//------------------------------------------------------------------
void FgridBase::Ffour2five(Vector *five, Vector *four, int s_u, int s_l, int Ncb)
{
  int x;
  int i;
  Float *field_4D;
  Float *field_5D;
  char *fname = "Ffour2five(V*,V*,i,i)";
  VRB.Func(cname,fname);


  //------------------------------------------------------------------
  // Initializations
  //------------------------------------------------------------------
  size_t f_size = GJP.VolNodeSites() * FsiteSize()*Ncb/2;
  if(GJP.Gparity()) f_size*=2;

  int ls = GJP.SnodeSites();
  int vol_4d = GJP.VolNodeSites()*Ncb/2;
  int ls_stride = 24 * vol_4d;
  if(GJP.Gparity()) ls_stride*=2;

  int s_u_local = s_u % GJP.SnodeSites();
  int s_l_local = s_l % GJP.SnodeSites();
  int s_u_node = s_u / GJP.SnodeSites();
  int s_l_node = s_l / GJP.SnodeSites();


  //------------------------------------------------------------------
  // Set *five using the 4D field *four. 
  //------------------------------------------------------------------

  // Set all components of the 5D field to zero.
  //---------------------------------------------------------------
  field_5D  = (Float *) five;
  for(i=0; i<f_size; i++){
    field_5D[i]  = 0.0;
  }

  // Do the two upper spin components if s_u is in the node
  //---------------------------------------------------------------
  if( s_u_node == GJP.SnodeCoor() ){
    field_4D  = (Float *) four;
    field_5D  = (Float *) five;
    field_5D  = field_5D  + s_u_local * ls_stride;
    for(x=0; x<vol_4d; x++){
      for(i=0; i<12; i++){
	field_5D[i]  = field_4D[i];
      }
      field_4D  = field_4D  + 24;
      field_5D  = field_5D  + 24;
    }
    if(GJP.Gparity()){ //CK:08/11 do second stacked field
      for(x=0; x<vol_4d; x++){
	for(i=0; i<12; i++){
	  field_5D[i]  = field_4D[i];
	}
	field_4D  = field_4D  + 24;
	field_5D  = field_5D  + 24;
      }
    }
  }

  // Do the two lower spin components if s_l is in the node
  //----------------------------------------------------------------
  if( s_l_node == GJP.SnodeCoor() ){
    field_4D  = (Float *) four;
    field_5D  = (Float *) five;
    field_4D  = field_4D  + 12;
    field_5D  = field_5D  + 12 + s_l_local * ls_stride;
    for(x=0; x<vol_4d; x++){
      for(i=0; i<12; i++){
	field_5D[i]  = field_4D[i];
      }
      field_4D  = field_4D  + 24;
      field_5D  = field_5D  + 24;
    }
    if(GJP.Gparity()){
      for(x=0; x<vol_4d; x++){
	for(i=0; i<12; i++){
	  field_5D[i]  = field_4D[i];
	}
	field_4D  = field_4D  + 24;
	field_5D  = field_5D  + 24;
      }
    }
  }

}


//------------------------------------------------------------------
/*!
  \param four The 4-dimensional field.
  \param five The 5-dimensional field.
  \param s_u The global 5th direction (s) coordinate where 
  the values of the upper two components (right chirality) of the 5-dim. field
  are taken by those of the 4-dim. field.
  \param s_l The global 5th direction coordinate (s) where the values of 
  the lower two components (left chirality) of the 5-dim. field
  are taken by  those of the 4-dim. field.
  \post The 5-dim field is zero everywhere except where the global
  5th direction coordinate is \a s_l or \a s_u, where it takes the values
  explained above.
  \post An identical 4-dim. field is reproduced on all nodes in the s
  direction.
*/
//------------------------------------------------------------------
void FgridBase::Ffive2four(Vector *four, Vector *five, int s_u, int s_l, int Ncb)
{
  int x;
  int i;
  Float *field_4D;
  Float *field_5D;
  char *fname = "Ffive2four(V*,V*,i,i)";
  VRB.Func(cname,fname);


  //------------------------------------------------------------------
  // Initializations
  //------------------------------------------------------------------
  int ls = GJP.SnodeSites();
  size_t f_size = GJP.VolNodeSites() * FsiteSize()*Ncb / (ls*2);
  if(GJP.Gparity()) f_size*=2;

  int vol_4d = GJP.VolNodeSites()*Ncb/2;
  int ls_stride = 24 * vol_4d;
  if(GJP.Gparity()) ls_stride*=2;

  int s_u_local = s_u % GJP.SnodeSites();
  int s_l_local = s_l % GJP.SnodeSites();
  int s_u_node = s_u / GJP.SnodeSites();
  int s_l_node = s_l / GJP.SnodeSites();


  //------------------------------------------------------------------
  // Set *four using the 5D field *five. 
  //------------------------------------------------------------------

  // Set all components of the 4D field to zero.
  //---------------------------------------------------------------
  field_4D  = (Float *) four;
  for(i=0; i<f_size; i++){
    field_4D[i]  = 0.0;
  }

  // Do the two upper spin components if s_u is in the node
  //---------------------------------------------------------------
  if( s_u_node == GJP.SnodeCoor() ){
    field_4D = (Float *) four;
    field_5D = (Float *) five;
    field_5D = field_5D + s_u_local * ls_stride;
    for(x=0; x<vol_4d; x++){
      for(i=0; i<12; i++){
	field_4D[i] = field_5D[i];
      }
      field_4D = field_4D + 24;
      field_5D = field_5D + 24;
    }
    if(GJP.Gparity()){
      for(x=0; x<vol_4d; x++){
	for(i=0; i<12; i++){
	  field_4D[i] = field_5D[i];
	}
	field_4D = field_4D + 24;
	field_5D = field_5D + 24;
      }
    }

  }
  // Do the two lower spin components if s_l is in the node
  //----------------------------------------------------------------
  if( s_l_node == GJP.SnodeCoor() ){
    field_4D = (Float *) four;
    field_5D = (Float *) five;
    field_4D = field_4D + 12;
    field_5D = field_5D + 12 + s_l_local * ls_stride;
    for(x=0; x<vol_4d; x++){
      for(i=0; i<12; i++){
	field_4D[i] = field_5D[i];
      }
      field_4D = field_4D + 24;
      field_5D = field_5D + 24;
    }
    if(GJP.Gparity()){
      for(x=0; x<vol_4d; x++){
	for(i=0; i<12; i++){
	  field_4D[i] = field_5D[i];
	}
	field_4D = field_4D + 24;
	field_5D = field_5D + 24;
      }
    }
  }

  // Sum along s direction to get the same 4D field in all 
  // s node slices.
  //----------------------------------------------------------------
  if( GJP.Snodes() > 1) {
    Float sum;
    field_4D  = (Float *) four;
    for(i=0; i<f_size; i++){
      sum = field_4D[i];
      glb_sum_dir(&sum, 4);
      field_4D[i] = sum;    
    }
  }

}

CPS_END_NAMESPACE
#endif
