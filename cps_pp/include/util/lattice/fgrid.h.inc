

#ifdef IF_FIVE_D
#define FIVE_GRID *FGrid,*FrbGrid,
#define FERM_GRID FGrid
#define F_RB_GRID FrbGrid
#else
#define FIVE_GRID 
#define FERM_GRID UGrid
#define F_RB_GRID UrbGrid
#endif

CPS_START_NAMESPACE
class FGRID : public virtual Lattice, public FgridParams,
	public virtual FgridBase{

   using RealD = Grid::RealD;
   using RealF = Grid::RealF;
   public:
	const char *cname;

   private:
	template <typename vobj,typename sobj>
	void ImpexFermion(Vector *cps_f, vobj &grid_f, int cps2grid, EvenOdd eo ){
		const char *fname="ImpexFermion()";
		unsigned long vol;
		Grid::GridBase *grid = grid_f._grid;
		unsigned long fourvol = GJP.VolNodeSites();
		int ncb =2 ;
		if (eo!=All) {fourvol /=2; ncb=1;}
//		if ((eo==All) && ((grid_f._grid)->lSites())!= (vol = GJP.VolNodeSites()*GJP.SnodeSites()))
		vol = fourvol*(2/ncb);
		if(F5D()) vol *=GJP.SnodeSites();
		if (((grid_f._grid)->lSites())!= vol)
			ERR.General(cname,fname,"numbers of grid(%d) and GJP(%d) does not match\n",grid->lSites(),vol);
		std::vector<int> grid_coor;
//		 iGparitySpinColourVector<Grid::ComplexD> siteGrid;
		 sobj siteGrid;
		for(int site=0;site<vol;site++){
			Grid::Lexicographic::CoorFromIndex(grid_coor,site,grid->_ldimensions);
			int pos[4],offset=0;
		if(F5D()) offset=1;
			for(int i=0;i<4;i++) pos[i] =grid_coor[i+offset];
		if ((ncb>1)||(pos[0]+pos[1]+pos[2]+pos[3])%2==eo)
		if(cps2grid){
//			printf("%d: %s:%s: site=%d grid_coor=%d %d %d %d %d 4dindex = %d \n", UniqueID(),cname,fname, site,grid_coor[0],grid_coor[1],grid_coor[2],grid_coor[3],grid_coor[4], (FsiteOffset(pos)/(2/ncb)) );
			for(int gp=0;gp<n_gp;gp++)
			for(int s=0;s<Ns;s++)
			for(int i=0;i<Nc;i++){
				int i_gp=0;
				if(F5D()) i_gp=grid_coor[0];// s coor when 5D
				int index = (FsiteOffset(pos)/(2/ncb)) + fourvol*(gp+n_gp*i_gp);
				Float *cps = (Float *)cps_f;
				cps += 2*(i+Nc*(s+Ns*(index)));
				std::complex<double> elem(*cps,*(cps+1));
//				if (norm(elem)>0.01) printf("%d %d %d %d: cps[%d][%d][%d][%d] = %g %g\n", GJP.NodeCoor(0),GJP.NodeCoor(1),GJP.NodeCoor(2),GJP.NodeCoor(3), site,gp,s,i,elem.real(),elem.imag());
				siteGrid(GP)(s)(i) = elem;
			}
			pokeLocalSite(siteGrid,grid_f,grid_coor);
		} else {
			Grid::Lexicographic::CoorFromIndex(grid_coor,site,grid->_ldimensions);
			peekLocalSite(siteGrid,grid_f,grid_coor);
			for(int gp=0;gp<n_gp;gp++)
			for(int s=0;s<Ns;s++)
			for(int i=0;i<Nc;i++){
				int i_gp=0;
				if(F5D()) i_gp=grid_coor[0];// s coor when 5D
				std::complex<double> elem;
				elem = siteGrid(GP)(s)(i);
//				if (norm(elem)>0.01) printf("%d %d %d %d %d: grid[%d][%d][%d][%d] = %g %g\n",
//grid_coor[0], grid_coor[1], grid_coor[2], grid_coor[3], grid_coor[4],
//site,gp,s,i,elem.real(),elem.imag());
				int index = (FsiteOffset(pos)/(2/ncb)) + fourvol*(gp+n_gp*i_gp);
				Float *cps = (Float *)cps_f;
				cps += 2*(i+Nc*(s+Ns*(index)));
				*cps=elem.real();
				*(cps+1)=elem.imag();
			}
		}
		} 
	}

   public:


	FGRID(FgridParams &params): cname(XSTR(FGRID)),FgridBase(params){
#ifdef GRID_GPARITY
	if(!GJP.Gparity()) ERR.General(cname, XSTR(FGRID) ,"Trying to instantiate Fgrid class with Gparity on non-Gparity lattice\n");
#endif
	}
	void ImportFermion(Vector *cps_f, LATTICE_FERMION &grid_f,EvenOdd eo = All){
	ImpexFermion<LATTICE_FERMION, SITE_FERMION>(cps_f,grid_f,0,eo);
	}
	void ImportFermion(LATTICE_FERMION &grid_f,Vector *cps_f,EvenOdd eo = All){
	ImpexFermion<LATTICE_FERMION,SITE_FERMION>(cps_f,grid_f,1,eo);
	}

#if 0
	void ImportGauge( Matrix *mom){ ImportGauge( NULL, mom,1); }

	void ImportGauge( LatticeGaugeFieldD *grid_lat)
		{ ImportGauge( grid_lat,NULL,1); }

	void ImportGauge( LatticeGaugeFieldD *grid_lat, Matrix *mom){
		ImpexGauge( grid_lat, mom, 1);
	}

	void ImportGauge( Matrix *mom, LatticeGaugeFieldD *grid_lat){
		ImpexGauge( grid_lat, mom, 0);
	}

	void ImportGauge() { ImpexGauge(NULL,NULL,1); }
		
	void ImpexGauge( LatticeGaugeFieldD *grid_lat, Matrix *mom, int cps2grid){

		Float *gauge = (Float *) mom;
		if(!mom)  gauge = (Float *)GaugeField();
		if (!grid_lat)  grid_lat = Umu;
		unsigned long vol;
		const char *fname="ImpexGauge()";
		GridBase *grid = grid_lat->_grid;
		if (grid_lat->_grid->lSites()!= (vol = GJP.VolNodeSites()))
			ERR.General(cname,fname,"numbers of grid(%d) and GJP(%d) does not match\n",grid_lat->_grid->lSites(),vol);
		std::vector<int> grid_coor;
		LorentzColourMatrixD siteGrid;
		for(int site=0;site<vol;site++)
		for(int mu=0;mu<4;mu++){
			if(cps2grid){
			for(int i=0;i<Nc;i++)
			for(int j=0;j<Nc;j++){
				Float *cps = gauge +18*(site*4+mu)+6*j+2*i;
				std::complex<double> elem(*cps,*(cps+1));
				siteGrid(mu)()(j,i) = elem;
//				if (norm(elem)>0.01) printf("gauge[%d][%d][%d][%d] = %g %g\n",site,mu,i,j,elem.real(),elem.imag());
			}
			Grid::Lexicographic::CoorFromIndex(grid_coor,site,grid->_ldimensions);
			pokeLocalSite(siteGrid,*grid_lat,grid_coor);
			} else {
			Grid::Lexicographic::CoorFromIndex(grid_coor,site,grid->_ldimensions);
			peekLocalSite(siteGrid,*grid_lat,grid_coor);
			for(int i=0;i<Nc;i++)
			for(int j=0;j<Nc;j++){
				std::complex<double> elem;
				elem  = siteGrid(mu)()(j,i);
				Float *cps = gauge +18*(site*4+mu)+6*j+2*i;
				*cps = elem.real();*(cps+1)=elem.imag();
//				if (norm(elem)>0.01) printf("gauge[%d][%d][%d][%d] = %g %g\n",site,mu,i,j,elem.real(),elem.imag());
			}
			}
		} 
	}
std::vector<int> SetTwist(){
	std::vector<int> twists(Nd,0);
	for(int i=0;i<3;i++){
		 twists[i] = (GJP.Bc(i) == BND_CND_GPARITY) ? 1 : 0 ;
		if(twists[i]) VRB.Result(cname,"SetTwist()","gparity[%d]=1\n",i);
	}
	return twists;
}
#endif

void  SetParams( DIRAC ::ImplParams &params){
#ifdef GRID_GPARITY
	std::vector<int> twists = SetTwist();
	params.twists = twists;
#endif

}



	void Fdslash(Vector *f_out, Vector *f_in, CgArg *cg_arg,
                    CnvFrmType cnv_frm, int dir_flag){
	const char *fname("Fdslash()");
#if 1
//	mass = cg_arg->mass;
	SetMass(cg_arg->mass);
	SetEpsilon(cg_arg->epsilon);
	VRB.Result(cname,fname,"mass=%0.14g epsilon=%g \n",mass,eps);
	RealD M5 = GJP.DwfHeight();
	ImportGauge();
	std::vector<int> twists = SetTwist();
	

	LATTICE_FERMION grid_in(FERM_GRID),grid_out(FERM_GRID);
	ImportFermion(grid_in, f_in);
	DIRAC ::ImplParams params;
	SetParams( params );
	DIRAC Ddwf(*Umu, FIVE_GRID *UGrid,*UrbGrid,mass  MOB PARAMS );
	Ddwf.M(grid_in,grid_out);
	ImportFermion(f_out, grid_out);
#endif
	}
  // It returns the type of fermion class
  FclassType Fclass()const {
    return CLASS_NAME;
  }
  
  int FsiteSize() const {
#ifdef IF_FIVE_D 
    int size= 24 * Ls;
#else
    int size= 24;
#endif
    return size;
  }
  // Returns the number of fermion field 
  // components (including real/imaginary) on a
  // site of the 4-D lattice.
  
  int FchkbEvl() const {
    return 1;
  }
  // Returns 0 => If no checkerboard is used for the evolution
  //      or the CG that inverts the evolution matrix.
  
  int FmatEvlInv(Vector *f_out, Vector *f_in, 
		 CgArg *cg_arg, 
		 Float *true_res,
		 CnvFrmType cnv_frm = CNV_FRM_YES)
  {ERR.NotImplemented(cname,"FmatEvlInv");}
  // It calculates f_out where A * f_out = f_in and
  // A is the preconditioned fermion matrix that appears
  // in the HMC evolution (even/odd preconditioning 
  // of [Dirac^dag Dirac]). The inversion is done
  // with the conjugate gradient. cg_arg is the structure
  // that contains all the control parameters, f_in is the
  // fermion field source vector, f_out should be set to be
  // the initial guess and on return is the solution.
  // f_in and f_out are defined on a checkerboard.
  // If true_res !=0 the value of the true residual is returned
  // in true_res.
  // *true_res = |src - MatPcDagMatPc * sol| / |src|
  // The function returns the total number of CG iterations.
  int FmatEvlInv(Vector *f_out, Vector *f_in, 
		 CgArg *cg_arg, 
		 CnvFrmType cnv_frm = CNV_FRM_YES)
  {ERR.NotImplemented(cname,"FmatEvlInv");}
  
  int FmatEvlMInv(Vector **f_out, Vector *f_in, Float *shift, 
		  int Nshift, int isz, CgArg **cg_arg, 
		  CnvFrmType cnv_frm, MultiShiftSolveType type, Float *alpha,
		  Vector **f_out_d)
	{
	const char *fname("FmatEvlMInv()");
#if 1
//	mass = cg_arg[0]->mass;
	SetMass(cg_arg[0]->mass);
	SetEpsilon(cg_arg[0]->epsilon);
	VRB.Result(cname,fname,"mass=%0.14g epsilon=%g \n",mass,eps);
	RealD M5 = GJP.DwfHeight();
	ImportGauge();
	std::vector<int> twists = SetTwist();

	DIRAC ::ImplParams params;
	SetParams( params );
	DIRAC Ddwf(*Umu, FIVE_GRID *UGrid,*UrbGrid,mass  MOB PARAMS );

	LATTICE_FERMION grid_in(FERM_GRID),psi(FERM_GRID);
	std::vector<LATTICE_FERMION> grid_rb_out(Nshift,F_RB_GRID);
	LATTICE_FERMION grid_rb_in(F_RB_GRID),grid_out(FERM_GRID);

//	Float stp_cnd = cg_arg->stop_rsd;
//	stp_cnd = stp_cnd*stp_cnd*norm2(grid_in);

#if 0
	ConjugateGradient<LATTICE_FERMION> CG(stp_cnd,cg_arg->max_num_iter);
	SchurRedBlackDiagMooeeSolve<LATTICE_FERMION> SchurSolver(CG);
	SchurSolver(Ddwf,grid_in,grid_out);
#else 
	Grid::MultiShiftFunction shifts;
	shifts.order=Nshift;
	shifts.poles.resize(Nshift);
	shifts.tolerances.resize(Nshift);
	shifts.residues.resize(Nshift);
	for(int i=0;i<Nshift;i++){
		shifts.poles[i]  = shift[i];
		shifts.residues[i]  =alpha[i];
		shifts.tolerances[i]  =cg_arg[i]->stop_rsd;
	}
	ImportFermion(grid_in, f_in,Odd);

	pickCheckerboard(Odd,grid_rb_in,grid_in);
	Grid::QCD::SchurDifferentiableOperator< IMPL > MdagM(Ddwf);
        Grid::ConjugateGradientMultiShift<LATTICE_FERMION> 
				MSCG(cg_arg[0]->max_num_iter,shifts);
	MSCG(MdagM,grid_rb_in,grid_rb_out); 
	for(int i=0;i<Nshift;i++){
	setCheckerboard(grid_out,grid_rb_out[i]);
	ImportFermion(f_out[i], grid_out,Odd);
	}
	
#endif
#if 0
	if(1){
		LATTICE_FERMION temp(FERM_GRID);
		Ddwf.M(grid_out,temp);
		temp = temp - grid_in;
		true_res = std::sqrt(norm2(temp)/norm2(grid_in));
		VRB.Result(cname,fname,"true_res=%g\n",true_res);
	}
	ImportFermion(f_out, grid_out,Odd);
#endif
#endif
	}
  
  void FminResExt(Vector *sol, Vector *source, Vector **sol_old, 
		  Vector **vm, int degree, CgArg *cg_arg, CnvFrmType cnv_frm)
  {ERR.NotImplemented(cname,"FminResExt");}
  
  int FmatInv(Vector *f_out, Vector *f_in, 
	      CgArg *cg_arg, 
	      Float *true_res,
	      CnvFrmType cnv_frm = CNV_FRM_YES,
	      PreserveType prs_f_in = PRESERVE_YES){
	const char *fname("FmatInv()");
#if 1
	mass = cg_arg->mass;
	VRB.Result(cname,fname,"mass=%0.14g\n",mass);
	RealD M5 = GJP.DwfHeight();
	ImportGauge();
	std::vector<int> twists = SetTwist();

	LATTICE_FERMION grid_in(FERM_GRID),grid_out(FERM_GRID);
	ImportFermion(grid_in, f_in);
	ImportFermion(grid_out, f_out);
	DIRAC ::ImplParams params;
	SetParams( params );
	DIRAC Ddwf(*Umu, FIVE_GRID *UGrid,*UrbGrid,mass  MOB PARAMS );

	Float stp_cnd = cg_arg->stop_rsd;
//	stp_cnd = stp_cnd*stp_cnd*norm2(grid_in);
#if 1

	Grid::ConjugateGradient<LATTICE_FERMION> CG(stp_cnd,cg_arg->max_num_iter);
	Grid::SchurRedBlackDiagMooeeSolve<LATTICE_FERMION> SchurSolver(CG);
	SchurSolver(Ddwf,grid_in,grid_out);
#else
	MGrid::dagMLinearOperator<DIRAC,LATTICE_FERMION> HermOp(Ddwf);
	Grid::ConjugateGradient<LATTICE_FERMION> CG(stp_cnd,cg_arg->max_num_iter);
	CG(HermOp,grid_in,grid_out);
#endif
	if(true_res){
		LATTICE_FERMION temp(FERM_GRID);
		Ddwf.M(grid_out,temp);
		temp = temp - grid_in;
		*true_res = std::sqrt(norm2(temp)/norm2(grid_in));
		VRB.Result(cname,fname,"true_res=%g\n",*true_res);
	}
	ImportFermion(f_out, grid_out);
#endif
	}
  int FmatInv(Vector *f_out, Vector *f_in, 
	      CgArg *cg_arg, 
	      CnvFrmType cnv_frm = CNV_FRM_YES,
	      PreserveType prs_f_in = PRESERVE_YES)
  {ERR.NotImplemented(cname,"FmatInv");}
  
  int FeigSolv(Vector **f_eigenv, Float *lambda,
	       Float *chirality, int *valid_eig,
	       Float **hsum,
	       EigArg *eig_arg, 
	       CnvFrmType cnv_frm = CNV_FRM_YES)
  {ERR.NotImplemented(cname,"FeigSolv");}
  
  Float SetPhi(Vector *phi, Vector *frm1, Vector *frm2,
	       Float mass, Float epsilon, DagType dag)
  {ERR.NotImplemented(cname,"SetPhi");}
  // It sets the pseudofermion field phi from frm1, frm2.

  Float SetPhi(Vector *phi, Vector *frm1, Vector *frm2,
	       Float mass, DagType dag)
  {ERR.NotImplemented(cname,"SetPhi");}
    
  void MatPc(Vector *out, Vector *in, Float mass, Float epsilon, DagType dag)
  {ERR.NotImplemented(cname,"MatPc");}

  void MatPc(Vector *out, Vector *in, Float mass, DagType dag)
  {ERR.NotImplemented(cname,"MatPc");}

  ForceArg EvolveMomFforce(Matrix *mom, Vector *frm,
                           Float mass, Float epsilon, Float step_size)
  {ERR.NotImplemented(cname,"EvolveMomFforce");}

  ForceArg EvolveMomFforce(Matrix *mom, Vector *frm,
                           Float mass, Float step_size){
    return EvolveMomFforce(mom,frm,mass,-12345,step_size);
  }
  // It evolves the canonical momentum mom by step_size
  // using the fermion force.

  ForceArg EvolveMomFforce(Matrix *mom, Vector *phi, Vector *eta,
                           Float mass, Float epsilon, Float step_size) {
    return EvolveMomFforceBase(mom, phi, eta, mass, epsilon, -step_size);
  }
  ForceArg EvolveMomFforce(Matrix *mom, Vector *phi, Vector *eta,
                           Float mass, Float step_size) {
    return EvolveMomFforceBase(mom, phi, eta, mass, -12345, -step_size);
  }

  // It evolves the canonical Momemtum mom:
  // mom += coef * (phi1^\dag e_i(M) \phi2 + \phi2^\dag e_i(M^\dag) \phi1)
  // note: this function does not exist in the base Lattice class.
  //CK: This function is not used, so I have not modified it for WilsonTM

  ForceArg EvolveMomFforceBase(Matrix *mom,
                               Vector *phi1,
                               Vector *phi2,
                               Float mass_, Float epsilon,
                               Float coef){
	const char *fname("EvolveMomFforceBase()");
	mass = mass_;
	VRB.Result(cname,fname,"mass=%0.14g\n",mass);
	RealD M5 = GJP.DwfHeight();

	ImportGauge();

	Grid::QCD::LatticeGaugeFieldD grid_mom(UGrid), dSdU(UGrid),force(UGrid);
	ImportGauge(&grid_mom, mom);

//	std::vector<int> twists = SetTwist();
	DIRAC ::ImplParams params;
	SetParams( params );

	LATTICE_FERMION grid_phi(FERM_GRID),grid_Y(FERM_GRID);
	LATTICE_FERMION X(F_RB_GRID),Y(F_RB_GRID);
	ImportFermion(grid_phi, phi1,Odd);
	ImportFermion(grid_Y, phi2,Odd);
	pickCheckerboard(Odd,X,grid_phi);
	pickCheckerboard(Odd,Y,grid_Y);

	DIRAC DenOp(*Umu, FIVE_GRID *UGrid,*UrbGrid,mass  MOB PARAMS );
	DIRAC NumOp(*Umu, FIVE_GRID *UGrid,*UrbGrid,1.  MOB PARAMS );

// not really used. Just to fill invoke TwoFlavourEvenOddRatioPseudoFermionAction
	Grid::ConjugateGradient<LATTICE_FERMION> CG(1e-8,10000);
	
	Grid::QCD::TwoFlavourEvenOddRatioPseudoFermionAction< IMPL > quo(NumOp,DenOp,CG,CG);
	
#if 0
	quo.cps_deriv(*Umu,X,Y,dSdU);
#else
	{
		NumOp.ImportGauge(*Umu);
		DenOp.ImportGauge(*Umu);
	
		Grid::QCD::SchurDifferentiableOperator< IMPL > Mpc(DenOp);
		Grid::QCD::SchurDifferentiableOperator< IMPL > Vpc(NumOp);

		force=0.;
		Mpc.MpcDeriv(force,X,Y);  	dSdU=force;
#if 1
		Mpc.MpcDagDeriv(force,Y,X); dSdU=dSdU+force;
#endif

//		LATTICE_FERMION X(FGrid),Y(FGrid);

	}
#endif
#if 1
	grid_mom +=2.*Ta(dSdU);
	ImportGauge(mom, &grid_mom);
#endif
}
  // It evolves the canonical Momemtum mom:
  // mom += coef * (phi1^\dag e_i(M) \phi2 + \phi2^\dag e_i(M^\dag) \phi1)
  // note: this function does not exist in the base Lattice class.
  ForceArg EvolveMomFforceBase(Matrix *mom,
                               Vector *phi1,
                               Vector *phi2,
                               Float mass,
                               Float coef){
    return EvolveMomFforceBase(mom,phi1,phi2,mass,-12345,coef);
  }

  // It evolve the canonical momentum mom  by step_size
  // using the bosonic quotient force.
  
  ForceArg RHMC_EvolveMomFforce(Matrix *mom, Vector **sol, int degree,
				int isz, Float *alpha, Float mass, Float epsilon, Float dt,
				Vector **sol_d, ForceMeasure measure)
  {
	const char *fname("RHMC_EvolveMomFforce()");
#if 0
//	mass = cg_arg[0]->mass;
	VRB.Result(cname,fname,"mass=%0.14g\n",mass);
	RealD M5 = GJP.DwfHeight();
	ImportGauge();
	std::vector<int> twists = SetTwist();
        LatticeGaugeFieldD mom_grid(UGrid);
	

	DIRAC ::ImplParams params;
	params.twists = twists;
//	DIRAC Ddwf(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass  MOB PARAMS );
	DIRAC Ddwf(*Umu, FIVE_GRID *UGrid,*UrbGrid,mass  MOB PARAMS );

	LATTICE_FERMION grid_in(FERM_GRID),psi(FERM_GRID);
	std::vector<LATTICE_FERMION> grid_rb_out(Nshift,F_RB_GRID);
	LATTICE_FERMION grid_rb_in(F_RB_GRID),grid_out(FERM_GRID);

//	Float stp_cnd = cg_arg->stop_rsd;
//	stp_cnd = stp_cnd*stp_cnd*norm2(grid_in);

#if 0
	ConjugateGradient<LATTICE_FERMION> CG(stp_cnd,cg_arg->max_num_iter);
	SchurRedBlackDiagMooeeSolve<LATTICE_FERMION> SchurSolver(CG);
	SchurSolver(Ddwf,grid_in,grid_out);
#else 
	MultiShiftFunction shifts;
	shifts.order=Nshift;
	shifts.poles.resize(Nshift);
	shifts.tolerances.resize(Nshift);
	shifts.residues.resize(Nshift);
	for(int i=0;i<Nshift;i++){
		shifts.poles[i]  = shift[i];
		shifts.residues[i]  =alpha[i];
		shifts.tolerances[i]  =cg_arg[i]->stop_rsd;
	}
	ImportFermion(grid_in, f_in,Odd);

	pickCheckerboard(Odd,grid_rb_in,grid_in);
	SchurDifferentiableOperator< IMPL > MdagM(Ddwf);
    ConjugateGradientMultiShift<LATTICE_FERMION> 
				MSCG(cg_arg[0]->max_num_iter,shifts);
	MSCG(MdagM,grid_rb_in,grid_rb_out); 
	for(int i=0;i<Nshift;i++){
	setCheckerboard(grid_out,grid_rb_out[i]);
	ImportFermion(f_out[i], grid_out,Odd);
	}
	
#endif
#if 0
	if(1){
		LATTICE_FERMION temp(FERM_GRID);
		Ddwf.M(grid_out,temp);
		temp = temp - grid_in;
		true_res = std::sqrt(norm2(temp)/norm2(grid_in));
		VRB.Result(cname,fname,"true_res=%g\n",true_res);
	}
	ImportFermion(f_out, grid_out,Odd);
#endif
#endif
  }
  ForceArg RHMC_EvolveMomFforce(Matrix *mom, Vector **sol, int degree,
				int isz, Float *alpha, Float mass, Float dt,
				Vector **sol_d, ForceMeasure measure) {
    return RHMC_EvolveMomFforce(mom,sol,degree,isz,alpha,mass,-12345,dt,sol_d,measure);
  }
  
  Float FhamiltonNode( Vector *phi,  Vector *chi)
  {ERR.NotImplemented(cname,"FhamiltonNode");}
  // The fermion Hamiltonian of the node sublattice.
  // chi must be the solution of Cg with source phi.	       
  
#if 1
  // doing nothing for now
  // Convert fermion field f_field from -> to
  void Fconvert(Vector *f_field,
		StrOrdType to,
		StrOrdType from)
  {}
#endif
  
  Float BhamiltonNode(Vector *boson, Float mass, Float epsilon)
  {ERR.NotImplemented(cname,"BhamiltonNode");}
  Float BhamiltonNode(Vector *boson, Float mass)
  {ERR.NotImplemented(cname,"BhamiltonNode");}
  // The boson Hamiltonian of the node sublattice
  int SpinComponents() const {
    return 4;
  }

  int ExactFlavors() const {
    return 2;
  }
    
  //!< Method to ensure bosonic force works (does nothing for Wilson
  //!< theories.
  void BforceVector(Vector *in, CgArg *cg_arg)
  {ERR.NotImplemented(cname,"BforceVec");}

// Moving to LatticeBase Temporarily
#if 0
   int F5D() const {
#ifdef IF_FIVE_D
     return 1;
#else
     return 0;
#endif
   }
#endif

};
#if 1

#if 0
#define PASTER(x,y) x ## y
#define EVALUATOR(x,y) PASTER(x,y)
#define GFCLASS(class) EVALUATOR(class, FGRID )
#define XSTR(s) STR(s)
#define STR(s) #s
#endif

//------------------------------------------------------------------
#define GFCLASS_NAME XSTR(GFCLASS(Gnone))
class GFCLASS(Gnone)
: public virtual Lattice,
    public virtual Gnone,
    public virtual FGRID,
    public virtual FgridBase
{
 private:
    const char *cname;    // Class name.

 public:
    GFCLASS(Gnone)(FgridParams &params):cname(GFCLASS_NAME), FGRID (params),FgridBase(params) {
      const char *fname = GFCLASS_NAME"()";
      VRB.Func(cname,fname);
    }

    ~GFCLASS(Gnone)() {
      const char *fname = "~"GFCLASS_NAME"()";
      VRB.Func(cname,fname);
    }
    
};

#undef GRID_GPARITY
#undef IF_FIVE_D
#undef FGRID
#undef CLASS_NAME
#undef GFCLASS_NAME 
#undef DIRAC
#undef MOB
#undef IMPL
#undef LATTICE_FERMION
#undef SITE_FERMION
#undef PARAMS
#undef GP
#undef FIVE_GRID
#undef FERM_GRID
#undef F_RB_GRID
#endif
CPS_END_NAMESPACE
