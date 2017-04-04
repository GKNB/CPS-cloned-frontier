int IRL (Vector ** f_out, Vector * f_in, CgArg * cg_arg,
	 int Nk, int Np, int Nstop, Float low, Float high, int order)
{
  const char *fname ("IRL()");
  SetMass (cg_arg->mass);
  SetEpsilon (cg_arg->epsilon);
  VRB.Result (cname, fname, "mass=%0.14g epsilon=%g \n", mass, eps);
//      VRB.Result(cname,fname,"max_iter=%d\n",cg_arg->max_num_iter);
  RealD M5 = GJP.DwfHeight ();

  ImportGauge ();
  std::vector < int >twists = SetTwist ();

  DIRAC::ImplParams params;
  SetParams (params);
  DIRAC DdwfD (*Umu, FIVE_GRID * UGridD, *UrbGridD, mass MOB PARAMS);
  Grid::SchurDiagTwoOperator < DIRAC, LATTICE_FERMION > HermOp (DdwfD);
  Grid::QCD::LatticeGaugeFieldF UmuF (UGridF);
  precisionChange (UmuF, *Umu);
  DIRAC_F DdwfF (UmuF, FIVE_GRID_F * UGridF, *UrbGridF, mass MOB PARAMS);
  Grid::SchurDiagTwoOperator < DIRAC_F, LATTICE_FERMION_F > HermOpF (DdwfF);

  LATTICE_FERMION grid_in (FERM_GRID), psi (FERM_GRID);
//      std::vector<LATTICE_FERMION> grid_rb_out(Nshift,F_RB_GRID);
  LATTICE_FERMION grid_rb_in (F_RB_GRID), grid_out (FERM_GRID);
  LATTICE_FERMION_F grid_rb_in_f (F_RB_GRID_F), grid_out_f (FERM_GRID_F);
  LatVector cps_temp (FsiteSize () / 6 * n_gp, GJP.VolNodeSites () / 2);
  VRB.Debug (cname, fname, "cps_temp.Size()=%d\n", cps_temp.Size ());
//      VRB.Debug(cname,fname,"Nshift=%d alpha=%p\n",Nshift,alpha);
  Vector *f_temp = cps_temp.Vec ();

  double fac = 1.;
  ImportFermion (grid_in, f_in, Odd);

  pickCheckerboard (Odd, grid_rb_in, grid_in);
  const int Nm = Nk + Np;
  const int MaxIt = 10000;
  RealD resid = 1.0e-8;
  RealD resid_f = 1.0e-6;

  std::vector < double >Coeffs
  {
  0., -1.};
  Grid::Polynomial < LATTICE_FERMION > PolyX (Coeffs);
  Grid::Chebyshev < LATTICE_FERMION_F > Cheb (low, high, order + 1);
//  ChebyshevLanczos<LatticeFermion> Cheb(9.,1.,0.,20);
//  Cheb.csv(std::cout);
//  exit(-24);
//  ImplicitlyRestartedLanczos<FermionField> IRL(HermOp,Cheb,Nstop,Nk,Nm,resid,MaxIt);
//  Grid::ImplicitlyRestartedLanczos<LATTICE_FERMION> IRL(HermOp,Cheb,Nstop,Nk,Nm,resid,MaxIt);
  Grid::ImplicitlyRestartedLanczos < LATTICE_FERMION_F > IRL (HermOpF, Cheb,
							      Nstop, Nk, Nm,
							      resid_f, MaxIt);
  precisionChange (grid_rb_in_f, grid_rb_in);

  std::vector < Grid::RealD > eval (Nm);
  std::vector < LATTICE_FERMION > evec (Nm, F_RB_GRID);
  std::vector < LATTICE_FERMION_F > evec_f (Nm, F_RB_GRID_F);

  int Nconv;
//  IRL.calc(eval,evec,grid_rb_in,Nconv);
  IRL.calc (eval, evec_f, grid_rb_in_f, Nconv);

  return Nconv;

}

#undef SINGLE

int FeigSolv (Vector ** f_eigenv, Float * lambda,
	      LanczosArg * eig_arg, CnvFrmType cnv_frm)
{
  const char *fname ("FeigSolv()");
  if (cnv_frm == CNV_FRM_YES)
    ERR.General (cname, fname, " only has support for checkerboarded\n");
  SetMass (eig_arg->mass);
  SetEpsilon (0.);
  VRB.Result (cname, fname, "mass=%0.14g epsilon=%g \n", mass, eps);
//      VRB.Result(cname,fname,"max_iter=%d\n",cg_arg->max_num_iter);
  RealD M5 = GJP.DwfHeight ();

  ImportGauge ();
  std::vector < int >twists = SetTwist ();

  DIRAC::ImplParams params;
  SetParams (params);

//      Staying with double precision, at least until gaining more confidence in single precision
  DIRAC DdwfD (*Umu, FIVE_GRID * UGridD, *UrbGridD, mass MOB PARAMS);
  Grid::SchurDiagTwoOperator < DIRAC, LATTICE_FERMION > HermOp (DdwfD);
#ifdef SINGLE
  Grid::QCD::LatticeGaugeFieldF UmuF (UGridF);
  precisionChange (UmuF, *Umu);
  DIRAC_F DdwfF (UmuF, FIVE_GRID_F * UGridF, *UrbGridF, mass MOB PARAMS);
  Grid::SchurDiagTwoOperator < DIRAC_F, LATTICE_FERMION_F > HermOpF (DdwfF);
#endif

  LATTICE_FERMION grid_in (FERM_GRID), psi (FERM_GRID);
  LATTICE_FERMION grid_rb_in (F_RB_GRID), grid_out (FERM_GRID);
#ifdef SINGLE
  LATTICE_FERMION_F grid_rb_in_f (F_RB_GRID_F), grid_out_f (FERM_GRID_F);
#endif
  LatVector cps_temp (FsiteSize () / 6 * n_gp, GJP.VolNodeSites () / 2);
  VRB.Debug (cname, fname, "cps_temp.Size()=%d\n", cps_temp.Size ());
//      VRB.Debug(cname,fname,"Nshift=%d alpha=%p\n",Nshift,alpha);
  Vector *f_temp = cps_temp.Vec ();
  RandGaussVector (f_temp, 0.5, 1);

//      double fac = 1.;


  ImportFermion (grid_in, f_temp, Odd);
  pickCheckerboard (Odd, grid_rb_in, grid_in);

  const int Np = eig_arg->np_lanczos_vectors;
  const int Nk = eig_arg->nk_lanczos_vectors;
  const int Nm = Np + Nk;
  const int Nstop = eig_arg->nt_lanczos_vectors;
  const int MaxIt = eig_arg->maxiters;
  RealD resid = eig_arg->stop_residual;

  Float low = eig_arg->matpoly_arg.params.params_val[1];
  low = low * low;
  Float high = eig_arg->matpoly_arg.params.params_val[0];
  high = high * high;
  int order = eig_arg->matpoly_arg.Npol + 1;	//convention difference
#ifndef SINGLE
  Grid::Chebyshev < LATTICE_FERMION > Cheb (low, high, order);
  Grid::ImplicitlyRestartedLanczos < LATTICE_FERMION > IRL (HermOp, Cheb, Nstop,
							    Nk, Nm, resid,
							    MaxIt);
#else
  Grid::Chebyshev < LATTICE_FERMION_F > Cheb (low, high, order);
  Grid::ImplicitlyRestartedLanczos<LATTICE_FERMION_F> IRL(HermOpF,Cheb,Nstop,Nk,Nm,resid,MaxIt);
  precisionChange(grid_rb_in_f,grid_rb_in);
#endif

  std::vector < Grid::RealD > eval (Nm);

  int Nconv;
#ifndef SINGLE
  std::vector < LATTICE_FERMION > evec (Nm, F_RB_GRID);
  for (int i = 0; i < Nm; i++)
    evec[i].checkerboard = Grid::Odd;
  IRL.calc (eval, evec, grid_rb_in, Nconv);
#else
  std::vector < LATTICE_FERMION_F > evec_f (Nm, F_RB_GRID_F);
  for (int i = 0; i < Nm; i++) evec_f[i].checkerboard = Grid::Odd;
  IRL.calc(eval,evec_f,grid_rb_in_f,Nconv);
#endif

  for (int i = 0; i < Nstop; i++) {
    if (f_eigenv[i] == NULL)
      ERR.General (cname, fname, "f_eigenv[%d] not allocated\n", i);
    lambda[i] = eval[i];
#ifndef SINGLE
    setCheckerboard (grid_out, evec[i]);
    ImportFermion (f_eigenv[i], grid_out, Odd);
#else
    setCheckerboard (grid_out_f, evec_f[i]);
    precisionChange(grid_out,grid_out_f);
    ImportFermion (f_eigenv[i], grid_out, Odd);
#endif
  }

  return Nconv;

}
#undef SINGLE
