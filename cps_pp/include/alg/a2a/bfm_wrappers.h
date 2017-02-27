#ifndef _BFM_WRAPPERS_H_
#define _BFM_WRAPPERS_H_
#ifdef USE_BFM

#include<util/lattice.h>
#include<util/lattice/bfm_mixed_solver.h>
#include<util/lattice/bfm_evo.h>
#include<alg/eigen/Krylov_5d.h>
#include<alg/a2a/CPSfield.h>

CPS_START_NAMESPACE

//Hold bfm instances
struct BFMsolvers{
  bfm_evo<double> dwf_d;
  bfm_evo<float> dwf_f;
  bfmarg dwfa;

  template<typename mf_Float>
  static void setMass(bfm_evo<mf_Float> &dwf, const double mass){
    dwf.mass = mass;
    dwf.GeneralisedFiveDimEnd(); // reinitialising since using a new mass
    dwf.GeneralisedFiveDimInit();
  }
  
  static void setup_bfmargs(bfmarg &dwfa, int nthread, const double mass, const double residual, const int max_iter, const BfmSolver &solver = HmCayleyTanh, const double mobius_scale = 1.){
    if(!UniqueID()) printf("Setting up bfmargs\n");

    omp_set_num_threads(nthread);

    dwfa.node_latt[0] = GJP.XnodeSites();
    dwfa.node_latt[1] = GJP.YnodeSites();
    dwfa.node_latt[2] = GJP.ZnodeSites();
    dwfa.node_latt[3] = GJP.TnodeSites();
    multi1d<int> ncoor(4);
    multi1d<int> procs(4);
    for(int i=0;i<4;i++){ ncoor[i] = GJP.NodeCoor(i); procs[i] = GJP.Nodes(i); }

    if(GJP.Gparity()){
      dwfa.gparity = 1;
      if(!UniqueID()) printf("G-parity directions: ");
      for(int d=0;d<3;d++)
	if(GJP.Bc(d) == BND_CND_GPARITY){ dwfa.gparity_dir[d] = 1; printf("%d ",d); }
	else dwfa.gparity_dir[d] = 0;
      for(int d=0;d<4;d++){
	dwfa.nodes[d] = procs[d];
	dwfa.ncoor[d] = ncoor[d];
      }
      if(!UniqueID()) printf("\n");
    }

    dwfa.verbose=1;
    dwfa.reproduce=0;
    bfmarg::Threads(nthread);
    bfmarg::Reproduce(0);
    bfmarg::ReproduceChecksum(0);
    bfmarg::ReproduceMasterCheck(0);
    bfmarg::Verbose(1);

    for(int mu=0;mu<4;mu++){
      if ( procs[mu]>1 ) {
	dwfa.local_comm[mu] = 0;
	if(!UniqueID()) printf("Non-local comms in direction %d\n",mu);
      } else {
	dwfa.local_comm[mu] = 1;
	if(!UniqueID()) printf("Local comms in direction %d\n",mu);
      }
    }

    dwfa.precon_5d = 1;
    if(solver == HmCayleyTanh){
      dwfa.precon_5d = 0; //mobius uses 4d preconditioning
      dwfa.mobius_scale = mobius_scale;
    }
    assert(GJP.Snodes() == 1);
    dwfa.Ls = GJP.SnodeSites();
    dwfa.solver = solver;
    dwfa.M5 = toDouble(GJP.DwfHeight());
    dwfa.mass = mass;
    dwfa.Csw = 0.0;
    dwfa.max_iter = max_iter;
    dwfa.residual = residual;
    if(!UniqueID()) printf("Finished setting up bfmargs\n");
  }
  
  BFMsolvers(const int nthreads, const double mass, const double residual, const int max_iters, const BfmSolverType bfm_solver, const double mobius_scale){
    //Initialize both a double and single precision instance of BFM
    BfmSolver solver;
    switch(bfm_solver){
    case BFM_DWF:
      solver = DWF; break;
    case BFM_HmCayleyTanh:
      solver = HmCayleyTanh; break;
    default:
      ERR.General("LatticeSolvers","constructor","Unknown solver\n");
    }
    setup_bfmargs(dwfa, nthreads, mass, residual, max_iters, solver, mobius_scale);
#ifdef USE_NEW_BFM_GPARITY
    dwf_d.threads = dwf_f.threads = nthreads;
#endif
    dwf_d.init(dwfa);
    dwf_d.comm_end(); dwf_f.init(dwfa); dwf_f.comm_end(); dwf_d.comm_init();
  }

  ~BFMsolvers(){
    dwf_d.end();
    dwf_f.end();
  }
  //Import CPS gauge field into BFM
  void importLattice(Lattice *lat){
    lat->BondCond(); //Apply the boundary conditions!
    Float* gauge = (Float*) lat->GaugeField();
    dwf_d.cps_importGauge(gauge);
    dwf_d.comm_end(); 
    dwf_f.comm_init(); dwf_f.cps_importGauge(gauge); dwf_f.comm_end(); 
    dwf_d.comm_init();
    lat->BondCond(); //Un-apply the boundary conditions! 
  }
};




//Wrap the Lanczos
struct BFMLanczosWrapper{
  BFM_Krylov::Lanczos_5d<double> *eig;

  BFMLanczosWrapper(): eig(NULL){}
  
  void compute(const LancArg &lanc_arg, BFMsolvers &solvers){
    eig = new BFM_Krylov::Lanczos_5d<double>(solvers.dwf_d,const_cast<LancArg&>(lanc_arg)); //sets up the mass of dwf_d correctly
    eig->Run();

    solvers.setMass(solvers.dwf_f, solvers.dwf_d.mass); //keep the single-prec solver in sync
    test_eigenvectors(*eig,solvers.dwf_d,false);
  }

  static void test_eigenvectors(BFM_Krylov::Lanczos_5d<double> &eig, bfm_evo<double> & dwf, bool singleprec_evecs){
    const int len = 24 * dwf.node_cbvol * (1 + dwf.gparity) * dwf.cbLs;
#ifdef USE_NEW_BFM_GPARITY
    omp_set_num_threads(dwf.threads);
    if(!UniqueID()) printf("test_eigenvectors set omp threads %d to bfm threads %d\n",omp_get_max_threads(),dwf.threads);
#else
    omp_set_num_threads(bfmarg::threads);	
#endif
  
    Fermion_t bq_tmp = singleprec_evecs ? dwf.allocCompactFermion() : dwf.allocFermion(); 
    Fermion_t tmp1 = dwf.allocFermion();
    Fermion_t tmp2 = dwf.allocFermion();
    Fermion_t tmp3 = dwf.allocFermion();

    if(!UniqueID()) printf("Computing eigenvector residuals\n");

    for(int i=0;i<eig.get;i++){
      if(singleprec_evecs){ // eig->bq is in single precision
#pragma omp parallel for  //Bet I could reduce the threading overheads by parallelizing this entire method
	for(int j = 0; j < len; j++) {
	  ((double*)bq_tmp)[j] = ((float*)(eig.bq[i][1]))[j];
	}
      }else{
#pragma omp parallel
	{
	  dwf.axpy(bq_tmp, eig.bq[i][1], eig.bq[i][1], 0.);
	}
      }
    
      double nrm_boss;
#pragma omp parallel
      {
	dwf.Mprec(bq_tmp,tmp1,tmp3, 0);
	dwf.Mprec(tmp1, tmp2, tmp3, 1); //tmp2 = M M^dag v
      
	//M M^dag v = lambda v
	dwf.set_zero(tmp1);	
	dwf.axpy(tmp3, bq_tmp, tmp1, eig.evals[i]); //tmp3 = lambda v
      
	double nrm = dwf.axpy_norm(tmp1, tmp2, tmp3, -1.); //tmp1 = tmp3 - tmp2
	if(dwf.isBoss()) nrm_boss = sqrt(nrm); //includes global sum
      }
      if(!UniqueID()) printf("%d %g\n",i,nrm_boss);
    }
  
    dwf.freeFermion(bq_tmp);
    dwf.freeFermion(tmp1);
    dwf.freeFermion(tmp2);
    dwf.freeFermion(tmp3);
  }
  
  void toSingle(){
    eig->toSingle(); 
    //Test the single-prec converted eigenvectors to make sure we haven't dropped too much precision
    test_eigenvectors(*eig,eig->dop,true);
  }

  void writeParallel(const std::string &file_stub, FP_FORMAT fileformat = FP_AUTOMATIC) const{
    ERR.General("BFMLanczosWrapper","writeParallel","Not yet implemented\n");
  }  
  void readParallel(const std::string &file_stub){
    ERR.General("BFMLanczosWrapper","readParallel","Not yet implemented\n");
  }
    
  
  void freeEvecs(){
    eig->free_bq();
  }

  void randomizeEvecs(const LancArg &lanc_arg, BFMsolvers &solvers){
    assert(eig == NULL);
    eig = new BFM_Krylov::Lanczos_5d<double>(solvers.dwf_d,const_cast<LancArg&>(lanc_arg));
    solvers.dwf_f.mass = solvers.dwf_d.mass; //keep the single-prec solver in sync
    solvers.dwf_f.GeneralisedFiveDimEnd(); // reinitialising since using a new mass
    solvers.dwf_f.GeneralisedFiveDimInit();

    eig->bq.resize(eig->get);
    eig->evals.resize(eig->get);
    
    CPSfermion5D<cps::ComplexD> tmp;
    if(!UniqueID()) printf("BFM gparity is %d, nsimd = %d, precon_5d = %d\n",solvers.dwf_d.gparity,solvers.dwf_d.simd(), solvers.dwf_d.precon_5d);

    IncludeCBsite<5> oddsites(1, solvers.dwf_d.precon_5d);
    IncludeCBsite<5> oddsitesf0(1, solvers.dwf_d.precon_5d, 0);
    IncludeCBsite<5> oddsitesf1(1, solvers.dwf_d.precon_5d, 1);
    IncludeCBsite<5> evensites(0, solvers.dwf_d.precon_5d);
    IncludeCBsite<5> evensitesf0(0, solvers.dwf_d.precon_5d, 0);
    IncludeCBsite<5> evensitesf1(0, solvers.dwf_d.precon_5d, 1);

    for(int i=0;i<eig->get;i++){
      eig->bq[i][0] = eig->bq[i][1] = NULL;
#pragma omp parallel
      {
	for(int p=eig->prec; p<2; p++) eig->bq[i][p] = solvers.dwf_d.threadedAllocCompactFermion();
      }
      tmp.setGaussianRandom();
      double nrmcps = tmp.norm2();
      double nrmoddcps = tmp.norm2(oddsites);
      double nrmoddf0cps = tmp.norm2(oddsitesf0);
      double nrmoddf1cps = tmp.norm2(oddsitesf1);

      double nrmevencps = tmp.norm2(evensites);
      double nrmevenf0cps = tmp.norm2(evensitesf0);
      double nrmevenf1cps = tmp.norm2(evensitesf1);
      
      for(int p=eig->prec; p<2; p++) tmp.exportFermion<double>(eig->bq[i][p], p, solvers.dwf_d);
      double nrm;
#pragma omp parallel
      {
	double tnrm =solvers.dwf_d.norm(eig->bq[i][1]);	
	if(eig->prec == 0) tnrm += solvers.dwf_d.norm(eig->bq[i][0]);
	if(omp_get_thread_num() == 0) nrm = tnrm;
      }

      eig->evals[i] = LRG.Lrand(10,0.1); //same on all nodes
      if(!UniqueID()) printf("random evec %d BFM norm %g CPS norm %g (odd %g : odd f0 %g, odd f1 %g) (even %g : even f0 %g, even f1 %g) and eval %g\n",i,nrm,nrmcps,nrmoddcps,nrmoddf0cps,nrmoddf1cps,nrmevencps,nrmevenf0cps,nrmevenf1cps,eig->evals[i]);
    }
  }

  
  ~BFMLanczosWrapper(){
    if(eig != NULL)
      delete eig;
  }
};

CPS_END_NAMESPACE


#endif
#endif
