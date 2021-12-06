#include <alg/a2a/ktopipi_gparity.h>
#include <alg/alg_tcharge.h>
#include <Grid.h>

using namespace cps;

inline int toInt(const char* a){
  std::stringstream ss; ss << a; int o; ss >> o;
  return o;
}

void setupDoArg(DoArg &do_arg, int size[5], int ngp, bool verbose = false){
  do_arg.x_sites = size[0];
  do_arg.y_sites = size[1];
  do_arg.z_sites = size[2];
  do_arg.t_sites = size[3];
  do_arg.s_sites = size[4];
  do_arg.x_node_sites = 0;
  do_arg.y_node_sites = 0;
  do_arg.z_node_sites = 0;
  do_arg.t_node_sites = 0;
  do_arg.s_node_sites = 0;
  do_arg.x_nodes = 0;
  do_arg.y_nodes = 0;
  do_arg.z_nodes = 0;
  do_arg.t_nodes = 0;
  do_arg.s_nodes = 0;
  do_arg.updates = 0;
  do_arg.measurements = 0;
  do_arg.measurefreq = 0;
  do_arg.cg_reprod_freq = 10;
  do_arg.x_bc = BND_CND_PRD;
  do_arg.y_bc = BND_CND_PRD;
  do_arg.z_bc = BND_CND_PRD;
  do_arg.t_bc = BND_CND_APRD;
  do_arg.start_conf_kind = START_CONF_ORD;
  do_arg.start_conf_load_addr = 0x0;
  do_arg.start_seed_kind = START_SEED_FIXED;
  do_arg.start_seed_filename = "../rngs/ckpoint_rng.0";
  do_arg.start_conf_filename = "../configurations/ckpoint_lat.0";
  do_arg.start_conf_alloc_flag = 6;
  do_arg.wfm_alloc_flag = 2;
  do_arg.wfm_send_alloc_flag = 2;
  do_arg.start_seed_value = 83209;
  do_arg.beta =   2.25;
  do_arg.c_1 =   -3.3100000000000002e-01;
  do_arg.u0 =   1.0000000000000000e+00;
  do_arg.dwf_height =   1.8000000000000000e+00;
  do_arg.dwf_a5_inv =   1.0000000000000000e+00;
  do_arg.power_plaq_cutoff =   0.0000000000000000e+00;
  do_arg.power_plaq_exponent = 0;
  do_arg.power_rect_cutoff =   0.0000000000000000e+00;
  do_arg.power_rect_exponent = 0;
  do_arg.verbose_level = -1202; //VERBOSE_DEBUG_LEVEL; //-1202;
  do_arg.checksum_level = 0;
  do_arg.exec_task_list = 0;
  do_arg.xi_bare =   1.0000000000000000e+00;
  do_arg.xi_dir = 3;
  do_arg.xi_v =   1.0000000000000000e+00;
  do_arg.xi_v_xi =   1.0000000000000000e+00;
  do_arg.clover_coeff =   0.0000000000000000e+00;
  do_arg.clover_coeff_xi =   0.0000000000000000e+00;
  do_arg.xi_gfix =   1.0000000000000000e+00;
  do_arg.gfix_chkb = 1;
  do_arg.asqtad_KS =   0.0000000000000000e+00;
  do_arg.asqtad_naik =   0.0000000000000000e+00;
  do_arg.asqtad_3staple =   0.0000000000000000e+00;
  do_arg.asqtad_5staple =   0.0000000000000000e+00;
  do_arg.asqtad_7staple =   0.0000000000000000e+00;
  do_arg.asqtad_lepage =   0.0000000000000000e+00;
  do_arg.p4_KS =   0.0000000000000000e+00;
  do_arg.p4_knight =   0.0000000000000000e+00;
  do_arg.p4_3staple =   0.0000000000000000e+00;
  do_arg.p4_5staple =   0.0000000000000000e+00;
  do_arg.p4_7staple =   0.0000000000000000e+00;
  do_arg.p4_lepage =   0.0000000000000000e+00;

  if(verbose) do_arg.verbose_level = VERBOSE_DEBUG_LEVEL;

  BndCndType* bc[3] = { &do_arg.x_bc, &do_arg.y_bc, &do_arg.z_bc };
  for(int i=0;i<ngp;i++){ 
    *(bc[i]) = BND_CND_GPARITY;
  }
}


namespace Grid{
  
  template <class Gimpl> class WilsonLoopsExt : public Gimpl {
public:
    INHERIT_GIMPL_TYPES(Gimpl);
    
    typedef typename Gimpl::GaugeLinkField GaugeMat;
    typedef typename Gimpl::GaugeField GaugeLorentz;

#define Fmu(A) Gimpl::CovShiftForward(Umu, mu, A)
#define Bmu(A) Gimpl::CovShiftBackward(Umu, mu, A)
#define Fnu(A) Gimpl::CovShiftForward(Unu, nu, A)
#define Bnu(A) Gimpl::CovShiftBackward(Unu, nu, A)
#define FmuI Gimpl::CovShiftIdentityForward(Umu, mu)
#define BmuI Gimpl::CovShiftIdentityBackward(Umu, mu)
#define FnuI Gimpl::CovShiftIdentityForward(Unu, nu)
#define BnuI Gimpl::CovShiftIdentityBackward(Unu, nu)

    inline static int Sign(int a){ 
      if(a < 0) return -1;
      else if(a == 0) return 0;
      else return 1;
    }
    static int LeviCivita(int mu, int nu, int rho, int sigma){
      return Sign(nu-mu)*Sign(rho-mu)*Sign(sigma-mu)*
	Sign(rho-nu)*Sign(sigma-nu)*
	Sign(sigma-rho);
    }


    //Reproduction of 1x1
    static void FieldStrength1x1_H(GaugeMat &FS, const GaugeMat &Umu, const GaugeMat &Unu, int mu, int nu){      
      // FS =  
      // 	Fnu( Fmu( Bnu( BmuI)))
      // 	+ 
      // 	Bmu( Fnu( Fmu( BnuI)))
      // 	+
      // 	Fmu( Bnu( Bmu( FnuI)))
      // 	+
      // 	Bnu( Bmu( Fnu( FmuI)));

      FS = Fmu( Fnu( Bmu( BnuI))) +
      	Fnu( Bmu( Bnu( FmuI))) +
      	Bmu( Bnu( Fmu( FnuI))) +
      	Bnu( Fmu( Fnu( BmuI))); 
    }

    //adj( F_munu ) = - F_munu
    //F_munu = -F_numu
    static void FieldStrength1x1_2(GaugeMat &FS, const GaugeLorentz &U, int mu, int nu){      
      GaugeMat Umu = PeekIndex<LorentzIndex>(U, mu);
      GaugeMat Unu = PeekIndex<LorentzIndex>(U, nu);
      GaugeMat horizontal(Umu.Grid()), vertical(Umu.Grid());
      FieldStrength1x1_H(horizontal, Umu, Unu, mu, nu);
      //FieldStrength1x1_H(vertical, Unu, Umu, nu, mu); // V_munu = adj(H_numu)
      //vertical = adj(vertical);     
      //FS = 0.125 * ( imag(horizontal) + imag(vertical) );
      //FS = 0.25 * imag(horizontal);
      FS = 0.125 * ( horizontal - adj(horizontal) );
      
    }

    static Real TopologicalCharge1x1(GaugeLorentz &U){
      std::vector<std::vector<GaugeMat*> > F(Nd,std::vector<GaugeMat*>(Nd));
      //Note F_numu = -F_munu
      for(int mu=0;mu<Nd-1;mu++){
	for(int nu=mu+1; nu<Nd; nu++){
	  F[mu][nu] = new GaugeMat(U.Grid());
	  FieldStrength1x1_2(*F[mu][nu], U, mu, nu);

	  F[nu][mu] = new GaugeMat(U.Grid());
	  //*F[nu][mu] = -(*F[mu][nu]);
	  FieldStrength1x1_2(*F[nu][mu], U, nu, mu);
	}
      }
      GaugeMat tmp(U.Grid());

      Real coeff = 1./(32 * M_PI*M_PI);

      ComplexField fsum(U.Grid());
      fsum = Zero();
      for(int mu=0;mu<Nd;mu++){
	for(int nu=0;nu<Nd;nu++){
	  for(int rho=0;rho<Nd;rho++){
	    for(int sigma=0;sigma<Nd;sigma++){
	      int eps = LeviCivita(mu,nu,rho,sigma);
	      //std::cout << "eps: " << mu << " " << nu << " " << rho << " " << sigma << " : " << eps << std::endl;

	      if(eps == 0) continue;
	      fsum = fsum + double(eps) * coeff * trace( (*F[mu][nu]) * (*F[rho][sigma]) );
	    }
	  }
	}
      }

      for(int mu=0;mu<Nd-1;mu++){
	for(int nu=mu+1; nu<Nd; nu++){
	  delete F[mu][nu];
	  delete F[nu][mu];
	}
      }

      auto Tq = sum(fsum);
      return TensorRemove(Tq).real();
    }















    //https://arxiv.org/pdf/hep-lat/9701012.pdf Eq 7  for 1x2 Wilson loop
    static void FieldStrength1x2(GaugeMat &FS, const GaugeLorentz &U, int mu, int nu){      
      GaugeMat Umu = PeekIndex<LorentzIndex>(U, mu);
      GaugeMat Unu = PeekIndex<LorentzIndex>(U, nu);     

      GaugeMat horizontal = 
	Fnu( Fmu( Fmu( Bnu( Bmu( BmuI)))))
	+ 
	Bmu( Bmu( Fnu( Fmu( Fmu( BnuI)))))
	+
	Fmu( Fmu( Bnu( Bmu( Bmu( FnuI)))))
	+
	Bnu( Bmu( Bmu( Fnu( Fmu( FmuI)))));

      GaugeMat vertical = 
	Fnu( Fnu( Fmu( Bnu( Bnu( BmuI)))))
	+
	Bmu( Fnu( Fnu( Fmu( Bnu( BnuI)))))
	+
	Fmu( Bnu( Bnu( Bmu( Fnu( FnuI)))))
	+
	Bnu( Bnu( Bmu( Fnu( Fnu( FmuI)))));
      
      //FS = 0.125 * ( imag(horizontal) + imag(vertical) );
      FS = 0.0625 * ( horizontal - adj(horizontal) + vertical - adj(vertical) );
      
    }

    //Horizontal Wilson loop component of 1x2 field strength
    //https://arxiv.org/pdf/hep-lat/9701012.pdf Eq 7  for 1x2 Wilson loop
    static void FieldStrength1x2_H(GaugeMat &FS, const GaugeMat &Umu, const GaugeMat &Unu, int mu, int nu){      
      FS =  
	Fnu( Fmu( Fmu( Bnu( Bmu( BmuI)))))
	+ 
	Bmu( Bmu( Fnu( Fmu( Fmu( BnuI)))))
	+
	Fmu( Fmu( Bnu( Bmu( Bmu( FnuI)))))
	+
	Bnu( Bmu( Bmu( Fnu( Fmu( FmuI)))));
    }

    //Note F_numu = - F_munu
    static void FieldStrength1x2_2(GaugeMat &FS, const GaugeLorentz &U, int mu, int nu){      
      GaugeMat Umu = PeekIndex<LorentzIndex>(U, mu);
      GaugeMat Unu = PeekIndex<LorentzIndex>(U, nu);
      GaugeMat horizontal(Umu.Grid()), vertical(Umu.Grid());
      FieldStrength1x2_H(horizontal, Umu, Unu, mu, nu);
      FieldStrength1x2_H(vertical, Unu, Umu, nu, mu); // V_munu = adj(H_numu)
      vertical = adj(vertical);     
      //FS = 0.125 * ( imag(horizontal) + imag(vertical) );
      FS = 0.0625 * ( horizontal - adj(horizontal) + vertical - adj(vertical) );
    }

    static Real TopologicalCharge1x2(GaugeLorentz &U){
      std::vector<std::vector<GaugeMat*> > F(Nd,std::vector<GaugeMat*>(Nd));
      //Note F_numu = - F_munu
      for(int mu=0;mu<Nd-1;mu++){
	for(int nu=mu+1; nu<Nd; nu++){
	  F[mu][nu] = new GaugeMat(U.Grid());
	  FieldStrength1x2_2(*F[mu][nu], U, mu, nu);

	  F[nu][mu] = new GaugeMat(U.Grid());
	  FieldStrength1x2_2(*F[nu][mu], U, nu, mu);
	  //*F[nu][mu] = -*F[mu][nu];
	}
      }
      GaugeMat tmp(U.Grid());

      Real coeff = 1./(32 * M_PI*M_PI *4);

      ComplexField fsum(U.Grid());
      fsum = Zero();
      for(int mu=0;mu<Nd;mu++){
	for(int nu=0;nu<Nd;nu++){
	  for(int rho=0;rho<Nd;rho++){
	    for(int sigma=0;sigma<Nd;sigma++){
	      int eps = LeviCivita(mu,nu,rho,sigma);
	      if(eps == 0) continue;
	      fsum = fsum + double(eps) * coeff * trace( (*F[mu][nu]) * (*F[rho][sigma]) );
	    }
	  }
	}
      }

      for(int mu=0;mu<Nd-1;mu++){
	for(int nu=mu+1; nu<Nd; nu++){
	  delete F[mu][nu];
	  delete F[nu][mu];
	}
      }

      auto Tq = sum(fsum);
      return TensorRemove(Tq).real();
    }







    
    //Cloverleaf field strength term for arbitrary mu-extent M and nu extent N
    //cf  https://arxiv.org/pdf/hep-lat/9701012.pdf Eq 7  for 1x2 Wilson loop    
    //Clockwise ordering
    //Vertical orientation can be obtained from the relation V_munu = adj(H_numu)    where H is the horizontal orientation and V the vertical
    static void CloverleafMxN(GaugeMat &FS, const GaugeMat &Umu, const GaugeMat &Unu, int mu, int nu, int M, int N){  
      //Upper right loop
      GaugeMat tmp = BmuI;
      for(int i=1;i<M;i++)
	tmp = Bmu(tmp);
      for(int j=0;j<N;j++)
	tmp = Bnu(tmp);
      for(int i=0;i<M;i++)
	tmp = Fmu(tmp);
      for(int j=0;j<N;j++)
	tmp = Fnu(tmp);
      
      FS = tmp;

      //Upper left loop
      tmp = BnuI;
      for(int j=1;j<N;j++)
	tmp = Bnu(tmp);
      for(int i=0;i<M;i++)
	tmp = Fmu(tmp);
      for(int j=0;j<N;j++)
	tmp = Fnu(tmp);
      for(int i=0;i<M;i++)
	tmp = Bmu(tmp);
      
      FS = FS + tmp;

      //Lower right loop
      tmp = FnuI;
      for(int j=1;j<N;j++)
	tmp = Fnu(tmp);
      for(int i=0;i<M;i++)
	tmp = Bmu(tmp);
      for(int j=0;j<N;j++)
	tmp = Bnu(tmp);
      for(int i=0;i<M;i++)
	tmp = Fmu(tmp);
      
      FS = FS + tmp;

      //Lower left loop
      tmp = FmuI;
      for(int i=1;i<M;i++)
	tmp = Fmu(tmp);
      for(int j=0;j<N;j++)
	tmp = Fnu(tmp);
      for(int i=0;i<M;i++)
	tmp = Bmu(tmp);
      for(int j=0;j<N;j++)
	tmp = Bnu(tmp);

      FS = FS + tmp;
    }

    //Field strength from MxN Wilson loop
    //Note F_numu = - F_munu
    static void FieldStrengthMxN(GaugeMat &FS, const GaugeLorentz &U, int mu, int nu, int M, int N){  
      GaugeMat Umu = PeekIndex<LorentzIndex>(U, mu);
      GaugeMat Unu = PeekIndex<LorentzIndex>(U, nu);
      if(M == N){
	GaugeMat F(Umu.Grid());
	CloverleafMxN(F, Umu, Unu, mu, nu, M, N);
	FS = 0.125 * ( F - adj(F) );
      }else{
	//Average over two different orientations
	GaugeMat horizontal(Umu.Grid()), vertical(Umu.Grid());
	CloverleafMxN(horizontal, Umu, Unu, mu, nu, M, N);
	CloverleafMxN(vertical, Umu, Unu, mu, nu, N, M);
	//vertical = adj(vertical);     
	FS = 0.0625 * ( horizontal - adj(horizontal) + vertical - adj(vertical) );
      }
    }

    static Real TopologicalChargeMxNbasic(GaugeLorentz &U, int M, int N){
      std::vector<std::vector<GaugeMat*> > F(Nd,std::vector<GaugeMat*>(Nd));
      //Note F_numu = - F_munu
      for(int mu=0;mu<Nd-1;mu++){
	for(int nu=mu+1; nu<Nd; nu++){
	  F[mu][nu] = new GaugeMat(U.Grid());
	  FieldStrengthMxN(*F[mu][nu], U, mu, nu, M, N);

	  F[nu][mu] = new GaugeMat(U.Grid());
	  FieldStrengthMxN(*F[nu][mu], U, nu, mu, M, N);
	  //*F[nu][mu] = -*F[mu][nu];
	}
      }
      GaugeMat tmp(U.Grid());

      Real coeff = -1./(32 * M_PI*M_PI * M*M * N*N); //overall sign to match CPS and Grid conventions, possibly related to time direction = 3 vs 0

      ComplexField fsum(U.Grid());
      fsum = Zero();
      for(int mu=0;mu<Nd;mu++){
	for(int nu=0;nu<Nd;nu++){
	  for(int rho=0;rho<Nd;rho++){
	    for(int sigma=0;sigma<Nd;sigma++){
	      int eps = LeviCivita(mu,nu,rho,sigma);
	      if(eps == 0) continue;
	      fsum = fsum + double(eps) * coeff * trace( (*F[mu][nu]) * (*F[rho][sigma]) );
	    }
	  }
	}
      }

      for(int mu=0;mu<Nd-1;mu++){
	for(int nu=mu+1; nu<Nd; nu++){
	  delete F[mu][nu];
	  delete F[nu][mu];
	}
      }

      auto Tq = sum(fsum);
      return TensorRemove(Tq).real();
    }

    //Topological charge contribution from MxN Wilson loops
    //cf  https://arxiv.org/pdf/hep-lat/9701012.pdf  Eq 6
    static Real TopologicalChargeMxN(GaugeLorentz &U, int M, int N){
      assert(Nd == 4);
      std::vector<std::vector<GaugeMat*> > F(Nd,std::vector<GaugeMat*>(Nd));
      //Note F_numu = - F_munu
      //hence we only need to loop over mu,nu,rho,sigma that aren't related by permuting mu,nu  or rho,sigma
      //Use nu > mu
      for(int mu=0;mu<Nd-1;mu++){
	for(int nu=mu+1; nu<Nd; nu++){
	  F[mu][nu] = new GaugeMat(U.Grid());
	  FieldStrengthMxN(*F[mu][nu], U, mu, nu, M, N);
	}
      }
      Real coeff = -1./(32 * M_PI*M_PI * M*M * N*N); //overall sign to match CPS and Grid conventions, possibly related to time direction = 3 vs 0

      static const int combs[3][4] = { {0,1,2,3}, {0,2,1,3}, {0,3,1,2} };
      static const int signs[3] = { 1, -1, 1 }; //epsilon_{mu nu rho sigma}

      ComplexField fsum(U.Grid());
      fsum = Zero();
      for(int c=0;c<3;c++){
	int mu = combs[c][0], nu = combs[c][1], rho = combs[c][2], sigma = combs[c][3];
	int eps = signs[c];
	fsum = fsum + (8. * coeff * eps) * trace( (*F[mu][nu]) * (*F[rho][sigma]) ); 
      }

      for(int mu=0;mu<Nd-1;mu++)
	for(int nu=mu+1; nu<Nd; nu++)
	  delete F[mu][nu];

      auto Tq = sum(fsum);
      return TensorRemove(Tq).real();
    }

    //Generate the contributions to the 5Li topological charge from Wilson loops of the following sizes
    //Use coefficients from hep-lat/9701012
    //1x1 : c1=(19.-55.*c5)/9.
    //2x2 : c2=(1-64.*c5)/9.
    //1x2 : c3=(-64.+640.*c5)/45.
    //1x3 : c4=1./5.-2.*c5
    //3x3 : c5=1./20.
    //Output array contains the loops in the above order
    static std::vector<Real> TopologicalCharge5LiContributions(GaugeLorentz &U){
      static const int exts[5][2] = { {1,1}, {2,2}, {1,2}, {1,3}, {3,3} };       
      std::vector<Real> out(5);
      std::cout << GridLogMessage << "Computing topological charge" << std::endl;
      for(int i=0;i<5;i++){	
	out[i] = TopologicalChargeMxN(U,exts[i][0],exts[i][1]);
	std::cout << GridLogMessage << exts[i][0] << "x" << exts[i][1] << " Wilson loop contribution " << out[i] << std::endl;
      }
      return out;
    }

    //Compute the 5Li topological charge
    static Real TopologicalCharge5Li(GaugeLorentz &U){
      std::vector<Real> loops = TopologicalCharge5LiContributions(U);

      double c5=1./20.;
      double c4=1./5.-2.*c5;
      double c3=(-64.+640.*c5)/45.;
      double c2=(1-64.*c5)/9.;
      double c1=(19.-55.*c5)/9.;

      double Q = c1*loops[0] + c2*loops[1] + c3*loops[2] + c4*loops[3] + c5*loops[4];

      std::cout << GridLogMessage << "5Li Topological charge: " << Q << std::endl;
      return Q;
    }





  };
}

template<typename ActionType>
struct setupAction{};

template<>
struct setupAction<GnoneFgridGparityMobius>{
  static inline GnoneFgridGparityMobius* run(FgridParams grid_params){
    //Gauge BCs
    std::vector<int> conjDirs(4,0);
    for(int i=0;i<3;i++) conjDirs[i] = (GJP.Bc(i) == BND_CND_GPARITY ? 1 : 0);
    Grid::ConjugateGimplD::setDirections(conjDirs);

    return new GnoneFgridGparityMobius(grid_params);
  }
};

template<>
struct setupAction<GnoneFgridMobius>{
  static inline GnoneFgridMobius* run(FgridParams grid_params){
    return new GnoneFgridMobius(grid_params);
  }
};




template<typename FgridType, typename Gimpl>
void run(int ngp){
  CommonArg common_arg;

  //Setup SIMD info
  int nsimd = Grid::vComplexD::Nsimd();
  typename SIMDpolicyBase<4>::ParamType simd_dims;
  SIMDpolicyBase<4>::SIMDdefaultLayout(simd_dims,nsimd,2); //only divide over spatial directions
  
  typename SIMDpolicyBase<3>::ParamType simd_dims_3d;
  SIMDpolicyBase<3>::SIMDdefaultLayout(simd_dims_3d,nsimd);
  
  //Setup lattice
  FgridParams grid_params; 
  grid_params.mobius_scale = 2.0;
  FgridType *lat_ptr = setupAction<FgridType>::run(grid_params);
  FgridType &lattice = *lat_ptr;

  lattice.SetGfieldDisOrd();
  lattice.ImportGauge(); //lattice -> Grid  

  typedef Grid::WilsonLoops<Gimpl> GridWloops;
  typedef Grid::WilsonLoopsExt<Gimpl> GridWloopsExt;

  double grid_avgP = GridWloops::avgPlaquette(*lattice.getUmu());
  
  double cps_avgP = lattice.SumReTrPlaq();
  int Nd = 4;
  int Nc = 3;
  cps_avgP /=  GJP.VolSites() *  ( (1.0 * Nd * (Nd - 1)) / 2.0 ) * Nc;
   
  std::cout << "Ensure plaquette agrees between Grid and CPS: " <<  grid_avgP << " " << cps_avgP << std::endl;
  assert( fabs( ( grid_avgP - cps_avgP )/cps_avgP ) < 1e-5 );

  std::vector<Float> Qmn_cps;
  std::vector<std::vector<Float> > Qmn_slice_cps;
  
  AlgTcharge alg_tcharge(lattice, &common_arg);
  alg_tcharge.run(Qmn_cps, Qmn_slice_cps);

  double Q11_grid = GridWloops::TopologicalCharge(*lattice.getUmu());
  double Q11_grid_repro = GridWloopsExt::TopologicalCharge1x1(*lattice.getUmu());
  double Q11_grid_gen = GridWloopsExt::TopologicalChargeMxN(*lattice.getUmu(),1,1);

  std::cout << "1x1 CPS: " << Qmn_cps[0] << " Grid: " << Q11_grid << " Grid repro: " << Q11_grid_repro << " Grid gen: " << Q11_grid_gen << std::endl;

  typedef typename Gimpl::GaugeLinkField GaugeMat;
  typedef typename Gimpl::GaugeLinkField GaugeLorentz;
  
  GaugeMat FS01(lattice.getUGrid()), FS01_2(lattice.getUGrid()), FS10(lattice.getUGrid());

  GridWloopsExt::FieldStrength1x2(FS01, *lattice.getUmu(), 0, 1);
  GridWloopsExt::FieldStrength1x2_2(FS01_2, *lattice.getUmu(), 0, 1);

  
  GaugeMat tmp = FS01 - FS01_2;
  std::cout << "Check 1x2 difference between implementations: " << norm2(tmp) << std::endl;

  GridWloopsExt::FieldStrength1x2(FS10, *lattice.getUmu(), 1, 0);  

  tmp = FS01 + FS10;
  std::cout << "Check 1x2  F_01 = -F_10 << " <<  norm2(FS01) << " " << norm2(FS10) << " Diff: " << norm2(tmp) << std::endl;
  

  double Q12_grid = GridWloopsExt::TopologicalCharge1x2(*lattice.getUmu());
  double Q12_grid_gen = GridWloopsExt::TopologicalChargeMxN(*lattice.getUmu(),1,2);
  
  std::cout << "1x2 CPS: " << Qmn_cps[1] << " Grid: " << Q12_grid << " Grid gen: " << Q12_grid_gen << std::endl;

  GaugeMat F1x1_01_orig(lattice.getUGrid()), F1x1_01_repro(lattice.getUGrid());
  GridWloops::FieldStrength(F1x1_01_orig, *lattice.getUmu(), 0, 1);
  GridWloopsExt::FieldStrength1x1_2(F1x1_01_repro, *lattice.getUmu(), 0, 1);
  //GridWloopsExt::FieldStrength1x1_2(F1x1_01_repro, *lattice.getUmu(), 1, 0);

  tmp = F1x1_01_orig - F1x1_01_repro;
  std::cout << "Check 1x1 field strength orig " <<  norm2(F1x1_01_orig) << " repro " << norm2(F1x1_01_repro) << " Diff: " << norm2(tmp) << std::endl;

  std::cout << "Full check Grid gen vs CPS:" << std::endl;

  const char* names[5] = { "1x1",
			   "1x2",
			   "2x2",
			   "3x3",
			   "1x3" };
  
  std::vector<std::pair<int,int> > exts = { {1,1}, {1,2}, {2,2}, {3,3}, {1,3} };
  for(int i=0;i<5;i++){
    double Q_grid_gen = GridWloopsExt::TopologicalChargeMxN(*lattice.getUmu(),exts[i].first,exts[i].second);
    std::cout << names[i] << " CPS: " << Qmn_cps[i] << " Grid: " << Q_grid_gen << std::endl;
  }

  std::vector<double> test = GridWloopsExt::TopologicalCharge5LiContributions(*lattice.getUmu());
  
  std::cout << "Test 1x1 Wloop output against Grid top Q: " << test[0] << " " << Q11_grid << std::endl;

  //Note ordering of Grid data is different from CPS:  
  //Grid 1x1, 2x2, 1x2, 1x3, 3x3
  //CPS  1x1, 1x2, 2x2, 3x3, 1x3

  int grid_cps_map[5] = { 0, 2, 1, 4, 3 };

  std::vector<double> test2 = GridWloops::TopologicalCharge5LiContributions(*lattice.getUmu());
  std::cout << "Test new topological charge code imported into Grid" << std::endl;
  for(int i=0;i<5;i++){
    std::cout << i << " Grid imported : " << test2[i] << "  Grid test code : " << test[i] << " CPS : " << Qmn_cps[ grid_cps_map[i] ] << std::endl;
  }

  std::cout << "Done" << std::endl;

  delete lat_ptr;
}

int main(int argc,char *argv[])
{
  Start(&argc, &argv);
  assert(argc >= 2);

  int ngp = std::stoi(argv[1]);
  int size[5] = {4,4,4,8,12};

  CommonArg common_arg;
  DoArg do_arg;  setupDoArg(do_arg,size,ngp);

  GJP.Initialize(do_arg);

  if(ngp>0){
    run<GnoneFgridGparityMobius, Grid::ConjugateGimplD>(ngp);
  }else{
    run<GnoneFgridMobius, Grid::PeriodicGimplD>(ngp);
  }

  End();
  return 0;
}
