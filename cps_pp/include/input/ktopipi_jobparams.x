class ZMobiusParams{
      A2A_ZMobiusGammaSource gamma_src;  //If A2A_ZMobiusGammaSourceInput use the array provided in this VML, else if A2A_ZMobiusGammaSourceCompute compute it directly
      double compute_lambda_max; //In generating ZMobius approx we need the upper bound of the eigenvalues of the Wilson kernel. 1.42 seems to be a standard number!    
      double gamma_real<>; //real parts of gamma if using A2A_ZMobiusGammaSourceInput
      double gamma_imag<>; //imaginary parts of gamma if using A2A_ZMobiusGammaSourceInput

  rpccommand GENERATE_PRINT_METHOD;
  rpccommand GENERATE_DEEPCOPY_METHOD;

};

class MADWFparams{
  int Ls_inner; //Inner Ls for MADWF
  double b_plus_c_inner; //Inner b+c for MADWF
  A2Apreconditioning precond; //SchurDiagTwo typically converges faster for MADWF (make sure your eigenvectors are also computed using this preconditioning!) 

  bool use_ZMobius; //Use ZMobius instead of Mobius (complex coefficients, allows smaller inner Ls)
  ZMobiusParams ZMobius_params;

  rpccommand GENERATE_PRINT_METHOD;
  rpccommand GENERATE_DEEPCOPY_METHOD;
};

class CGcontrols{
  A2ACGalgorithm CGalgorithm;
  double CG_tolerance;
  int CG_max_iters;
  double mixedCG_init_inner_tolerance; //mixed precision restarted CG inner CG initial tolerance

  double reliable_update_delta; //'delta' parameter of reliable update CG, controlling how often the reliable update is performed
  double reliable_update_transition_tol; //tolerance at which the single precision linear operator is replaced by the 'fallback' operator (use 0 to not use the fallback)

  int multiCG_block_size; //if using multi-RHS or split-CG, how many solves are we performing at once?
  int split_grid_geometry<>; //if using a split-Grid technique, how do you want to divide up the lattice?
  
  MADWFparams madwf_params; //if CGalgorithm ==  AlgorithmMixedPrecisionMADWF, get the MADWF parameters from here
  
  A2AhighModeSourceType highmode_source;

  rpccommand GENERATE_PRINT_METHOD;
  rpccommand GENERATE_DEEPCOPY_METHOD;
};

class LanczosControls{
  A2AlanczosType lanczos_type; //what Lanczos implementation to use
  int block_lanczos_split_grid_geometry<>; //if using a block Lanczos split-Grid technique, how do you want to divide up the lattice?

  rpccommand GENERATE_PRINT_METHOD;
  rpccommand GENERATE_DEEPCOPY_METHOD;
};

class JobParams{
  BfmSolverType solver; //BFM solver type
  double mobius_scale; //if solver == BFM_HmCayleyTanh

  LanczosControls lanczos_controls;
  
  CGcontrols cg_controls;
  
  double pion_rad; //radius of pion Hydrogen wavefunction source
  double kaon_rad; //radius of kaon Hydrogen wavefunction source

  int pipi_separation; //timeslice separation of pions in pipi src/snk
  int tstep_pipi; //time increment on which we place pipi sources in pipi 2pt function calc. Default 1.

  int k_pi_separation<>; //values of K->pi separations in K->pipi (looped over)
  int xyzstep_type1; //spatial increment on which we place the 4-quark operator in type1 K->pipi. Default 1
  int tstep_type12; //time increment on which we place pipi sources in type 1 and 2 K->pipi calc. Default 1.

 rpccommand GENERATE_PRINT_METHOD;
 rpccommand GENERATE_DEEPCOPY_METHOD;
};
