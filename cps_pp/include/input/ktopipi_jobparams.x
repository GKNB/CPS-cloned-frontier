class CGcontrols{
  A2ACGalgorithm CGalgorithm;
  double CG_tolerance;
  int CG_max_iters;
  double mixedCG_init_inner_tolerance; //mixed precision restarted CG inner CG initial tolerance
  double reliable_update_delta; //'delta' parameter of reliable update CG, controlling how often the reliable update is performed

  rpccommand GENERATE_PRINT_METHOD;
  rpccommand GENERATE_DEEPCOPY_METHOD;
};

class JobParams{
  BfmSolverType solver; //BFM solver type
  double mobius_scale; //if solver == BFM_HmCayleyTanh

  bool convert_evecs_to_single_precision; //if not using a single-precision Lanczos, convert the double precision evecs to single prec to save memory
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
