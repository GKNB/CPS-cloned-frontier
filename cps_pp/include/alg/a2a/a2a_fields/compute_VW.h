//Can do Lanczos in BFM or Grid, and A2A in BFM or Grid. I have a BFM Lanczos -> Grid interface

#if defined(USE_BFM_A2A)
# warning "Using BFM A2A"

# ifndef USE_BFM
#  error "Require BFM for USE_BFM_A2A"
# endif

# ifdef USE_GRID_LANCZOS
#  error "No Grid Lanczos -> BFM A2A interface implemented"
# endif

# include "compute_VW/compute_VW_bfmLanczos.h"

#elif defined(USE_GRID_A2A)
# warning "Using Grid A2A"

# ifndef USE_GRID
#  error "Require Grid for USE_GRID_A2A"
# endif

# if defined(USE_BFM_LANCZOS) && !defined(USE_BFM)
#  error "BFM Lanczos -> Grid A2A interface requires BFM!"
# endif

# include "compute_VW/compute_VW_gridA2A.h"
# include "compute_VW/compute_VW_bfmLanczos.h"

#else

# error "Need either BFM or Grid to compute A2A vectors"

#endif
