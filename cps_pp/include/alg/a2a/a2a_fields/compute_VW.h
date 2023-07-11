#include "compute_VW/randomize_VW.h"

//Grid only supported
#if defined(USE_GRID_A2A)
# warning "Using Grid A2A"

# ifndef USE_GRID
#  error "Require Grid for USE_GRID_A2A"
# endif

# include "compute_VW/compute_VW_gridA2A.h"

#else

# error "Need Grid to compute A2A vectors"

#endif
