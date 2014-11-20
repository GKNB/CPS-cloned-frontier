#ifndef INCLUDED_BFM_MADWF_H
#define INCLUDED_BFM_MADWF_H

/*#include <qdp_multi.h>
#include <QDPOperators.h>

#include <omp.h>

namespace madwf {

    // The MADWF algorithm:
    //
    // We want to solve 
    //                
    //     D_DWF(m) x = b
    //
    // We have access to a cheaper DWF operator D_DWF'
    //
    // 1. Construct 
    //        c = P^-1 D_DWF(1)^-1 b
    //
    // 2. Make a smaller vector 
    //        c' = (c_0, 0, 0, ...) 
    //    which can serve as an RHS for the cheaper operator D_DWF'
    //
    // 3. Solve 
    //       y' = P^-1 D_DWF'(m)^-1 D_DWF'(1) P c'
    //
    // 4. Construct the large vector 
    //        f = (-y'_0, c_1, c_2, ...)
    //
    // 5. Solve
    //        y = P^-1 D_DWF(1)^-1 D_DWF(m) P f
    //
    // 6. Replace 
    //      y_0 = y'_0
    //
    // 7. Construct
    //        x = P y  
    //
    // Repeat as necessary




    


}*/

#endif