// -*- mode:c++; c-basic-offset:4 -*-
#include <vector>
#include <string>
#include <alg/qpropw.h>
#include <util/qcdio.h>
#include <util/time_cps.h>
#include "prop_container.h"
#include "run_wsnk.h"
#include "run_3pt.h"

CPS_START_NAMESPACE

////////////////////////////////////////////////////////////////////////
// Operators that we calculate for the kl3 contraction.  There are 5
// contractions:

// 0: gamma_0 (x)
// 1: gamma_1 (y)
// 2: gamma_2 (z)
// 3: gamma_3 (t)
// 4: I       (identity matrix)
const int total_ops = 5;

static const WilsonMatrix &apply_op(WilsonMatrix &out,
                                    const WilsonMatrix &in,
                                    int op_id)
{
    switch(op_id) {
    case 0:
    case 1:
    case 2:
    case 3:
        return out.glV(in, op_id);
    case 4:
        return out = in;
    default:
        ERR.General("", "apply_op", "Invalid op_id = %d\n", op_id);
        return out;
    }
}

class RunKl3 {
public:
    RunKl3(const AllProp &sp,
           const AllProp &lp,
           const AllProp &lt,
           PROP_TYPE _ptype)
        :sprop(sp),lprop(lp),ltwst(lt),ptype(_ptype)
    {
        run_wall_snk(&lwsnk, lprop, ptype);
    }

    // computable
    bool operator()(int tk, int tp)const
    {
        if(sprop.empty(tk, ptype)) return false;
        if(lwsnk[tp].empty()) return false;
        if(ltwst.empty(tp, ptype)) return false;
        return true;
    }

    // eval_site
    void operator()(Rcomplex *ret, int tk, int tp, int i)const
    {
        WilsonMatrix p[3] = {0, ltwst(i, tp, ptype), 0};
        p[0].glV(sprop(i, tk, ptype), -5);
        p[2].glV(lwsnk[tp][tk], -5);
        p[0].hconj();
        p[2].hconj();

        p[1] *= p[2] * p[0];

        for(int op_id = 0; op_id < total_ops; ++op_id) {
            apply_op(p[0], p[1], op_id);
            ret[op_id] += p[0].Trace();
        }
    }
private:
    const AllProp &sprop;
    const AllProp &lprop;
    const AllProp &ltwst;
    PROP_TYPE ptype;
    std::vector<std::vector<WilsonMatrix> > lwsnk;
};

// Compute kl3 correlation functions, source type is determined
// implicitly by the propagator.
void run_kl3(const AllProp &sprop,
             const AllProp &lprop,
             const AllProp &ltwst,
             const std::string &fn,
             PROP_TYPE ptype)
{
    const RunKl3 run_obj(sprop, lprop, ltwst, ptype);
    run_3pt<total_ops>(run_obj, run_obj, fn, ptype);
}

CPS_END_NAMESPACE
