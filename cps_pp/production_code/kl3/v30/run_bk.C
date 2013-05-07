// -*- mode:c++; c-basic-offset:4 -*-
#include <vector>
#include <string>
#include <alg/qpropw.h>
#include <util/qcdio.h>
#include <util/time_cps.h>
#include "my_util.h"
#include "prop_container.h"
#include "run_3pt.h"

CPS_START_NAMESPACE

class RunBK {
public:
    RunBK(const AllProp &lA,
          const AllProp &sA,
          const AllProp &lB,
          const AllProp &sB,
          PROP_TYPE _ptype)
        :lpropA(lA),spropA(sA),lpropB(lB),spropB(sB),ptype(_ptype)
    {
    }

    // computable
    bool operator()(int ka, int kb)const
    {
        if(lpropA.empty(ka, ptype)) return false;
        if(spropA.empty(ka, ptype)) return false;
        if(lpropB.empty(kb, ptype)) return false;
        if(spropB.empty(kb, ptype)) return false;
        return true;
    }

    // eval_site
    void operator()(Rcomplex *ret, int ka, int kb, int i)const
    {
        // K & Kb
        WilsonMatrix K[2]  = {
            lpropA(i, ka, ptype), spropA(i, ka, ptype)
        };
        WilsonMatrix Kb[2] = {
            lpropB(i, kb, ptype), spropB(i, kb, ptype)
        };

        K[1].hconj();
        Kb[1].hconj();

        WilsonMatrix lines[2] = {K[0] * K[1], Kb[0] * Kb[1]};

        // type 0 = Tr(loop0) * Tr(loop1)
        // type 1 = Tr(loop0 * loop1)
        for(int mu = 0; mu < 4; ++mu) {
            WilsonMatrix tmp[2];
            // AA term
            tmp[0].glA(lines[0], mu);
            tmp[1].glA(lines[1], mu);

            ret[0] += tmp[0].Trace() * tmp[1].Trace();
            ret[1] += Trace(tmp[0], tmp[1]);

            // VV term
            tmp[0].glV(lines[0], mu);
            tmp[1].glV(lines[1], mu);

            ret[0] += tmp[0].Trace() * tmp[1].Trace();
            ret[1] += Trace(tmp[0], tmp[1]);
        } // mu
    }
private:
    const AllProp &lpropA, &lpropB, &spropA, &spropB;
    PROP_TYPE ptype;
};

// Compute Bk correlation functions, source type is determined
// implicitly by the propagator.
void run_bk(const AllProp &lpropA,
            const AllProp &spropA,
            const AllProp &lpropB,
            const AllProp &spropB,
            const std::string &fn,
            PROP_TYPE ptype)
{
    const int total_ops = 2;
    const RunBK run_obj(lpropA, spropA, lpropB, spropB, ptype);
    run_3pt<total_ops>(run_obj, run_obj, fn, ptype);
}

CPS_END_NAMESPACE
