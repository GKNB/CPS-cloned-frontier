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

// The 8 different types of contractions related to I=2 K to pi pi are
// defined using Qi's convention. However 1-8 are relabelled as 0-7.
//
// Contractions written in LR style (i.e., operators are
// gamma(i)(1+/-gamma(5))).
static inline void cal_8types_LR(Rcomplex *ret,
                                 const WilsonMatrix p[2])
{
    for(int mu = 0; mu < 4; ++mu) {
        WilsonMatrix l[2] = {p[0].glL(mu), p[1].glL(mu)};
        WilsonMatrix r[2] = {p[0].glR(mu), p[1].glR(mu)};

        ret[0] += l[0].Trace() * l[1].Trace(); // 1
        ret[2] += l[0].Trace() * r[1].Trace(); // 3
        ret[4] += Trace(l[0], l[1]);           // 5
        ret[6] += Trace(r[0], l[1]);           // 7

        ret[1] += (SpinTrace(l[0]) * SpinTrace(l[1])).Tr();  // 2
        ret[3] += (SpinTrace(l[0]) * SpinTrace(r[1])).Tr();  // 4
        ret[5] += Trace(ColorTrace(l[0]), ColorTrace(l[1])); // 6
        ret[7] += Trace(ColorTrace(r[0]), ColorTrace(l[1])); // 8
    }
}

// The 8 different types of contractions related to I=2 K to pi pi are
// defined using Qi's convention. However 1-8 are relabelled as 0-7.
//
// Contractions written in VA style (i.e., operators are gamma(i) and
// gamma(i)gamma(5)).
static inline void cal_8types_VA(Rcomplex *ret,
                                 const WilsonMatrix p[2])
{
    for(int mu = 0; mu < 4; ++mu) {
        WilsonMatrix v[2] = {p[0].glV(mu), p[1].glV(mu)};
        WilsonMatrix a[2] = {p[0].glA(mu), p[1].glA(mu)};

        ret[0] += v[0].Trace() * a[1].Trace(); // 1
        ret[2] += a[0].Trace() * v[1].Trace(); // 3
        ret[4] += Trace(v[0], a[1]);           // 5
        ret[6] += Trace(a[0], v[1]);           // 7

        ret[1] += (SpinTrace(v[0]) * SpinTrace(a[1])).Tr();  // 2
        ret[3] += (SpinTrace(a[0]) * SpinTrace(v[1])).Tr();  // 4
        ret[5] += Trace(ColorTrace(v[0]), ColorTrace(a[1])); // 6
        ret[7] += Trace(ColorTrace(a[0]), ColorTrace(v[1])); // 8
    }
}

class RunKpp {
public:
    RunKpp(const AllProp &sp,
           const AllProp &up,
           const AllProp &dp,
           PROP_TYPE _ptype)
        :sprop(sp),uprop(up),dprop(dp),ptype(_ptype)
    {
        run_wall_snk(&uwsnk, uprop, ptype);
    }

    // computable
    bool operator()(int tk, int tp)const
    {
        if(sprop.empty(tk, ptype)) return false;
        if(uwsnk[tp].empty()) return false;
        if(dprop.empty(tp, ptype)) return false;
        return true;
    }

    // eval_site
    void operator()(Rcomplex *ret, int tk, int tp, int i)const
    {
        WilsonMatrix p[3];
        p[1] = sprop(i, tk, ptype);
        p[1].hconj();
        p[2].glV(p[1], -5);
        p[2].gr(-5);
        p[1] = uwsnk[tp][tk];
        p[1].hconj();
        
        p[0] = dprop(i, tp, ptype) * p[1] * p[2]; // part 0
        
        p[2] = uprop(i, tp, ptype);
        p[2].hconj();
        p[2].gr(-5);
        p[1] = dprop(i, tp, ptype) * p[2]; // part 1
        
        // p[2] is not used in the following function.
        cal_8types_VA(ret, p);
    }
private:
    const AllProp &sprop, &uprop, &dprop;
    PROP_TYPE ptype;
    std::vector<std::vector<WilsonMatrix> > uwsnk;
};

// Run I=2 K to pi pi.
void run_k2pipi(const AllProp &sprop,
                const AllProp &uprop,
                const AllProp &dprop,
                const std::string &fn,
                PROP_TYPE ptype)
{
    const int total_ops = 8;
    const RunKpp run_obj(sprop, uprop, dprop, ptype);
    run_3pt<total_ops>(run_obj, run_obj, fn, ptype);
}

CPS_END_NAMESPACE
