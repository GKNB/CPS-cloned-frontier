// -*- mode:c++; c-basic-offset:4 -*-
#ifndef INCLUDED_RUN_3PT_H_KL3
#define INCLUDED_RUN_3PT_H_KL3

#include <vector>
#include <string>
#include <qmp.h>
#include <alg/qpropw.h>
#include <util/qcdio.h>
#include <util/time_cps.h>
#include "prop_container.h"

CPS_START_NAMESPACE

// Compute 3 point correlation functions, source type is determined
// implicitly by the propagator.
//
// total_ops: total number of output complex numbers for any site. For
// example, for I=2 K->pi pi there are 8 contractions so this number
// is 8.
//
// FComputable: a function object that accepts 2 integers (locations
// of the left and right wall particles) and returns a bool indicating
// if we can compute the contraction. It tests if we have computed the
// relevant propagators.
//
// FEval: a function object that computes the contraction for a given
// site. It accepts the following parameters:
// 
//    output pointer: an array of complex numbers accepting the
//    results. The function must write total_ops complex numbers to
//    this location.
//
//    ta: the location of the left wall particle (for example, in Kl3
//    this is a kaon).
//
//    tb: the location of the right wall particle (for example, in Kl3
//    this is a pion).
//
//    i: the location of the point operator.
//
// The following code requires that both FComputable and FEval are
// thread safe.
template<int total_ops,
         typename FComputable,
         typename FEval>
void run_3pt(FComputable computable,
             FEval eval_site,
             const std::string &fn,
             PROP_TYPE ptype)
{
    const char *fname = "run_3pt()";
    const int t_scale = ptype == PROP_PA ? 2 : 1;
    const int t_size = GJP.TnodeSites() * GJP.Tnodes();
    const int t_size_ap = t_scale * t_size;
    const int lcl[4] = {GJP.XnodeSites(), GJP.YnodeSites(),
                        GJP.ZnodeSites(), GJP.TnodeSites(),};
    const int lcl_vol = lcl[0] * lcl[1] * lcl[2] * lcl[3];
    const int t_node_off = GJP.TnodeSites() * GJP.TnodeCoor();

    const int total_size = t_size_ap * t_size * t_size_ap * total_ops;
    std::vector<Rcomplex> retv(total_size, 0);
    Rcomplex *ret = retv.data();

    Float dtime0 = dclock();

#pragma omp parallel for
    for(int sep_ta = 0; sep_ta < t_size_ap * t_size; ++sep_ta) {
        // for(int sep = 0; sep < t_size_ap; ++sep) {
        //     for(int ta = 0; ta < t_size; ++ta) {
        const int sep = sep_ta / t_size;
        const int ta = sep_ta % t_size;
        const int tb = (ta + sep) % t_size_ap;
        const int offset = sep_ta * t_size_ap * total_ops;
        if(! computable(ta, tb)) continue;

        for(int i = 0; i < t_scale * lcl_vol; ++i) {
            int x[4];
            // Note: since i will never be larger than lcl_vol
            // in P or A case, compute_coord_ap() suits all
            // ptypes.
            compute_coord_ap(x, lcl, i, t_size);
            const int t_glb = x[3] + t_node_off;
            const int t_delta = (t_glb + t_size_ap - ta) % t_size_ap;
            
            int site_off = offset + t_delta * total_ops;
            eval_site(ret + site_off, ta, tb, i);
        } // operator site
    } //sep_ta

    // FIXME
    assert(GJP.Snodes() == 1);
    QMP_sum_double_array((double*)ret, total_size * 2);

    Float dtime1 = dclock();

    // binary dump
    {
        const string fn_bin = fn + ".bin";
        FILE *fp = Fopen(fn_bin.c_str(), "w");
        Fwrite(ret, total_size * sizeof(Rcomplex), 1, fp);
        Fclose(fp);
    }

    // output
    const int n_nodes = GJP.Xnodes() * GJP.Ynodes() * GJP.Znodes() * GJP.Tnodes();

    for(int sep = UniqueID(); sep < t_size_ap; sep += n_nodes) {
        const string fn_node = fn + tostring(".") + tostring(sep);
        FILE *fp = fopen(fn_node.c_str(), "w");

        for(int ta = 0; ta < t_size; ++ta) {
            const int tb = (ta + sep) % t_size_ap;
            if(! computable(ta, tb)) continue;

            const int offset = (sep * t_size + ta) * t_size_ap * total_ops;

            for(int t = 0; t < t_size_ap; ++t) {
                std::fprintf(fp, "%3d %3d %3d", sep, ta, t);

                for(int op_id = 0; op_id < total_ops; ++op_id) {
                    std::fprintf(fp, " %17.10e %17.10e",
                                 real(ret[offset + t * total_ops + op_id]),
                                 imag(ret[offset + t * total_ops + op_id]));
                }
                std::fprintf(fp, "\n");
            }
        }

        std::fclose(fp);
    }

    Float dtime2 = dclock();

    VRB.Result("", fname, "time1-0 = %17.10e seconds\n", dtime1  - dtime0);
    VRB.Result("", fname, "time2-1 = %17.10e seconds\n", dtime2  - dtime1);
    VRB.Result("", fname, "time2-0 = %17.10e seconds\n", dtime2  - dtime0);
}

CPS_END_NAMESPACE

#endif
