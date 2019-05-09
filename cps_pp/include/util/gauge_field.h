// vim: set ts=2 sw=2 expandtab:
#ifndef INCLUDED_GAUGE_FIELD_H
#define INCLUDED_GAUGE_FIELD_H

#include <util/lattice.h>
#include <util/vector.h>

#include <cassert>
#include <vector>
#include <set>

CPS_START_NAMESPACE

#ifdef USE_QMP
#define GAUGE_FIELD_COMM
#endif

//#define PROFILE_COMM

/*
 * Why do we need multiple instance of GaugeField objects?
 * 1. AlgApeSmear, AlgWilsonFlow
 * 2. FixGauge
 * 3. HMC accept/reject
 * 4. LinkBuffer
 * Then we would not need the work around LatticeContainer any more.
 *
 */

class Geometry {

  public :

    // About qmp geometry.
    int sizeN[4];
    int coorN[4];
    // 0 <= coorN[i] < sizeN[i]
    int numN;
    // numN = NumNodes()
    // numN = sizeN[0] * sizeN[1] * sizeN[2] * sizeN[3]
    int idN;
    // idN = UniqueID()
    // 0 <= idN < numN

    int node_sites[4];
    //!< The local lattice dimensions.
    /*!<
      A reimplementation of GlobalJobParameter::XnodeSite(), \e etc:
      -   	 node_sites[0] = GJP.XnodeSite();
      -    	 node_sites[1] = GJP.YnodeSite();
      -    	 node_sites[2] = GJP.ZnodeSite();
      -    	 node_sites[3] = GJP.TnodeSite();
      */
    int left_expansion[4];
    int right_expansion[4];

    int multiplicity; // number of elements on each lattice site
    int dir_offset[5]; // dir_offset[4] = total number of elements

    void setParameter(const int *_total_sites, const int _multiplicity);

    void setParameter(const int _numN, const int _idN, const int *_sizeN, const int *_coorN,
        const int _multiplicity,
        const int *_node_sites, const int *_left_expansion, const int *_right_expansion);
    // very unsafe
    // use with great care

    void setParameter(const Geometry& _geo, const int *_left_expansion, const int *_right_expansion);

    int siteOffset(const int *x) const {
#ifdef GAUGE_FIELD_COMM
      return (x[0] + left_expansion[0]) * dir_offset[0]
           + (x[1] + left_expansion[1]) * dir_offset[1]
           + (x[2] + left_expansion[2]) * dir_offset[2]
           + (x[3] + left_expansion[3]) * dir_offset[3];
#else
      int offset = 0;
      for (int i = 0; i < 4; i++) {
        int on_node_site = x[i] % node_sites[i];
        if (on_node_site < 0) {
          on_node_site += node_sites[i] ;
        }
        offset += on_node_site * dir_offset[i];
      }
      return offset;
#endif
    }
    //!< Gets the array index of a gauge link.
    /*!<
      Specifically, the internal array index of the first of the four gauge
      field links at lattice site x for the canonical storage order.
      \param x The lattice site coordinates.
      \return The array index.
      */

    void coordinatesFromOffset(int *x, int offset) const {
      x[3] = offset / dir_offset[3] - left_expansion[3];
      offset %= dir_offset[3];
      x[2] = offset / dir_offset[2] - left_expansion[2];
      offset %= dir_offset[2];
      x[1] = offset / dir_offset[1] - left_expansion[1];
      offset %= dir_offset[1];
      x[0] = offset / dir_offset[0] - left_expansion[0];
    }
    // 0 <= offset < siteSize

    int siteIndex(const int *x) const {
      return (((x[3] * node_sites[2]) + x[2]) * node_sites[1] + x[1]) * node_sites [0] + x[0];
    }
    // 0 <= index < localVolume

    void coordinatesFromIndex(int *x, int index) const {
      x[0] = index % node_sites[0];
      index /= node_sites[0];
      x[1] = index % node_sites[1];
      index /= node_sites[1];
      x[2] = index % node_sites[2];
      index /= node_sites[2];
      x[3] = index % node_sites[3];
    }
    // get local coordinates from index
    // 0 <= index < localVolume

    bool isOnNode(const int *x) const {
#ifdef GAUGE_FIELD_COMM
      return 0 <= x[0] && x[0] < node_sites[0]
          && 0 <= x[1] && x[1] < node_sites[1]
          && 0 <= x[2] && x[2] < node_sites[2]
          && 0 <= x[3] && x[3] < node_sites[3];
#else
      return true;
#endif
    }

    int localVolume() const {
      return node_sites[0] * node_sites[1] * node_sites[2] * node_sites[3];
    }

    int siteSize() const {
      return dir_offset[4] / multiplicity;
    }
    //!< Gets the number of gauge field  components per lattice site.

    int nodeSites(int mu) const {
      return node_sites[mu];
    }

    int coorNode(int mu) const {
      return coorN[mu];
    }

    int sizeNodes(int mu) const {
      return sizeN[mu];
    }

    int numNodes() const {
      return numN;
    }

    int idNode() const {
      return idN;
    }
};

inline bool operator==(const Geometry& geo1, const Geometry& geo2) {
  return 0 == memcmp(&geo1, &geo2, sizeof(Geometry));
}

inline bool operator!=(const Geometry& geo1, const Geometry& geo2) {
  return ! (geo1 == geo2);
}

class Offsets {

  public :

    Geometry geo;
    std::vector<std::vector<int> > keys;
    std::vector<std::vector<int> > offsetsVec, loffsetsVec;
    std::vector<std::vector<Matrix> > sendVec;
    std::vector<Matrix> recv_vec;

};

class GaugeField {

  private:

    const char *cname;    // Class name.

    Geometry geo;

    Matrix* gauge_field;
    // Pointer to the gauge field configuration.

    bool record_offset;
    bool record_local_links, record_neighbor_links;
    std::set<int> recorded_offsets;

  public:

    GaugeField(const int *_total_sites) {
      cname = "GaugeField";
      const char * fname = "GaugeField(const int * _total_sites)";
      geo.setParameter(_total_sites, 4);
      recordOffsetEnd();
      gauge_field = (Matrix *) malloc(geo.dir_offset[4] * sizeof(Matrix));
    }

    GaugeField(const Geometry& _geo) {
      cname = "GaugeField";
      geo = _geo;
      recordOffsetEnd();
      gauge_field = (Matrix *) malloc(geo.dir_offset[4] * sizeof(Matrix));
    }

    GaugeField(GaugeField& gf) {
      cname = "GaugeField";
      geo = gf.geo;
      recordOffsetEnd();
      gauge_field = (Matrix *) malloc(geo.dir_offset[4] * sizeof(Matrix));
      memcpy(gauge_field, gf.gauge_field, geo.dir_offset[4] * sizeof(Matrix));
    }
    // Exact Copy

    void swap(GaugeField &gf) {
      Geometry _geo;
      _geo = gf.geo;
      gf.geo = geo;
      geo = _geo;
      gf.recordOffsetEnd();
      recordOffsetEnd();
      Matrix* _gauge_field;
      _gauge_field = gf.gauge_field;
      gf.gauge_field = gauge_field;
      gauge_field = _gauge_field;
    }

    void fetchLinks(GaugeField &gf, Offsets& offsets);
    // only make use of the local links in gf
    // do not change current size of GaugeField

    void fetchLinks(GaugeField &gf, bool fetch_local_links = true, bool fetch_neighbor_links = true);
    // only make use of the local links in gf
    // do not change current size of GaugeField

    void fetchLocalLinks(GaugeField& gf);

    void refresh(Offsets& offsets) {
#ifdef GAUGE_FIELD_COMM
      fetchLinks(*this, offsets);
#endif
    }

    void refresh() {
#ifdef GAUGE_FIELD_COMM
      fetchLinks(*this, false, true);
#endif
    }

    GaugeField(GaugeField &gf, const int * _left_expansion, const int * _right_expansion) {
      cname = "GaugeField";
      geo.setParameter(gf.geo, _left_expansion, _right_expansion);
      gauge_field = (Matrix *) malloc(geo.dir_offset[4] * sizeof(Matrix));
      fetchLinks(gf, true, false);
    }
    // Be careful
    // Won't fetch neighbor links.
    // please call refresh if needed.

    GaugeField(GaugeField &gf, const int thick = 0) {
      int _left[4] = { thick, thick, thick, thick };
      int _right[4] = { thick, thick, thick, thick };
      geo.setParameter(gf.geo, _left, _right);
      gauge_field = (Matrix *) malloc(geo.dir_offset[4] * sizeof(Matrix));
      fetchLinks(gf, true, false);
    }

    ~GaugeField() {
      free(gauge_field);
    }

    void resize(const int * _left_expansion, const int * _right_expansion) {
#ifdef GAUGE_FIELD_COMM
      bool same = true;
      for (int i = 0; i < 4; i++) {
        if (_left_expansion[i] != geo.left_expansion[i] || _right_expansion[i] != geo.right_expansion[i]) {
          same = false;
          break;
        }
      }
      if (!same) {
        GaugeField temp(*this, _left_expansion, _right_expansion);
        swap(temp);
      }
#endif
    }
    // Be careful
    // Won't fetch neighbor links.
    // please call refresh if needed.

    void require(const int * _left_expansion, const int * _right_expansion) {
      const char * fname = "require(left, right)";
#ifdef GAUGE_FIELD_COMM
      bool fit = true;
      for (int i = 0; i < 4; i++) {
        if (_left_expansion[i] > geo.left_expansion[i] || _right_expansion[i] > geo.right_expansion[i]) {
          fit = false;
          break;
        }
      }
      if (!fit) {
        VRB.Result(cname, fname, "!!!! Size does not meet requirement.\n");
        VRB.Result(cname, fname, "!!!! You should avoid this as much as possible.\n");
        VRB.Result(cname, fname, "!!!! Will expand the lattice size.\n");
        GaugeField temp(*this, _left_expansion, _right_expansion);
        swap(temp);
      }
#endif
    }
    // Be careful
    // Won't fetch neighbor links.
    // please call refresh if needed.


    void resize(const int thick = 0) {
      int _expansion[4] = { thick, thick, thick, thick };
      resize(_expansion, _expansion);
    }

    void require(const int thick = 0) {
      int _expansion[4] = { thick, thick, thick, thick };
      require(_expansion, _expansion);
    }

    GaugeField(Lattice& lat);

    void getLattice(Lattice& lat) {
#ifdef GAUGE_FIELD_COMM
      int _expansion[4] = { 0, 0, 0, 0 };
      GaugeField temp(*this, _expansion, _expansion);
      memcpy(temp.gauge_field, lat.GaugeField(), temp.geo.dir_offset[4] * sizeof(Matrix));
      fetchLinks(temp, true, false);
#else
      memcpy(gauge_field, lat.GaugeField(), geo.dir_offset[4] * sizeof(Matrix));
#endif
    }

    void setLattice(Lattice& lat) {
#ifdef GAUGE_FIELD_COMM
      int _expansion[4] = { 0, 0, 0, 0 };
      GaugeField temp(*this, _expansion, _expansion);
      memcpy(lat.GaugeField(), temp.gauge_field, temp.geo.dir_offset[4] * sizeof(Matrix));
#else
      memcpy(lat.GaugeField(), gauge_field, geo.dir_offset[4] * sizeof(Matrix));
#endif
    }

    // start record AFTER refresh()!
    void recordOffsetStart(bool local = true, bool neighbor = true) {
      record_offset = true;
      record_local_links = local;
      record_neighbor_links = neighbor;
      recorded_offsets.clear();
    }

    void recordOffsetEnd() {
      record_offset = false;
      record_local_links = true;
      record_neighbor_links = true;
      recorded_offsets.clear();
    }

    Offsets getRecordedOffsets();

    // offset is NOT the same as index
    Matrix * unsafeGetLink(const int offset) {
      if (record_offset) {
        int site[4];
        coordinatesFromOffset(site, offset);
        bool on_node = isOnNode(site);
        if (on_node && record_local_links || !on_node && record_neighbor_links) {
#pragma omp critical
          recorded_offsets.insert(offset);
        }
      }
      return gauge_field + offset;
    }

    Matrix * unsafeGetLink(const int * site, int mu) {
      int offset = siteOffset(site) + mu;
      return unsafeGetLink(offset);
    }

    Matrix * localGetLink(const int * site, int mu) const {
      if (isOnNode(site)) {
        return gauge_field + siteOffset(site) + mu;
      } else {
        return NULL;
      }
    }

    Matrix getLink(const int * site, int mu) const;
    //!< Gets the gauge link U_mu(x).
    // defined relative to the local site (0,0,0,0).
    // If the link is on-node, it returns a reference to
    // to the link.  If the link is off-node, it retrieves the link
    // into a static buffer and returns the reference to the buffer.
    // Since the buffer can be used by other routines as well as other
    // calls to this routine, as a general rule, the link should be
    // used immediately or else copied.

    Matrix * getGaugeField() const {
      return gauge_field;
    }
    //!< Returns the pointer to the gauge field configuration.

    void setGaugeField(Matrix *u);
    //!< Copies an array into the gauge configuration.

    const Geometry& getGeo() const {
      return geo;
    }

    int siteOffset(const int *x) const {
      return geo.siteOffset(x);
    }

    void coordinatesFromOffset(int *x, int offset) const {
      return geo.coordinatesFromOffset(x, offset);
    }
    // 0 <= offset < siteSize

    int siteIndex(const int *x) const {
      return geo.siteIndex(x);
    }
    // 0 <= index < localVolume

    void coordinatesFromIndex(int *x, int index) const {
      return geo.coordinatesFromIndex(x, index);
    }
    // get local coordinates from index
    // 0 <= index < localVolume

    bool isOnNode(const int *x) const {
      return geo.isOnNode(x);
    }

    int localVolume() const {
      return geo.localVolume();
    }

    int siteSize() const {
      return geo.siteSize();
    }
    //!< Gets the number of gauge field  components per lattice site.

    int nodeSites(int mu) const {
      return geo.nodeSites(mu);
    }

    int coorNode(int mu) const {
      return geo.coorNode(mu);
    }

    int sizeNodes(int mu) const {
      return geo.sizeNodes(mu);
    }

    int numNodes() const {
      return geo.numNodes();
    }

    int idNode() const {
      return geo.idNode();
    }

};

class GaugeActionNone;
class GaugeActionWilson;
class GaugeActionImprRect;

class GaugeAction {

  public :

    GaugeAction(GclassType _gclass, double _beta) {
      cname = "GaugeAction";
      gclass = _gclass;
      beta = _beta;
    }

    GaugeAction& getGaugeAction() {
      return *this;
    }

    GaugeActionNone& getGaugeActionNone() {
      return *(GaugeActionNone *)this;
    }

    GaugeActionWilson& getGaugeActionWilson() {
      return *(GaugeActionWilson *)this;
    }

    GaugeActionImprRect& getGaugeActionImprRect() {
      return *(GaugeActionImprRect *)this;
    }

    GclassType Gclass() const {
      return gclass;
    }

    double getBeta() const {
      return beta;
    }

  private :

    const char * cname;

    double beta;

    GclassType gclass;
};

class GaugeActionNone : public GaugeAction {

  public :

    GaugeActionNone() : GaugeAction(G_CLASS_NONE, 0.0) {
      cname = "GaugeActionNone";
    }

  private:

    const char * cname;
};

class GaugeActionWilson : public GaugeAction {

  public:

    GaugeActionWilson(double _beta) : GaugeAction(G_CLASS_WILSON, _beta) {
      cname = "GaugeActionWilson";
    }

  private:

    const char * cname;
};

class GaugeActionImprRect : public GaugeAction {

  public:

    GaugeActionImprRect(double _beta, double _c1) : GaugeAction(G_CLASS_IMPR_RECT, _beta) {
      cname = "GaugeActionImprRect";
      c1 = _c1;
    }

    double getC1() {
      return c1;
    }

  private:

    const char * cname;

    double c1;
};

// the pointer need to be deleted manually.
inline GaugeAction * getGaugeAction(Lattice& lat) {
  switch (lat.Gclass()) {
    case G_CLASS_NONE :
      return new GaugeActionNone();
    case G_CLASS_WILSON :
      return new GaugeActionWilson(GJP.Beta());
    case G_CLASS_IMPR_RECT:
      return new GaugeActionImprRect(GJP.Beta(), GJP.C1());
    default :
      ERR.NotImplemented("GaugeAction", "getGaugeAction");
      return new GaugeActionNone();
  }
}

inline int gaInteractionRange(GaugeAction& ga) {
  switch (ga.Gclass()) {
    case G_CLASS_NONE :
      return 0;
    case G_CLASS_WILSON :
      return 1;
    case G_CLASS_IMPR_RECT:
      return 2;
    default :
      ERR.NotImplemented("GaugeAction", "gaInteractionRange");
      return 0;
  }
}

void gfPathOrdProdPlus(Matrix & mat, GaugeField& gf, const int *x, const int * dirs, int n);
//!< Computes the product of links along a path and adds it to a matrix.
//given the starting point x, the directions of each step on the path
//and the number of steps. calculate the path ordered product of
//all the links along the path and add the result to mat
//each direction could be {0,1,2,3,4,5,6,7} which coresponds to
//the directions {n_x, n_y, n_z, n_t, -n_x, -n_y, -n_z, -n_t}
//the result is returned in mat.

inline void gfPathOrdProd(Matrix & mat, GaugeField& gf, const int *x, const int * dirs, int n) {
  mat.ZeroMatrix();
  gfPathOrdProdPlus(mat, gf, x, dirs, n);
}
//!< Computes the product of links along a path.
//also calculates the path ordered product, but the result returned
//is that product of the path that end on the local node

void gfPlaqStaple(Matrix& stap, GaugeField& gf, const int *x, int mu);
//!< Calculates the gauge field square staple sum around a link

void gfRectStaple(Matrix& stap, GaugeField& gf, const int *x, int mu);
//!< Calculates the rectangle staple sum around a link.
// The rectangle field is:
//
// \sum_{v != u} {
//     U_u(x+u)    U_v(x+2u)    U_u(x+u+v)~ U_u(x+v)~  U_v(x)~
//   + U_u(x+u)    U_v(x+2u-v)~ U_u(x+u-v)~ U_u(x-v)~  U_v(x-v)
//   + U_v(x+u)    U_u(x+v)~    U_u(x-u+v)~ U_v(x-u)~  U_u(x-u)
//   + U_v(x+u-v)~ U_u(x-v)~    U_u(x-u-v)~ U_v(x-u-v) U_u(x-u)
//   + U_v(x+u)    U_v(x+u+v)   U_u(x+2v)~  U_v(x+v)~  U_v(x)~
//   + U_v(x+u-v)~ U_v(x+u-2v)~ U_u(x-2v)~  U_v(x-2v)  U_v(x-v)
// }


Float gfSumReTrPlaqNode(GaugeField &gf);
//!< Sums the real part of the trace of all plaquettes based on this 
//node, including all six orientations.

Float gfSumReTrRectNode(GaugeField &gf);
//!< Sums the real part of the trace of all 2x1 rectangles based on this 
//node, including all twelve orientations.

inline void gfAllStaple(Matrix& stap, GaugeField& gf, GaugeAction& ga, const int *x, int mu) {
  switch (ga.Gclass()) {
    case G_CLASS_NONE :
      stap.ZeroMatrix();
      break;
    case G_CLASS_WILSON :
      gfPlaqStaple(stap, gf, x, mu);
      break;
    case G_CLASS_IMPR_RECT :
      {
        double c1 = ga.getGaugeActionImprRect().getC1();
        Matrix mat;
        gfPlaqStaple(mat, gf, x, mu);
        vecTimesEquFloat((IFloat*)&mat, 1-8*c1, sizeof(Matrix) / sizeof(IFloat));
        gfRectStaple(stap, gf, x, mu);
        vecTimesEquFloat((IFloat*)&stap, c1, sizeof(Matrix) / sizeof(IFloat));
        vecAddEquVec((IFloat*)&stap, (IFloat*)&mat, sizeof(Matrix) / sizeof(IFloat));
      }
      break;
    default :
      ERR.NotImplemented("","","AllStaple(Matrix& stap, GaugeField& gf, GaugeAction& ga, const int *x, int mu)\n");
  }
}

void gfUnitMatrix(GaugeField& gf);

inline void setGaugeFieldOrd(GaugeField& gf) {
  gfUnitMatrix(gf);
}

void gfUnitarize(GaugeField& gf);
//!< Re-unitarize the gauge field configuration.

void gfConstruct3rdRow(GaugeField& gf);

void gfDagger(GaugeField& gf);

// Remember to gf.refresh(offset) before call this function.
ForceArg gfEvolveMomGforce(Matrix *mom, GaugeField& gf, GaugeAction& ga, Float dt);

// Remember to gf.refresh(offset) before call this function.
Float gfGhamiltonNode(GaugeField& gf, GaugeAction& ga);

CPS_END_NAMESPACE

#endif
