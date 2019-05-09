// vim: set ts=2 sw=2 expandtab:

#include <util/gauge_field.h>
#include <util/time_cps.h>

#ifdef GAUGE_FIELD_COMM
#include <comms/scu.h>
#endif
#include <comms/sysfunc_cps.h>

#include <iostream>
#include <vector>
#include <map>

CPS_START_NAMESPACE

using namespace std;

void Geometry::
setParameter(const int *_total_sites, const int _multiplicity) {
  int _expansion[] = { 0, 0, 0, 0 };
  int _sizeN[] = { SizeX(), SizeY(), SizeZ(), SizeT() };
  int _coorN[] = { CoorX(), CoorY(), CoorZ(), CoorT() };
  int _node_sites[4];
  for (int i = 0; i < 4; i++) {
    _node_sites[i] = _total_sites[i] / _sizeN[i];
    if (0 != _total_sites[i] % _sizeN[i]) {
      ERR.General("Geometry", "setParameter(int*, int)", "dir %d : total_site = %d while sizeN = %d\n",
          i, _total_sites[i], _sizeN[i]);
    }
  }
  setParameter(NumNodes(), UniqueID(), _sizeN, _coorN,
      _multiplicity,
      _node_sites, _expansion, _expansion);
}

void Geometry::
setParameter(const int _numN, const int _idN,
    const int *_sizeN,
    const int *_coorN,
    const int _multiplicity,
    const int *_node_sites,
    const int *_left_expansion,
    const int *_right_expansion) {
  numN = _numN;
  idN = _idN;
  multiplicity = _multiplicity;
  int vol = multiplicity;
  for (int i = 0; i < 4; i++) {
#ifdef GAUGE_FIELD_COMM
    left_expansion[i] = _left_expansion[i];
    right_expansion[i] = _right_expansion[i];
#else
    left_expansion[i] = 0;
    right_expansion[i] = 0;
#endif
    sizeN[i] = _sizeN[i];
    coorN[i] = _coorN[i];
    node_sites[i] = _node_sites[i];
    dir_offset[i] = vol;
    vol *= left_expansion[i] + node_sites[i] + right_expansion[i];
  }
  dir_offset[4] = vol;
}

void Geometry::
setParameter(const Geometry& _geo, const int *_left_expansion, const int *_right_expansion) {
  setParameter(_geo.numN, _geo.idN, _geo.sizeN, _geo.coorN,
      _geo.multiplicity,
      _geo.node_sites, _left_expansion, _right_expansion);
}

GaugeField::
GaugeField(Lattice &lat) {
  cname = "GaugeField";
  gauge_field = NULL;
  int expansion[] = { 0, 0, 0, 0 };
  int _sizeN[] = { SizeX(), SizeY(), SizeZ(), SizeT() };
  int _coorN[] = { CoorX(), CoorY(), CoorZ(), CoorT() };
  geo.setParameter(NumNodes(), UniqueID(), _sizeN, _coorN, 4,
      lat.node_sites, expansion, expansion);
  recordOffsetEnd();
  gauge_field = (Matrix *) malloc(geo.dir_offset[4] * sizeof(Matrix));
  memcpy(gauge_field, lat.gauge_field, geo.dir_offset[4] * sizeof(Matrix));
}

Offsets GaugeField::
getRecordedOffsets() {
  Offsets offsets;
  offsets.geo = geo;
  map<vector<int>, vector<int> > offsetsMap;
  for (set<int>::iterator it = recorded_offsets.begin(); it != recorded_offsets.end(); it++) {
    int offset = *it;
    vector<int> key(4, 0), pos(4, 0), lpos(4, 0);
    coordinatesFromOffset(pos.data(), offset);
    for (int k = 0; k < 4; k++) {
      lpos[k] = pos[k] % geo.node_sites[k];
      key[k] = pos[k] / geo.node_sites[k];
      if (lpos[k] < 0) {
        lpos[k] += geo.node_sites[k];
        key[k]--;
      }
    }
    offsetsMap[key].push_back(offset);
  }
  for (map<vector<int>, vector<int> >::iterator it = offsetsMap.begin();
      it != offsetsMap.end(); it++) {
    offsets.keys.push_back(it->first);
    offsets.offsetsVec.push_back(it->second);
    vector<int> loffs;
    for (int i = 0; i < (it->second).size(); i++) {
      int offset = (it->second)[i];
      vector<int> key(4, 0), pos(4, 0), lpos(4, 0);
      coordinatesFromOffset(pos.data(), offset);
      for (int k = 0; k < 4; k++) {
        lpos[k] = pos[k] % geo.node_sites[k];
        key[k] = pos[k] / geo.node_sites[k];
        if (lpos[k] < 0) {
          lpos[k] += geo.node_sites[k];
          key[k]--;
        }
      }
      int loffset = siteOffset(lpos.data()) + offset % 4;
      loffs.push_back(loffset);
    }
    offsets.loffsetsVec.push_back(loffs);
  }
  offsets.sendVec.resize(offsets.keys.size());
  for (int k = 0; k < offsets.keys.size(); k++) {
    size_t size = offsets.offsetsVec[k].size();
    offsets.sendVec[k].resize(size);
    offsets.recv_vec.resize(max(offsets.recv_vec.size(), size));
  }
  return offsets;
}

void GaugeField::
fetchLinks(GaugeField &gf, Offsets& offsets) {
  const char * fname = "fetchLinks(gf, offsets)";
  recordOffsetEnd();

#ifdef GAUGE_FIELD_COMM

#ifdef PROFILE_COMM
  Float dtime = -dclock();
#endif

    // Checking geometric parameters
  if (offsets.geo != geo || offsets.geo != gf.geo) {
    ERR.General(cname, fname, "Geometric parameters does not match! Usually means offsets used in a wrong setting.\n");
  }

#pragma omp parallel for
  for (int k = 0; k < offsets.keys.size(); k++) {
    const vector<int>& key = offsets.keys[k];
    const vector<int>& loffsets = offsets.loffsetsVec[k];
    vector<Matrix>& send_vec = offsets.sendVec[k];
    //VRB.Result(cname, fname, "key = { %d, %d, %d, %d } size = %d\n",
    //    key[0], key[1], key[2], key[3], loffsets.size());
    for (int i = 0; i < loffsets.size(); i++) {
      send_vec[i] = *gf.unsafeGetLink(loffsets[i]);
    }
  }

#ifdef PROFILE_COMM
  Float comtime = -dclock();
#endif
  vector<Matrix>& recv_vec = offsets.recv_vec;
  for (int k = 0; k < offsets.keys.size(); k++) {
    const vector<int>& key = offsets.keys[k];
    //VRB.Result(cname, fname, "communicating : %d %d %d %d\n",
    //    key[0], key[1], key[2], key[3]);
    vector<Matrix>& send_vec = offsets.sendVec[k];
    int size = send_vec.size();
    size_t size_bytes = size * sizeof(Matrix);
    Matrix * send = send_vec.data();
    Matrix * recv = recv_vec.data();
    for (int i = 0; i < 4; i++) {
      int dis = key[i];
      if (dis < 0) {
        while (dis != 0) {
          getMinusData((IFloat *)recv, (IFloat *)send, size * sizeof(Matrix) / sizeof(IFloat), i);
          memcpy(send, recv, size_bytes);
          dis++;
        }
      } else if (dis > 0) {
        while (dis != 0) {
          getPlusData((IFloat *)recv, (IFloat *)send, size * sizeof(Matrix) / sizeof(IFloat), i);
          memcpy(send, recv, size_bytes);
          dis--;
        }
      }
    }
  }
#ifdef PROFILE_COMM
  comtime += dclock();
  print_time(cname, "fL-com", comtime);
#endif

#pragma omp parallel for
  for (int k = 0; k < offsets.keys.size(); k++) {
    const vector<int>& key = offsets.keys[k];
    const vector<int>& offs = offsets.offsetsVec[k];
    vector<Matrix>& send_vec = offsets.sendVec[k];
    //VRB.Result(cname, fname, "key = { %d, %d, %d, %d } size = %d\n",
    //    key[0], key[1], key[2], key[3], offsets.size());
    for (int i = 0; i < offs.size(); i++) {
      *unsafeGetLink(offs[i]) = send_vec[i];
    }
  }

#ifdef PROFILE_COMM
  dtime += dclock();
  print_time(cname, "fL-tot", dtime);
#endif

#else
  if (this != &gf) {
    for (int k = 0; k < offsets.keys.size(); k++) {
      const vector<int>& key = offsets.keys[k];
      const vector<int>& loffs = offsets.loffsetsVec[k];
      //VRB.Result(cname, fname, "key = { %d, %d, %d, %d } size = %d\n",
      //    key[0], key[1], key[2], key[3], offsets.size());
      for (int i = 0; i < loffs.size(); i++) {
        *unsafeGetLink(loffs[i]) = *gf.unsafeGetLink(loffs[i]);
      }
    }
  }
#endif
}

void GaugeField::
fetchLinks(GaugeField &gf, bool fetch_local_links, bool fetch_neighbor_links) {
  const char * fname = "fetchLinks";
  recordOffsetEnd();

#ifdef GAUGE_FIELD_COMM
  VRB.Result(cname, fname, "fetch_local_links=%d, fetch_neighbor_links=%d\n", fetch_local_links, fetch_neighbor_links);
  if (!fetch_neighbor_links) {
    if (fetch_local_links) {
      fetchLocalLinks(gf);
    }
    return;
  }

#ifdef PROFILE_COMM
  Float dtime = -dclock();
#endif
  //the key of send map is a 4-element vector giving the offset of the node to
  //receive from. The value is a vector of all the Matrices to send.
  map<vector<int>, vector<Matrix> > sendmap;
  map<vector<int>, int > sendmap_consume;
  vector<int> pos(4, 0);  //coordinates position of a site relative to this node
  vector<int> lpos(4, 0); //coordinates of a site relative to its home node
  vector<int> key(4, 0);  //offset of a site's home node from this node

  //Populate sendmap with the data that we need to send to other nodes
  for (int offset = 0; offset < geo.dir_offset[4]; offset += 4) {
    coordinatesFromOffset(pos.data(), offset);
    for (int i = 0; i < 4; i++) {
      lpos[i] = pos[i] % geo.node_sites[i];
      key[i] = pos[i] / geo.node_sites[i];
      if (lpos[i] < 0) {
        lpos[i] += geo.node_sites[i];
        key[i]--;
      }
    }
    bool on_node = key[0] == 0 && key[1] == 0 && key[2] == 0 && key[3] == 0;
    if (on_node && fetch_local_links || !on_node && fetch_neighbor_links) {
      vector<Matrix> & vec = sendmap[key];
      for (int mu = 0; mu < 4; mu++) {
        vec.push_back(*gf.unsafeGetLink(lpos.data(), mu));
      }
    }
  }

#ifdef PROFILE_COMM
  Float comtime = -dclock();
#endif
  vector<Matrix> recv_vec; //will store data received from other nodes
  //Iterate over all the nodes to which we need to send data.
  //We ultimately copy the received data into the corresponding
  //value of sendmap.
  map<vector<int>, vector<Matrix> >::iterator it;
  for (it = sendmap.begin(); it != sendmap.end(); it++) {
    key = it->first; //offset of node
    vector<Matrix> & send_vec = it->second;

    int size = send_vec.size();
    size_t size_bytes = size * sizeof(Matrix);
    recv_vec.resize(max(2500, size)); //Ask Luchang why max()

    //Get data from the node with offset given by key.
    //This may take several rounds of communications.
    Matrix * send = send_vec.data();
    Matrix * recv = recv_vec.data();
    for (int i = 0; i < 4; i++) {
      int dis = key[i];
      if (dis < 0) {
        while (dis != 0) {
          getMinusData((IFloat *)recv, (IFloat *)send, size * sizeof(Matrix) / sizeof(IFloat) , i);
          memcpy(send, recv, size_bytes);
          dis++;
        }
      } else if (dis > 0) {
        while (dis != 0) {
          getPlusData((IFloat *)recv, (IFloat *)send, size * sizeof(Matrix) / sizeof(IFloat), i);
          memcpy(send, recv, size_bytes);
          dis--;
        }
      }
    }
    sendmap_consume[key] = 0;
  }

#ifdef PROFILE_COMM
  comtime += dclock();
  print_time(cname, "fL-cub-com", comtime);
#endif

  //now sendmap[key] is the vector of data recieved from the node
  //pointed to by key.

  //Copy the received data into bufferred_gauge_field
  for (int offset = 0; offset < geo.dir_offset[4]; offset += 4) {
    coordinatesFromOffset(pos.data(), offset);
    for (int i = 0; i < 4; i++) {
      lpos[i] = pos[i] % geo.node_sites[i];
      key[i] = pos[i] / geo.node_sites[i];
      if (lpos[i] < 0) {
        lpos[i] += geo.node_sites[i];
        key[i]--;
      }
    }
    bool on_node = key[0] == 0 && key[1] == 0 && key[2] == 0 && key[3] == 0;
    if (on_node && fetch_local_links || !on_node && fetch_neighbor_links) {
      //sendmap_consume[key] keeps track of our progress in consuming the
      //received data in sendmap[key], so that we know which offset of
      //sendmap[key] corresponds to which site.
      int consume = sendmap_consume[key];
      vector<Matrix> & vec = sendmap[key];
      for (int mu = 0; mu < 4; mu++) {
        gauge_field[offset+mu] = vec[consume];
        consume++;

        /* Debug code
           Matrix diff_matrix = gf.getLink(pos.data(), mu) - gauge_field[offset+mu];
           if (diff_matrix.ReTr() != 0) {
           VRB.Result(cname, fname, "Mismatch: %d %d %d %d -- %d %d %d %d\n",
           key[0], key[1], key[2], key[3],
           pos[0], pos[1], pos[2], pos[3]);
           }
        // */

      }
      sendmap_consume[key] = consume;
    }

  }
#ifdef PROFILE_COMM
  dtime += dclock();
  print_time(cname, "fL-cub-tot", dtime);
#endif
#else
  if (this != &gf && (fetch_local_links || fetch_neighbor_links)) {
    memcpy(gauge_field, gf.gauge_field, geo.dir_offset[4] * sizeof(Matrix));
  }
#endif
}

void GaugeField::
fetchLocalLinks(GaugeField &gf) {
  const char * fname = "fetchLocalLinks()";
  recordOffsetEnd();
  if (this == &gf) {
    return;
  }

#ifdef GAUGE_FIELD_COMM

#ifdef PROFILE_COMM
  Float dtime = -dclock();
#endif
#pragma omp parallel for
  for (int index = 0; index < localVolume(); index++) {
    int x[4];
    coordinatesFromIndex(x, index);
    for (int mu = 0; mu < 4; mu++) {
      *unsafeGetLink(x, mu) = *gf.unsafeGetLink(x, mu);
    }
  }
#ifdef PROFILE_COMM
  dtime += dclock();
  print_time(cname, fname, dtime);
#endif

#else
  memcpy(gauge_field, gf.gauge_field, geo.dir_offset[4] * sizeof(Matrix));
#endif
}

Matrix GaugeField::
getLink(const int * site, int mu) const {
  int on_node_site[4];
#ifdef GAUGE_FIELD_COMM
  bool on_node = true;
  const Matrix *on_node_link;
  for (int i = 0; i < 4; i++) {
    on_node_site[i] = site[i] % geo.node_sites[i];
    if (on_node_site[i] < 0) {
      on_node_site[i] += geo.node_sites[i] ;
    }
    if (on_node_site[i] != site[i]) {
      on_node = false;
    }
  }
  on_node_link = gauge_field + siteOffset(on_node_site) + mu;
  if (on_node) {
    return *on_node_link;
  } else {
    Matrix send = *on_node_link;
    Matrix recv;
    for (int i = 0; i < 4; i++) {
      while (site[i] != on_node_site[i]) {
        if (site[i] < 0) {
          getMinusData((IFloat *)&recv, (IFloat *)&send, sizeof(Matrix) / sizeof(IFloat), i);
          on_node_site[i] -= geo.node_sites[i];
        } else {
          getPlusData((IFloat *)&recv, (IFloat *)&send, sizeof(Matrix) / sizeof(IFloat), i);
          on_node_site[i] += geo.node_sites[i];
        }
        send = recv;
      }
    }
    return recv ;
  }
#else
  return *(gauge_field + siteOffset(site) + mu);
#endif
}

CPS_END_NAMESPACE
