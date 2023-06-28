template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _trV{
  typedef typename VectorMatrixType::scalar_type OutputType;
  accelerator_inline void operator()(OutputType &out, const VectorMatrixType &in, const int lane) const{ 
    Trace(out, in, lane);
  }
};

template<int Index, typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _trIndexV{
  typedef typename _PartialTraceFindReducedType<VectorMatrixType,Index>::type OutputType;
  accelerator_inline void operator()(OutputType &out, const VectorMatrixType &in, const int lane) const{ 
    TraceIndex<Index>(out, in, lane);
  }
};

template<int Index1, int Index2, typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _trTwoIndicesV{
  typedef typename _PartialDoubleTraceFindReducedType<VectorMatrixType,Index1,Index2>::type OutputType;
  accelerator_inline void operator()(OutputType &out, const VectorMatrixType &in, const int lane) const{ 
    TraceTwoIndices<Index1,Index2>(out, in, lane);
  }
};

template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _transposeV{
  typedef VectorMatrixType OutputType;
  accelerator_inline void operator()(VectorMatrixType &out, const VectorMatrixType &in, const int lane) const{ 
    Transpose(out, in, lane);
  }
};

template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _daggerV{
  typedef VectorMatrixType OutputType;
  accelerator_inline void operator()(VectorMatrixType &out, const VectorMatrixType &in, const int lane) const{ 
    Transpose(out, in, lane);
    cconj(out,lane);
  }
};


template<int Index, typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _transIdx{
  typedef VectorMatrixType OutputType;
  accelerator_inline void operator()(VectorMatrixType &out, const VectorMatrixType &in, const int lane) const{ 
    TransposeOnIndex<Index>(out, in, lane);
  }
};

template<typename ComplexType>
struct _gl_r_V{
  typedef CPSspinMatrix<ComplexType> OutputType;
  int dir;
  _gl_r_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinMatrix<ComplexType> &out, const CPSspinMatrix<ComplexType> &in, const int lane) const{ 
    gl_r(out, in, dir, lane);
  }
};
template<typename ComplexType>
struct _gl_r_scf_V{
  typedef CPSspinColorFlavorMatrix<ComplexType> OutputType;
  int dir;
  _gl_r_scf_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &out, const CPSspinColorFlavorMatrix<ComplexType> &in, const int lane) const{ 
    gl_r(out, in, dir, lane);
  }
};

template<typename ComplexType>
struct _gr_r_V{
  typedef CPSspinMatrix<ComplexType> OutputType;
  int dir;
  _gr_r_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinMatrix<ComplexType> &out, const CPSspinMatrix<ComplexType> &in, const int lane) const{ 
    gr_r(out, in, dir, lane);
  }
};
template<typename ComplexType>
struct _gr_r_scf_V{
  typedef CPSspinColorFlavorMatrix<ComplexType> OutputType;
  int dir;
  _gr_r_scf_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &out, const CPSspinColorFlavorMatrix<ComplexType> &in, const int lane) const{ 
    gr_r(out, in, dir, lane);
  }
};



template<typename ComplexType>
struct _glAx_r_V{
  typedef CPSspinMatrix<ComplexType> OutputType;
  int dir;
  _glAx_r_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinMatrix<ComplexType> &out, const CPSspinMatrix<ComplexType> &in, const int lane) const{ 
    glAx_r(out, in, dir, lane);
  }
};
template<typename ComplexType>
struct _glAx_r_scf_V{
  typedef CPSspinColorFlavorMatrix<ComplexType> OutputType;
  int dir;
  _glAx_r_scf_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &out, const CPSspinColorFlavorMatrix<ComplexType> &in, const int lane) const{ 
    glAx_r(out, in, dir, lane);
  }
};


template<typename ComplexType>
struct _grAx_r_V{
  typedef CPSspinMatrix<ComplexType> OutputType;
  int dir;
  _grAx_r_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinMatrix<ComplexType> &out, const CPSspinMatrix<ComplexType> &in, const int lane) const{ 
    grAx_r(out, in, dir, lane);
  }
};
template<typename ComplexType>
struct _grAx_r_scf_V{
  typedef CPSspinColorFlavorMatrix<ComplexType> OutputType;
  int dir;
  _grAx_r_scf_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &out, const CPSspinColorFlavorMatrix<ComplexType> &in, const int lane) const{ 
    grAx_r(out, in, dir, lane);
  }
};

template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _cconjV{
  typedef VectorMatrixType OutputType;
  accelerator_inline void operator()(VectorMatrixType &out, const VectorMatrixType &in, const int lane) const{ 
    cconj(out, in, lane);
  }
};

template<typename ComplexType>
struct _gl_r_scf_V_2d{
  typedef CPSspinColorFlavorMatrix<ComplexType> OutputType;
  int dir;
  _gl_r_scf_V_2d(int dir): dir(dir){}

  accelerator_inline int nParallel() const{ return 144; } //spin, (color * flavor)^2
  
  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &out, const CPSspinColorFlavorMatrix<ComplexType> &in, const int widx, const int lane) const{
    int ffcc = widx % 36;
    int s2 = widx / 36;
    gl_r(out, in, dir, s2, ffcc, lane);
  }
};
template<typename ComplexType>
struct _glAx_r_scf_V_2d{
  typedef CPSspinColorFlavorMatrix<ComplexType> OutputType;
  int dir;
  _glAx_r_scf_V_2d(int dir): dir(dir){}

  accelerator_inline int nParallel() const{ return 144; } //spin, (color * flavor)^2
  
  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &out, const CPSspinColorFlavorMatrix<ComplexType> &in, const int widx, const int lane) const{
    int ffcc = widx % 36;
    int s2 = widx / 36;
    glAx_r(out, in, dir, s2, ffcc, lane);
  }
};


template<typename ComplexType>
struct _gr_r_scf_V_2d{
  typedef CPSspinColorFlavorMatrix<ComplexType> OutputType;
  int dir;
  _gr_r_scf_V_2d(int dir): dir(dir){}

  accelerator_inline int nParallel() const{ return 144; } //spin, (color * flavor)^2
  
  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &out, const CPSspinColorFlavorMatrix<ComplexType> &in, const int widx, const int lane) const{
    int ffcc = widx % 36;
    int s1 = widx / 36;
    gr_r(out, in, dir, s1, ffcc, lane);
  }
};
template<typename ComplexType>
struct _grAx_r_scf_V_2d{
  typedef CPSspinColorFlavorMatrix<ComplexType> OutputType;
  int dir;
  _grAx_r_scf_V_2d(int dir): dir(dir){}

  accelerator_inline int nParallel() const{ return 144; } //spin, (color * flavor)^2
  
  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &out, const CPSspinColorFlavorMatrix<ComplexType> &in, const int widx, const int lane) const{
    int ffcc = widx % 36;
    int s1 = widx / 36;
    grAx_r(out, in, dir, s1, ffcc, lane);
  }
};

template<typename VectorMatrixType>
struct _timesV{
  typedef VectorMatrixType OutputType;
  accelerator_inline void operator()(VectorMatrixType &out, const VectorMatrixType &a, const VectorMatrixType &b, const int lane) const{ 
    mult(out, a, b, lane);
  }
};

template<typename ScalarType, typename VectorMatrixType>
struct _scalarTimesVPre{
  typedef VectorMatrixType OutputType;
  accelerator_inline void operator()(VectorMatrixType &out, const ScalarType &a, const VectorMatrixType &b, const int lane) const{ 
    scalar_mult_pre(out, a, b, lane);
  }
};



template<typename VectorMatrixType>
struct _addV{
  typedef VectorMatrixType OutputType;
  accelerator_inline void operator()(VectorMatrixType &out, const VectorMatrixType &a, const VectorMatrixType &b, const int lane) const{ 
    add(out, a, b, lane);
  }
};

template<typename VectorMatrixType>
struct _subV{
  typedef VectorMatrixType OutputType;
  accelerator_inline void operator()(VectorMatrixType &out, const VectorMatrixType &a, const VectorMatrixType &b, const int lane) const{ 
    sub(out, a, b, lane);
  }
};

template<typename VectorMatrixType>
struct _traceProdV{
  typedef typename VectorMatrixType::scalar_type OutputType;
  accelerator_inline void operator()(OutputType &out, const VectorMatrixType &a, const VectorMatrixType &b, const int lane) const{ 
    Trace(out, a, b, lane);
  }
};

template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _unitV{
  accelerator_inline void operator()(VectorMatrixType &in, const int lane) const{ 
    unit(in, lane);
  }
};

template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _timesIV{
  accelerator_inline void operator()(VectorMatrixType &in, const int lane) const{ 
    timesI(in, in, lane);
  }
};

template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _timesMinusIV{
  accelerator_inline void operator()(VectorMatrixType &in, const int lane) const{ 
    timesMinusI(in, in, lane);
  }
};

template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _timesMinusOneV{
  accelerator_inline void operator()(VectorMatrixType &in, const int lane) const{ 
    timesMinusOne(in, in, lane);
  }
};


template<typename ComplexType>
struct _gl_V{
  int dir;
  _gl_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinMatrix<ComplexType> &in, const int lane) const{ 
    gl(in, dir, lane);
  }
};
template<typename ComplexType>
struct _gl_scf_V{
  int dir;
  _gl_scf_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &in, const int lane) const{ 
    gl(in, dir, lane);
  }
};

template<typename ComplexType>
struct _gr_V{
  int dir;
  _gr_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinMatrix<ComplexType> &in, const int lane) const{ 
    gr(in, dir, lane);
  }
};
template<typename ComplexType>
struct _gr_scf_V{
  int dir;
  _gr_scf_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &in, const int lane) const{ 
    gr(in, dir, lane);
  }
};


template<typename ComplexType>
struct _glAx_V{
  int dir;
  _glAx_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinMatrix<ComplexType> &in, const int lane) const{ 
    glAx(in, dir, lane);
  }
};
template<typename ComplexType>
struct _glAx_scf_V{
  int dir;
  _glAx_scf_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &in, const int lane) const{ 
    glAx(in, dir, lane);
  }
};

template<typename ComplexType>
struct _grAx_V{
  int dir;
  _grAx_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinMatrix<ComplexType> &in, const int lane) const{ 
    grAx(in, dir, lane);
  }
};
template<typename ComplexType>
struct _grAx_scf_V{
  int dir;
  _grAx_scf_V(int dir): dir(dir){}

  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &in, const int lane) const{ 
    grAx(in, dir, lane);
  }
};


template<typename ComplexType>
struct _pl_V{
  const FlavorMatrixType type;
  _pl_V(const FlavorMatrixType type): type(type){}

  accelerator_inline void operator()(CPSflavorMatrix<ComplexType> &in, const int lane) const{ 
    pl(in, type, lane);
  }
};
template<typename ComplexType>
struct _pl_scf_V{
  const FlavorMatrixType type;
  _pl_scf_V(const FlavorMatrixType type): type(type){}

  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &in, const int lane) const{ 
    pl(in, type, lane);
  }
};

template<typename ComplexType>
struct _pr_V{
  const FlavorMatrixType type;
  _pr_V(const FlavorMatrixType type): type(type){}

  accelerator_inline void operator()(CPSflavorMatrix<ComplexType> &in, const int lane) const{ 
    pr(in, type, lane);
  }
};
template<typename ComplexType>
struct _pr_scf_V{
  const FlavorMatrixType type;
  _pr_scf_V(const FlavorMatrixType type): type(type){}

  accelerator_inline void operator()(CPSspinColorFlavorMatrix<ComplexType> &in, const int lane) const{ 
    pr(in, type, lane);
  }
};

template<typename T>
struct _setUnit{
  accelerator_inline void operator()(T &m) const{
    m.unit();
  } 
};
