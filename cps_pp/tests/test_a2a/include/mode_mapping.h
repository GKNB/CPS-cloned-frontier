#pragma once

CPS_START_NAMESPACE

void testModeContractionIndices(const A2AArg &a2a_arg){
  LOGA2A << "Starting testModeContractionIndices" << std::endl;
  A2Aparams a2a_params(a2a_arg);
  modeIndexSet il, ir;
  int nl = a2a_params.getNl();
  int nf = a2a_params.getNflavors();
  int nt = a2a_params.getNtBlocks();
  int nhit = a2a_params.getNhits();
  int nsc = a2a_params.getNspinColor();
  int nv =a2a_params.getNv();

  //Check fully diluted mapping indicates contraction on every index
  {
    ModeContractionIndices<StandardIndexDilution,StandardIndexDilution> m(a2a_params);
    assert(m.tensorSize() == 1);
    
    //Expect the same index contractions for all time,spin_color,flavor coordinates
    for(il.hit =0; il.hit < nhit; il.hit++){
      for(il.time =0; il.time < nt; il.time++){
	for(il.flavor =0; il.flavor < nf; il.flavor++){
	  for(il.spin_color =0; il.spin_color < nsc; il.spin_color++){

	    for(ir.hit =0; ir.hit < nhit; ir.hit++){
	      for(ir.time =0; ir.time < nt; ir.time++){
		for(ir.flavor =0; ir.flavor < nf; ir.flavor++){
		  for(ir.spin_color =0; ir.spin_color < nsc; ir.spin_color++){
		
		    auto const & v = m.getIndexVector(il,ir);
		    assert(v.size() == nv);
		    for(int i=0;i<nv;i++)
		      assert( v[i].first == i && v[i].second == i );
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
  //Check time packed contractions. The implied delta in time is   delta_{t,t_l}delta_{t,t_r}
  {
    ModeContractionIndices<TimePackedIndexDilution,TimePackedIndexDilution> m(a2a_params);
    assert(m.tensorSize() == a2a_params.getNtBlocks()*a2a_params.getNtBlocks());
    TimePackedIndexDilution tdil(a2a_params);

    //These indices represent the required evaluations to compute the matrix product
    // \sum_{j} A_{il,j} B_{j,ir}
    //The implied delta functions here assert that only  il_t == j_t  and  j_t == ir_t  are non-zero

    //Expect the same index contractions for all time,spin_color,flavor coordinates
    for(il.hit =0; il.hit < nhit; il.hit++){
      for(il.time =0; il.time < nt; il.time++){
	for(il.flavor =0; il.flavor < nf; il.flavor++){
	  for(il.spin_color =0; il.spin_color < nsc; il.spin_color++){

	    for(ir.hit =0; ir.hit < nhit; ir.hit++){
	      for(ir.time =0; ir.time < nt; ir.time++){
		for(ir.flavor =0; ir.flavor < nf; ir.flavor++){
		  for(ir.spin_color =0; ir.spin_color < nsc; ir.spin_color++){
		    auto const & v = m.getIndexVector(il,ir);
		    // std::cout << il << " " << ir << " size " << v.size() << std::endl;
		    // for(int i=0;i<v.size();i++){
		    //   modeIndexSet jl, jr;
		    //   tdil.indexUnmap(v[i].first,jl);
		    //   tdil.indexUnmap(v[i].second,jr);
		    //   std::cout << i << " " << jl << " " << jr << std::endl;
		    // }

		    if(il.time == ir.time){
		      //They are both unpacked in spin_color, flavor so we expect 12*nf*nhit high mode columns / rows will still need to be contracted
		      assert(v.size() == nl + nsc*nf*nhit);
		    }else{
		      assert(v.size() == nl);
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }


  //Check time/spin/color packed contractions
  {
    ModeContractionIndices<TimeSpinColorPackedIndexDilution,TimeSpinColorPackedIndexDilution> m(a2a_params);
    assert(m.tensorSize() == nt*nt*nsc*nsc);
    TimePackedIndexDilution tdil(a2a_params);

    //These indices represent the required evaluations to compute the matrix product
    // \sum_{j} A_{il,j} B_{j,ir}
    //The implied delta functions here assert that only  il_t == j_t  &&  j_t == ir_t && il_sc == j_sc && ir_sc == j_sc   are non-zero

    //Expect the same index contractions for all time,spin_color,flavor coordinates
    for(il.hit =0; il.hit < nhit; il.hit++){
      for(il.time =0; il.time < nt; il.time++){
	for(il.flavor =0; il.flavor < nf; il.flavor++){
	  for(il.spin_color =0; il.spin_color < nsc; il.spin_color++){

	    for(ir.hit =0; ir.hit < nhit; ir.hit++){
	      for(ir.time =0; ir.time < nt; ir.time++){
		for(ir.flavor =0; ir.flavor < nf; ir.flavor++){
		  for(ir.spin_color =0; ir.spin_color < nsc; ir.spin_color++){
		    auto const & v = m.getIndexVector(il,ir);

		    if(il.time == ir.time && il.spin_color == ir.spin_color){
		      assert(v.size() == nl + nf*nhit);
		    }else{
		      assert(v.size() == nl);
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }



  LOGA2A << "testModeContractionIndices passed" << std::endl;
}

CPS_END_NAMESPACE
