import copy


class Globals:
    bc = ["BND_CND_PRD","BND_CND_PRD","BND_CND_PRD","BND_CND_APRD"]
    @staticmethod
    def set_bc(ibc):
        Globals.bc = ibc
    
class PointSourceAttrArg:
    def __init__(self,pos):
        self.pos = pos
           
    def write(self,fhandle):
        fhandle.write("AttrType type = POINT_SOURCE_ATTR\n")
        fhandle.write("struct PointSourceAttrArg point_source_attr = {\n")
        fhandle.write("Vector pos[4] = {\n")
        for i in range(4):
            fhandle.write("int pos[%d] = %d\n" % (i,self.pos[i]) )
        fhandle.write("}\n");
        fhandle.write("}\n");

class WallSourceAttrArg:
    def __init__(self,t):
        self.t = t
    def write(self,fhandle):
        fhandle.write("AttrType type = WALL_SOURCE_ATTR\n")
        fhandle.write("struct WallSourceAttrArg wall_source_attr = {\n")
        fhandle.write("int t = %d\n" % self.t)
        fhandle.write("}\n");

  #NOTE: Momentum units are:
  #                         Periodic  2\pi/L
  #                     Antiperiodic   \pi/L
  #                         G-parity   \pi/2L
  #In G-parity directions, the propagators are antiperiodic in 2L
class MomentumAttrArg:
    def __init__(self,p): #p in units of pi
        self.p = p
    def write(self,fhandle):
        fhandle.write("AttrType type = MOMENTUM_ATTR\n")
        fhandle.write("struct MomentumAttrArg momentum_attr = {\n")
        fhandle.write("Vector p[3] = {\n")
        for i in range(3):
            fhandle.write("int p[%d] = %d\n" % (i,self.p[i]) )
        fhandle.write("}\n");
        fhandle.write("}\n");

class GparityFlavorAttrArg:
    def __init__(self,flav):
        self.flav = flav
    def write(self,fhandle):
        fhandle.write("AttrType type = GPARITY_FLAVOR_ATTR\n")
        fhandle.write("struct GparityFlavorAttrArg gparity_flavor_attr = {\n")
        fhandle.write("int flavor = %d\n" % self.flav)
        fhandle.write("}\n");

class GaugeFixAttrArg:
    def __init__(self,gauge_fix_src):
        self.gauge_fix_src = gauge_fix_src
        self.gauge_fix_snk = 0 #sink gauge fixing is handled in contract args now, this uses the crappy old QpropW method
    def write(self,fhandle):
        fhandle.write("AttrType type = GAUGE_FIX_ATTR\n")
        fhandle.write("struct GaugeFixAttrArg gauge_fix_attr = {\n")
        fhandle.write("int gauge_fix_src = %d\n" % self.gauge_fix_src)
        fhandle.write("int gauge_fix_snk = %d\n" % self.gauge_fix_snk)
        fhandle.write("}\n");

class MomCosAttrArg:
    def __init__(self):
        pass
    def write(self,fhandle):
        fhandle.write("AttrType type = MOM_COS_ATTR\n")
        fhandle.write("struct MomCosAttrArg mom_cos_attr = {\n")
        fhandle.write("}\n");

class GparityOtherFlavPropAttrArg:
    def __init__(self, tag):
        self.tag = tag
    def write(self,fhandle):
        fhandle.write("AttrType type = GPARITY_OTHER_FLAV_PROP_ATTR\n")
        fhandle.write("struct GparityOtherFlavPropAttrArg gparity_other_flav_prop_attr = {\n")
        fhandle.write("string tag = \"%s\"\n" % self.tag)
        fhandle.write("}\n");  

class PropCombinationAttrArg:
    def __init__(self,prop_A,prop_B, comb):
        self.prop_A = prop_A
        self.prop_B = prop_B
        self.comb = comb
    def write(self,fhandle):
        fhandle.write("AttrType type = PROP_COMBINATION_ATTR\n")
        fhandle.write("struct PropCombinationAttrArg prop_combination_attr = {\n")
        fhandle.write("string prop_A = \"%s\"\n" % self.prop_A)
        fhandle.write("string prop_B = \"%s\"\n" % self.prop_B)
        fhandle.write("PropCombination combination = %s\n" % self.comb)
        fhandle.write("}\n");  

class CGAttrArg:
    def __init__(self,**args):
        self.max_num_iter = 10000
        self.stop_rsd = 1e-08
        self.true_rsd = 1e-08
        if("max_num_iter" in args):
            self.max_num_iter = args["max_num_iter"]
        if("stop_rsd" in args):
            self.stop_rsd = args["stop_rsd"]
        if("true_rsd" in args):
            self.true_rsd = args["true_rsd"]

    def write(self,fhandle):
        fhandle.write("AttrType type = CG_ATTR\n")
        fhandle.write("struct CGAttrArg cg_attr = {\n")
        fhandle.write("int max_num_iter = %d\n" % self.max_num_iter)
        fhandle.write("double stop_rsd = %f\n" % self.stop_rsd)
        fhandle.write("double true_rsd = %f\n" % self.true_rsd)
        fhandle.write("}");  



class PropagatorArg:
    def __init__(self,tag, mass, time_bc = "BND_CND_APRD"):
        self.tag = tag
        self.mass = mass
        self.bc = copy.deepcopy(Globals.bc)
        self.bc[3] = time_bc
        self.attr = []
    def write(self,fhandle, vname, array_idx):
        fhandle.write("class PropagatorArg %s[%d] = {\n" % (vname,array_idx) )
        fhandle.write("struct GenericPropAttrArg generics = {\n")
        fhandle.write("string tag = \"%s\"\n" % self.tag)
        fhandle.write("double mass = %e\n" % self.mass)
        fhandle.write("Vector bc[4] = {\n")
        for i in range(4):
            fhandle.write("BndCndType bc[%d] = %s\n" % (i,self.bc[i]) )
        fhandle.write("}\n")
        fhandle.write("}\n")
        fhandle.write("Array attributes[%d] = {\n" % len(self.attr) )
        for i in range(len(self.attr)):
            self.attr[i].write(fhandle)
        fhandle.write("}\n")
        fhandle.write("}\n")

    def setGparityFlavor(self,f):
        self.attr.append( GparityFlavorAttrArg(f))
    def setWallSource(self,t,flav=None):
        self.attr.append( WallSourceAttrArg(t))
        if(flav != None): 
            self.setGparityFlavor(flav)
    def setMomentumSource(self,t,p, flav=None):
        self.attr.append( WallSourceAttrArg(t))
        self.attr.append( MomentumAttrArg(p))
        if(flav != None): 
            self.setGparityFlavor(flav)
    def setMomCosSource(self,t,p, flav=None):
        self.attr.append( WallSourceAttrArg(t))
        self.attr.append( MomentumAttrArg(p))
        self.attr.append( MomCosAttrArg())
        if(flav != None): 
            self.setGparityFlavor(flav)
    def setCombinationSource(self,prop_A,prop_B,comb):
        self.attr.append( PropCombinationAttrArg(prop_A,prop_B,comb))
               

class JobPropagatorArgs:
    def __init__(self):
        self.props = []        
    def write(self,filename):
        fhandle = open(filename, "w")
        fhandle.write("class JobPropagatorArgs prop_arg = {\n")
        fhandle.write("Array props[%d] = {\n" % len(self.props))
        for i in range(len(self.props)):
            self.props[i].write(fhandle,"props",i)
        fhandle.write("}\n")
        fhandle.write("}\n")
        fhandle.close()

    def addPropagator(self,prop):
        self.props.append(prop)




class ContractionTypeLLMesons:
    def __init__(self,prop_L,sink_mom, file):
        self.prop_L = prop_L
        self.sink_mom = copy.deepcopy(sink_mom)
        self.file = file

    def write(self,fhandle):
        fhandle.write("ContractionType type = CONTRACTION_TYPE_LL_MESONS\n")
        fhandle.write("struct ContractionTypeLLMesons contraction_type_ll_mesons = {\n")
        fhandle.write("string prop_L = \"%s\"\n" % self.prop_L)
        fhandle.write("Vector sink_mom[3] = {\n")
        for i in range(3):
            fhandle.write("int sink_mom[%d] = %s\n" % (i,self.sink_mom[i]) )
        fhandle.write("}\n");  
        fhandle.write("string file = \"%s\"\n" % self.file)
        fhandle.write("}\n");  

class ContractionTypeHLMesons:
    def __init__(self,prop_H,prop_L,sink_mom, file):
        self.prop_H = prop_H
        self.prop_L = prop_L
        self.sink_mom = copy.deepcopy(sink_mom)
        self.file = file

    def write(self,fhandle):
        fhandle.write("ContractionType type = CONTRACTION_TYPE_HL_MESONS\n")
        fhandle.write("struct ContractionTypeHLMesons contraction_type_hl_mesons = {\n")
        fhandle.write("string prop_H = \"%s\"\n" % self.prop_H)
        fhandle.write("string prop_L = \"%s\"\n" % self.prop_L)
        fhandle.write("Vector sink_mom[3] = {\n")
        for i in range(3):
            fhandle.write("int sink_mom[%d] = %s\n" % (i,self.sink_mom[i]) )
        fhandle.write("}\n");  
        fhandle.write("string file = \"%s\"\n" % self.file)
        fhandle.write("}\n"); 

class ContractionTypeOVVpAA:
    def __init__(self,prop_H_t0,prop_L_t0,prop_H_t1,prop_L_t1, file):
        self.prop_H_t0 = prop_H_t0
        self.prop_L_t0 = prop_L_t0
        self.prop_H_t1 = prop_H_t1
        self.prop_L_t1 = prop_L_t1
        self.file = file

    def write(self,fhandle):
        fhandle.write("ContractionType type = CONTRACTION_TYPE_O_VV_P_AA\n")
        fhandle.write("struct ContractionTypeOVVpAA contraction_type_o_vv_p_aa = {\n")
        fhandle.write("string prop_H_t0 = \"%s\"\n" % self.prop_H_t0)
        fhandle.write("string prop_L_t0 = \"%s\"\n" % self.prop_L_t0)
        fhandle.write("string prop_H_t1 = \"%s\"\n" % self.prop_H_t1)
        fhandle.write("string prop_L_t1 = \"%s\"\n" % self.prop_L_t1)
        fhandle.write("string file = \"%s\"\n" % self.file)
        fhandle.write("}\n"); 

class MomArg:
    def __init__(self,p):
        self.p = copy.deepcopy(p)

    def write(self,fhandle, vname, array_idx):
        fhandle.write("struct MomArg %s[%d] = {\n" % (vname,array_idx) )
        fhandle.write("Vector p[3] = {\n")
        for i in range(3):
            fhandle.write("double p[%d] = %g\n" % (i,self.p[i]) )
        fhandle.write("}\n"); 
        fhandle.write("}\n"); 

class MomPairArg:
    def __init__(self,p1,p2):
        self.p1 = copy.deepcopy(p1)
        self.p2 = copy.deepcopy(p2)
    def write(self,fhandle, vname, array_idx):
        fhandle.write("struct MomPairArg %s[%d] = {\n" % (vname,array_idx) )
        fhandle.write("Vector p1[3] = {\n")
        for i in range(3):
            fhandle.write("double p1[%d] = %g\n" % (i,self.p1[i]) )
        fhandle.write("}\n");
        fhandle.write("Vector p2[3] = {\n")
        for i in range(3):
            fhandle.write("double p2[%d] = %g\n" % (i,self.p2[i]) )
        fhandle.write("}\n");
        fhandle.write("}\n"); 


class ContractionTypeAllBilinears:
    def __init__(self,prop_1,prop_2,momenta, file): #momenta elements should be 3-component lists
        self.prop_1 = prop_1
        self.prop_2 = prop_2

        self.momenta = []
        for i in range(len(momenta)):
            self.momenta.append( MomArg(momenta[i]) )
        self.file = file

    def write(self,fhandle):
        fhandle.write("ContractionType type = CONTRACTION_TYPE_ALL_BILINEARS\n")
        fhandle.write("struct ContractionTypeAllBilinears contraction_type_all_bilinears = {\n")
        fhandle.write("string prop_1 = \"%s\"\n" % self.prop_1)
        fhandle.write("string prop_2 = \"%s\"\n" % self.prop_2)
        fhandle.write("Array momenta[%d] = {\n" % len(self.momenta) )
        for i in range(len(self.momenta)):
            self.momenta[i].write(fhandle,"momenta",i)
        fhandle.write("}\n"); 
        fhandle.write("string file = \"%s\"\n" % self.file)
        fhandle.write("}\n"); 

class ContractionTypeAllWallSinkBilinears:
    def __init__(self,prop_1,prop_2,momenta, file): #momenta elements should be 3-component lists
        self.prop_1 = prop_1
        self.prop_2 = prop_2
        self.momenta = []
        for i in range(len(momenta)):
            self.momenta.append( MomArg(momenta[i]) )
        self.file = file

    def write(self,fhandle):
        fhandle.write("ContractionType type = CONTRACTION_TYPE_ALL_WALLSINK_BILINEARS\n")
        fhandle.write("struct ContractionTypeAllWallSinkBilinears contraction_type_all_wallsink_bilinears = {\n")
        fhandle.write("string prop_1 = \"%s\"\n" % self.prop_1)
        fhandle.write("string prop_2 = \"%s\"\n" % self.prop_2)
        fhandle.write("Array momenta[%d] = {\n" % len(self.momenta) )
        for i in range(len(self.momenta)):
            self.momenta[i].write(fhandle,"momenta",i)
        fhandle.write("}\n"); 
        fhandle.write("string file = \"%s\"\n" % self.file)
        fhandle.write("}\n"); 


class ContractionTypeAllWallSinkBilinearsSpecificMomentum:
    def __init__(self,prop_1,prop_2,momenta, cosine_sink, file): #momenta elements should each be a 2 component containing a 3-component list
        self.prop_1 = prop_1
        self.prop_2 = prop_2
        self.momenta = []
        for i in range(len(momenta)):
            self.momenta.append( MomPairArg(momenta[i][0],momenta[i][1]) )
        self.cosine_sink = cosine_sink
        self.file = file

    def write(self,fhandle):
        fhandle.write("ContractionType type = CONTRACTION_TYPE_ALL_WALLSINK_BILINEARS_SPECIFIC_MOMENTUM\n")
        fhandle.write("struct ContractionTypeAllWallSinkBilinearsSpecificMomentum contraction_type_all_wallsink_bilinears_specific_momentum = {\n")
        fhandle.write("string prop_1 = \"%s\"\n" % self.prop_1)
        fhandle.write("string prop_2 = \"%s\"\n" % self.prop_2)
        fhandle.write("Array momenta[%d] = {\n" % len(self.momenta) )
        for i in range(len(self.momenta)):
            self.momenta[i].write(fhandle,"momenta",i)
        fhandle.write("}\n"); 
        fhandle.write("int cosine_sink = %d\n" % self.cosine_sink)
        fhandle.write("string file = \"%s\"\n" % self.file)
        fhandle.write("}\n"); 

class ContractionTypeFourierProp:
    def __init__(self,prop,gauge_fix,momenta, file): #momenta elements should be 3-component lists
        self.prop = prop
        self.gauge_fix = gauge_fix
        self.momenta = []
        for i in range(len(momenta)):
            self.momenta.append( MomArg(momenta[i]) )
        self.file = file

    def write(self,fhandle):
        fhandle.write("ContractionType type = CONTRACTION_TYPE_FOURIER_PROP\n")
        fhandle.write("struct ContractionTypeFourierProp contraction_type_fourier_prop = {\n")
        fhandle.write("string prop = \"%s\"\n" % self.prop)
        fhandle.write("int gauge_fix = %d\n" % self.gauge_fix)
        fhandle.write("Array momenta[%d] = {\n" % len(self.momenta) )
        for i in range(len(self.momenta)):
            self.momenta[i].write(fhandle,"momenta",i)
        fhandle.write("}\n"); 
        fhandle.write("string file = \"%s\"\n" % self.file)
        fhandle.write("}\n"); 


class ContractionTypeBilinearVertex:
    def __init__(self,prop_1,prop_2,momenta, file): #momenta elements should be 3-component lists
        self.prop_1 = prop_1
        self.prop_2 = prop_2
        self.momenta = []
        for i in range(len(momenta)):
            self.momenta.append( MomArg(momenta[i]) )
        self.file = file

    def write(self,fhandle):
        fhandle.write("ContractionType type = CONTRACTION_TYPE_BILINEAR_VERTEX\n")
        fhandle.write("struct ContractionTypeBilinearVertex contraction_type_bilinear_vertex = {\n")
        fhandle.write("string prop_1 = \"%s\"\n" % self.prop_1)
        fhandle.write("string prop_2 = \"%s\"\n" % self.prop_2)
        fhandle.write("Array momenta[%d] = {\n" % len(self.momenta) )
        for i in range(len(self.momenta)):
            self.momenta[i].write(fhandle,"momenta",i)
        fhandle.write("}\n"); 
        fhandle.write("string file = \"%s\"\n" % self.file)
        fhandle.write("}\n"); 

class QuadrilinearSpinStructure:
    def __init__(self,Gamma1, Gamma2, Sigma1, Sigma2):
        self.Gamma1 = copy.deepcopy(Gamma1)
        self.Gamma2 = copy.deepcopy(Gamma2)
        self.Sigma1 = copy.deepcopy(Sigma1)
        self.Sigma2 = copy.deepcopy(Sigma2)

    def write(self,fhandle,vname,array_idx):
        fhandle.write("struct QuadrilinearSpinStructure %s[%d] = {\n" % (vname,array_idx) )

        fhandle.write("Vector Gamma1[%d] = {\n" % len(self.Gamma1) )
        for i in range(len(self.Gamma1)):
            fhandle.write("int Gamma1[%d] = %s\n" % (i,self.Gamma1[i]) )
        fhandle.write("}\n");

        fhandle.write("Vector Gamma2[%d] = {\n" % len(self.Gamma2) )
        for i in range(len(self.Gamma2)):
            fhandle.write("int Gamma2[%d] = %s\n" % (i,self.Gamma2[i]) )
        fhandle.write("}\n");

        fhandle.write("Vector Sigma1[%d] = {\n" % len(self.Sigma1) )
        for i in range(len(self.Sigma1)):
            fhandle.write("int Sigma1[%d] = %s\n" % (i,self.Sigma1[i]) )
        fhandle.write("}\n");

        fhandle.write("Vector Sigma2[%d] = {\n" % len(self.Sigma2) )
        for i in range(len(self.Sigma2)):
            fhandle.write("int Sigma2[%d] = %s\n" % (i,self.Sigma2[i]) )
        fhandle.write("}\n");

        fhandle.write("}\n"); 


class ContractionTypeQuadrilinearVertex:
    def __init__(self,prop_1,prop_2,prop_3,prop_4,momenta,file, spin_structs):
        self.prop_1 = prop_1
        self.prop_2 = prop_2
        self.prop_3 = prop_3
        self.prop_4 = prop_4

        self.momenta = []
        for i in range(len(momenta)):
            self.momenta.append( MomArg(momenta[i]) )
        self.file = file

        self.spin_structs = copy.deepcopy(spin_structs)

    def write(self,fhandle):
        fhandle.write("ContractionType type = CONTRACTION_TYPE_QUADRILINEAR_VERTEX\n")
        fhandle.write("struct ContractionTypeBilinearVertex contraction_type_quadrilinear_vertex = {\n")
        fhandle.write("string prop_1 = \"%s\"\n" % self.prop_1)
        fhandle.write("string prop_2 = \"%s\"\n" % self.prop_2)
        fhandle.write("string prop_3 = \"%s\"\n" % self.prop_3)
        fhandle.write("string prop_4 = \"%s\"\n" % self.prop_4)

        fhandle.write("Array momenta[%d] = {\n" % len(self.momenta) )
        for i in range(len(self.momenta)):
            self.momenta[i].write(fhandle,"momenta",i)
        fhandle.write("}\n"); 

        fhandle.write("string file = \"%s\"\n" % self.file)

        fhandle.write("Array spin_structs[%d] = {\n" % len(self.spin_structs) )
        for i in range(len(self.spin_structs)):
            self.spin_structs[i].write(fhandle,"spin_structs",i)
        fhandle.write("}\n"); 

        fhandle.write("}\n"); 

class ContractionTypeTopologicalCharge:
    def __init__(self,n_ape_smearing_cycles,ape_smear_su3_project,ape_su3_proj_tolerance,ape_orthog, ape_coef ,file):
        self.n_ape_smearing_cycles = n_ape_smearing_cycles
        self.ape_smear_su3_project = ape_smear_su3_project
        self.ape_su3_proj_tolerance = ape_su3_proj_tolerance
        self.ape_orthog = ape_orthog
        self.ape_coef = ape_coef
        self.file = file

    def write(self,fhandle):
        fhandle.write("ContractionType type = CONTRACTION_TYPE_TOPOLOGICAL_CHARGE\n")
        fhandle.write("struct ContractionTypeTopologicalCharge contraction_type_topological_charge = {\n")
        fhandle.write("int n_ape_smearing_cycles = %s\n" % self.n_ape_smearing_cycles)
        fhandle.write("int ape_smear_su3_project = %s\n" % self.ape_smear_su3_project)
        fhandle.write("double ape_su3_proj_tolerance = %s\n" % self.ape_su3_proj_tolerance)
        fhandle.write("int ape_orthog = %s\n" % self.ape_orthog)
        fhandle.write("double ape_coef = %s\n" % self.ape_coef)
        fhandle.write("string file = \"%s\"\n" % self.file)
        fhandle.write("}\n")


class FixGaugeArg:
    def __init__(self,fix_gauge_kind, hyperplane_start, hyperplane_step, hyperplane_num, stop_cond, max_iter_num):
        self.fix_gauge_kind = fix_gauge_kind 
        self.hyperplane_start = hyperplane_start
        self.hyperplane_step = hyperplane_step
        self.hyperplane_num = hyperplane_num
        self.stop_cond = stop_cond
        self.max_iter_num = max_iter_num
        
    def write(self,fhandle,vname):
        fhandle.write("struct FixGaugeArg %s = {\n" % vname)
        fhandle.write("FixGaugeType fix_gauge_kind = %s\n" % self.fix_gauge_kind)
        fhandle.write("int hyperplane_start = %s\n" % self.hyperplane_start)
        fhandle.write("int hyperplane_step = %s\n" % self.hyperplane_step)
        fhandle.write("int hyperplane_num = %s\n" % self.hyperplane_num)
        fhandle.write("double stop_cond = %s\n" % self.stop_cond)
        fhandle.write("int max_iter_num = %s\n" % self.max_iter_num)
        fhandle.write("}\n")





class GparityContractArg:
    def __init__(self,config_fmt,conf_start,conf_incr,conf_lessthan,fix_gauge):
        self.config_fmt = config_fmt
        self.conf_start = conf_start
        self.conf_incr = conf_incr
        self.conf_lessthan = conf_lessthan
        self.fix_gauge = fix_gauge
        self.meas = []
        
    def addMeas(self,job):
        self.meas.append(job)
    def write(self,filename):
        fhandle = open(filename, "w")
        fhandle.write("class GparityContractArg contract_arg = {\n")
        fhandle.write("Array meas[%d] = {\n" % len(self.meas))
        for i in range(len(self.meas)):
            self.meas[i].write(fhandle)
        fhandle.write("}\n")

        fhandle.write("string config_fmt = \"%s\"\n" % self.config_fmt)
        fhandle.write("int conf_start = %s\n" % self.conf_start)
        fhandle.write("int conf_incr = %s\n" % self.conf_incr)
        fhandle.write("int conf_lessthan = %s\n" % self.conf_lessthan)
        self.fix_gauge.write(fhandle,"fix_gauge")
        
        fhandle.write("}\n")
        fhandle.close()
   




