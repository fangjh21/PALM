import math
import simpy
import numpy as np
from typing import List,Union
from util import *
from hardware import Hardware
from macro import *
from op import Op
from contextlib import nullcontext

class Tile():
    def __init__(self,env,hd_config,st_config,sim_config) -> None:
        #hardware configs
        self.name=hd_config["t_name"]
        self.t_INT8=hd_config["t_INT8(TOPs)"]
        self.t_FP16=hd_config["t_FP16(TFLOPs)"]
        self.t_FP32=hd_config["t_FP32(TFLOPs)"]
        self.t_sram_cap=hd_config["t_sram_cap(MB)"]
        self.freq=hd_config["clk_freq(GHz)"]

        #software configs
        self.recompute=st_config["recompute"]
        self.zero=st_config["zero"]
        self.pipeline=st_config["pipeline"]
        #optimizer==none means inference
        self.opti=st_config["optimizer"]
        self.mode=st_config["mode"]
        #self.BYTES={'NONE':0,'INT8':1,'FP16':2,'TF32':2.375,'FP32':4,'FP64':5}
        #Error correction factor
        self.tile_factor={
            'gpu':0.5,
            'wafer':0.005,
            'default':0.1,
            }
        #env 
        self.env=env
        self.analytical=sim_config['analytical']
        self.cp_trace=[]
        if not self.analytical:
            self.cp_worker= simpy.Resource(env, capacity=1)
            self.cm_worker= simpy.Resource(env, capacity=1)

    def compute_cycles(self,param:List[int]):
        assert(len(param)>=3)
        cost=0
        coe=1 if len(param)==3 else sizeof(param[:-3],1)
        [SR,SC,T]=param[-3:]
        [R,C]=self.array_shape
        [PR,PC]=self.array_group
        if self.cp_model==comp_model.scale_sim:
            sr=math.ceil(SR/PR)
            sc=math.ceil(SC/PC)
            cost=(2*R+C+T-2)*math.ceil(sr/R)*math.ceil(sc/C)
        elif self.cp_model==comp_model.simple:
            cost=T*math.ceil(SR/(R*PR))*math.ceil(SC/(C*SC))
        elif self.cp_model==comp_model.abrupt_curve:
            cost=T*math.ceil(SR/(R*PR))*math.ceil(SC/(C*SC))
            if SR%(PR*R)==0 and SC%(PC*C)==0:
                cost=1.0*cost
            elif SR%(PR*R)==0 :
                cost=1.2*cost
            elif SR%R==0 and SC%C==0:
                cost=1.8*cost
            elif SR%R==0 :
                cost=2.0*cost
            else:
                cost=2.5*cost
            return int(cost)
        else:
            raise NotImplementedError
        return cost*coe
    def computation(self,macs_m,compute_power_t):
        exetime = 2 * macs_m *Ks/ compute_power_t
        #exetime = self.compute_cycles(macs_m) / self.freq 
        if not self.analytical:
            with self.cp_worker.request() as req:
                    yield req
                    t_last=self.env.now
                    yield self.env.timeout(exetime)
                    self.cp_trace.append(self.env.now-t_last)
                    #print(self.env.now-t_last)
        else:
            yield self.env.timeout(exetime)
            self.cp_trace.append(exetime)   
    def communication(self,comm_ops,comm_bytes,hd:Hardware,dram=True,overlap=False):
        #commops tuple(data_size,comm. group,type)
        #[(524288.0, [[0, 2], [1, 3]], AR)]),()]
        while True:
            #yield self.env.timeout(0.00001)
            
            if self.analytical:
                    events = []
                    for comm_op in comm_ops:
                        comm_mbytes = comm_op[0] * comm_bytes*MB
                        comm_groups = comm_op[1]
                        comm_type = comm_op[2]
                        for cg in comm_groups:
                            events.append(self.env.process(hd.collective_comm(cg, comm_mbytes, hd.all_in_one_node(cg), comm_type)))
                            break
                    yield simpy.AllOf(self.env, events) 
            else:
                with (self.cp_worker.request() if not overlap else self.cm_worker )  as req:
                    yield req
                    events = []
                    for comm_op in comm_ops:
                        comm_mbytes = comm_op[0] * comm_bytes*MB
                        comm_groups = comm_op[1]
                        comm_type = comm_op[2]
                        for cg in comm_groups:
                            events.append(self.env.process(hd.collective_comm(cg, comm_mbytes, hd.all_in_one_node(cg), comm_type)))
                    yield simpy.AllOf(self.env, events) 
            
            break

    def tile_dataflow(self,all_ops,tiles,pp_infs:int=1):
        tile_num=len(tiles)
        t_sgy={}
        t_sgy['dram_cap_req_gb_max']=0
        ops_size=0
        i_size,w_size,o_size,r_size=0,0,0,0
        t_sgy['ior_wsg_bytes']=[1,1,1,1,1,1]
        sram_weight_bytes=1
        if self.mode==mode.INT8 or self.opti==optimizer.none :
            t_sgy['ior_wsg_bytes']=[1,1,0,1,0,0]
            sram_weight_bytes=1
            t_sgy['compute_power_t']=self.t_INT8
        elif self.mode==mode.FP16 and self.opti==optimizer.SGD:
            t_sgy['ior_wsg_bytes']=[2,2,2,2,0,2]
            sram_weight_bytes=2
            t_sgy['compute_power_t']=self.t_FP16
        elif self.mode==mode.FP16 and self.opti==optimizer.adam:
            t_sgy['ior_wsg_bytes']=[2,2,2,2+4,4+4,2]#权重：2byte+备份4byte,优化器：4+4, 梯度：2
            sram_weight_bytes=2
            t_sgy['compute_power_t']=self.t_FP16
        elif self.mode==mode.FP32 and self.opti==optimizer.adam:
            t_sgy['ior_wsg_bytes']=[4,4,4,4,4+4,4]
            sram_weight_bytes=4
            t_sgy['compute_power_t']=self.t_FP32
        elif self.mode==mode.FP32 and self.opti==optimizer.SGD:
            t_sgy['ior_wsg_bytes']=[4,4,4,4,0,4]
            sram_weight_bytes=4
            t_sgy['compute_power_t']=self.t_FP32
        t_sgy['sram_weight_bytes']=sram_weight_bytes
        t_sgy['transformer_num']=0
        op_cnt=0
        for op in all_ops:
            if op.type==OP.Encoder:
                t_sgy['transformer_num']+=1
            if op_cnt==0:
                i_size+=(op.iwor_size[0])*t_sgy['ior_wsg_bytes'][0]#reduce
            w_size+=(op.iwor_size[1])*t_sgy['ior_wsg_bytes'][3]
            o_size+=(op.iwor_size[2])*t_sgy['ior_wsg_bytes'][1]
            r_size+=(op.iwor_size[3])*t_sgy['ior_wsg_bytes'][2]
            ops_size+=op.iwor_size[1]
            op_cnt+=1
        ior_recompute_coe=[1,1,1]
        if t_sgy['transformer_num']>0 :
            if self.recompute==recompute.full:
                ior_recompute_coe=[1/t_sgy['transformer_num'],1/t_sgy['transformer_num'],1/t_sgy['transformer_num']]
            elif self.recompute==recompute.one:
                ior_recompute_coe=[1,1/t_sgy['transformer_num'],1/t_sgy['transformer_num']]
            elif self.recompute==recompute.half:
                half=math.ceil(t_sgy['transformer_num']/2)/t_sgy['transformer_num']
                ior_recompute_coe=[half,half,half]
        t_sgy['dram_cap_req_gb_max']=tile_num*(i_size*ior_recompute_coe[0]*pp_infs+w_size+o_size*ior_recompute_coe[1]*pp_infs+r_size*ior_recompute_coe[2]*pp_infs)*GB
        sram_cap=tile_num*self.t_sram_cap/1024
        sram_weight_size=ops_size*GB*sram_weight_bytes*tile_num
        act_size=(i_size+o_size+r_size)*GB*tile_num
        t_sgy['dataflow']=None
        if sram_weight_size<sram_cap:
            t_sgy['dataflow']=dataflow.ActStream
        elif act_size<sram_cap:
            t_sgy['dataflow']=dataflow.WeightStream
        else:
            t_sgy['dataflow']=dataflow.Stationary
        '''
        elif act_size>sram_weight_size:
            t_sgy['dataflow']=dataflow.IS
        else:
            t_sgy['dataflow']=dataflow.WS
        '''
        t_sgy['dram_cap_req_gb_max']+=(i_size+r_size)*GB

        tile_info='sram_cap={:.3f} GB, sram_weight_size={:.3f} GB, act_size={:.3f} GB, '.format(sram_cap,sram_weight_size,act_size)
        tile_info+='dram_cap_req={:.3f} GB, '.format(t_sgy['dram_cap_req_gb_max'])
        tile_info+='ops_size_all_tile ={:.3f} B, '.format(ops_size*Gs*tile_num)
        tile_info+='dataflow={}, '.format(t_sgy['dataflow'])
        t_sgy['dataflow']=t_sgy['dataflow']
        print(t_sgy)
        print(tile_info)
        return t_sgy
    

    def op_events(self,op:Op,hd:Hardware,t_sgy,state=state.forward):
        compute_power_t=t_sgy['compute_power_t']
        sbytes=t_sgy['sram_weight_bytes']
        dbytes=t_sgy['ior_wsg_bytes']
        df=t_sgy['dataflow']
        factor=self.tile_factor[hd.name]
        events=[]
        transformer_num=t_sgy['transformer_num']
        transformer_cnt=0
        iwor=op.iwor_size
        #dbytes[0]+dbytes[1]+dbytes[2]+dbytes[3]+dbytes[4]++dbytes[5]
        while True:
            if state==state.forward:
                macs_m=op.fd_macs*Ms
                write,read=0,0
                if df==dataflow.ActStream:
                    write=(iwor[2]*dbytes[1]+iwor[3]*dbytes[2])*MB
                    read=(iwor[2]*dbytes[0]+iwor[3]*dbytes[2])*MB
                elif df==dataflow.WeightStream:
                    write,read=0,(iwor[1])*dbytes[3]*MB
                else:
                    if op.type!=OP.Encoder:
                        write=iwor[2]*dbytes[1]*MB
                        read=min(math.ceil(iwor[0]*dbytes[0]*MB/self.t_sram_cap)*iwor[1]*dbytes[3]*MB,\
                                 math.ceil(iwor[1]*dbytes[3]*MB/self.t_sram_cap)*iwor[0]*dbytes[0]*MB)
                    else:
                        coe=factor*math.ceil(iwor[0]*dbytes[0]*MB/self.t_sram_cap)
                        write=coe*(iwor[2]*dbytes[1]+iwor[3]*dbytes[2])*MB
                        read=coe*(iwor[0]*dbytes[0]+iwor[1]*dbytes[3]+iwor[3]*dbytes[2])*MB
                events.append(self.env.process(self.computation(macs_m,compute_power_t)))
                events.append(self.env.process(hd.tile_gd_access(write,read,op.devices,write=1,read=1)))
                events=[simpy.AllOf(self.env, events)]
                if op.d4d_comm['f']!=[]:
                    events.append(self.env.process(self.communication(op.d4d_comm['f'],dbytes[0],hd)))
                    
            elif state==state.backward:
                times_cp=2
                if op.type==OP.Encoder:
                    transformer_cnt+=1
                    if (transformer_cnt % 2 ==1 and self.recompute==recompute.half) or self.recompute==recompute.full:
                        times_cp=3
                macs_m=times_cp*op.fd_macs*Ms
                if df==dataflow.ActStream:
                    write=(iwor[2]*dbytes[0]+iwor[3]*dbytes[2])*MB
                    read=(iwor[2]*dbytes[0]+iwor[3]*dbytes[2])*MB+(iwor[2]*dbytes[1]+iwor[3]*dbytes[2])*MB
                elif df==dataflow.WeightStream:
                    write,read=0,(iwor[1])*dbytes[3]*MB
                else:
                    if op.type!=OP.Encoder:
                        write=iwor[0]*dbytes[0]*MB
                        read=min(math.ceil(iwor[2]*dbytes[1]*MB/self.t_sram_cap)*iwor[1]*dbytes[3]*MB,\
                                 math.ceil(iwor[1]*dbytes[3]*MB/self.t_sram_cap)*iwor[2]*dbytes[1]*MB)
                        read+=min(math.ceil(iwor[0]*dbytes[0]*MB/self.t_sram_cap)*iwor[3]*dbytes[2]*MB,\
                                 math.ceil(iwor[3]*dbytes[2]*MB/self.t_sram_cap)*iwor[0]*dbytes[0]*MB)
                    else:
                        coe=factor**math.ceil(iwor[0]*dbytes[0]*MB/self.t_sram_cap)
                        write=coe*(iwor[3]*dbytes[2]+iwor[0]*dbytes[0])*MB
                        read=coe*(iwor[0]*dbytes[0]+iwor[1]*dbytes[3]+2*iwor[3]*dbytes[0])*MB
                events.append(self.env.process(self.computation(macs_m,compute_power_t)))
                events.append(self.env.process(hd.tile_gd_access(write,read,op.devices,write=1,read=1)))
                events=[simpy.AllOf(self.env, events)]
                if op.d4d_comm['b']!=[]:
                    events.append(self.env.process(self.communication(op.d4d_comm['b'],sbytes,hd)))
                        
            elif state==state.param_sync:
                events.append(self.env.process(self.communication(op.d4d_comm['u'],sbytes,hd)))
                events=[simpy.AllOf(self.env, events)]
                write,read=(iwor[1])*(dbytes[3])*MB,(iwor[1])*(dbytes[3]+dbytes[4]+dbytes[4])*MB
                events.append(self.env.process(hd.tile_gd_access(write,read,op.devices,write=1,read=1))) 
            else:#recompute
                pass
            yield simpy.AllOf(self.env, events)
            break

    def ops_events(self,ops,hd:Hardware,t_sgy,state=state.forward):
        while(True):
            for op in ops:
                yield  self.env.process(self.op_events(op,hd,t_sgy,state))
            break

if __name__ == "__main__":
    pass
