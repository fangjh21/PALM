import simpy
from resource import MonitoredResource as Resource
from typing import List
import random
from util import *
from functools import wraps
from macro import *
import numpy as np
Topology =['mesh', 'torus', 'gpu_like']
class Packet:
    def __init__(self, id, shape: List[int], bits=16,meta_data="test") -> None:
        self.id = id
        self.shape = shape
        self.size = sizeof(shape,coe=bits)
        self.meta_data = meta_data
    def __str__(self):
        return "Packet:(id:{},shape:{},size:{} MByte,meta:{})".format(self.id, self.shape, self.size, self.meta_data)
    @staticmethod
    def random_gen():
        id = random.randint(0, 10000)
        shape = []
        shape_dim = random.randint(1, 2)
        for i in range(shape_dim):
            shape.append(random.randint(1, 128))
        return Packet(id=id, shape=shape)

class Dram_model:
    def __init__(self,env,name='dram',bw=256,cap=2,rlt=0,wlt=0,cap_analytical=False) -> None:
        self.name = name
        self.bw_GB = bw
        self.read_latency = rlt
        self.write_latency = wlt
        self.capacity = cap
        self.env = env
        self.link=Resource(self.env,capacity=1)
        if cap_analytical:
            self.container = simpy.Container(self.env,capacity=cap*1024,init=0)
        else:  
            self.container=None
    '''
    def access(self, data_size_mb,write=True,debug=True,analytical=True):
        info=''
        latency = data_size_mb / self.bw_GB
        latency += self.write_latency if write else self.read_latency
        while True:
            if analytical:
                yield self.env.timeout(latency)
            else:
                with self.link.request() as req: 
                    yield req
                    if self.container!=None:
                        if write:
                            yield self.container.put(data_size_mb)
                        else:
                            yield self.container.get(data_size_mb)
                        info+='dram rest capacity={}\n'.format(self.container.capacity-self.container.level)
                    yield self.env.timeout(latency)
            info+='Event {} finished at {} ms\n'.format("write" if write else "read",env.now) 
            if debug:
                print(info)
            break
    '''
class Hardware:
    def __init__(self,env,hd_config, sim_config) -> None:
        self.name = hd_config['name']
        #link
        self.X0Y0 = hd_config['intra_s']
        self.X1Y1 = hd_config['inter_s']
        self.intra_bw =  hd_config['intra_bw(GB/s)']
        self.inter_bw = hd_config['inter_bw(GB/s)']
        self.intra_link_l=hd_config['intra_link_lty(us)']
        self.inter_link_l=hd_config['inter_link_lty(us)']
        self.route_XY = "X"
        self.topo_tpye=hd_config['topology_tpye']
        self.tile_num=0
        assert(self.topo_tpye in Topology)
        self.topo_adj=[]

        
        #dram
        self.d_per_tile = bool(hd_config['d_per_tile'])#TODO
        self.t_d_bw = hd_config['t_d_bw(GB/s)']
        self.t_d_cap = hd_config['t_d_cap(GB)']
        self.e_d_cap = hd_config['e_d_cap(GB)']
        self.e_d_bw=hd_config['e_d_bw(GB/s)']
        self.d_l=hd_config['d_lty(us)']
        self.clk_freq= hd_config['clk_freq(GHz)']

        self.device_dist = {}
        # simpy env and resource define 
        self.env = env
        self.analytical = sim_config['analytical']
        self.debug=sim_config['debug']
        self.link=[]
        self.dram=[]
        self.link_map=[]
        self.dram_map=[]

        '''
        self.noc_util=[]
        self.edge_dram_util=[]
        self.tile_dram_util=[]
        '''

        self.__topolopy()
    def wafer_info(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            print("----------hardware info----------")
            print("2D {} {}:{}x{},{}x{}".format(self.topo_tpye,self.name,self.X1Y1[0],self.X1Y1[1],self.X0Y0[0],self.X0Y0[1],))
            return func(self, *args, **kwargs)

        return wrapper
    def link_gen(self,src,des):#4维度坐标(x1,y1,x0,y0)
        #print(src,des)
        links=[]
        inter_link_num=0
        assert(self.topo_tpye in ['torus','mesh'])
        X1,Y1,X0,Y0,=self.X1Y1[0],self.X1Y1[1],self.X0Y0[0],self.X0Y0[1]
        #print(src,des)
        #assert(src!=des and (src[0]< X1 and src[2] <Y1 and src[1]< X0 and src[3]< Y0 ) and (des[0]< X1 and des[2] <Y1 and des[1]< X0 and des[3]< Y0 ))
        src_x,src_y=src[0]*X0+src[2],src[1]*Y0+src[3]
        des_x,des_y=des[0]*X0+des[2],des[1]*Y0+des[3]
        #print(src_x,src_y,des_x,des_y)
        if self.topo_tpye=='mesh':
            step_x=1 if src_x<=des_x else -1
            step_y=1 if src_y<=des_y else -1
            cur_y,cur_x=src_y,src_x
            for step_x0 in range(src_x+step_x,des_x+step_x,step_x):
                next_x=step_x0
                #print(cur_x,next_x)
                links.append(int(self.link_map[cur_y,cur_y,cur_x,next_x]))
                if self.topo_adj[cur_y,cur_y,cur_x,next_x]==self.inter_bw:
                    inter_link_num+=1
                cur_x=next_x
            for step_y0 in range(src_y+step_y,des_y+step_y,step_y):
                next_y=step_y0
                #print(cur_y,next_y)
                links.append(int(self.link_map[cur_y,next_y,des_x,des_x]))
                if self.topo_adj[cur_y,cur_y,cur_x,cur_x]==self.inter_bw:
                    inter_link_num+=1
                cur_y=next_y
        else:
            NotImplementedError 
        return links,inter_link_num
    def nearest_dram_of(self,device):
        tile_x=self.X0Y0[0]-1 if (device[2]+1)>self.X0Y0[0]//2 else 0
        tile_y=1 if device[3]==0 else (self.X0Y0[1]-2 if device[3]==self.X0Y0[1]-1 else device[3])
        nearest_device_id=[device[0],device[1],tile_x,tile_y]
        dram_id=2*(self.X1Y1[1]*(self.X0Y0[1]-2)*device[0]+(self.X0Y0[1]-2)*device[1]+tile_y-1)+(0 if tile_x==0 else 1)
        #print(device,nearest_device_id)
        return nearest_device_id,dram_id
    
    @wafer_info
    def __topolopy(self):
        assert self.topo_tpye in Topology
        X1,Y1,X0,Y0,=self.X1Y1[0],self.X1Y1[1],self.X0Y0[0],self.X0Y0[1]
        X=X1*X0
        Y=Y0*Y1
        self.tile_num=X*Y
        self.topo_adj=np.zeros((Y,Y,X,X))
        self.link_map=-np.ones((Y,Y,X,X))
        link_num=0
        if self.topo_tpye in ['torus','mesh']:
            assert Y0>2
            for y in range(Y1):
                for x in range(X1):
                    for y in range(Y0-2):
                        self.dram.append(Dram_model(env=self.env,bw=self.t_d_bw,cap=self.t_d_cap,rlt=self.d_l,wlt=self.d_l))
                        self.dram.append(Dram_model(env=self.env,bw=self.t_d_bw,cap=self.t_d_cap,rlt=self.d_l,wlt=self.d_l))
            for yj in range(Y):
                for yi in range(Y):
                    for xj in range(X):
                        for xi in range(X):
                            if yj==yi:
                                if abs(xi-xj)==1:
                                    if (min(xi,xj)+1)%X0==0:
                                        self.topo_adj[yj,yi,xj,xi]=self.inter_bw
                                    else:
                                        self.topo_adj[yj,yi,xj,xi]=self.intra_bw
                                    self.link_map[yj,yi,xj,xi]=link_num
                                    link_num+=1
                                    if not self.analytical:
                                        self.link.append(Resource(self.env, capacity=1))
                                elif (abs(xi-xj)+1)% X==0 and self.topo_tpye=='torus':
                                    if X1==1:
                                        self.topo_adj[yj,yi,xj,xi]=self.intra_bw
                                    else:
                                        self.topo_adj[yj,yi,xj,xi]=self.inter_bw   
                                    self.link_map[yj,yi,xj,xi]=link_num
                                    link_num+=1
                                    if not self.analytical:
                                        self.link.append(Resource(self.env, capacity=1))
                            elif abs(yj-yi)==1:
                                if xi==xj:
                                    if (min(yi,yj)+1)%Y0==0:
                                        self.topo_adj[yj,yi,xj,xi]=self.inter_bw
                                    else:
                                        self.topo_adj[yj,yi,xj,xi]=self.intra_bw
                                    self.link_map[yj,yi,xj,xi]=link_num
                                    link_num+=1
                                    if not self.analytical:
                                        self.link.append(Resource(self.env, capacity=1))
                                elif (abs(yi-yj)+1)% Y==0 and self.topo_tpye=='torus':
                                    if Y1==1:
                                        self.topo_adj[yj,yi,xj,xi]=self.intra_bw
                                    else:
                                        self.topo_adj[yj,yi,xj,xi]=self.inter_bw  
                                    self.link_map[yj,yi,xj,xi]=link_num 
                                    link_num+=1
                                    if not self.analytical:
                                        self.link.append(Resource(self.env, capacity=1))
        elif self.topo_tpye=="gpu_like":
            #https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf
            alpha=0.87
            beta=1.0
            gamma=0.5
            node_size=sizeof(self.X0Y0)
            node_num=sizeof(self.X1Y1)
            self.inter_ring_bw=gamma*beta*self.inter_bw
            self.intra_ring_bw=gamma*alpha*self.intra_bw
        else:
            raise NotImplementedError
    def id_transfer(self,pos):#4维度坐标(x1,y1,x0,y0) or 1维坐标 or 2维度坐标
        if (type(pos)==list):
            if len(pos)==4:
                return self.X1Y1[0]*self.X0Y0[0]*self.X0Y0[1]*pos[1]+self.X0Y0[0]*self.X0Y0[1]*pos[0]+self.X0Y0[0]*pos[3]+pos[2]
            elif len(pos)==2:#(x,y)
                return self.X0Y0[0]*self.X1Y1[0]*pos[1]+pos[0]
            else:
                raise ImportWarning
        else:
            y1=pos // (self.X1Y1[0]*self.X0Y0[0]*self.X0Y0[1])
            tp=pos -y1*self.X1Y1[0]*self.X0Y0[0]*self.X0Y0[1]
            x1=tp// (self.X0Y0[0]*self.X0Y0[1])
            tp=tp-x1* (self.X0Y0[0]*self.X0Y0[1])
            y0=tp // self.X0Y0[0]
            x0= tp-y0*self.X0Y0[0]
            return [x1,y1,x0,y0]
    def all_in_one_node(self,devices):
        node_ids=[]
        for device in devices:
            node_id=device[0]+device[1]*self.X1Y1[1]
            if node_id not in node_ids:
                node_ids.append(node_id)
        #print(node_ids)
        return len(node_ids)==1
    def send_recv(self,src,des,data_size_mb,task_id='send_recv'):
        list_id ,inter_link_num= self.link_gen(src,des)
        link_bw=self.intra_bw if inter_link_num==0 else self.inter_bw
        time_ms = data_size_mb / link_bw+(len(list_id)-inter_link_num)*self.intra_link_l/1000+inter_link_num*self.inter_link_l/1000
        info=''
        while True:
            t_ori=self.env.now
            if  self.analytical:
                info+='Event {} started at {:.3f} ms\n'.format(task_id,t_ori) 
                yield self.env.timeout(time_ms)
                info+='Event {} finished at {:.3f} ms\n'.format(task_id,env.now)
            else:
                requests = [self.link[i].request() for i in list_id]
                yield simpy.AllOf(self.env, requests)
                info+='Event {} started at {:.3f} ms\n'.format(task_id,t_ori) 
                # 等待所有请求完成
                #yield env.all_of(requests)
                yield self.env.timeout(time_ms)
                # 处理请求
                for req in requests:
                    req.resource.release(req)  # 释放资源
                info+='Event {} finished at {:.3f} ms\n'.format(task_id,self.env.now) 
            break
        if self.debug:
            print(info)
    def tile_reduce(self,devices,data_size_mb):
        pass
    def collective_comm(self,devices,data_size_mb,all_in_one_node_flag=None,comm_type=COMM.AR):
        debug=self.debug
        P=len(devices)
        coe=2 if comm_type==COMM.AR else 1
        coe=P-1 if comm_type==COMM.AA else coe
        data_s=coe*(P-1)/P*data_size_mb
        info=''
        if all_in_one_node_flag==None:
            all_in_one_node_flag=self.all_in_one_node(devices)
            info+='all_in_one_node:{}\n'.format(all_in_one_node_flag)
        while True:
            #t_ori=self.env.now
            if self.topo_tpye=='gpu_like' or self.analytical:
                if all_in_one_node_flag:
                    bw=self.intra_ring_bw if self.topo_tpye=='gpu_like' else self.intra_bw
                else:
                    bw=self.inter_ring_bw if self.topo_tpye=='gpu_like' else self.inter_bw
                info+='all-reduce bindwidth(GB/s):{}\n'.format(bw)
                yield self.env.timeout(data_s/bw)
                info+='Event {} finished at {:.3f} ms\n'.format(comm_type,self.env.now)
            else:
                chunk_size = data_size_mb / P if comm_type!=COMM.AA else (P-1)* data_size_mb / P 
                if comm_type in [COMM.AR ,COMM.RS]:
                    for i in range(P-1):
                        event_list=[]
                        for id_idx in range(P-1):
                            event_list.append(self.env.process(self.send_recv(devices[id_idx],devices[id_idx+1],chunk_size,)))
                        event_list.append(self.env.process(self.send_recv(devices[-1],devices[0],chunk_size )))
                        yield simpy.AllOf(self.env, event_list)
                if comm_type in [COMM.AR ,COMM.AG,COMM.AA]:
                    for i in range(P - 1):
                        event_list = []
                        for id_idx in range(P - 1):
                            event_list.append(self.env.process(self.send_recv(devices[id_idx], devices[id_idx + 1],chunk_size, )))
                        event_list.append(self.env.process(self.send_recv(devices[-1], devices[0],chunk_size, )))
                        yield simpy.AllOf(self.env, event_list)
                info+='Event {} finished at {:.3f} ms\n'.format(comm_type,self.env.now)
            break 
        if debug:
            print(info)

    def tile_d_access(self,data_size_mb,device,task_id='dram',write=1,read=0):
        info=''
        times=write+read
        ''''
        dram_id=self.id_transfer(device)
        t_ori=self.env.now
        time_ms=times*(data_size_mb / self.e_d_bw+self.d_l)
        info+='Event {} started at {:.3f} ms\n'.format(task_id,t_ori) 
        yield self.env.timeout(time_ms)
        info+='Event {} finished at {:.3f} ms\n'.format("write" if write>read else "read",self.env.now) 
        '''
        if self.topo_tpye in ['torus','mesh']:
            src=device
            des,dram_id=self.nearest_dram_of(device)
            #print(device,des,dram_id,task_id)
            if write>0:
                list_id ,inter_link_num= self.link_gen(src,des)
            else:
                list_id ,inter_link_num= self.link_gen(des,src)
            #print(list_id ,inter_link_num)
            link_bw=min(self.intra_bw if inter_link_num==0 else self.inter_bw,self.e_d_bw) 
            #print(link_bw,self.inter_bw,self.t_d_bw,self.e_d_bw)
            time_ms = times*(data_size_mb / link_bw+(len(list_id)-inter_link_num)*self.intra_link_l/1000+inter_link_num*self.inter_link_l/1000+self.d_l)
            #print(time_ms,data_size_mb)
            while True:
                t_ori=self.env.now
                if self.analytical:
                    yield self.env.timeout(time_ms)
                else:
                    requests = [self.link[i].request() for i in list_id]
                    #print(len(self.dram))
                    requests+=[self.dram[dram_id].link.request()]
                    yield simpy.AllOf(self.env, requests)
                    info+='Event {} started at {:.3f} ms\n'.format(task_id,t_ori) 
                    if self.dram[dram_id].container!=None:
                        if write:
                            yield self.dram[dram_id].container.put(data_size_mb)
                        else:
                            yield self.dram[dram_id].container.get(data_size_mb)
                        info+='dram rest capacity={} GB\n'.format((self.dram[dram_id].container.capacity-self.dram[dram_id].container.level)/1024)
                    yield self.env.timeout(time_ms)
                    for req in requests:
                        req.resource.release(req)  # 释放资源
                info+='Event {} finished at {:.3f} ms\n'.format("write" if write>read else "read",self.env.now) 
                break  
         
        elif self.topo_tpye== "gpu_like":
            dram_id=self.id_transfer(device)
            t_ori=self.env.now
            time_ms=times*(data_size_mb / self.t_d_bw+self.d_l)
            info+='Event {} started at {:.3f} ms\n'.format(task_id,t_ori) 
            yield self.env.timeout(time_ms)
            info+='Event {} finished at {:.3f} ms\n'.format("write" if write>read else "read",self.env.now) 
          
        if self.debug:
            print(info)
    def tile_gd_access(self,data_size_mb_of_each,devices,task_id='dram_group',write=1,read=0):
        while True:
            #now=self.env.now
            #print(devices)
            events = [self.env.process(self.tile_d_access(data_size_mb=data_size_mb_of_each,device=device,\
                                                     write=write,read=read,task_id=task_id)) for device in devices]
            yield simpy.AllOf(self.env, events)
            #print(self.env.now-now)
            break
    def stage_data_tranfer(self,src_g,des_g,data_size_mb_of_each,store_flag=True):
        pass

    def tile_split_by_pp(self,pp_tiles_num=[1,2,3,4]):
        X1,Y1,X0,Y0,=self.X1Y1[0],self.X1Y1[1],self.X0Y0[0],self.X0Y0[1]
        #tile_num=X1*X0*Y0*Y1
        tile_num_req=sum(pp_tiles_num)
        tiles=[]
        pp_tiles=[]
        #优先划分完整的Die  
        prior='X' #prior='Y'
        if tile_num_req>self.tile_num:
            print("pp needs {} tiles, but hardware only has {} tiles".format(tile_num_req,self.tile_num))
        tiles = []
        y_in_order=True
        x_in_order=True
        y_edge=False
        x_edge=False
        for x1 in range(X1):
            y_range = range(Y1) if y_in_order else range(Y1 - 1, -1, -1)
            for y1 in y_range:
                y0_range = range(Y0) if y_in_order else range(Y0 - 1, -1, -1)
                for y0 in y0_range:
                    y_edge=(y0==Y0-1 and y1==Y1-1)
                    x_range = range(X0) if (x_in_order or y_edge) else range(X0 - 1, -1, -1)
                    for x0 in x_range:
                            tiles.append([x1, y1, x0, y0])
                    x_in_order = not x_in_order
            y_in_order = not y_in_order
        oft=0
        for idx,i in enumerate(pp_tiles_num):
            if idx==len(pp_tiles_num)-1:
                pp_tiles.append(tiles[oft:])
            else:
                pp_tiles.append(tiles[oft:oft+i])
            #print(tiles[oft:oft+i])
            oft=i+oft

        return pp_tiles#,tiles

    def edge_d_access(self):
        pass
    def edge_gd_access(self):
        pass

if __name__ == "__main__":
    env = simpy.Environment()
    hd_config=load_config('config/wafer.json')
    sim_config={        
        "analytical":False,
        "tile_aggregate":True,
        "pipe_boost":True,
        'debug':False
        }
    #hd_config=load_config('config/wafer.json')
    #dr=Dram_model(env)
    wd = Hardware(env,hd_config,sim_config)
    pp_tiles=wd.tile_split_by_pp(pp_tiles_num=[16]*20)
    #print(wd.topo_adj[...,0,0,0])
    #wd.link_gen(src=[0,0,0,0],des=[4,3,3,3])
    #env.process(wd.send_recv(src=[0,0,0,0],des=[4,3,3,3],data_size_mb=1024,))
    #env.process(wd.send_recv(src=[0,0,0,0],des=[4,3,3,3],data_size_mb=1024,))
    #env.process(wd.collective_comm(devices=[[0,0,0,0],[0,0,0,1],[0,0,0,2],[0,0,0,3]],data_size_mb=1024,comm_type=COMM.AR))
    #env.process(wd.all_reduce(devices=[[0,0,0,0],[4,3,3,3],[0,0,3,3]],data_size_mb=1024,))
    #env.process(wd.tile_d_access(data_size_mb=1024,device=[4,3,3,3]))
    #env.process(wd.tile_gd_access(data_size_mb_of_each=1024,devices=[[4,3,3,3],[4,3,3,2]]))
    #env.process(dr.access(data_size_mb=1024,))
    #env.process(dr.access(data_size_mb=1024,))
    print(len(pp_tiles[0]))  
    env.process(wd.tile_gd_access(1,pp_tiles[0],task_id='dram_group',write=1,read=0))
    env.process(wd.tile_gd_access(1,pp_tiles[1],task_id='dram_group1',write=1,read=0))
    env.run(until=10000)
    '''
    tp=wd.id_transfer([2,1,1,3])
    print(tp)
    print(wd.id_transfer(tp))
    print(wd.id_transfer([0,15]))
    print(wd.tile_split_by_pp())
    '''


