import simpy
import math
import time
from visualize import *
from util import *
from tile import *
from hardware import *
from macro import *
import numpy as np
class Stage():
    __stage_id=0
    def __init__(self,env,st_config,hd_config,sim_config,ops,L_devices,C_devices,N_devices)-> None:
        self.ops=ops
        self.i_size_mb=0#0.000001#TODO
        self.o_size_mb=0#0.000001
        self.N_devices=N_devices
        self.L_devices=L_devices
        self.C_devices=C_devices
        self.tile=[]
        for i in range(len(self.C_devices)):
            self.tile.append(Tile(env,hd_config,st_config,sim_config))
            if sim_config['tile_aggregate']:
                break

        self.stage_info=[]
        self.tile_strategy=None
        
        #simpy env 
        self.env=env
        self.qu=simpy.PriorityResource(env, capacity=1)
        self.trace=[]
        self.fd_cnt=0
        self.__class__.__stage_id+=1

    def tiles_init(self,micro_batch,cur_stage,stage_num,pipe_strategy,train):
        for op in self.ops:
            op.dims[0]=micro_batch
            op.analysis()
        pp_infs=stage_num-cur_stage if pipe_strategy==pipe.Dreampipe1F1B else  stage_num
        pp_infs=pp_infs if train else 1
        #print('pp_infs',pp_infs)  
        self.tile_strategy=self.tile[0].tile_dataflow(self.ops,self.C_devices,pp_infs=pp_infs)
    def up_state(self,hd,c_type=state.forward,wait=1e-15):
        while True:
            with self.qu.request() as req:
                    yield req
                    t_last=self.env.now
                    event_list=[self.env.process(tile.ops_events(self.ops,hd,self.tile_strategy,c_type)) for tile in self.tile]
                    yield  self.env.all_of(event_list)
                    self.trace.append((t_last,self.env.now,c_type))    
                    if c_type==state.forward:
                        self.fd_cnt+=1
                    elif c_type==state.backward:
                        self.fd_cnt-=1
                    else:
                        pass
            break
                    
class Pipeline():
    def __init__(self,env,stage_devices,stage_ops,hardware,hd_config,st_config,sim_config,pipe_config) -> None:

        #simpy env 
        self.env=env
        self.reg=[]
        self.cur_fd_times=0
        self.cur_bd_times=0
        self.one_epoch_finish=simpy.Store(self.env,capacity=1)
        self.one_fd_finish=simpy.Store(self.env,capacity=1)
        self.one_data_fetch=simpy.Store(self.env,capacity=1)

        #stage
        self.stage_num=len(stage_ops)
        self.hd=hardware
        self.mini_batch=pipe_config['mini_batch_size']
        self.micro_batch=pipe_config['micro_batch_size']
        self.micro_batch_num=math.ceil(self.mini_batch/self.micro_batch)
        self.stages=[]
        for i in range(self.stage_num):
            self.stages.append(Stage(self.env,st_config,hd_config,sim_config,stage_ops[i],\
                                     L_devices=None if i==0 else stage_devices[i-1] ,\
                                     C_devices=stage_devices[i],\
                                     N_devices=stage_devices[i+1] if i<self.stage_num-1 else None))

        self.pipe_strategy=st_config['pipeline']
        self.st_config=st_config
        #sim config
        self.boost_mode=sim_config['pipe_boost']
        self.train=(st_config['optimizer']!=optimizer.none) and (st_config['mode']!=mode.INT8)
        self.boost_times=6 #  if sim_config['topy_analytical']  else 6

        #dram analysis
        self.strategy={}
        self.__set_stage()
    def __set_stage(self):
        for cur_stage in range(self.stage_num):
            self.reg.append(simpy.PriorityStore(self.env, capacity=self.stage_num-cur_stage))
            self.stages[cur_stage].tiles_init(self.micro_batch,cur_stage,self.stage_num,self.pipe_strategy,self.train)
            self.strategy['stage_'+str(cur_stage)+'_dram_req']=self.stages[cur_stage].tile_strategy['dram_cap_req_gb_max']
        #print(self.strategy)
    def forward(self,times):
        with self.one_data_fetch.get() as get:
            a=yield get
            for i,stg in enumerate(self.stages):
                #if any other pipe can change code here
                if self.pipe_strategy==pipe.Dreampipe1F1B and self.train:
                    yield self.reg[i].put(1)
                elif self.pipe_strategy==pipe.GPipe or not self.train:
                    if i==self.stage_num-1:
                        self.cur_fd_times+=1
                    if self.cur_fd_times==times:
                        self.one_fd_finish.put(1)
                else:
                    raise NotImplementedError 
                yield self.env.process(stg.up_state(self.hd,c_type=state.forward,wait=1e-15))       
    def backward(self,times): 
        for i in range(self.stage_num-1,-1,-1):
            if self.pipe_strategy==pipe.Dreampipe1F1B:
                with self.reg[i].get() as get:
                    a=yield get
                    stg=self.stages[i]
                    yield self.env.process(stg.up_state(self.hd,c_type=state.backward,wait=1e-15))  
                    if i==0:
                        self.cur_bd_times+=1
                    if self.cur_bd_times==times:
                        self.one_epoch_finish.put(1)
            elif  self.pipe_strategy==pipe.GPipe:  
                stg=self.stages[i]
                yield self.env.process(stg.up_state(self.hd,c_type=state.backward,wait=1e-15))  
                if i==0:
                    self.cur_bd_times+=1
                if self.cur_bd_times==times:
                    self.one_epoch_finish.put(1)  
    def parameter_syn(self):
        while(True):
            with self.one_epoch_finish.get() as get:
                a=yield get
                for stg in self.stages:
                    self.env.process(stg.up_state(self.hd,c_type=state.param_sync,wait=1e-15))
                break
    def start(self):
        times=self.boost_times if self.boost_mode else self.micro_batch_num
        for i in range(times):
            #task_info='input_data_fetch_'+str(i)
            with self.one_data_fetch.put(1)as put:
                yield put
                yield self.env.process(self.hd.tile_gd_access(self.stages[0].i_size_mb,self.stages[0].C_devices))
                
    def register(self): 
        print('----------pipe_info----------')
        print('stage num={}, extute times={}'.format(len(self.stages),self.micro_batch_num))
        print('mini batch={}, micro batch={}'.format(self.mini_batch,self.micro_batch))
        self.boost_times=min(self.boost_times,self.micro_batch_num)
        times=self.boost_times if self.boost_mode else self.micro_batch_num
        #self.boost_times=1 
        def all_backward(times):
            while(True):
                with self.one_fd_finish.get() as get:
                    a=yield get    
                    for i in range(times):
                        self.env.process(self.backward(times)) 
                break
        self.env.process(self.start())
        for i in range(times):
            self.env.process(self.forward(times))
        if self.train:
            if self.pipe_strategy==pipe.GPipe:  
                self.env.process(all_backward(times))
            elif self.pipe_strategy==pipe.Dreampipe1F1B:  
                for i in range(times):
                    self.env.process(self.backward(times))
            self.env.process(self.parameter_syn())
        
    def simpy_run(self,until_ms=2000):
        self.register()
        print('----------simpy_run----------')
        sim_start_t=time.time()
        print('start simpy simulation...')
        self.env.run(until=until_ms)
        sim_end_t=time.time()
        print('finish simpy simulation with {:.3f}s\n'.format(sim_end_t-sim_start_t))
    def sim_visualize(self,path='./sim_visualize/pipeline/',draw_pipe=True,write_log=False,clear=True):
        results=''
        exe_mode='training' if self.train else  'inference'
        tm=time.strftime('_%m_%d_%H_%M_%S',time.localtime())
        name='pipeline'+str(tm)
        name_log=name+'.log'
        all_trace=[]
        utilization=[0] * len(self.stages)
        utilization_tile_cp=[0] * len(self.stages)
        tile_num_list=[0] * len(self.stages)
        pipe_endtime=0
        title=str(self.pipe_strategy) if self.train else 'Inference'
        for i,stage in enumerate(self.stages):
            #print(stage.trace)
            all_trace.append(stage.trace)
            if stage.trace[-1][1]>pipe_endtime:
                pipe_endtime=stage.trace[-1][1]
            for item in stage.trace:
                utilization[i]+=(item[1]-item[0])
            tile_num_list[i]=len(stage.C_devices)
        corr_coe=0
        if self.boost_mode :
            #add boosted time
                max_unit_time_1F_1B=max_ave_1F_1B_time(all_trace,self.train)#all_trace[-1][1][1]-all_trace[-1][0][0]#
                add_time=(self.micro_batch_num-self.boost_times)*max_unit_time_1F_1B 
                pipe_endtime=pipe_endtime+add_time
                corr_coe=(self.micro_batch_num-self.boost_times)/self.boost_times
                for i in range(len(utilization)):
                    utilization[i]+=(utilization[i]*corr_coe)
                    utilization_tile_cp[i]+=(sum(self.stages[i].tile[0].cp_trace)*(corr_coe+1))
                    utilization_tile_cp[i]/=pipe_endtime
                    utilization[i]/=pipe_endtime
        else:
                for i in range(len(utilization)):
                    utilization[i]/=pipe_endtime
                    utilization_tile_cp[i]+=(sum(self.stages[i].tile[0].cp_trace))
                    utilization_tile_cp[i]/=pipe_endtime
        #hd.resource_util(corr_coe,pipe_endtime)  
        mini_batch=self.micro_batch_num*self.micro_batch
        endtime_secs=pipe_endtime/1000
        endtime_days=endtime_secs/60/60/24
        if not os.path.exists(path):
            os.makedirs(path)
        elif clear:
            ls = os.listdir(path)
            for i in ls: 
                f_path = os.path.join(path, i)
                #print(f_path)    
                os.remove(f_path)
        if write_log:
            with open(path+name_log, 'w') as f:
                for i in range(len(all_trace)):
                    f.write(str(all_trace[i]))
                    f.write('\n')

        if draw_pipe:
            draw_pipeline(all_trace,path=path,title=title,throughout=mini_batch/endtime_secs,name=name)
        #print(self.strategy)
        draw_dram_req(self.strategy,path=path,name=title,info=self.st_config)
        results+='{} {} pipeline endtime {:.4f} days [{:.4f}s]\n'.format(title,exe_mode,endtime_days,endtime_secs)
        results+='{} {} pipeline throughout= {:.4f} sample/s\n'.format(title,exe_mode,mini_batch/endtime_secs)
        results+=draw_util(utilization,'utilization',path=path)
        results+=draw_util(utilization_tile_cp,'computational_utilization',path=path)
        #draw_util(hd.tile_dram_util,'hd.tile_dram_util')
        #draw_util(hd.hd_util,'hd.hd_util')
        #draw_util(hd.edge_dram_util,'hd.edge_dram_util')
        
        return results#endtime_days,utilization,utilization_tile_cp


