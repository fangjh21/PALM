import simpy
from util import *
from hardware import Hardware
from macro import *
from tile import *
from dl_graph import *
from mapping import *
from pipeline import *
if __name__ == "__main__":
    env = simpy.Environment()
    hd_config=load_config('config/gpu.json')
    #hd_config=load_config('config/wafer.json')
    st_config={
        "recompute":recompute.full,
        "zero":zero.none,
        "pipeline":pipe.Dreampipe1F1B,#pipe.Dreampipe1F1B,#pipe.GPipe,#
        "optimizer":optimizer.adam,#optimizer.adam, 
        "mode":mode.FP16# FP16 fixed,FP32 full,INT8
    } 
    sim_config={        
        "analytical":False,
        "tile_aggregate":True,
        "pipe_boost":True,
        'debug':False
        }
    #model=Graph.Encoder(name='T-18B')
    model=Graph.Encoder(name='T-530B')
    PP,DP,TP=35,9,8
    hd_config["inter_s"]=[PP, DP]
    pipe_config={
        'mini_batch_size':model.batch_size,
        'pp_stage_num':PP,
        'micro_batch_size':int(model.batch_size/PP),#1
    }
    print(model.batch_size)
    hd = Hardware(env,hd_config,sim_config)
    tx8=Tile(env,hd_config,st_config,sim_config)
    stage_devices,stage_ops=auto_mapping_average_tile(model=model,hardware=hd,Parallism=[PP,DP,TP,1,1])
    pipe=Pipeline(env,stage_devices,stage_ops,hd,hd_config,st_config,sim_config,pipe_config)
    pipe.simpy_run(until_ms=ONE_WEEK_MS*10)
    results=pipe.sim_visualize(draw_pipe=True,clear=True,write_log=False)
    print(results)
