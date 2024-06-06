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
    #hd_config=load_config('config/gpu.json')
    #hd_config=load_config('config/wafer.json')
    hd_config=load_config('config/tile.json')
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
    matrix=Graph.matrix()
    print(len(matrix))
    pipe_config={
        'mini_batch_size':matrix.batch_size,
        'pp_stage_num':2,
        'micro_batch_size':int(matrix.batch_size/2),#1
    }
    print(matrix.batch_size)
    hd = Hardware(env,hd_config,sim_config)
    tx8=Tile(env,hd_config,st_config,sim_config)
    stage_devices,stage_ops=auto_mapping_average_tile(model=matrix,hardware=hd,Parallism=[pipe_config['pp_stage_num'],1,1,1,1],auto_split=True)
    pipe=Pipeline(env,stage_devices,stage_ops,hd,hd_config,st_config,sim_config,pipe_config)
    pipe.simpy_run(until_ms=ONE_WEEK_MS*10)
    results=pipe.sim_visualize(draw_pipe=True,clear=True,write_log=False)
    print(results)
