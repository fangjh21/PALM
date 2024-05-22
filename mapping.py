
from util import *
from visualize import *
#import math
def auto_mapping_average_tile(model,hardware,pp_num=6):
    assert pp_num<=len(model)
    total_cp=0
    for op in model:
        total_cp+=op.fd_macs
    total_cp=total_cp*Gs
    cp_per_stage=total_cp/pp_num
    #pp_tiles_num = [hardware.tile_num//pp_num for _ in range(pp_num)]
    stage_ops = [[] for _ in range(pp_num)]
    #tile_num_per_stage=[0]*pp_num
    cur_cp_status=0
    cur_id=0
    #mapping
    for op in model:
        cur_cp_status+=(op.fd_macs*Gs)
        if cur_cp_status<=cp_per_stage or cur_id==pp_num-1:
            stage_ops[cur_id].append(op)
        else:
            stage_ops[cur_id+1].append(op)
            cur_cp_status=(op.fd_macs*Gs)
            cur_id+=1 
    pp_tiles_num=[]
    for i in range(cur_id+1):
        cur_cp_status=0
        for op in stage_ops[i]:
            cur_cp_status+=(op.fd_macs*Gs)
        #print(cur_cp_status)
        pp_tiles_num.append(int(cur_cp_status/total_cp*hardware.tile_num))
    stage_devices=hardware.tile_split_by_pp(pp_tiles_num)
    #parallelism
    for i in range(cur_id+1):
        for op in stage_ops[i]:
            #op.dpmap(devices=stage_devices[cur_id],p_sgy=[len(stage_devices[cur_id]),1,1,1])
            #print(stage_devices[i],)
            op.dpmap(devices=stage_devices[i],p_sgy=[1,len(stage_devices[i]),1,1])
    draw_mapping(hardware,'test',tiles=stage_devices,path='sim_visualize',ori=False)
    return stage_devices,stage_ops
        
    