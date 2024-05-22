import matplotlib.pyplot as plt
import os
from macro import *
from typing import List
from queue import Queue
import numpy as np
MY_COLOR=['#63b2ee','#76da91','#f8cb7f','#f89588','#7cd6cf','#9192ab','#7898e1','#efa666','#eddd86','#9987ce','#63b2ee','#76da91',]#maktalong
MY_COLOR1 = ['#DE6E66', '#5096DE', '#CBDE3A','#DE66C2', '#5096DE', '#DEA13A', '#61DE45']
def draw_dram_req(dram_dist,name,info,path='status',):
    name=name+'_dram_max_req'
    keys = list(dram_dist.keys())
    keys+=['total req']
    values = list(dram_dist.values())
    average_v=sum(values)/len(values)
    values+=[average_v]
    index = np.arange(len(keys))
    #colors = []
    # 创建双坐标轴
    fig, ax1 = plt.subplots(figsize=(10,8))
    # 在第一个坐标轴上绘制条形图
    bar_width=0.5
    ax1.bar(index,values , color=MY_COLOR1[0], width=bar_width, edgecolor='black')
    ax1.text(index[-1], values[-1], "{:.1f} TB".format(values[-1]*(len(values)-1)/1000), ha='center', va='bottom', fontsize=12, rotation=80)
    #ax1.set_ylim(-1, 0)
    ax1.set_xticks(index)  # 设置x轴的刻度为x列表中的值
    ax1.set_xticklabels(keys,rotation=75)  # 设置x轴的刻度标签为x列表中的值
    ax1.set_ylabel('DRAM Capacity GB')
    plt.title('{}'.format(str(info))[1:-1])
    # 调整图表布局
    fig.tight_layout()
    #ax1.legend(loc='upper left')  # 位置: 左上角
    plt.savefig(os.path.join(path,name+'.pdf'))
def draw_pipeline(trace_list,path,title,throughout,name='pipeline'):
    #print(trace_list)
    #[[(s,e),(s,e),(s,e)],[],[]], []=stages,s=micro_start_time,e=micro_end_time
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #COLOR = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#10c020', '#D20F99','#FFFFFF','#000000']
    #COLOR=['#63b2ee','#76da91','#f8cb7f','#f89588','#7cd6cf','#9192ab','#7898e1','#efa666','#eddd86','#9987ce','#63b2ee','#76da91','#100000']#maktalong
    num=len(trace_list)
    leng=len(trace_list[0])
    width_scale=0
    for trace in trace_list:
        if trace[-1][1]>width_scale:
            width_scale=trace[-1][1]
    #width_scale=max(trace_list[0][-1][1],trace_list[-1][-1][1])
    height_scale=4
    single_height=1
    start_height=single_height/2
    for j in range(num):
        k=0
        m=0
        leng=len(trace_list[j])
        for i in range(leng):
            x=trace_list[j][i]
            #facecolor=color[0] if x[2]==0 else color[5]
            if x[2]==state.forward:
                facecolor=MY_COLOR[k % len(MY_COLOR)]
                k+=1
            elif x[2]==state.backward:
                facecolor=MY_COLOR[m % len(MY_COLOR)]
                m+=1
            else:
                facecolor=MY_COLOR[0]
            edgecolor=MY_COLOR[-1]
            rect = plt.Rectangle((x[0],start_height+single_height*j),(x[1]-x[0]),single_height,fill=True,edgecolor='black',facecolor=facecolor,linewidth=0.5)
            ax.add_patch(rect)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title('{} stages ML {} pipeline [{:.3f} sample/s]'.format(num,title,throughout))
    ax.set_ylim(0, num+1)
    ax.set_yticks([num-i for i in range(num)])
    ax.set_xlim(0, width_scale)
    #ax.set_aspect(20)
    plt.xlabel("Time(ms)")
    plt.ylabel("Stage")
    plt.savefig(os.path.join(path,name+'.pdf'))
    plt.close()

def max_ave_1F_1B_time(trace,train):
    max_1F1B_time=0
    tp_time=0 
    stage_num=len(trace)
    fb_num=len(trace[0])
    #print(stage_num,fb_num)
    for i in range(stage_num):
        tp_time=0
        fb_num=len(trace[i])-1 if train else len(trace[i])
        for j in range(fb_num):
            tp_time+=(trace[i][j][1]-trace[i][j][0])
        if train:
            tp_time/=(fb_num /2)
        else:
            tp_time/=fb_num
        #print(i,tp_time)
        if max_1F1B_time< tp_time:
            max_1F1B_time=tp_time
            #print('max_index=',i)
    return max_1F1B_time

def visualize_resource(data:List,path,name,clear_redundance=True,max_resource=256,ave_unit_ms=1):
    #[(req_flag,req_time,len_q),(res_flag,res_time,len_q)]
    #print(data)
    if data==[]:
        return None
    q_req=Queue()
    occupy_list=[]
    for item in data:
        if item[0]=='req':
            q_req.put(item)
        elif item[0]=='res':
            req_item=q_req.get()
            res_item=item
            occupy_list.append([req_item[1],res_item[1],max_resource])
    if clear_redundance:
        leng=len(occupy_list)
        del_list=[]
        for i in range(1,leng):
            if occupy_list[i][0]==occupy_list[i-1][0]:
                del_list.append(i-1)
            elif occupy_list[i][0]<occupy_list[i-1][1]:
                del_list.append(i-1)
                occupy_list[i][0]=occupy_list[i-1][0]
        new_list=[]
        for i in range(leng):
            if i not in del_list:
                new_list.append(occupy_list[i])
    #print(data_list)  
    if ave_unit_ms!=1:
        list_ave=[]
        occupy_time=0
        time=ave_unit_ms
        for data in new_list:
            if data[1]<time and data!=new_list[-1]:
                occupy_time+=data[1]-data[0]
            else:
                if data==new_list[-1]:
                    occupy_time+=data[1]-data[0]
                ave_resource=max_resource*occupy_time/ave_unit_ms
                list_ave.append((time-ave_unit_ms,time,ave_resource))
                time+=ave_resource
                occupy_time=data[1]-data[0]
    data_list=list_ave if ave_unit_ms!=1 else new_list
    #[(start_time,end_time,resource_occupy)]
    #print(data_list)  

    fig = plt.figure()
    ax = fig.add_subplot(111)
    data0=0
    for data in data_list:
        #plt.plot([data[0],data[1]],[data[2],data[2]],color='r',linewidth=2)
        plt.scatter(data[0],data[2],color='b')
        plt.scatter(data[1],data[2],color='r')
        #print(data[1],data[0])
        if data[0]>data0:
            plt.plot([data0,data[0]],[0,0],color='black',linewidth=1)
            data0=data[0]
    plt.xlabel("Time(ms)")
    plt.ylabel("Bandwidth(GB/s)")
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path,name+'.png'))
    plt.close()
    return data_list
def draw_util(util,name,path='status'):
    fig = plt.figure()
    plt.bar(list(range(len(util))),util,edgecolor='black')
    ave_util=sum(util)/len(util)
    plt.bar(len(util),ave_util,color='red',edgecolor='black')
    plt.title('average_{}={:.3f}%'.format(name,ave_util*100))
    plt.xlabel('id')
    plt.ylabel('utilization')
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path,name+'.pdf'))
    plt.close()
    return 'average_{}={:.3f}%\n'.format(name,ave_util*100)
def draw_mapping(wd,ml_name,tiles=[],path='sim_visualize',ori=False):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    x0=wd.X0Y0[0]
    x1=wd.X1Y1[0]
    y0=wd.X0Y0[1]
    y1=wd.X1Y1[1]
    max_s=max(y0*y1,x0*x1)
    core_weight=0.8
    core_high=0.8

    #print(x1,x0)
    for xj in range(x1):
        for yj in range(y1):
            facecolor=MY_COLOR[(yj+xj*y1)%len(MY_COLOR)]
            for xi in range(x0):
                for yi in range(y0):
                    xx=(xi+xj*x0-0.4)
                    yy=(yi+yj*y0)-0.4
                    rect = plt.Rectangle((xx,yy),core_weight,core_high,fill=ori,facecolor=facecolor,linewidth=0.1)
                    ax.add_patch(rect)
    #yi+yj*y0+xi*y1*y0+xj*y0*y1*x0
   
    if(tiles!=[]):
        for ids,tile in enumerate(tiles) :
            for id in tile:#4维度坐标(x1,y1,x0,y0)
                [xi,yi,xj,yj]=id
                xx=(xj+xi*x0-0.4)
                yy=(yj+yi*y0)-0.4
                #print(yy,xx)         
                plt.text(x=xx+0.4, y=yy+0.5, s=ids, rotation=1,ha='center',fontsize=12)  # 
                facecolor=MY_COLOR[ids%len(MY_COLOR)]
                rect = plt.Rectangle((xx,yy),core_weight,core_high,fill=not ori,facecolor=facecolor,linewidth=0.1)
                ax.add_patch(rect)
    
    # 设置整数范围
    ylim_lower = int(x0 * x1) - 1
    ylim_upper = 0
    xlim_lower = 0
    xlim_upper = int(y0 * y1) - 1

    # 设置整数范围
    ax.set_ylim(int(ylim_lower), int(ylim_upper))  # 确保传递整数值
    ax.set_xlim(int(xlim_lower), int(xlim_upper))  # 确保传递整数值

    # 设置坐标轴上的数字显示的位置，top:显示在顶部  bottom:显示在底部,默认是none
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    name=ml_name+'_map'
    plt.axis('equal')
    ax.set_xlabel("x_direct", fontsize=20)  # 设置 x 轴标签和字体大小
    ax.set_ylabel("y_scale", fontsize=20)   # 设置 y 轴标签和字体大小

    # 设置 x 和 y 轴的刻度标签字体大小
    ax.tick_params(axis='x', labelsize=16)  # 设置 x 轴刻度标签字体大小为 12
    ax.tick_params(axis='y', labelsize=16)  # 设置 y 轴刻度标签字体大小为 12
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path,name+'.pdf'))
    plt.close()
