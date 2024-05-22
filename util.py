

from macro import *
import json
import pandas as pd
import numpy as np
def load_config(input_path):
    # 读取json配置文件
    with open(input_path, 'r') as file:
        # 从文件中加载 JSON 数据
        config = json.load(file)
    return config
def store_config(path,config):
    # 读取json配置文件
    with open(path, 'w') as file:
        # 从文件中加载 JSON 数据
        json.dump(config, file,default=convert_to_str, indent=4)
def save_file(data, file_path):
    # 转换为list+dict类型
    df = pd.DataFrame(data)
    # 将DataFrame保存到Excel中，index参数用于指定是否包含行索引
    df.to_excel(file_path, index=False)

def convert_to_str(obj):
    if isinstance(obj, Enum):
        return obj.name
    else:
        return obj

def sizeof(dim_list,coe=1):
    if dim_list==None:
        size=0
    else:
        size=1
        for dim in dim_list:
            size*=dim
    return size*coe

def str2list(string):
    ls=[]
    string=string.split('[')[1].split(']')[0]
    string=string.split(',')

    for num_str in string:
        #print(int(num_str))
        num_str.split(',')
        ls.append(int(num_str))
    return ls

def str2strlist(string):
    ls=[]
    string=string.split('[')[1].split(']')[0]
    string=string.split(',')
    for str_ in string:
        if str_!='':
            str_str=str_.split('\'')
            ls.append(str_str[1])
    return ls
def split_cg(Group_Id,parall_dims):
    ''' 
    Here is an example :
    suppose Group_Id=[0,1,2,3,...,15],len=16
    1.if parall_dims=[16,1,1,1],group=[[0:15],[],[],[]]
    2.if parall_dims=[1,16,1,1],group=[[],[0:15],[],[]]
    3.if parall_dims=[8,2,1,1],group=
    [[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]
    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
    []
    []
    '''
    Group_Size=len(Group_Id)
    total_dims=1
    split_group=[]
    for dim in parall_dims:
        split_group.append(total_dims)
        total_dims*=dim
    assert Group_Size==total_dims,'Group_Size={},but total_dims={} '.format(Group_Size,total_dims)
    num_dims=len(parall_dims)
    groups=[]
    offset=Group_Size
    #print(split_group)
    for k in range(num_dims):
        temp_group_size=parall_dims[k]
        #print(temp_group_size)
        temp_group=[]
        if temp_group_size!=1:
            offset//=parall_dims[k]
            #print("offset",offset)
            for j in range(split_group[k]):
                #print(k,offset,j)
                for i in range(offset):
                    #print(i+j*(Group_Size//split_group[k]),(j+1)*Group_Size//split_group[k],offset)
                    temp_group.append(Group_Id[i+j*(Group_Size//split_group[k]):(j+1)*Group_Size//split_group[k]:offset])
        groups.append(temp_group)
    '''
    if parall_dims[0]==4 and parall_dims[1]==4:
        return [[
            [Group_Id[0], Group_Id[1], Group_Id[5], Group_Id[4]], [Group_Id[2], Group_Id[3], Group_Id[7], Group_Id[6]]\
             ,[Group_Id[8], Group_Id[9], Group_Id[13], Group_Id[12]], [Group_Id[10], Group_Id[11], Group_Id[15], Group_Id[14]]],\
                  [[Group_Id[0], Group_Id[2], Group_Id[8], Group_Id[10]], [Group_Id[1], Group_Id[3], Group_Id[9], Group_Id[11]], [Group_Id[4], Group_Id[6], Group_Id[12], Group_Id[14]], [Group_Id[5], Group_Id[7], Group_Id[13], Group_Id[15]]]
                ]
    '''
    return groups
if __name__ == "__main__":
    cg=split_cg([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[2,2,2,2])
    print(cg[0])
    print(cg[1])
    print(cg[2])
    print(cg[3])
