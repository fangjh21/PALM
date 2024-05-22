
from enum import Enum

class Enum(Enum):
    def __str__(self):
        return self.name
    def __repr__(self):
        return f'{self.name}'

ONE_WEEK_MS=24*60*60*7*1000
INT_MAX=2**64-1
TB=1/(1024**4)
GB=1/(1024**3)
MB=1/(1024**2)
KB=1/1024
Ts=1/(1000**4)
Gs=1/(1000**3)
Ms=1/(1000**2)
Ks=1/(1000)
ms=1000
us=1000*1000
h=1/60
OP = Enum('OP', ('Linear', 'Conv2', 'Embedding', 'Softmax','LayerNorm','Encoder','Pool','Concat','Sum',))
COMM=Enum('COMM',('NONE','AR','AA','AG','RS'))
optimizer=Enum('optimizer',('none','SGD','adam'))
mode=Enum('mode',('INT8','FP16','FP32'))
state=Enum('state',('forward','backward','param_sync','recompute'))
dataflow=Enum('dataflow',('IS','WS','WeightStream','ActStream'))
comp_model=Enum('comp_model',('simple','scale_sim','abrupt_curve'))

store=Enum('store',('cache','weight','ACT','ACT_weight','none'))
recompute=Enum('recompute',('none','one','half','full'))

pipe=Enum('pipe',('GPipe','Dreampipe1F1B','Interleaved1F1B'))
zero=Enum('zero',('none','s1','s2','s3','sr'))

event=Enum('event',('act_store','act_fetch','comm','act_fd','grad_fetch','grad_store','wt_load','wt_store','opt_load','opt_store','dloss_load','dloss_store'))

def str2enum(str,type='OP'):
    try:
        if type=='OP':
            enum_value = OP[str]
        elif type=='recompute':
            enum_value = recompute[str]
        elif type=='pipe':
            enum_value = pipe[str]
        elif type=='zero':
            enum_value = zero[str]
        else:
            raise NotImplementedError
        return enum_value
    except KeyError:
        raise NotImplementedError
