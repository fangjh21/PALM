from macro import *
from util import *
from typing import List
class Op():
    def __init__(self,hint_name:str,op_type:OP,op_param:List[int],p_sgy=[1,1,1,1]) -> None:
        #base info 
        self.hint=hint_name
        self.type=op_type
        self.dims=op_param
        self.p_sgy=p_sgy

        self.zero=zero.none
        self.abcd=[1,1,1,1]

        self.o_shape=[]
        self.i_shape=[]
        self.w_shape=[]
        self.r_size=0

        self.fd_macs=0  

        self.dpmap_flag=False
        self.devices=[]
        self.cg=[]
        self.d4d_comm={'f':[],'b':[],'u':[]}
        self.zero_comm={'f':[],'b':[],'u':[]}
        self.analysis()
    def __str__(self):
        return '{}:{},{}),p_sgy={},macs={},\ndevice={},4d_comm={},zero_comm={})'\
            .format(self.hint,self.type,self.dims,self.p_sgy,self.fd_macs,self.devices,self.d4d_comm,self.zero_comm)
    def analysis(self,abcd=[1,1,1,1],zero=zero.none):
        self.abcd=abcd
        self.zero=zero
        cg=self.cg
        if self.type==OP.Linear:
            [B,M,N,K]=self.dims
            [b,m,n,k]=self.p_sgy
            self.o_shape=[B/b,M/m,N/n]
            self.i_shape=[B/b,M/m,K/k]
            self.w_shape=[N/n,K/k]
            self.r_size=0
            self.fd_macs=B*M*N*K/b/m/n/k
            if self.dpmap_flag:
                self.d4d_comm={'f':[(B*M*N/(b*m*n),cg[3],COMM.AR)],'b':[(B*M*K/(b*m*k),cg[2],COMM.AR)],'u':[(N*K/(n*k),cg[0],COMM.AR),(N*K/(n*k),cg[1],COMM.AR)]}
            #self.zero_comm={'f':[],'b':[],'u':[]}
        elif self.type==OP.Conv2:
            [B,H,W,C,R,S,K] =self.dims
            [b,c,i,k]=self.p_sgy
            Ho=H/i-S+1
            Wo=W-R+1
            self.o_shape=[B/b,Ho,Wo,K/k]
            self.i_shape=[B/b,H/i,W,C/c]
            self.w_shape=[C,R,S,K]
            self.r_size=0
            self.fd_macs=B*H*W*R*S*C*K/b/c/i/k
            if self.dpmap_flag:
                self.d4d_comm={'f':[(B*Ho*Wo*K/(b*i*k),cg[1],COMM.AR)],'b':[(B*H*W*C/(b*i*c),cg[3],COMM.AR)],'u':[(R*S*C*K/(c*k),cg[0],COMM.AR),(R*S*C*K/(c*k),cg[2],COMM.AR)]}
            #self.zero_comm={'f':[],'b':[],'u':[]}
        elif self.type==OP.Pool:
            [B,H,W,C,R,S] =self.dims
            [b,c,i,_]=self.p_sgy
            Ho=H/i-S+1
            Wo=W-R+1
            self.o_shape=[B/b,Ho,Wo,C/c]
            self.i_shape=[B/b,H/i,W,C/c]
            self.w_shape=[0]
            self.r_size=0
            self.fd_macs=B*H*W*R*S*C/b/c/i
            if self.dpmap_flag:
                self.d4d_comm={'f':[],'b':[],'u':[]}
            #self.zero_comm={'f':[],'b':[],'u':[]}
        elif self.type==OP.Embedding:
            #TODO
            [B,M,N,K]=self.dims
            [b,m,n,k]=self.p_sgy
            self.o_shape=[B/b,M/m,N/n]
            self.i_shape=[B/b,M/m,K/k]
            self.w_shape=[N/n,K/k]
            self.r_size=0
            self.fd_macs=0#B*M*N*K/b/m/n/k
            if self.dpmap_flag:
                self.d4d_comm={'f':[],'b':[],'u':[]}
            #self.zero_comm={'f':[],'b':[],'u':[]}
        elif self.type==OP.Encoder:
            [B,S,H,A,H1]=self.dims
            [Nd,Nm,_,_]=self.p_sgy
            self.o_shape=[B/Nd,S,H]
            self.i_shape=[B/Nd,S,H]
            self.w_shape=[H/Nm,4*H+2*H1]
            self.r_size=3*B*S*H/Nd+(2*B*S*H1+4*B*S*H)/Nd/Nm+2.5*B*S*S*A/Nd/Nm
            self.fd_macs=(4*B*S*H*H+2*B*S*H*H1+2*B*S*S*H)/Nd/Nm
            if self.dpmap_flag:
                self.d4d_comm={'f':[(B*S*H/Nd,cg[1],COMM.AR)],'b':[(B*S*H/Nd,cg[1],COMM.AR)],'u':[(H*4*H+(H*2*H1)/Nm,cg[0],COMM.AR)]}
                if self.zero==zero.s3:
                    self.w_shape=[H/Nd/Nm,4*H+2*H1]
                    self.zero_comm={'f':[(H*4*H+(H*2*H1)/Nm,cg[0],COMM.AG)],'b':[(H*4*H+(H*2*H1)/Nm,cg[0],COMM.AG)],'u':[]}
                else:
                    self.zero_comm={'f':[],'b':[],'u':[]}
        else:
            raise NotImplementedError
        self.iwor_size=[sizeof(self.i_shape),sizeof(self.w_shape),sizeof(self.o_shape),self.r_size]
    def dpmap(self,devices:List[int],p_sgy:List):
        assert len(p_sgy)==4
        self.devices=devices
        if self.type==OP.Linear :
            self.p_sgy=p_sgy
        elif self.type==OP.Conv2:
            self.p_sgy=p_sgy
        elif self.type==OP.Pool:
            self.p_sgy=[p_sgy[0]*p_sgy[3],p_sgy[1],p_sgy[2],1]
        elif self.type==OP.Embedding:
            self.p_sgy=p_sgy
        elif self.type==OP.Encoder:
            self.p_sgy=[p_sgy[0]*p_sgy[2]*p_sgy[3],p_sgy[1],1,1]
        else:
            raise NotImplementedError
        assert sizeof(self.p_sgy)==len(devices),'sizeof(p_sgy)={},len(device_id)={}'.format(sizeof(p_sgy),len(devices))
        self.dpmap_flag=True  
        self._comm_set()
        self.analysis(abcd=[1,1,1,1],zero=zero.none)
        return True

    def _comm_set(self):
        self.cg=split_cg(self.devices,self.p_sgy)
        
if __name__ == '__main__':
    op1=Op(hint_name='t1',op_type=OP.Encoder,op_param=[1,4096,4096,4096,4096*4])
    op2=Op(hint_name='s1',op_type=OP.Linear,op_param=[1,1024,1024,1024])
    #op1.dpmap(p_sgy=[1,2,1,2],devices=[0,1,2,3])
    op2.dpmap(p_sgy=[1,2,1,2],devices=[0,1,2,3])
    print(op1)
    print(op2)