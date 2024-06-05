from collections import defaultdict
from op import *
class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.schedule = []
        self.visited = set()
        self.__iter_index=0
        self.batch_size=0
    def add_edge(self, son,parent):
        self.graph[son].append(parent)
    def __len__(self):
        return len( self.graph)
    def __iter__(self):
        return self
    def __next__(self):
        if self.__iter_index==0:
            self.__iter_items=iter(self.schedule)
        elif self.__iter_index==len(self.schedule):
             self.__iter_index=0
             raise StopIteration
        self.__iter_index+=1
        return next(self.__iter_items)
    def __contains__(self, item):
            return item in self.graph
    def topological_sort(self):
        for node in list(self.graph.keys()):
            #print(node)
            if node not in self.visited:
                self._topological_sort(node)
    def _topological_sort(self, node):
        self.visited.add(node)
        for neighbor in self.graph[node]:
            if neighbor not in self.visited:
                self._topological_sort(neighbor)
        self.schedule.append(node)

    def training_graph(self):
        pass

    @staticmethod
    def matrix():
        op0=Op(hint_name='linear_0',op_type=OP.Linear,op_param=[512,1024,1024,1024])
        op1=Op(hint_name='linear_1',op_type=OP.Linear,op_param=[512,1024,1024,1024])
        graph = Graph()
        graph.add_edge(op1,op0)
        graph.topological_sort()
        graph.batch_size=512
        return graph
    #80,2304,2048,12288,96
    def Encoder(name='Bert-base',B=512,S=512,H=1024,A=12,H1=1024*4,L=12):
        if name in ['Bert-base','Bert-large','T-18B','T-39B','T-76B','T-145B','T-310B','T-530B','T-1T','GPT3-175B','LLaMA2-7B']:
            if name=='Bert-base':
                L,B,S,H,A,H1=12,512,512,768,12,768*4
            elif name=='Bert-large':
                L,B,S,H,A,H1=24,512,512,1024,16,1024*4
            elif name=='T-18B':
                L,B,S,H,A,H1=40,1024,2048,6144,48,6144*4     
            elif name=='T-39B':
                L,B,S,H,A,H1=48,2304,2048,8192,64,8192*4   
            elif name=='T-76B':
                L,B,S,H,A,H1=60,2304,2048,10240,80,10240*4   
            elif name=='T-145B':
                L,B,S,H,A,H1=80,2304,2048,12288,96,12288*4   
            elif name=='T-310B':
                L,B,S,H,A,H1=96,2304,2048,16384,128,16384*4   
            elif name=='T-530B':
                L,B,S,H,A,H1=105,2304,2048,20480,128,20480*4     
            elif name=='T-1T':
                L,B,S,H,A,H1=128,2304,2048,25600,128,25600*4  
            elif name=='GPT3-175B':
                L,B,S,H,A,H1=96,2304,2048,12288,96,12288*4  
            elif name=='LLaMA2-7B':
                L,B,S,H,A,H1=32,1024,4096,4096,32,11008
            else:
                print('warning: not predefined LLM model')
                pass
        layers_id=0
        graph = Graph()
        for i in range(L):
            cur_op=Op(hint_name='encoder_'+str(layers_id),op_type=OP.Encoder,op_param=[B,S,H,A,H1])
            if layers_id>0:
                graph.add_edge(cur_op,last_op)
            last_op=cur_op
            layers_id+=1
        graph.topological_sort()
        graph.batch_size=B
        return graph
if __name__ == "__main__":
    test=Graph.Encoder()
    for op in test:
        print(op)