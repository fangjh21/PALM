#### Simpy event registration 事件注册机制

```
import simpy
class comm_overlap():
    def __init__(self,env) -> None:
        self.env=env
        self.cp_worker= simpy.Resource(env, capacity=1)#定义计算和通信process的资源,并发执行的容量为1
        self.cm_worker= simpy.Resource(env, capacity=1)
    def cp_process(self):
        '''
        process 1 
        '''
        with self.cp_worker.request() as req:
                yield req
                yield self.env.timeout(20)#计算时间为20个单位
        print('process 1 done @{:.3f} '.format(self.env.now))
    def cm_process(self):
        '''
        process 2
        '''
        with self.cm_worker.request() as req:
                yield req
                yield self.env.timeout(30)#通信时间为20个单位
        print('process 2 done @{:.3f} '.format(self.env.now))
    def overlap_process(self):
         event_list=[]
         while(True):
              event_list.append(self.env.process(self.cp_process()))
              event_list.append(self.env.process(self.cm_process()))
              yield simpy.AllOf(env,event_list)#等待两个可以overlap的事件完成
              print('process overlap_process done @{:.3f} '.format(self.env.now))
              break
    def order_process(self):
         while(True):
              yield self.env.process(self.cp_process())
              yield self.env.process(self.cm_process())#等待事件顺序完成
              print('process order_process done @{:.3f} '.format(self.env.now))
              break
    def short_process(self):
        while(True):
              yield self.env.timeout(20)
              yield self.env.timeout(30)#等待事件顺序完成
              print('process short_process done @{:.3f} '.format(self.env.now))
              break
    def test_process(self):
        while(True):
              print('process test_process start @{:.3f} '.format(self.env.now))
              yield self.env.process(self.short_process())
              yield self.env.process(self.short_process())
              print('process test_process done @{:.3f} '.format(self.env.now))
              break   
if __name__ == '__main__':
    env=simpy.Environment()
    test=comm_overlap(env)
    #顶层事件并行注册
    env.process(test.overlap_process())
    env.process(test.order_process())
    env.process(test.short_process())
    env.process(test.test_process())
    #运行200时间单位
    env.run(until=200)

```

#### Result:

```
#process print
process test_process start @0.000 
process 1 done @20.000 
process 2 done @30.000 
process overlap_process done @30.000 
process 1 done @40.000 
process short_process done @50.000 
process short_process done @50.000 
process 2 done @70.000 
process order_process done @70.000 
process short_process done @100.000
process test_process done @100.000
```
