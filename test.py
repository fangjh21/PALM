import simpy
env = simpy.Environment()
def test_event(env):
    while True:
        print(env.now)
        yield env.timeout(10)
        print(env.now)
        break
event1=test_event(env)
env.process(event1)
#env.process(event1)
env.run(until=100)