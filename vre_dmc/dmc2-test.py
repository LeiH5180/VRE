import dmc2gym
import numpy

print(numpy.__version__)

env = dmc2gym.make(domain_name='point_mass', task_name='easy', seed=1)

done = False
obs = env.reset()
while not done:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
  print(action, reward, done)