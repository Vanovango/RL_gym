import gym

env = gym.make('CarRacing-v2')

env.reset()
while True:
    env.render()
