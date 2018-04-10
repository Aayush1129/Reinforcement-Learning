import gym
import numpy as np

env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n])

n_episodes = 4000
lr = 0.7
y = 0.95

rList = []

for i in range(n_episodes):
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    while j<=99:
        j+=1
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        s1, r, d, _ = env.step(a)
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        rAll+=r
        s=s1
        if d==True:
            break
    rList.append(rAll)

print("score over time: "+ str(sum(rList)/n_episodes))
print("Final Q Table values: ")
print(Q)