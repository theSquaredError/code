import gym
import numpy as np
import torch
import matplotlib.pyplot as plt 
import torch.optim as optim
from collections import deque
import sys  

from policy import PolicyNetwork

# env = gym.make("Breakout-v0")

# observation space and action space
# obs_space = env.observation_space
# action_space = env.action_space

# print("The observation space: {}".format(obs_space))
# print("The action space: {}".format(action_space))

# env_screen = env.render(mode = 'rgb_array')
# env.close()

# plt.imshow(env_screen)
# plt.show()

# state = env.reset()
# print(state)

# observation size of atari is --> (210,160,3)

GAMMA = 0.9


def reinforce(n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        state = torch.tensor (state)
        for t in range(max_t):
            action, log_prob = policy.get_action(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break 
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)])
        
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque)>=195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            break
        
    return scores

def update_policy(policy_network, rewards, log_probs):
    discounted_rewards = []   # storing discounted rewards from each timestep
    timesteps = len(rewards)
    for t in range(timesteps):
        Gt = 0 # discounted from timstep t
        pw = 0  
        for r in rewards[t:]:
            Gt = Gt + GAMMA**pw*r
            pw = pw+1
        discounted_rewards.append(Gt)

    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards-discounted_rewards.mean())/(discounted_rewards.std() + 1e-9)

    policy_gradient = []
    
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob*Gt)
    
    optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    optimizer.step()




# training part
if __name__ == '__main__':
    env = gym.make("ALE/Breakout-v5")
    print(env.env.get_action_meanings())
    print(env.action_space.n)
    policy = PolicyNetwork()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    max_episode_num = 5000
    max_steps = 10000
    num_steps = []
    avg_numsteps = []
    all_rewards = []

    for episode in range(max_episode_num):
        state = env.reset()
        log_probs = []
        
        rewards = []

        for steps in range(max_steps):
            env.render()
            state = torch.tensor(state).float()
            # print(state.size)
            action, log_prob = policy.get_action(state)
            new_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            if done:
                update_policy(policy, rewards, log_probs)
                num_steps.append(steps)
                avg_numsteps.append(np.mean(num_steps[-10:0]))
                all_rewards.append(np.sum(rewards))
                
                if episode %1 == 0:
                    sys.stdout.write("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3),  np.round(np.mean(all_rewards[-10:]), decimals = 3), steps))
                break
            state = new_state
    plt.plot(num_steps)
    plt.plot(avg_numsteps)
    plt.xlabel('Episode')
    plt.show()