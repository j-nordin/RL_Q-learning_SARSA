import argparse
import gymnasium as gym
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as st
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agent.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)




try:
    env = gym.make(args.env)
    print("Loaded ", args.env)
except:
    file_name, env_name = args.env.split(":")
    gym.envs.register(
        id=env_name + "-v0",
        entry_point=args.env,
    )
    env = gym.make(env_name + "-v0")
    print("Loaded", args.env)


action_dim = env.action_space.n
state_dim = env.observation_space.n


num_runs = 5
num_eps = 10000
num_steps = 15000


print(f"Running {args.env} with agent {args.agentfile}")

# Run trainings
for run in range(num_runs):
    agent = agentfile.Agent(state_dim, action_dim)
    
    
    if "FrozenLake" in args.env:
        rewards = np.zeros(shape=(num_runs, num_eps))
        
        # Run episodes
        for ep in range(num_eps):
            print(f"Run {run+1}: {100*(ep/num_eps):.1f}%", end="\r")
            observation = env.reset()
            
            done = False
            steps = 0
            
            # Run steps until done
            while not done: 
                #env.render()
                action = agent.act(observation) # your agent here (this currently takes random actions)
                observation, reward, done, truncated, info = env.step(action)
                # Compute new avg reward
                rewards[run, ep] = (rewards[run, ep] * (steps) + reward) / (steps + 1)
                agent.observe(observation, reward, done)
                
                if done:
                    observation, info = env.reset() 
                    
                steps += 1
    else:
        # Riverswim
        rewards = np.zeros(shape=(num_runs, num_steps))
        observation = env.reset()
        
        # Run episodes
        steps = 0
        
        # Run steps until done
        for step in range(num_steps): 
            #env.render()
            action = agent.act(observation) # your agent here (this currently takes random actions)
            observation, reward, done, truncated, info = env.step(action)
            # Compute new avg reward
            rewards[run, step] = reward
            agent.observe(observation, reward, done)
            steps += 1
        
    print(f"Run {run+1}: done")


# Calculate moving averages for the runs
runs_moving_avgs = pd.DataFrame(rewards).rolling(window=int(num_eps/10), min_periods=1, axis=1).mean()
runs_moving_avgs = runs_moving_avgs.to_numpy()
moving_avg = np.mean(runs_moving_avgs, axis=0)

# Calculate error bars
n = num_runs
std_err = np.std(runs_moving_avgs, axis=0) / np.sqrt(n)
t_value = st.stats.t.ppf((1 + 0.95) / 2., n - 1)
confidence_intervals = t_value * std_err

if args.env == "FrozenLake-v1":
    if args.agentfile == "agent_Q_learning.py":
        Q_values = agent.Q_table
        print(np.reshape(np.argmax(agent.Q_table, axis=1), (4,4)))
        Q_policy = np.argmax(agent.Q_table, axis=1)
        titel ='Q-learning agent'

    if args.agentfile == "agent_double_Q_learning.py":
        print(np.reshape(np.argmax(agent.Q1_table, axis=1), (4,4)))
        print(np.reshape(np.argmax(agent.Q2_table, axis=1), (4,4)))

        titel ='Double Q-learning agent'
        Q_policy = np.argmax(agent.Q_table, axis=1)
        Q_values = agent.Q_table

    if args.agentfile == "agent_SARSA.py":
        print(np.reshape(np.argmax(agent.Q, axis=1), (4,4)))
        titel ='SARSA agent'
        Q_policy = np.argmax(agent.Q, axis=1)
        Q_values = agent.Q

    if args.agentfile == "agent_Expected_SARSA.py":
        print(np.reshape(np.argmax(agent.Q, axis=1), (4,4)))
        titel ='Expected SARSA agent'
        Q_policy = np.argmax(agent.Q, axis=1)
        Q_values = agent.Q

else:
    if args.agentfile == "agent_Q_learning.py":
        print(np.reshape(np.argmax(agent.Q_table, axis=1), (6,1)))
        Q_policy = np.argmax(agent.Q_table, axis=1)
        titel ='Q-learning agent'
        Q_values = agent.Q_table

    if args.agentfile == "agent_double_Q_learning.py":
        print(np.reshape(np.argmax(agent.Q1_table, axis=1), (6,1)))
        print(np.reshape(np.argmax(agent.Q2_table, axis=1), (6,1)))
        titel ='Double Q-learning agent'
        Q_policy = np.argmax(agent.Q_table, axis=1)
        Q_values = agent.Q_table

    if args.agentfile == "agent_SARSA.py":
        print(np.reshape(np.argmax(agent.Q, axis=1), (6,1)))
        titel ='SARSA agent'
        Q_policy = np.argmax(agent.Q, axis=1)
        Q_values = agent.Q

    if args.agentfile == "agent_Expected_SARSA.py":
        print(np.reshape(np.argmax(agent.Q, axis=1), (6,1)))
        titel ='Expected SARSA agent'
        Q_policy = np.argmax(agent.Q, axis=1)
        Q_values = agent.Q


# VISUALIZATIONS
data = []
matrix = []

if args.env == "FrozenLake-v1":

    #POLICY
    for i in range(len(Q_policy)):
        if i % 4 == 0 and i != 0:
            matrix.append(data)
            data = []
        data.append(Q_policy[i])

    matrix.append(data)

    # heatmap = sns.heatmap(matrix, cmap='coolwarm', annot=True, cbar=False)
    heatmap = sns.heatmap(matrix, cmap='coolwarm', annot=True, cbar=False)

    # Define custom labels for the legend
    legend_labels = ['0 = Left', '1 = Down', '2 = Right', '3 = Up']

    # Get the current axes and create a colorbar
    ax = plt.gca()
    cbar = ax.figure.colorbar(heatmap.collections[0])

    # # Set the tick positions and labels for the colorbar
    # cbar.set_ticks([0, 1, 2, 3])
    # cbar.set_ticklabels(legend_labels)
    titel_policy = "Policy for " + titel
    heatmap.set_yticklabels([' ', ' ', ' ', ' '])
    heatmap.set_xticklabels([' ', ' ', ' ', ' '])
    plt.title(titel_policy)
    plt.show()

    #Q-VALUES
    title_value = "Q-values for " + titel
    heatmap = sns.heatmap(Q_values, cmap='coolwarm', annot=True, cbar=False)
    plt.title(title_value)
    heatmap.set_xticklabels(['Left', 'Down', 'Right', 'Up'])
    # heatmap.set_yticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.show()

else:

    #POLICY
    data = []
    matrix = []
    for i in range(len(Q_policy)):
        data.append(Q_policy[i])
        matrix.append(data)
        data = []

    # heatmap_policy = sns.heatmap(matrix, cmap='coolwarm', annot=True, cbar=False)
    heatmap = sns.heatmap(matrix, cmap='coolwarm', annot=True, cbar=False)

    # Define custom labels for the legend
    legend_labels = ['0 = Up', '1 = Down']

    # Get the current axes and create a colorbar
    ax = plt.gca()
    cbar = ax.figure.colorbar(heatmap.collections[0])

    # Set the tick positions and labels for the colorbar
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(legend_labels)
    titel_policy = "Policy for " + titel
    heatmap.set_yticklabels([' ', ' ', ' ', ' ', ' ', ' '])
    heatmap.set_xticklabels([' '])
    plt.title(titel_policy)
    plt.show()

    #Q-VALUES
    title_value = "Q-values for " + titel
    heatmap = sns.heatmap(Q_values, cmap='coolwarm', annot=True, cbar=False)
    plt.title(title_value)
    heatmap.set_xticklabels(['Left', 'Right'])
    # heatmap.set_yticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.show()



# Plot moving avg
plt.errorbar(range(len(moving_avg)), moving_avg, yerr=confidence_intervals, ecolor='gray', c='black', label="Moving average (95% CI)")
#for i in range(num_runs):
#    plt.plot(pd.Series(rewards[i,:]).rolling(window=int(num_eps/10), min_periods=1).mean())
plt.title(titel)
plt.xlabel(('Episodes' if 'FrozenLake' in args.env else 'Steps'))
plt.ylabel('Rewards (moving average)')
plt.legend()
plt.show()

env.close()
