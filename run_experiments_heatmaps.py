import argparse
import gymnasium as gym
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agent.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)


try:
    env = gym.make(args.env, is_slippery=True)
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
num_eps = 10
num_steps = 15000
rewards = np.zeros(shape=(num_runs, num_eps))


# Run trainings
for run in range(num_runs):
    agent = agentfile.Agent(state_dim, action_dim)
    
    # Run episodes
    for ep in range(num_eps):
        observation = env.reset()
        
        # Run steps
        for _ in range(num_steps): 
            #env.render()
            action = agent.act(observation) # your agent here (this currently takes random actions)
            observation, reward, done, truncated, info = env.step(action)
            rewards[run, ep] += reward
            agent.observe(observation, reward, done)
            
            if done:
                observation, info = env.reset() 

avg_rewards = np.average(rewards, axis=0)
std_rewards = np.std(rewards, axis=0)
t_val = 2.776
confidence_intervals = t_val * std_rewards / np.sqrt(len(rewards))

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


# Plot moving avg
#plt.plot(range(len(avg_rewards)), avg_rewards)
# print(rewards)
# print(avg_rewards)
# print(std_rewards)

# plt.errorbar(range(len(confidence_intervals)), avg_rewards, yerr=confidence_intervals)
# plt.title(titel)
# plt.xlabel('Episodes')
# plt.ylabel('Rewards (moving average)')
# plt.show()

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


env.close()
