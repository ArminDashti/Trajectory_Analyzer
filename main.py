import minari
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def load_and_collect(dataset_name, attribute):
    # Load the dataset
    dataset = minari.load_dataset(dataset_name, download=True)

    # Collect specified attribute from all episodes
    collected_data = []
    for episode in dataset:
        collected_data.append(getattr(episode, attribute))

    # Return stacked data or list of sums/variances
    if attribute in ['observations', 'actions']:
        return np.vstack(collected_data)
    elif attribute == 'rewards':
        return [np.sum(episode.rewards) for episode in dataset], [np.var(episode.rewards) for episode in dataset]

def plot_variance(data, title, xlabel):
    variances = np.var(data, axis=0)
    plt.figure()
    plt.bar(range(len(variances)), variances)
    plt.xlabel(xlabel)
    plt.ylabel('Variance')
    plt.title(title)
    plt.show()

def plot_dimensional_variance(dataset_name):
    data = load_and_collect(dataset_name, 'observations')
    plot_variance(data, 'Variance of Each Dimension (Observations)', 'Dimension Number')

def plot_action_variance(dataset_name):
    data = load_and_collect(dataset_name, 'actions')
    plot_variance(data, 'Variance of Each Dimension (Actions)', 'Dimension Number')

def plot_rewards_by_episode(dataset_name):
    total_rewards, _ = load_and_collect(dataset_name, 'rewards')
    plt.figure()
    plt.plot(range(len(total_rewards)), total_rewards, marker='o')
    plt.xlabel('Episode Number')
    plt.ylabel('Total Rewards')
    plt.title('Total Rewards per Episode')
    plt.show()

def plot_reward_variance_by_episode(dataset_name):
    total_rewards, reward_variances = load_and_collect(dataset_name, 'rewards')
    episode_numbers = range(len(total_rewards))

    # Normalize variances for coloring
    norm = plt.Normalize(min(reward_variances), max(reward_variances))
    colors = cm.RdBu(norm(reward_variances))

    plt.figure()
    scatter = plt.scatter(episode_numbers, total_rewards, s=np.array(reward_variances) * 100, c=colors, alpha=0.5, cmap='RdBu')
    plt.colorbar(scatter, label='Reward Variance')
    plt.xlabel('Episode Number')
    plt.ylabel('Total Rewards')
    plt.title('Reward Variance Across Episodes (Bubble Plot)')
    plt.show()

def plot_average_reward_by_step(dataset_name):
    # Load the dataset
    dataset = minari.load_dataset(dataset_name, download=True)

    # Collect rewards across all episodes
    all_rewards = []
    for episode in dataset:
        all_rewards.append(episode.rewards)

    # Determine the maximum number of steps
    max_steps = max(len(rewards) for rewards in all_rewards)

    # Initialize an array for averaging rewards per step
    rewards_per_step = np.zeros(max_steps)
    counts_per_step = np.zeros(max_steps)

    for rewards in all_rewards:
        for step, reward in enumerate(rewards):
            rewards_per_step[step] += reward
            counts_per_step[step] += 1

    # Calculate average rewards per step
    average_rewards = rewards_per_step / counts_per_step

    # Plot average rewards per step
    plt.figure()
    plt.plot(range(len(average_rewards)), average_rewards, marker='o')
    plt.xlabel('Step Number')
    plt.ylabel('Average Reward')
    plt.title('Average Reward per Step Across Episodes')
    plt.show()

# Example usage
plot_dimensional_variance('D4RL/door/expert-v2')
plot_action_variance('D4RL/door/expert-v2')
plot_rewards_by_episode('D4RL/door/expert-v2')
plot_reward_variance_by_episode('D4RL/door/expert-v2')
plot_average_reward_by_step('D4RL/door/expert-v2')
