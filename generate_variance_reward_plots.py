import matplotlib.pyplot as plt
import numpy as np

# Example: Replace this with your actual rewards data
num_episodes = 100
time_steps_per_episode = 100
rewards = np.random.normal(0, 1, size=(num_episodes, time_steps_per_episode))

# Calculate mean and variance across episodes for each time step
mean_rewards = np.mean(rewards, axis=0)
variance_rewards = np.var(rewards, axis=0)

# Create time steps array
time_steps = np.arange(time_steps_per_episode)

# Plot mean reward
plt.plot(time_steps, mean_rewards, label="Mean Reward", color="blue")

# Plot variance as a shaded region around the mean
plt.fill_between(
    time_steps,
    mean_rewards - np.sqrt(variance_rewards),
    mean_rewards + np.sqrt(variance_rewards),
    color="blue",
    alpha=0.2,
    label="Variance",
)

# Customize the plot
plt.title("Reward Plot with Variance")
plt.xlabel("Time Steps")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.show()
