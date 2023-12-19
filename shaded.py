import matplotlib.pyplot as plt
import numpy as np
import json

file_paths = ["./results/2023-11-13--09-52-54", "./results/2023-12-06--16-14-30", "./results/2023-12-07--23-47-10"]
# 0 Example data (replace this with your actual data)
seeds = [42, 43, 44]
reward_data = {42: [], 43: [], 44: []}

objective_mcts_data = {42: [], 43: [], 44: []}

for file_path, seed in zip(file_paths, seeds):
    # Create an empty dictionary to store the merged result
    merged_dict = {}
    eval_mcts = {"Total num played games": [], "Total num trained steps": []}
    eval_avg_obj = {"Total num played games": []}
    # Open the file and read line by line
    with open(file_path + "/log.txt", "r") as file:
        for line in file:
            try:
                # Parse each line as a JSON dictionary
                line_dict = json.loads(line)
                # Merge the parsed dictionary into the result dictionary
                for key, value in line_dict.items():
                    if key == "logtype" and value == "evaluation":
                        eval_mcts["Total num played games"].append(line_dict["Total num played games"])
                        eval_mcts["Total num trained steps"].append(line_dict["Total num trained steps"])
                    if key == "Avg objective":
                        eval_avg_obj["Total num played games"].append(line_dict["Total num played games"])

                    if key in merged_dict:
                        # If the key already exists, append the value to a list
                        if isinstance(merged_dict[key], list):
                            merged_dict[key].append(value)
                        else:
                            merged_dict[key] = [merged_dict[key], value]
                    else:
                        # If the key doesn't exist, create it with the value
                        merged_dict[key] = value
            except json.JSONDecodeError as e:
                # Handle any JSON decoding errors here if needed
                print(f"Error decoding JSON on line: {line}")
                print(e)
        reward_data[seed] = merged_dict["Avg objective"]
        objective_mcts_data[seed] = merged_dict["Evaluation MCTS"][:-2]


min_comm_len = min(len(reward_data[42]), len(reward_data[43]), len(reward_data[44]))
mean_rewards = np.mean(
    [
        np.array(reward_data[42][:min_comm_len]),
        np.array(reward_data[43][:min_comm_len]),
        np.array(reward_data[44][:min_comm_len]),
    ],
    axis=0,
)
std_rewards = np.std(
    [
        np.array(reward_data[42][:min_comm_len]),
        np.array(reward_data[43][:min_comm_len]),
        np.array(reward_data[44][:min_comm_len]),
    ],
    axis=0,
)


# Plot the first line
x = eval_avg_obj["Total num played games"][:min_comm_len]

# Plot the sum of rewards with shaded regions
plt.figure(figsize=(10, 5))
# Plot mean line
plt.plot(x, mean_rewards, label="Mean", color="blue")
# Plot shaded region
plt.fill_between(x, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, color="blue")
plt.xlabel("Played Episodes")
plt.ylabel("Average Objective")
plt.legend()
plt.savefig("seeds_tr")
plt.show()

min_comm_len = min(len(objective_mcts_data[42]), len(objective_mcts_data[43]), len(objective_mcts_data[44]))
mean_rewards = np.mean(
    [
        np.array(objective_mcts_data[42][:min_comm_len]),
        np.array(objective_mcts_data[43][:min_comm_len]),
        np.array(objective_mcts_data[44][:min_comm_len]),
    ],
    axis=0,
)
std_rewards = np.std(
    [
        np.array(objective_mcts_data[42][:min_comm_len]),
        np.array(objective_mcts_data[43][:min_comm_len]),
        np.array(objective_mcts_data[44][:min_comm_len]),
    ],
    axis=0,
)
# Plot the first line
x = eval_mcts["Total num trained steps"][:min_comm_len]
# Plot the sum of rewards with shaded regions
plt.figure(figsize=(10, 5))
# Plot mean line
plt.plot(x, mean_rewards, label="Mean", color="blue")
# Plot shaded region
plt.fill_between(x, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, color="blue")

# plt.title('Sum of Rewards Over Time with Shaded Regions (Mean ± Std)')
plt.xlabel("Training Steps")
plt.ylabel("Average Objective")
plt.legend()
plt.savefig("seeds_eval")
plt.show()
