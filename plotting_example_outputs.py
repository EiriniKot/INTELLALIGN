import numpy as np
import json, os
import torch
from tabulate import tabulate
from tools.dataset_generator import Decoder, flatten_input_tool
from tools.reward_functions import Reward
from tools.encoding_tools import pos_encoding_2

exp_path = "/home/eirini/PycharmProjects/GAZ_MSA/results/2023-11-20--16-25-54"
f = open(os.path.join(exp_path, "config_options.json"))
cfg = json.load(f)
test_set_path = cfg["test_set_path"].split("/")[-1]

clustal_results = os.path.join(f"sets/clustalomega_results_{test_set_path}")
f = open(clustal_results)
clustal_examples = json.load(f)
clustal_examples = flatten_input_tool(clustal_examples)

dec = Decoder(tokens_set=cfg["msa_conf"]["tokens_set"])
examples_path = os.path.join(exp_path, "examples/examples.json")
f = open(examples_path)
loaded_examples = json.load(f)

reward_func = Reward(
    method="1.0*SumOfPairs",
    reward_values={"GG": 0, "GL": 0, "LL": 1, "LDL": 0}
)
reward_func2 = Reward(
    method="1.0*TotalColumn",
    reward_values={"GG": 0, "GL": 0, "LL": 1, "LDL": 0}
)
x = 0
print(len(loaded_examples))
all_rewards = []
all_rewards2 = []

for example in loaded_examples:
    num_sequences = len(np.where(torch.IntTensor(example["init_state"]) == 5)[0])
    init_state = dec.encoder.batch_decode(torch.IntTensor(example["init_state"]))[1:]
    last = dec.encoder.batch_decode(torch.IntTensor(example["last"]))

    state = np.array(example["last"])
    pos_enc = pos_encoding_2(last, stop_move=True, newline="END")
    full_state = np.stack([state, pos_enc[0], pos_enc[1]])
    rew_intell = reward_func(full_state)
    rew_intell2 = reward_func2(full_state)
    all_rewards.append(rew_intell)
    all_rewards2.append(rew_intell2)
    last = last[1:]
    for entity in clustal_examples:
        if entity["init_flatten"] == init_state:
            final_msa = entity["final_msa"]
            final_msa_tbl = []
            for seq in final_msa:
                split = list(seq)
                final_msa_tbl.append(split)
            rew_clustal = entity["1.0*SumOfPairs"]

    init_state_reshaped = np.array(init_state).reshape(num_sequences, -1)
    init_state_reshaped = np.delete(init_state_reshaped, -1, axis=1)

    last_reshaped = np.array(last).reshape(num_sequences, -1)
    last_reshaped = np.delete(last_reshaped, -1, axis=1)

    print(tabulate(init_state_reshaped, tablefmt="latex"))
    print(tabulate(last_reshaped, tablefmt="latex"))
    print(tabulate(final_msa_tbl, tablefmt="latex"))
    print("Reward", rew_intell)
    print("Reward 2", rew_clustal)
    print("-" * 20)
print(all_rewards)
#
# with open(os.path.join(file_path, "examples/examples.json")) as f:
#     data_list = json.load(f)
# init_state = data_list[0]["init_state"]  # Replace with your data
# last = data_list[0]["last"]  # Replace with your data
#
# # Drop the first row from both arrays
# init_state = init_state[1:]
# last = last[1:]
#
# # Reshape the arrays into an orthogonal parallelogram
# num_rows = len(init_state)
# init_state_reshaped = np.array(init_state).reshape(5, -1)
# init_state_reshaped = np.delete(init_state_reshaped, -1, axis=1)
#
# last_reshaped = np.array(last).reshape(5, -1)
# last_reshaped = np.delete(last_reshaped, -1, axis=1)
#
# print('init_state_reshaped', init_state_reshaped.shape)
# print(tabulate(init_state_reshaped, tablefmt="latex"))


#2023-11-13--09-52-54
#[28.0, 18.0, 27.0, 14.0, 20.0, 28.0, 29.0, 25.0, 17.0, 28.0, 29.0, 19.0, 21.0, 26.0, 18.0, 21.0, 34.0, 24.0, 22.0, 30.0, 36.0, 33.0, 22.0, 39.0, 13.0, 15.0, 31.0, 24.0, 35.0, 28.0, 28.0, 19.0, 17.0, 16.0, 33.0, 33.0, 25.0, 22.0, 21.0, 25.0, 19.0, 29.0, 32.0, 36.0, 29.0, 25.0, 18.0, 19.0, 24.0, 22.0, 20.0, 32.0, 19.0, 24.0, 22.0, 26.0, 25.0, 27.0, 31.0, 32.0, 25.0, 33.0, 36.0, 30.0, 19.0, 21.0, 22.0, 28.0, 23.0, 30.0, 31.0, 22.0, 16.0, 13.0, 34.0, 19.0, 23.0, 23.0, 23.0, 26.0, 17.0, 18.0, 24.0, 25.0, 17.0, 15.0, 18.0, 23.0, 26.0, 37.0, 18.0, 22.0, 19.0, 29.0, 21.0, 33.0, 33.0, 24.0, 30.0, 22.0, 33.0, 31.0, 29.0, 30.0, 31.0, 14.0, 20.0, 18.0, 19.0, 24.0, 25.0, 22.0, 28.0, 21.0, 16.0, 34.0, 27.0, 29.0, 19.0, 19.0, 24.0, 31.0, 21.0, 36.0, 16.0, 25.0, 18.0, 15.0, 16.0, 17.0, 26.0, 18.0, 28.0, 30.0, 34.0, 29.0, 35.0, 19.0, 18.0, 34.0, 15.0, 34.0, 35.0, 15.0, 16.0, 21.0, 25.0, 17.0, 19.0, 33.0, 20.0, 20.0, 35.0, 20.0, 31.0, 18.0, 21.0, 20.0, 20.0, 26.0, 35.0, 35.0, 20.0, 15.0, 20.0, 28.0, 15.0, 34.0, 34.0, 25.0, 25.0, 28.0, 33.0, 32.0, 15.0, 18.0, 19.0, 30.0, 30.0, 20.0, 27.0, 38.0, 28.0, 30.0, 22.0, 13.0, 22.0, 21.0, 19.0, 20.0, 23.0, 17.0, 15.0, 29.0, 22.0, 17.0, 29.0, 21.0, 27.0, 19.0, 18.0, 21.0, 38.0, 21.0, 25.0, 20.0, 31.0, 16.0, 20.0, 25.0, 25.0, 26.0, 27.0, 22.0, 27.0, 31.0, 20.0, 26.0, 15.0, 19.0, 40.0, 24.0, 25.0, 37.0, 27.0, 16.0, 34.0, 30.0, 24.0, 35.0, 33.0, 17.0, 27.0, 29.0, 23.0, 18.0, 24.0, 26.0, 22.0, 34.0, 20.0, 19.0, 26.0, 23.0, 17.0, 16.0, 20.0, 20.0, 30.0, 18.0, 37.0, 18.0, 23.0, 24.0, 26.0, 23.0]
#[27.0, 19.0, 27.0, 12.0, 20.0, 26.0, 24.0, 23.0, 14.0, 23.0, 29.0, 20.0, 21.0, 26.0, 14.0, 19.0, 34.0, 22.0, 23.0, 28.0, 28.0, 29.0, 25.0, 26.0, 15.0, 17.0, 27.0, 25.0, 34.0, 23.0, 26.0, 14.0, 17.0, 14.0, 30.0, 32.0, 21.0, 22.0, 21.0, 24.0, 16.0, 32.0, 26.0, 34.0, 31.0, 27.0, 18.0, 16.0, 14.0, 22.0, 20.0, 23.0, 19.0, 25.0, 24.0, 36.0, 33.0, 27.0, 31.0, 26.0, 23.0, 26.0, 37.0, 28.0, 19.0, 25.0, 24.0, 27.0, 21.0, 33.0, 32.0, 22.0, 16.0, 19.0, 35.0, 18.0, 23.0, 16.0, 23.0, 26.0, 18.0, 16.0, 31.0, 24.0, 17.0, 15.0, 18.0, 23.0, 26.0, 37.0, 18.0, 17.0, 11.0, 27.0, 21.0, 33.0, 33.0, 21.0, 29.0, 23.0, 34.0, 31.0, 29.0, 29.0, 23.0, 18.0, 20.0, 18.0, 19.0, 28.0, 21.0, 23.0, 22.0, 22.0, 17.0, 32.0, 18.0, 25.0, 18.0, 19.0, 24.0, 27.0, 21.0, 31.0, 17.0, 18.0, 21.0, 14.0, 14.0, 17.0, 23.0, 18.0, 35.0, 30.0, 37.0, 37.0, 38.0, 19.0, 18.0, 24.0, 16.0, 37.0, 35.0, 15.0, 18.0, 22.0, 32.0, 18.0, 18.0, 37.0, 20.0, 17.0, 35.0, 18.0, 30.0, 20.0, 12.0, 25.0, 22.0, 25.0, 26.0, 38.0, 18.0, 16.0, 20.0, 28.0, 19.0, 31.0, 25.0, 25.0, 20.0, 28.0, 33.0, 24.0, 15.0, 17.0, 19.0, 30.0, 31.0, 19.0, 22.0, 34.0, 33.0, 32.0, 22.0, 13.0, 22.0, 18.0, 13.0, 20.0, 22.0, 19.0, 12.0, 27.0, 19.0, 17.0, 25.0, 15.0, 27.0, 19.0, 16.0, 21.0, 35.0, 21.0, 28.0, 21.0, 40.0, 16.0, 16.0, 22.0, 29.0, 26.0, 32.0, 24.0, 27.0, 31.0, 20.0, 31.0, 22.0, 23.0, 32.0, 24.0, 28.0, 39.0, 28.0, 16.0, 37.0, 26.0, 26.0, 35.0, 34.0, 19.0, 30.0, 24.0, 24.0, 17.0, 24.0, 28.0, 26.0, 34.0, 21.0, 15.0, 24.0, 23.0, 17.0, 16.0, 20.0, 20.0, 16.0, 18.0, 35.0, 23.0, 23.0, 27.0, 26.0, 23.0]
