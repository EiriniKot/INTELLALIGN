import numpy as np
import json, os
import torch
from tools.dataset_generator import Decoder
from tools.reward_functions import Reward
from tools.encoding_tools import pos_encoding_2
from tools.generic_tools import compare_with

# experiments = ["./results/2023-11-11--11-23-11", #3
#                "./results/2023-11-13--09-52-54", #4
#                "./results/2023-12-05--15-58-59"] #5
#                 "./results/2023-11-20--16-25-54", #6
#                 "./results/2023-11-23--10-26-36", #7
#                 "./results/2023-12-04--16-33-39"  #8
#             ]

# experiments = ["./results/2023-12-02--21-25-26", #9
#                "./results/2023-12-01--09-37-34", #10
#               ]
experiments = ["./results/2023-11-08--18-12-16", #2
               "./results/2023-11-07--11-19-08"] #1
# experiments = ["./results/2023-11-13--09-52-54",
#                "./results/2023-12-06--16-14-30", # 4 seed 43
#                "./results/2023-12-07--23-47-10"] # 4 seed 44

experiments = ["./results/2023-12-05--15-58-59"]
for exp in experiments:
    f = open(os.path.join(exp, "config_options.json"))
    cfg = json.load(f)
    test_set_path = cfg["test_set_path"].split("/")[-1]

    clustal_results = os.path.join(f"sets/clustalomega_results_{test_set_path}")
    mafft_results = os.path.join(f"sets/mafft5_results_{test_set_path}")
    muscle_results = os.path.join(f"sets/muscle5_results_{test_set_path}")

    dec = Decoder(tokens_set=cfg["msa_conf"]["tokens_set"])
    examples_path = os.path.join(exp, "examples/examples.json")
    f = open(examples_path)
    loaded_examples = json.load(f)

    sp = Reward(
        method="1.0*SumOfPairs",
        reward_values={"GG": 0, "GL": 0, "LL": 1, "LDL": 0}
    )
    tc = Reward(
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

        rew_intell = sp(full_state)
        rew_intell2 = tc(full_state)

        all_rewards.append(rew_intell)
        all_rewards2.append(rew_intell2)
        last = last[1:]


    print(exp)
    print('SumOfPairs')
    compare_with(np.array(all_rewards), ["1.0*SumOfPairs"], aligner="clustal", set_name=clustal_results)
    print('TotalColumn')
    compare_with(np.array(all_rewards2), ["1.0*TotalColumn"], aligner="clustal", set_name=clustal_results)
    print('-'*50)
    compare_with(np.array(all_rewards), ["1.0*SumOfPairs"], aligner="mafft", set_name=mafft_results)
    compare_with(np.array(all_rewards2), ["1.0*TotalColumn"], aligner="mafft", set_name=mafft_results)
    print('-'*50)
    compare_with(np.array(all_rewards), ["1.0*SumOfPairs"], aligner="muscle", set_name=muscle_results)
    compare_with(np.array(all_rewards2), ["1.0*TotalColumn"], aligner="muscle", set_name=muscle_results)
    print('-'*50)
    print('-' * 50)
    print('-' * 50)

