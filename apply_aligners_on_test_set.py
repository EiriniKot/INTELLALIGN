import os
import json
from os.path import dirname, abspath
import numpy as np

from tools.aligners import Aligners
from tools.dataset_generator import Decoder
from gaz_singleplayer.config_msa import Config
from tools.reward_functions import Reward
from tools.dataset_generator import ExamplesCreator

d = dirname(dirname(abspath(__file__)))
aligners = Aligners(
    methods_executables=[
        ("MafftApp", None),
        ("MuscleApp", '/home/eirini/Downloads/muscle/muscle3.8.31_i86linux64'),
        ("ClustalOmegaApp", None),
    ]
)
cfg = Config(save_model=False, cost_values={"GOC": 1, "GEC": 0.1}, reward_method="AffineSumOfPairs+20*TotalColumn")

if __name__ == "__main__":
    filename = "8f7c1202-203f-42b4-ac70-4077fa264774_test_512_numseq_(4, 5)_lenseq(10, 11).npy"
    path_to_load = os.path.join(d, f"GAZ_MSA/sets/{filename}")
    print(f"Path to load set {path_to_load}")
    test_set = np.load(path_to_load, allow_pickle=True)

    dec = Decoder(tokens_set=cfg.msa_conf["tokens_set"])
    # reward = cfg.msa_conf["reward"]
    reward_sum = Reward(
        tokens_set=cfg.msa_conf["tokens_set"],
        reward_values={"GG": 0, "GL": 0, "LL": 1, "LDL": 0},
        method="1.0*SumOfPairs",
    )
    reward_tc = Reward(
        tokens_set=cfg.msa_conf["tokens_set"],
        method="1.0*TotalColumn",
    )
    reward_from_training = Reward(
        tokens_set=cfg.msa_conf["tokens_set"],
        reward_values={"GG": 0, "GL": 0, "LL": 1, "LDL": 0},
        cost_values=cfg.msa_conf["reward"].cost_values,
        method=cfg.msa_conf["reward"].method,
    )

    test_set = dec(test_set)
    # Apply aligners into the set
    result = list(map(aligners, test_set))
    mafft5, muscle5, clustalomega = list(zip(*result))

    ex_creator = ExamplesCreator(
        letters_set=cfg.msa_conf["letters_set"],
        tokens_set=cfg.msa_conf["tokens_set"],
        stop_move=cfg.msa_conf["stop_move"],
        method=cfg.msa_conf["method"],
        path_set=cfg.msa_conf["path_set"],
    )

    mafft5_dict = []
    muscle5_dict = []
    clustalomega_dict = []
    tt = 0
    for indx, (msa, time) in enumerate(mafft5):
        post_msa = ex_creator.post_process(msa)
        rewardSumOfPairs = reward_sum(post_msa)
        rewardTotalColumn = reward_tc(post_msa)
        reward_result = reward_from_training(post_msa)
        mafft5_dict.append(
            {
                "initial_msa": test_set[indx],
                "time": time,
                cfg.msa_conf["reward"].method: reward_result,
                "1.0*SumOfPairs": rewardSumOfPairs,
                "1.0*TotalColumn": rewardTotalColumn,
                "final_msa": msa,
            }
        )
        tt += rewardSumOfPairs

    with open(f"sets/mafft5_results_{filename}", "w") as file:
        file.write(json.dumps(mafft5_dict, indent=4))
    print('Sum of rewardSumOfPairs', tt)

    for indx, (msa, time) in enumerate(muscle5):
        post_msa = ex_creator.post_process(msa)
        rewardSumOfPairs = reward_sum(post_msa)
        rewardTotalColumn = reward_tc(post_msa)
        reward_result = reward_from_training(post_msa)
        muscle5_dict.append(
            {
                "initial_msa": test_set[indx],
                "time": time,
                cfg.msa_conf["reward"].method: reward_result,
                "1.0*SumOfPairs": rewardSumOfPairs,
                "1.0*TotalColumn": rewardTotalColumn,
                "final_msa": msa,
            }
        )

    with open(f"sets/muscle5_results_{filename}", "w") as file:
        file.write(json.dumps(muscle5_dict, indent=4))

    for indx, (msa, time) in enumerate(clustalomega):
        post_msa = ex_creator.post_process(msa)
        rewardSumOfPairs = reward_sum(post_msa)
        rewardTotalColumn = reward_tc(post_msa)
        reward_result = reward_from_training(post_msa)
        clustalomega_dict.append(
            {
                "initial_msa": test_set[indx],
                "time": time,
                cfg.msa_conf["reward"].method: reward_result,
                "1.0*SumOfPairs": rewardSumOfPairs,
                "1.0*TotalColumn": rewardTotalColumn,
                "final_msa": msa,
            }
        )

    with open(f"sets/clustalomega_results_{filename}", "w") as file:
        file.write(json.dumps(clustalomega_dict, indent=4))
