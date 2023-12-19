import json, os
import ray
from launcher import GAZLauncher

from gaz_singleplayer.config_msa import Config

test_20 = ["./results/2023-12-06--16-14-30", "./results/2023-12-07--23-47-10"]
# test_10 = ["./results/2023-11-11--11-23-11", #3
#             "./results/2023-11-13--09-52-54", #4
            # "./results/2023-12-05--15-58-59", #5
            # "./results/2023-11-20--16-25-54", #6
            # "./results/2023-11-23--10-26-36", #7
            # "./results/2023-12-04--16-33-39"  #8
            # ]

# test_1 = ["./results/2023-12-02--21-25-26", #9
#           "./results/2023-12-01--09-37-34", #10
#             ]


if __name__ == "__main__":
    for exp in test_20:
        print(exp)
        checkpoint_path = os.path.join(exp, "best_model.pt")
        f = open(os.path.join(exp, "config_options.json"))
        data = json.load(f)

        config = Config(
            save_model=False,
            reward_method=data["reward_method"],
            reward_values=data["reward_values"],
            exp_name=exp,
            run_id=0,
            results_path=data["results_path"],
            training_games=0,
        )
        for i, val in data["msa_conf"].items():
            config.msa_conf[i] = val
        config.checkpoint_pth = checkpoint_path
        gaz = GAZLauncher(config)
        gaz.setup_workers()
        gaz.test(-1, save_results=True)
        ray.shutdown()
