import os, json
import numpy as np
from gaz_singleplayer.config_msa import Config
from gaz_singleplayer.experience_worker import ExperienceWorker
from gaz_singleplayer.msa_env import MSAEnvironment
import threading
from shared_storage import SharedStorage
from model.msa_network import StackGlimbseModel

test_experiment_3 = "./results/2023-10-17--16-26-23"
test_all = [test_experiment_3]

if __name__ == "__main__":
    for exp in test_all:
        print(exp)
        checkpoint_path = os.path.join(exp, "best_model.pt")
        f = open(os.path.join(exp, "config_options.json"))
        data = json.load(f)

        config = Config(
            save_model=False,
            reward_method="0.7*AffineSumOfPairs_scaled+0.3*TotalColumn_scaled",
            exp_name=exp,
            run_id=0,
            results_path=data["results_path"],
            training_games=0,
        )

        config.num_experience_workers = 2
        for i, val in data["msa_conf"].items():
            config.msa_conf[i] = val

        config.checkpoint_pth = checkpoint_path
        # gaz = GAZLauncher(config)
        validation_instances = np.load(config.test_set_path, allow_pickle=True)
        instance_list = [(ep_n, validation_instances[ep_n], "test") for ep_n in range(2)]
        threads = []
        for i in range(2):
            shared_storage = SharedStorage.remote(config.checkpoint_pth, config.results_path)
            exp = ExperienceWorker.remote(
                actor_id=i,
                config=config,
                shared_storage=shared_storage,
                inference_device="cpu",
                network_class=StackGlimbseModel,
                random_seed=config.seed + i,
                cpu_core="cpu",
            )
            eval_idcs, instances, _ = list(zip(*instance_list))
            env = MSAEnvironment(
                initial_state=instances[i],
                tokens_set=config.msa_conf["tokens_set"],
                reward_as_difference=config.msa_conf["reward_as_difference"],
                reward=config.msa_conf["reward"],
                stop_move=config.msa_conf["stop_move"],
                steps_ratio=config.msa_conf["steps_ratio"],
                complete_column_gaps=config.msa_conf["complete_column_gaps"],
            )
            history_list = [None]
            stats_list = [None]
            t = threading.Thread(
                target=exp.play_episode.remote,
                args=(i, env, history_list, stats_list, True),
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
