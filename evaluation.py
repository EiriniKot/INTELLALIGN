import ray
import json
import os, time
from tqdm import tqdm
import numpy as np

from shared_storage import SharedStorage


class Evaluation:
    def __init__(self, config, shared_storage: SharedStorage):
        self.cfg = config
        self.shared_storage = shared_storage

    def start_evaluation(self):
        ray.get(self.shared_storage.set_evaluation_mode.remote(True))

    def stop_evaluation(self):
        ray.get(self.shared_storage.set_evaluation_mode.remote(False))

    def evaluate(self, n_episodes: int, validation_instances, save_results: bool = False):
        print("Perfoming evaluation..")
        mcts_objectives = []
        game_examples = []

        if n_episodes == -1:
            n_episodes = validation_instances.shape[0]
        instance_list = [(ep_n, validation_instances[ep_n], "test") for ep_n in range(n_episodes)]
        # Del to clear some ram
        del validation_instances
        ray.get(self.shared_storage.set_to_evaluate.remote(instance_list))
        eval_results = [None] * n_episodes
        with tqdm(total=n_episodes) as progress_bar:
            while None in eval_results:
                time.sleep(0.1)
                fetched_results = ray.get(self.shared_storage.fetch_evaluation_results.remote())
                for i, result in fetched_results:
                    eval_results[i] = result
                progress_bar.update(len(fetched_results))

        for i, result in enumerate(eval_results):
            mcts_objectives.append(result["objective"])
            # Create a list of dicts of init last states and final objective
            game_examples.append(
                {
                    "init_state": result["init_state"].tolist(),
                    "last": result["last"].tolist(),
                    "objective": result["objective"],  # .tolist(),
                }
            )

        objectives = np.array(mcts_objectives)

        # Save the objectives for computing margins
        if save_results:
            print("Results path save", self.cfg.results_path)
            os.makedirs(self.cfg.results_path, exist_ok=True)
            np.save(os.path.join(self.cfg.results_path, f"eval_{self.cfg.reward_method}.npy"), objectives)
            out = os.path.join(self.cfg.results_path, "examples")
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "examples.json"), "w", encoding="utf-8") as f:
                json.dump(game_examples, f, indent=2)
        # Compute some stats
        stats = {"type": "Validation", "avg_objective": objectives.mean()}
        return stats
