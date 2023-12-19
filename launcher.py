import os, time, json

import numpy as np
import torch
import ray

from model.network_trainer import NetworkTrainer
from replay_buffer import ReplayBuffer
from logger import Logger
from evaluation import Evaluation
from shared_storage import SharedStorage
from model.msa_network import StackGlimbseModel
from gaz_singleplayer.experience_worker import ExperienceWorker
from tools.model_tools import set_checkpoint
from tools.generic_tools import compare_with


class GAZLauncher:
    """
    Main class which builds up everything and spawns training
    """

    def __init__(self, config):
        """
        Parameters:
            config: Config object
            network_class: Problem specific network module in "model"
        """
        self.cfg = config

        # Fix random number generator seed
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

        cuda_avail = torch.cuda.is_available()
        # Get devices and number of processes which need gpus
        self.gpu_access = dict()
        self.training_device = torch.device(self.cfg.cuda_device if self.cfg.cuda_device and cuda_avail else "cpu")
        if cuda_avail:
            if self.cfg.cuda_device:
                self.gpu_access[self.cfg.cuda_device] = 1
            for inference_device in self.cfg.cuda_devices_for_inferencers:
                if inference_device and inference_device != "cpu":
                    if not inference_device in self.gpu_access:
                        self.gpu_access[inference_device] = 0
                    self.gpu_access[inference_device] += 1

        print(f"{len(self.gpu_access.keys())} GPU devices are accessed by number of processes: {self.gpu_access}")
        context = ray.init(num_gpus=len(self.gpu_access.keys()))
        # ray.init(num_gpus=len(self.gpu_access.keys()), local_mode=True)
        # print("Dashboard", context.dashboard_url)
        print("Ray initialized")
        self.checkpoint = set_checkpoint(
            checkpoint_pth=self.cfg.checkpoint_pth,
            only_load_model_weights=self.cfg.only_load_model_weights,
        )
        self.model_inference_workers = None
        self.loaded = False

    def setup_workers(self):
        """
        Sets up all workers except the training worker.
        """
        core = 0  # CPU which is passed to each worker so that they can pin themselves to core
        self.shared_storage_worker = SharedStorage.remote(self.checkpoint, self.cfg.results_path)
        self.shared_storage_worker.set_info.remote("terminate", False)
        self.replay_buffer_worker = ReplayBuffer.remote(self.checkpoint, self.cfg, self.shared_storage_worker)
        if self.cfg.pin_workers_to_core:
            print("Experience workers are pinned to CPU cores.")

        self.exp_workers = []

        affinity = list(os.sched_getaffinity(0))
        workers = affinity[: self.cfg.num_experience_workers if len(affinity) > self.cfg.num_experience_workers else -1]
        print("Experience workers are : ", workers)
        for index, name in enumerate(workers):
            # get the correct inference worker.
            inference_device = "cpu"
            if (
                self.cfg.cuda_devices_for_inferencers is not None
                and len(self.cfg.cuda_devices_for_inferencers) == self.cfg.num_experience_workers
            ):
                inference_device = self.cfg.cuda_devices_for_inferencers[index] if torch.cuda.is_available() else "cpu"
            self.exp_workers.append(
                ExperienceWorker.options(name=f"experience_worker_{index}", max_concurrency=2).remote(
                    actor_id=name,
                    config=self.cfg,
                    shared_storage=self.shared_storage_worker,
                    inference_device=inference_device,
                    network_class=StackGlimbseModel,
                    random_seed=self.cfg.seed + name,
                    cpu_core=name,
                )
            )

        self.logging_worker = Logger.remote(
            self.cfg,
            self.shared_storage_worker,
            self.model_inference_workers,
        )

    def train(self):
        """
        Spawn ray workers, load models, launch training
        """
        training_gpu_share = (
            1 / self.gpu_access[self.cfg.cuda_device] if torch.cuda.is_available() and self.cfg.cuda_device else 0
        )

        self.training_net_worker = NetworkTrainer.options(num_cpus=1, num_gpus=training_gpu_share).remote(
            self.cfg,
            self.shared_storage_worker,
            StackGlimbseModel,
            self.checkpoint,
            self.training_device,
        )

        for experience_worker in self.exp_workers:
            experience_worker.continuous_play.remote(self.replay_buffer_worker, self.logging_worker)

        self.training_net_worker.continuous_update_weights.remote(self.replay_buffer_worker, self.logging_worker)

        # Loop to check if we are done with training and evaluation
        last_evaluation_at_step = self.checkpoint["training_step"]
        while ray.get(self.shared_storage_worker.get_info.remote("num_played_games")) < self.cfg.training_games:
            # check if we need to evaluate
            training_step = ray.get(self.shared_storage_worker.get_info.remote("training_step"))
            if training_step - last_evaluation_at_step >= self.cfg.evaluate_every_n_steps:
                # otherwise evaluate
                self.perform_evaluation(
                    n_episodes=self.cfg.num_evaluation_games,
                    set_path=self.cfg.validation_set_path,
                    save_model=self.cfg.save_model,
                    save_results=False,
                )
                last_evaluation_at_step = training_step
            time.sleep(2)

        print("Done Training. Evaluating last model.")
        self.perform_evaluation(
            n_episodes=self.cfg.num_evaluation_games,
            set_path=self.cfg.validation_set_path,
            save_model=self.cfg.save_model,
            save_results=False,
        )

        if self.cfg.save_model:
            path = os.path.join(self.cfg.results_path, "best_model.pt")
            self.checkpoint = torch.load(path)
            ray.get(self.shared_storage_worker.set_checkpoint.remote(self.checkpoint))

        model_type = "best" if self.cfg.save_model else "last"
        print(f"Evaluating {model_type} model on test set...")
        # wait until the best model has propagated
        self.perform_evaluation(n_episodes=-1, set_path=self.cfg.test_set_path, save_model=False, save_results=False)
        self.terminate_workers()

    def test(self, n_episodes=-1, save_results=True):
        print("Testing model")
        ray.get(self.shared_storage_worker.set_evaluation_mode.remote(True))

        ## Launch all the workers
        for experience_worker in self.exp_workers:
            experience_worker.continuous_play.remote(self.replay_buffer_worker, self.logging_worker)

        if self.model_inference_workers is not None:
            for model_inference_worker in self.model_inference_workers:
                model_inference_worker.continuous_inference.remote()

        # in test we save results
        self.perform_evaluation(
            n_episodes=n_episodes, set_path=self.cfg.test_set_path, save_model=False, save_results=True
        )


    def perform_evaluation(self, n_episodes: int, set_path: str, save_model=True, save_results=False):
        """
        Performs evaluation on a n_episodes number

        """
        evaluator = Evaluation(self.cfg, self.shared_storage_worker)
        evaluator.start_evaluation()
        validation_instances = np.load(set_path, allow_pickle=True)
        print(f"Loaded {len(validation_instances)} samples from validation file {set_path}")
        stats = evaluator.evaluate(
            n_episodes=n_episodes, validation_instances=validation_instances, save_results=save_results
        )
        stats["n_games"] = ray.get(self.shared_storage_worker.get_info.remote("num_played_games"))
        ray.get(self.logging_worker.evaluation_run.remote(stats))

        if save_model:
            print(
                f"""Current objective {stats['avg_objective']}
            best objective {ray.get(self.shared_storage_worker.get_info.remote('best_eval_score'))}"""
            )
            if stats["avg_objective"] > ray.get(self.shared_storage_worker.get_info.remote("best_eval_score")):
                print("Saving as best model...")
                ray.get(self.shared_storage_worker.set_info.remote("best_eval_score", stats["avg_objective"]))
                # Save the current model as best model
                ray.get(self.shared_storage_worker.save_checkpoint.remote("best_model.pt"))
            # Save the current model as last model
            ray.get(self.shared_storage_worker.save_checkpoint.remote("last_model.pt"))

        evaluator.stop_evaluation()
        return stats

    def terminate_workers(self):
        """
        Softly terminate workers and garbage collect them.
        Also update self.checkpoint by last checkpoint of shared storage.
        """
        if self.shared_storage_worker:
            self.shared_storage_worker.set_info.remote("terminate", True)
            # get last checkpoint
            self.checkpoint = ray.get(self.shared_storage_worker.get_checkpoint.remote())
        if self.replay_buffer_worker:
            self.replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())

        print("\nShutting down workers...")
        self.experience_workers = None
        self.training_net_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None
        self.logging_worker = None
        self.model_inference_workers = None
