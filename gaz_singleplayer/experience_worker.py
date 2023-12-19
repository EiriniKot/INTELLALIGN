import threading
import sys

import math, psutil, os
import torch

import numpy as np
import ray
import time

from local_inferencer import LocalInferencer
from gumbel_mcts import GumbelMCTS, GumbelNode
from shared_storage import SharedStorage
from gaz_singleplayer.msa_env import MSAEnvironment
from tools.dataset_generator import ExamplesCreator
from typing import Dict, Optional, Union, List


@ray.remote
class ExperienceWorker:
    """
    Experience Worker.
    Instances of this class run in separate processes and continuously play episodes.
    The episode history is saved to the global replay buffer, which is accessed by the training process which optimizes
    the networks.
    """

    def __init__(
        self,
        actor_id: int,
        config,
        shared_storage: SharedStorage,
        inference_device: str,
        network_class,
        random_seed: int = 42,
        cpu_core: int = None,
    ):
        self.actor_id = actor_id
        self.cfg = config
        self.ec = ExamplesCreator(
            letters_set=config.msa_conf["letters_set"],
            tokens_set=config.msa_conf["tokens_set"],
            method=config.msa_conf["method"],
            stop_move=config.msa_conf["stop_move"],
            path_set=config.msa_conf["path_set"],
            random_seed=random_seed,
        )

        if config.pin_workers_to_core and sys.platform == "linux" and cpu_core is not None:
            os.sched_setaffinity(0, {cpu_core})
            psutil.Process().cpu_affinity([cpu_core])

        if self.cfg.CUDA_VISIBLE_DEVICES:
            # override ray's limiting of GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg.CUDA_VISIBLE_DEVICES

        self.inferencer = LocalInferencer(
            config=self.cfg,
            network_class=network_class,
            shared_storage=shared_storage,
            initial_checkpoint=None,  # is set in CPU Inferencer
            device=torch.device(inference_device),
        )

        self.shared_storage = shared_storage
        self.n_games_played = 0
        self.thread_lock = threading.Lock()
        self.save_examples = False

        self.stats = {
            "objective": 0,
            "max_search_depth": 0,
            "policies_for_selected_moves": dict(
                zip(
                    self.cfg.log_policies_for_moves,
                    [[-1000.0] for _ in range(len(self.cfg.log_policies_for_moves))],
                )
            ),
            "baseline_objective": 0,
        }

        # Set the random seed for the worker
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    def play_episode(
        self,
        thread_id: int,
        env,
        results_history_list,
        results_stats_list,
        greedy: bool = False,
    ):
        """
        Runs single episode with MCTS
        """
        history = EpisodeHistory()
        history.observations.append(env.get_state_for_history())
        episode_time = time.perf_counter()  # track how long is needed for an episode
        episode_done = False
        move_counter = 0

        stats = self.stats.copy()
        self.inferencer.register_thread(thread_id)  # is thread-safe

        with torch.no_grad():
            tree = GumbelMCTS(
                thread_id=thread_id,
                config=self.cfg,
                inferencer=self.inferencer,
                deterministic=greedy,
                min_max_normalization=True,
            )

            while not episode_done:
                # run the simulation
                root, max_search_depth = tree.run_at_root(env)
                # Store maximum search depth for inspection
                stats["max_search_depth"] = max(stats["max_search_depth"], max_search_depth)
                action = root.sequential_halving_chosen_action
                # Make the chosen move
                episode_done, reward, why_done = env.transition(action)
                # store statistics in the history, as well as the next observation
                history.actions.append(action)
                history.store_gumbel_search_statistics(tree, self.cfg.gumbel_simple_loss)
                # Append next observation after transition
                history.observations.append(env.get_state_for_history())

                move_counter += 1
                if move_counter in self.cfg.log_policies_for_moves:
                    stats["policies_for_selected_moves"][move_counter] = root.children_prior_probs.copy()

                # important: shift must happen after storing search statistics
                tree.shift(action)

                if episode_done:
                    history.objective = reward
                    episode_time = time.perf_counter() - episode_time
                    stats["id"] = self.actor_id  # identify from which actor this game came from
                    stats["objective"] = history.objective
                    stats["game_time"] = episode_time
                    stats["waiting_time"] = tree.waiting_time
                    stats["why_done"] = why_done

        with self.thread_lock:
            self.n_games_played += 1

        self.inferencer.unregister_thread(thread_id)  # thread safe
        results_history_list[thread_id] = history
        results_stats_list[thread_id] = stats

    def eval_mode(self, include_episodes=False):
        """
        In evaluation mode, data to evaluate is pulled from shared storage until evaluation mode is unlocked.
        """
        while ray.get(self.shared_storage.in_evaluation_mode.remote()):
            to_evaluate_list = ray.get(
                self.shared_storage.get_to_evaluate.remote(n_items=self.cfg.max_num_threads_per_experience_worker)
            )
            if to_evaluate_list is not None:
                # We have something to evaluate
                eval_idcs, instances, _ = list(zip(*to_evaluate_list))
                envs = self.precomputed_env_from_instances(instances)
                # we will keep the evaluation examples init , last and objective here
                eval_examples = [
                    {"init_state": math.inf, "last": math.inf, "objective": math.inf} for _ in range(len(envs))
                ]
                history_list, stats_list = self.play_bunch_of_episodes(envs, greedy=True)
                # Save games initial and last sate
                for i, stats in enumerate(stats_list):
                    if include_episodes:
                        # [0][0] stands for the first episode and the first part (state_0) , the rest is the encoding
                        eval_examples[i]["init_state"] = history_list[i].observations[0][0]
                        eval_examples[i]["last"] = history_list[i].observations[-1][0]
                    eval_examples[i]["objective"] = stats["objective"]

                self.shared_storage.push_evaluation_results.remote(
                    [(eval_idcs[i], eval_examples[i]) for i, stats in enumerate(stats_list)]
                )
            else:
                time.sleep(1)

    def precomputed_env_from_instances(self, instances: Union[List[np.array], np.array]):
        """
        Initialize one environment for each instance..
        Return a list of those envs
        """
        return_list = []
        n = instances.shape[0] if type(instances) == np.ndarray else len(instances)
        for i in range(n):
            env = MSAEnvironment(
                initial_state=instances[i],
                tokens_set=self.cfg.msa_conf["tokens_set"],
                reward_as_difference=self.cfg.msa_conf["reward_as_difference"],
                reward=self.cfg.msa_conf["reward"],
                stop_move=self.cfg.msa_conf["stop_move"],
                steps_ratio=self.cfg.msa_conf["steps_ratio"],
                complete_column_gaps=self.cfg.msa_conf["complete_column_gaps"],
            )
            return_list.append(env)
        return return_list

    def play_bunch_of_episodes(self, envs: List, greedy=False):
        n = len(envs)
        history_list = [None] * n
        stats_list = [None] * n
        threads = []
        for i in range(n):
            env = envs[i]
            # print(f'Create thread {i} for episode')
            t = threading.Thread(
                target=self.play_episode,
                args=(i, env, history_list, stats_list, greedy),
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return history_list, stats_list

    def continuous_play(self, replay_buffer, logger=None):
        # Start the thread which continuously performs inference
        inference_thread = threading.Thread(target=self.inferencer.continuous_inference)
        inference_thread.start()
        while not ray.get(self.shared_storage.get_info.remote("terminate")):
            if ray.get(self.shared_storage.in_evaluation_mode.remote()):
                self.eval_mode(include_episodes=True)

            random_instances = self.ec.generate_random_instance(
                n_instances=self.cfg.max_num_threads_per_experience_worker,
                num_sequences_range=self.cfg.msa_conf["num_sequences_range"],
                length_range=self.cfg.msa_conf["length_range"],
                ratio_sequences_real=self.cfg.msa_conf["ratio_sequences_real"],
            )
            histories, stats = self.play_bunch_of_episodes(
                envs=self.precomputed_env_from_instances(random_instances),
                greedy=False,
            )
            # save game to the replay buffer and notify logger
            replay_buffer.save_episodes.remote(histories)
            if logger is not None:
                logger.played_episodes.remote(stats)

            if self.cfg.ratio_range:
                infos: Dict = ray.get(
                    self.shared_storage.get_info.remote(["training_step", "num_played_games", "terminate"])
                )
                num_played_games = infos["num_played_games"]
                num_games_in_replay_buffer = ray.get(replay_buffer.get_length.remote())
                ratio = infos["training_step"] / max(1, num_played_games - self.cfg.start_train_after_episodes)

                while (
                    ratio < self.cfg.ratio_range[0]
                    and num_games_in_replay_buffer > self.cfg.start_train_after_episodes
                    and not infos["terminate"]
                    and not ray.get(self.shared_storage.in_evaluation_mode.remote())
                ):
                    infos: Dict = ray.get(
                        self.shared_storage.get_info.remote(["training_step", "num_played_games", "terminate"])
                    )
                    num_games_in_replay_buffer = ray.get(replay_buffer.get_length.remote())
                    ratio = infos["training_step"] / max(
                        1,
                        infos["num_played_games"] - self.cfg.start_train_after_episodes,
                    )
                    time.sleep(0.010)  # wait for 10ms
        self.inferencer.terminate = True

    def select_action_from_priors(self, node: GumbelNode, deterministic: bool) -> int:
        """
        Select discrete action according to visit count distribution or directly from the node's prior probabilities.
        """
        priors = node.children_prior_probs[node.feasible_actions]
        if deterministic:
            return node.feasible_actions[np.argmax(priors)]
        return np.random.choice(node.feasible_actions, p=priors)


class EpisodeHistory:
    """
    Stores information about the moves in a game.
    When its filled observations will be 1 element more than root_policies
    """

    def __init__(self):
        # Observation is a np.array containing the state of the env of the current player
        self.observations = []
        # i-th entry corresponds to the action the player took in i-th state.
        self.actions = []
        # stores the action policy of the root node at i-th observation after the tree search, depending on visit
        # counts of children. Each element is a list of length number of actions on level, and sums to 1.
        self.root_policies = []
        self.objective: Optional[float] = None  # stores the final objective

    def store_gumbel_search_statistics(self, mcts: GumbelMCTS, for_simple_loss: bool = False):
        """
        Stores the improved policy of the root node.
        """
        root = mcts.root
        if for_simple_loss:
            # simple loss is where we assign probability one to the chosen action
            action = root.sequential_halving_chosen_action
            policy = np.zeros_like(root.children_prior_probs)
            policy[action] = 1.0
        else:
            policy = mcts.get_improved_policy(root)
        self.root_policies.append(policy)
        return policy
