import os

import ray
import time
import json

# import mlflow
from typing import Dict
from collections import Counter


@ray.remote
class Logger:
    def __init__(self, config, shared_storage, inferencers):
        self.cfg = config
        self.shared_storage = shared_storage

        self.n_played_games = 0
        # Check number of games played before this run (if a training is resumed from some checkpoint)
        self.n_played_games_previous = ray.get(shared_storage.get_info.remote("num_played_games"))
        self.rolling_game_stats = None
        self.play_took_time = 0
        self.reset_rolling_game_stats()

        self.n_trained_steps = 0
        self.n_trained_steps_previous = ray.get(shared_storage.get_info.remote("training_step"))
        self.rolling_loss_stats = None
        self.reset_rolling_loss_stats()

        self.inferencers = inferencers

        self.file_log_path = os.path.join(self.cfg.results_path, "log.txt")

    def reset_rolling_game_stats(self):
        self.play_took_time = time.perf_counter()
        self.rolling_game_stats = {
            "max_policies_for_selected_moves": {},
            "max_search_depth": 0,
            "game_time": 0,
            "waiting_time": 0,
            "objective": 0,
            "greedy_objective": 0,
            "why_done": Counter(
                {
                    "agent_stopped_0": 0,
                    "agent_stopped_more": 0,
                    "reached_maximum_moves": 0,
                    "number_of_moves": 0,
                    "gap_at_the_end": 0,
                }
            ),
        }

        for n_actions in self.cfg.log_policies_for_moves:
            self.rolling_game_stats["max_policies_for_selected_moves"][n_actions] = 0

    def reset_rolling_loss_stats(self):
        self.rolling_loss_stats = {"loss": 0, "value_loss": 0, "policy_loss": 0}

    def played_episodes(self, stats_list):
        for stats in stats_list:
            self.played_game(stats)

    def played_game(self, game_stats: Dict):
        self.n_played_games += 1
        self.rolling_game_stats["game_time"] += game_stats["game_time"]
        self.rolling_game_stats["max_search_depth"] += game_stats["max_search_depth"]
        if "waiting_time" in game_stats:
            self.rolling_game_stats["waiting_time"] += game_stats["waiting_time"]

        self.rolling_game_stats["objective"] += game_stats["objective"]
        self.rolling_game_stats["why_done"] += game_stats["why_done"]
        try:
            for action_key in self.rolling_game_stats["max_policies_for_selected_moves"].keys():
                self.rolling_game_stats["max_policies_for_selected_moves"][action_key] += max(
                    game_stats["policies_for_selected_moves"][action_key]
                )
        except KeyError as exp:
            raise KeyError(f'{exp} {game_stats["policies_for_selected_moves"]} for {action_key}')

        if self.n_played_games % self.cfg.log_avg_stats_every_n_episodes == 0:
            games_took_time = time.perf_counter() - self.play_took_time

            print(f"Num played games : {self.n_played_games}, Time : {games_took_time:.1f} secs")

            # Get time it took for models on average
            avg_model_inference_time = 0
            avg_objective = self.rolling_game_stats["objective"] / self.cfg.log_avg_stats_every_n_episodes
            avg_max_depth = self.rolling_game_stats["max_search_depth"] / self.cfg.log_avg_stats_every_n_episodes

            # Average maximum probability for selected moves
            for n_actions in self.cfg.log_policies_for_moves:
                self.rolling_game_stats["max_policies_for_selected_moves"][
                    n_actions
                ] /= self.cfg.log_avg_stats_every_n_episodes

            avg_time_per_game = self.rolling_game_stats["game_time"] / self.cfg.log_avg_stats_every_n_episodes

            avg_reasons_for_stopping = {
                key: round(value / self.cfg.log_avg_stats_every_n_episodes, 2)
                for key, value in self.rolling_game_stats["why_done"].items()
            }
            print(f"Avg max search depth per move: {avg_max_depth:.1f}")
            print(f"Avg objective: {avg_objective:.3f}")
            print(f"Avg reasons for ending game: {avg_reasons_for_stopping}")
            # print(f'Max Policies for selected moves: {self.rolling_game_stats["max_policies_for_selected_moves"]}')

            metrics_to_log = {
                "Avg objective": avg_objective,
                "Games time in secs": games_took_time,
                "Avg game time in secs": avg_time_per_game,
                "Avg Inferencer Time in secs": avg_model_inference_time,
                "Avg max search depth per move": avg_max_depth,
                "Avg reasons for ending game": avg_reasons_for_stopping,
            }

            # mlflow.set_experiment(experiment_name=self.cfg.exp_name)
            # mlflow.set_tag("mlflow.runName", "run_name")
            # mlflow.log_metric("Num played games total", self.n_played_games)
            # mlflow.log_metric("Time", games_took_time)
            # for k, v in avg_reasons_for_stopping.items():
            #     mlflow.log_metric(f"Avg ended {k}", v)
            self.reset_rolling_game_stats()
            if self.cfg.do_log_to_file:
                # Additional things for logging to file
                metrics_to_log["Total num played games"] = self.n_played_games
                metrics_to_log["Total num trained steps"] = self.n_trained_steps
                metrics_to_log["Timestamp in ms"] = int(time.time() * 1000)
                metrics_to_log["logtype"] = "played_game"
                metrics_to_log["Avg reasons for ending game"] = avg_reasons_for_stopping
                metrics_to_log["Avg reasons for ending game"] = avg_reasons_for_stopping
                metrics_to_log["Avg game time in secs"] = avg_time_per_game

                with open(self.file_log_path, "a+") as f:
                    f.write(json.dumps(metrics_to_log))
                    f.write("\n")

    def training_step(self, loss_dict: Dict):
        """
        Notify logger of performed training step. loss_dict has keys "loss", "value_loss" and "policy_loss" (all floats)
        for a batch on which has been trained.
        """
        self.n_trained_steps += 1
        self.rolling_loss_stats["loss"] += loss_dict["loss"]
        self.rolling_loss_stats["value_loss"] += loss_dict["value_loss"]
        self.rolling_loss_stats["policy_loss"] += loss_dict["policy_loss"]

        if self.n_trained_steps % self.cfg.log_avg_loss_every_n_steps == 0:
            # Also get training_steps to played_steps ratio
            training_steps = ray.get(self.shared_storage.get_info.remote("training_step"))
            played_games = ray.get(self.shared_storage.get_info.remote("num_played_games"))
            avg_loss = self.rolling_loss_stats["loss"] / self.cfg.log_avg_loss_every_n_steps
            avg_value_loss = self.rolling_loss_stats["value_loss"] / self.cfg.log_avg_loss_every_n_steps
            avg_policy_loss = self.rolling_loss_stats["policy_loss"] / self.cfg.log_avg_loss_every_n_steps
            ratio_steps_games = training_steps / played_games

            print(
                f"Train steps: {self.n_trained_steps}, ",
                f"Ratio train steps/played games: {ratio_steps_games:.4f}, ",
                f"Avg loss: {avg_loss:.4f}, Avg value Loss: {avg_value_loss:.4f}, ",
                f"Avg policy loss: {avg_policy_loss:.4f}",
            )

            self.reset_rolling_loss_stats()

            metrics_to_log = {
                "Ratio train steps to played games": ratio_steps_games,
                "Avg loss": avg_loss,
                "Avg value loss": avg_value_loss,
                "Avg policy loss": avg_policy_loss,
            }

            if self.cfg.do_log_to_file:
                # Additional things for logging to file
                metrics_to_log["Total num played games"] = self.n_played_games
                metrics_to_log["Total num trained steps"] = self.n_trained_steps
                metrics_to_log["Timestamp in ms"] = int(time.time() * 1000)
                metrics_to_log["logtype"] = "training_step"

                with open(self.file_log_path, "a+") as f:
                    f.write(json.dumps(metrics_to_log))
                    f.write("\n")

    def evaluation_run(self, stats_dict: Dict):
        print(f"Evaluation Run , Avg objective: {stats_dict['avg_objective']}")

        if self.cfg.do_log_to_file:
            # Additional things for logging to file
            metrics_to_log = {
                "Total num played games": self.n_played_games,
                "Total num trained steps": self.n_trained_steps,
                "Timestamp in ms": int(time.time() * 1000),
                "logtype": "evaluation",
                "Evaluation Type": stats_dict["type"],
                "Evaluation MCTS": stats_dict["avg_objective"],
            }

            with open(self.file_log_path, "a+") as f:
                f.write(json.dumps(metrics_to_log))
                f.write("\n")
