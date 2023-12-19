import copy
import os
import time

import ray
import torch

from typing import Dict


@ray.remote
class SharedStorage:
    """
    Class which run in a dedicated process to store the network weights and some information about played games.
    """

    def __init__(self, checkpoint: Dict, results_path):
        self.results_path = results_path
        self.current_checkpoint = copy.deepcopy(checkpoint)

        # Variables for evaluation mode
        self.evaluate_list = []  # Stores flowrates which should be evaluated
        self.evaluation_results = []
        self.example_results = []
        self.evaluation_mode = False

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.results_path, filename)
        torch.save(self.current_checkpoint, path)

    def set_checkpoint(self, checkpoint: Dict):
        self.current_checkpoint = copy.deepcopy(checkpoint)

    def get_checkpoint(self):
        return copy.deepcopy(self.current_checkpoint)

    def get_info(self, keys):
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys}
        else:
            raise TypeError

    def set_info(self, keys, values=None):
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError

    def in_evaluation_mode(self):
        return self.evaluation_mode

    def set_evaluation_mode(self, value: bool):
        self.evaluation_mode = value

    def get_to_evaluate(self, n_items: int):
        if len(self.evaluate_list) > 0:
            items = self.evaluate_list[:n_items]
            self.evaluate_list = self.evaluate_list[n_items:]
            return [copy.deepcopy(item) for item in items]
        else:
            # print(Warning('Evaluate list is empty'))
            return None

    def set_to_evaluate(self, evaluate_list):
        self.evaluate_list = evaluate_list
        time.sleep(2)

    def push_evaluation_results(self, eval_result_list):
        self.evaluation_results = self.evaluation_results + eval_result_list

    def fetch_evaluation_results(self):
        results = self.evaluation_results
        self.evaluation_results = []
        return results
