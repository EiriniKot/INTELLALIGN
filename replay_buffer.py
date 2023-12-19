import copy
import collections

import numpy as np
import ray
import torch
from typing import Dict, List

from gaz_singleplayer.experience_worker import EpisodeHistory
from torch.nn.utils.rnn import pad_sequence


@ray.remote
class ReplayBuffer:
    """
    Stores played episodes and generates batches for training the network.
    Runs in separate process, agents store their episodes in it asynchronously, while the
    trainer pulls batches from it.
    """

    def __init__(
        self,
        initial_checkpoint: Dict,
        config,
        shared_storage,
        prefilled_buffer: collections.deque = None,
    ):
        self.cfg = config
        self.shared_storage = shared_storage

        # copy buffer if it has been provided
        if prefilled_buffer is not None:
            self.buffer = copy.deepcopy(prefilled_buffer)
        else:
            self.buffer = collections.deque([], maxlen=self.cfg.replay_buffer_size)

        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_checkpoint["num_played_steps"]

        # total samples keeps track of number of "available" total samples in the buffer (i.e. regarding only episodes
        # in buffer
        self.total_samples = sum([len(episode_history.root_values) for episode_history in self.buffer])

        if self.total_samples != 0:
            print(
                f"Replay buffer initialized with prefilled buffer: {self.total_samples} samples ({self.num_played_games} games)"
            )

        # Set random seed
        np.random.seed(self.cfg.seed)

    def save_episodes(self, histories: List[EpisodeHistory]):
        for history in histories:
            self.save_episode(history)

    def save_episode(self, history: EpisodeHistory):
        # Store an episode in the buffer.
        # As we are using `collections.deque, older entries get thrown out of the buffer
        self.num_played_games += 1
        self.num_played_steps += len(history.actions)
        self.total_samples += len(history.actions)
        if len(self.buffer) == self.cfg.replay_buffer_size:
            self.total_samples -= len(self.buffer[0].actions)

        self.buffer.append(history)
        self.shared_storage.set_info.remote("num_played_games", self.num_played_games)
        self.shared_storage.set_info.remote("num_played_steps", self.num_played_steps)
        return self.num_played_games, self.num_played_steps, self.total_samples

    def sample_batch(self):
        observation_batch_0 = []
        observation_batch_1 = []
        observation_batch_2 = []
        value_batch = []
        policy_batch = []

        histories = np.random.choice(self.buffer, size=self.cfg.batch_size)
        for i, history in enumerate(histories):
            # From each game pick a random screenshot
            position = np.random.choice(len(history.actions))  # - 1  # we do not include the last auto-completed policy
            target_value, target_policy = self.make_target(history, position)
            state = history.observations[position]  # 3,length

            assert state.shape[0] == 3, "Error in sample batch "

            if state[0].shape[0] != target_policy.shape[0]:
                print("Function : sample_batch. state.shape[1] != target_policy.shape[1]")
                raise Exception(
                    f"Function sample_batch {i}: {state[0].shape[0]} , {target_policy.shape[0]} at position, {position}"
                )

            observation_batch_0.append(torch.from_numpy(state[0].copy()))
            observation_batch_1.append(torch.from_numpy(state[1].copy()))
            observation_batch_2.append(torch.from_numpy(state[2].copy()))
            value_batch.append(target_value)
            policy_batch.append(target_policy)

        value_batch_tensor = torch.cat(value_batch, dim=0)
        policy_batch = pad_sequence(policy_batch, batch_first=True, padding_value=-1000.0)

        observation_batch_0 = pad_sequence(
            observation_batch_0,
            batch_first=True,
            padding_value=self.cfg.msa_conf["num_tokens"],
        )
        observation_batch_1 = pad_sequence(
            observation_batch_1,
            batch_first=True,
            padding_value=self.cfg.msa_conf["max_length_per_sequence"],
        )
        observation_batch_2 = pad_sequence(
            observation_batch_2,
            batch_first=True,
            padding_value=self.cfg.msa_conf["max_number_of_sequences"],
        )
        observation_batch = (
            observation_batch_0,
            observation_batch_1,
            observation_batch_2,
        )

        return (observation_batch, value_batch_tensor, policy_batch)

    def get_length(self):
        return len(self.buffer)

    def make_target(self, history: EpisodeHistory, state_index: int):
        """
        Generates targets (value and policy) for each observation.

        Parameters
            history: Episode history
            state_index [int]: Position in episode to sample
        Returns:
            target_value: Float Tensor of shape (1, 1)
            target_policy: Tensor of shape (policy length)
        """
        target_value = torch.FloatTensor([history.objective]).unsqueeze(0) * self.cfg.value_target_scaling
        target_policy = torch.from_numpy(
            history.root_policies[state_index].copy()
        ).float()  # copy array to make it writable
        return target_value, target_policy

    def get_buffer(self):
        return self.buffer
