import os
import datetime
import json, copy
from tools.reward_functions import Reward


class Config:
    """
    MSA
    """

    def __init__(
        self,
        save_model=True,
        reward_method="0.9*SumOfPairs+0.1*TotalColumn",
        results_path=None,
        exp_name="0",
        run_id=None,
        training_games=0,
    ):
        super().__init__()

        tokens_set = [
            "STOP",
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
            "END",
            "-",
        ]

        self.reward_method = reward_method
        self.cost_values = {"GOC": 1, "GEC": 0.8}
        # print(reward_method)
        reward_func = Reward(
            method=self.reward_method,
            tokens_set=tokens_set,
            reward_values={"GG": 0, "GL": -1, "LL": 2, "LDL": -2},
            cost_values=self.cost_values,
        )

        self.exp_name = exp_name
        self.run_id = run_id
        self.msa_conf = {
            "num_tokens": len(tokens_set),
            "max_length_per_sequence": 200,
            "max_number_of_sequences": 8,
            "emb_depth": 138,
            "encoder_hyper": {"trf_blocks": 4, "trf_heads": 6},
            "trf_heads_pol": 6,
            "trf_heads_val": 6,
            "pad_policy": -10000.0,
            "letters_set": [
                "A",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "K",
                "L",
                "M",
                "N",
                "P",
                "Q",
                "R",
                "S",
                "T",
                "V",
                "W",
                "Y",
            ],
            "tokens_set": tokens_set,
            "num_sequences_range": (4, 5),
            "length_range": (7, 12),  # Currently there are no sequences of length <6 in my library
            "method": "ranged",
            "reward": reward_func,
            "reward_as_difference": False,
            "positional_embeddings": True,  # we can keep only token embedding layer if we want
            "stop_move": True,
            "ratio_sequences_real": 0.8,  # 70 percent of sequences will be real from genbank
            "steps_ratio": 0.7,
            "path_set": "dataset_protein_sequences/output.csv",
        }

        # Gumbel AlphaZero specific parameters
        self.value_target_scaling = 1.0
        self.gumbel_sample_n_actions = (
            30  # (Max) number of actions to sample at the root for the sequential halving procedure
        )
        self.gumbel_c_visit = 50.0  # constant c_visit in sigma-function.
        self.gumbel_c_scale = 1.0  # constant c_scale in sigma-function.
        self.gumbel_simple_loss = (
            False  # If True, KL divergence is minimized w.r.t. one-hot-encoded move, and not improved policy
        )
        self.gumbel_intree_sampling = False  # If True, not the deterministic action selection, but simple sampling from the improved policy is performed.
        self.num_simulations = 200  # Number of search simulations in GAZ's tree search
        self.seed = 42  # Random seed for torch, numpy, initial states.

        # --- Inferencer and experience generation options --- #
        self.num_experience_workers = 26  # Number of actors (processes) which generate experience.
        self.max_num_threads_per_experience_worker: int = 3
        self.check_for_new_model_every_n_seconds = 240  # Check the storage for a new model every n seconds

        self.pin_workers_to_core = True  # If True, workers are pinned to specific CPU cores, starting to count from 0.
        self.CUDA_VISIBLE_DEVICES = "0"
        self.cuda_device = "cuda:0"
        cuda_workers = 1
        self.cuda_devices_for_inferencers = ["cpu"] * (self.num_experience_workers - cuda_workers) + [
            self.cuda_device
        ] * cuda_workers
        # Number of most recent games to store in the replay buffer
        self.replay_buffer_size = 1000

        # --- Training / Optimizer specifics --- #
        # Tries to keep the ratio of training steps to number of episodes within the given range.
        self.ratio_range = [10, 20]
        # Total number of batches to train the network on
        self.training_games: int = training_games
        self.start_train_after_episodes: int = 300

        assert (
            self.replay_buffer_size > self.start_train_after_episodes
        ), "Buffer should be bigger than start_train_after_episodes continuous_update_weights line 86"

        self.batch_size = 386  # for training
        self.inference_max_batch_size = 118  # for inferencing on experience workers
        self.lr_init = 0.0001  # Initial learning rate
        self.weight_decay = 1e-4  # L2-regularization for adam

        self.gradient_clipping = 1  # Clip gradient to given L2-norm. Set to 0 if no clipping should be performed.
        self.lr_decay_rate = 0.95  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = (
            350e3  # means that after `decay_steps` training steps, the learning rate has decayed by `decay_rate`
        )

        self.value_loss_weight = 1.0  # Linear scale of value loss
        self.checkpoint_interval = 60  # Number of training steps before using the model for generating experience.

        self.results_path = (
            results_path
            if results_path
            else os.path.join("./results", datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
        )

        self.checkpoint_pth = None
        self.only_load_model_weights = False  # If True, only the model weights are loaded from `checkpoint_pth`

        # --- Logging --- #
        self.log_avg_stats_every_n_episodes = 80  # Compute average game statistics over last n games and log them
        self.log_avg_loss_every_n_steps = 80  # Compute average loss over last n training steps and log them
        self.log_policies_for_moves = [
            2,
            5,
        ]  # Logs probability distributions for numbered moves
        self.do_log_to_file = True
        # --- Evaluation --- #
        self.num_evaluation_games = 220  # For each validation run, how many instances should be solved in the
        # of the validation set (taken from the start)
        self.validation_set_path = (
            "./sets/7ea2f4c6-acb8-4e4a-a78a-149731f2c326_valid_protein_220_numseq_(4, 6)_lenseq(7, 13).npy"
        )
        self.test_set_path = (
            "./sets/c3db57b7-aa99-4b1e-8f14-4f4283288d37_test_protein_300_numseq_(4, 6)_lenseq(7, 13).npy"
        )
        self.evaluate_every_n_steps = 400
        self.save_model = save_model

        if self.save_model:
            os.makedirs(self.results_path, exist_ok=True)
            config_save_path = os.path.join(self.results_path, "config_options.json")
            self.save_to_file(config_save_path)

    def save_to_file(self, file_path):
        # Convert instance attributes to a dictionary
        data = vars(self)
        data2 = copy.deepcopy(data)
        # Write the dictionary to a JSON file
        data2["msa_conf"].pop("reward")
        with open(file_path, "w") as file:
            json.dump(data2, file)
