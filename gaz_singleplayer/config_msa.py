import os
import datetime
import json
import copy

# import mlflow
from tools.reward_functions import Reward


class Config:
    """
    MSA
    """

    # 10 512 200 True False 0 0 0.4 0.9 SEED
    def __init__(
        self,
        save_model=True,
        reward_method="0.9*SumOfPairs+0.1*TotalColumn",
        reward_values={"GG": 0, "GL": 0, "LL": 1, "LDL": 0},
        cost_values={"GOC": 1, "GEC": 0.1},
        results_path=None,
        exp_name="0",
        run_id=None,
        training_games=0,
    ):
        super().__init__()
        tokens_set = ["STOP", "A", "C", "G", "T", "END", "-"]
        self.reward_method = reward_method
        self.reward_values = reward_values
        self.cost_values = cost_values
        reward_func = Reward(
            method=self.reward_method,
            tokens_set=tokens_set,
            reward_values=reward_values,
            cost_values=cost_values,
            # MAFFT paper page 3/Scoring system section
        )

        self.exp_name = exp_name
        self.run_id = run_id
        self.msa_conf = {
            "num_tokens": len(tokens_set),
            "max_length_per_sequence": 65,
            "max_number_of_sequences": 8,
            "emb_depth": 256,
            "encoder_hyper": {"trf_blocks": 5, "trf_heads": 8},
            "trf_heads_pol": 8,
            "trf_heads_val": 8,
            "pad_policy": -10000.0,
            "letters_set": ["A", "C", "G", "T"],
            "tokens_set": tokens_set,
            "num_sequences_range": (4, 5),
            "complete_column_gaps": False,
            "length_range": (
                10,
                11,
            ),  # Currently there are no sequences of length <6 in my library
            "method": "ranged",
            "positional_embeddings": True,  # we can keep only token embedding layer if we want
            "reward": reward_func,
            "reward_as_difference": False,
            "stop_move": True,
            "ratio_sequences_real": 1,  # 80 percent of sequences will be real from genbank
            "steps_ratio": 0.4,
            "path_set": "dataset_dna_sequences/output.csv",
        }

        # Gumbel AlphaZero specific parameters
        self.value_target_scaling = 1.0
        self.gumbel_sample_n_actions = (
            16  # (Max) number of actions to sample at the root for the sequential halving procedure
        )
        self.gumbel_c_visit = 50.0  # constant c_visit in sigma-function.
        self.gumbel_c_scale = 1.0  # constant c_scale in sigma-function.
        self.gumbel_simple_loss = (
            False  # If True, KL divergence is minimized w.r.t. one-hot-encoded move, and not improved policy
        )
        self.gumbel_intree_sampling = False  # If True, not the deterministic action selection, but simple sampling from the improved policy is performed.
        self.num_simulations = 200  # Number of search simulations in GAZ's tree search
        self.seed = 43  # Random seed for torch, numpy, initial states.

        # --- Inferencer and experience generation options --- #
        self.num_experience_workers = 47  # Number of actors (processes) which generate experience.
        self.max_num_threads_per_experience_worker: int = 2
        self.check_for_new_model_every_n_seconds = 30  # Check the storage for a new model every n seconds

        self.pin_workers_to_core = True  # If True, workers are pinned to specific CPU cores, starting to count from 0.
        self.CUDA_VISIBLE_DEVICES = "0"
        self.cuda_device = "cuda:0"
        cuda_workers = 1
        self.cuda_devices_for_inferencers = ["cpu"] * (self.num_experience_workers - cuda_workers) + [
            "cuda:0",
            # "cuda:1",
        ]

        # Number of most recent games to store in the replay buffer
        self.replay_buffer_size = 1500

        # --- Training / Optimizer specifics --- #
        # Tries to keep the ratio of training steps to number of episodes within the given range.
        self.ratio_range = [2, 5]
        # Total number of batches to train the network on
        self.training_games: int = training_games
        self.start_train_after_episodes: int = 256

        assert (
            self.replay_buffer_size > self.start_train_after_episodes
        ), "Buffer should be bigger than start_train_after_episodes continuous_update_weights line 86"

        self.batch_size = 512  # for training
        self.inference_max_batch_size = 256  # for inferencing on experience workers
        self.lr_init = 0.0001  # Initial learning rate
        self.weight_decay = 1e-4  # L2-regularization for adam

        self.gradient_clipping = 1  # Clip gradient to given L2-norm. Set to 0 if no clipping should be performed.
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 300000

        self.value_loss_weight = 1  # Linear scale of value loss
        self.checkpoint_interval = 1  # Number of training steps before using the model for generating experience.

        self.results_path = (
            results_path
            if results_path
            else os.path.join("./results", datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
        )

        self.checkpoint_pth = "./results/2023-12-05--15-58-59/best_model.pt"
        self.only_load_model_weights = False  # If True, only the model weights are loaded from `checkpoint_pth`

        # --- Logging --- #
        self.log_avg_stats_every_n_episodes = 128  # Compute average game statistics over last n games and log them
        self.log_avg_loss_every_n_steps = 128  # Compute average loss over last n training steps and log them
        self.log_policies_for_moves = [
            2,
            5,
        ]  # Logs probability distributions for numbered moves
        self.do_log_to_file = True

        # --- Evaluation --- #
        self.num_evaluation_games = 256  # For each validation run, how many instances should be solved in the
        # of the validation set (taken from the start)
        self.validation_set_path = (
            "./sets/aeaf818e-a0fb-450b-b9e1-aba8f000595b_valid_512_numseq_(4, 5)_lenseq(10, 11).npy"
        )
        self.test_set_path = "./sets/8f7c1202-203f-42b4-ac70-4077fa264774_test_512_numseq_(4, 5)_lenseq(10, 11).npy"
        self.evaluate_every_n_steps = 600

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


# Saved ./sets/59d4658f-8cb5-48e6-9ec9-656d4274ac5b ./sets/edfa202d-b8e5-4140-bab3-a6e139070d6b
