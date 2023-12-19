import torch
import os
import numpy as np
import ray
import time
import copy

# import mlflow
from typing import Dict
from shared_storage import SharedStorage
from tools.generic_tools import dict_to_cpu


@ray.remote
class NetworkTrainer:
    """
    One instance of this class runs in a separate process, continuously training the policy/value-network using the
    experience sampled from the playing actors and saving the weights in the shared storage.
    """

    def __init__(
        self,
        config,
        shared_storage: SharedStorage,
        network_class,
        initial_checkpoint: Dict = None,
        device: torch.device = None,
    ):
        self.cfg = config
        self.shared_storage = shared_storage
        self.device = device

        if self.cfg.CUDA_VISIBLE_DEVICES:
            # override ray's limiting of GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg.CUDA_VISIBLE_DEVICES

        # Fix random generator seed
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

        # Initialize the network
        self.model = network_class(
            num_tokens=config.msa_conf["num_tokens"],
            max_length_per_sequence=config.msa_conf["max_length_per_sequence"],
            max_number_of_sequences=config.msa_conf["max_number_of_sequences"],
            trf_heads_pol=config.msa_conf["trf_heads_pol"],
            trf_heads_val=config.msa_conf["trf_heads_val"],
            encoder_hyper=config.msa_conf["encoder_hyper"],
            emb_depth=config.msa_conf["emb_depth"],
            positional_embeddings=config.msa_conf["positional_embeddings"],
            device=self.device,
        )

        if initial_checkpoint["weights"] is not None:
            self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
        else:
            # If we do not have an initial checkpoint, we set the random weights both to 'newcomer' and 'best'.
            print("Setting random weights to model...")
            self.shared_storage.set_info.remote(
                {
                    "weights_timestamp": round(time.time() * 1000),
                    "weights": copy.deepcopy(self.model.get_weights()),
                }
            )
        self.model.to(self.device)
        self.model.train()
        self.training_step = initial_checkpoint["training_step"]

        if "cuda" not in str(next(self.model.parameters()).device):
            print("NOTE: You are not training on GPU.\n")

        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr_init,
            weight_decay=self.cfg.weight_decay,
        )

        # Load optimizer state if available
        if initial_checkpoint["optimizer_state"]:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(copy.deepcopy(initial_checkpoint["optimizer_state"]))
        print("Successfully set up network trainer")

    def continuous_update_weights(self, replay_buffer, logger=None):
        """
        Continuously samples batches from the replay buffer, make an optimization step, repeat...
        """

        # Wait for replay buffer to contain at least a certain number of episodes.
        while ray.get(replay_buffer.get_length.remote()) < max(1, self.cfg.start_train_after_episodes):
            time.sleep(1)

        next_batch = replay_buffer.sample_batch.remote()

        # Main training loopplay_bunch_of_episodes
        while not ray.get(self.shared_storage.get_info.remote("terminate")):
            # If we should pause, sleep for a while and then continue
            if ray.get(self.shared_storage.in_evaluation_mode.remote()):
                time.sleep(1)
                continue
            batch = ray.get(next_batch)
            # already prepare next batch in the replay buffer worker, so we minimize waiting times
            next_batch = replay_buffer.sample_batch.remote()
            # perform exponential learning rate decay based on training steps. See config for more info
            self.update_lr()
            self.training_step += 1

            # loss for this batch
            policy_loss, value_loss = self.get_loss(batch)

            # combine policy and value loss
            loss = policy_loss + self.cfg.value_loss_weight * value_loss
            loss = loss.mean()

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()

            if self.cfg.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.cfg.gradient_clipping,
                )

            self.optimizer.step()

            # Save model to shared storage so it can be used by the actors
            if self.training_step % self.cfg.checkpoint_interval == 0:
                self.shared_storage.set_info.remote(
                    {
                        "weights_timestamp": round(time.time() * 1000),
                        "weights": copy.deepcopy(self.model.get_weights()),
                        "optimizer_state": copy.deepcopy(dict_to_cpu(self.optimizer.state_dict())),
                    }
                )

            # Send results to logger
            if logger is not None:
                logger.training_step.remote(
                    {
                        "loss": loss.item(),
                        "value_loss": value_loss.mean().item(),
                        "policy_loss": policy_loss.mean().item(),
                    }
                )

            # Inform shared storage of training step.
            self.shared_storage.set_info.remote({"training_step": self.training_step})

            # Managing the episode / training ratio
            if self.cfg.ratio_range:
                infos: Dict = ray.get(
                    self.shared_storage.get_info.remote(["training_step", "num_played_games", "terminate"])
                )
                ratio = infos["training_step"] / max(
                    1,
                    infos["num_played_games"] - self.cfg.start_train_after_episodes,
                )

                while (
                    ratio > self.cfg.ratio_range[1]
                    and not infos["terminate"]
                    and not ray.get(self.shared_storage.in_evaluation_mode.remote())
                ):
                    infos: Dict = ray.get(
                        self.shared_storage.get_info.remote(["training_step", "num_played_games", "terminate"])
                    )
                    ratio = infos["training_step"] / max(
                        1,
                        infos["num_played_games"] - self.cfg.start_train_after_episodes,
                    )
                    time.sleep(0.010)

    def get_loss(self, batch):
        """
        Parameters:
            for_value [bool]: If True, value loss is returned, else policy loss.

        Returns:
            [torch.Tensor] of shape (batch size,)
        """
        state_batch, value_batch_tensor, policy_batch_tensor = batch
        value_batch_tensor = value_batch_tensor.to(self.device)  # copy to device
        policy_batch_tensor = policy_batch_tensor.to(self.device)
        state_batch = [state_.to(self.device) for state_ in state_batch]
        # Generate predictions
        policy_logits, pred_value_batch = self.model(state_batch)
        padding_mask = self.model.compute_mask(state_batch)

        # Compute loss per step
        # Compute loss for each step

        value_loss, policy_loss = self.loss_function(
            pred_value_batch,
            policy_logits,
            value_batch_tensor,
            policy_batch_tensor,
            padding=padding_mask,
            use_kl=not self.cfg.gumbel_simple_loss,
        )

        return policy_loss, value_loss

    def loss_function(
        self,
        value,
        policy_logits,
        target_value,
        target_policy_tensor,
        padding=None,
        use_kl=False,
    ):
        """
        Parameters
            value: Tensor of shape (batch_size, 1)
            policy_logits: Policy logits which are padded to have the same size.
                Tensor of shape (batch_size, <maximum policy size>)
            target_value: Tensor of shape (batch_size, 1)
            target_policy_tensor: Tensor of shape (batch_size, <max policy len in batch>)

        Returns
            value_loss, policy_loss
        """
        value_loss = torch.square(value - target_value).sum(dim=1)
        log_softmax_policy_masked = self.masked_log_softmax(policy_logits, padding)

        if not use_kl:
            # Cross entropy loss between target distribution and predicted one
            policy_loss = torch.nansum(
                -target_policy_tensor * log_softmax_policy_masked * (1.0 - padding.float()),
                dim=1,
            )
        else:
            # Kullback-Leibler
            kl_loss = torch.nn.KLDivLoss(reduction="none")
            policy_loss = kl_loss(log_softmax_policy_masked, target_policy_tensor)
            policy_loss = torch.nansum(policy_loss, dim=1)
        return value_loss, policy_loss

    def masked_log_softmax(self, x, mask):
        x_masked = x.clone()
        x_masked[mask == 1] = -float("inf")
        return torch.log_softmax(x_masked, 1)

    def update_lr(self):
        """
        Update learning rate with an exponential scheme.
        """
        lr = self.cfg.lr_init * self.cfg.lr_decay_rate ** min(1.0, (self.training_step / self.cfg.lr_decay_steps))
        # mlflow.set_experiment(experiment_name=self.cfg.exp_name)
        # mlflow.log_param("lr", lr)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
