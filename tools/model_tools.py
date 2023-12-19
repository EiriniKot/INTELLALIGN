"""Module Model Tools."""
import torch


def set_checkpoint(checkpoint_pth=None, only_load_model_weights=None):
    # Initialize checkpoint and replay buffer. Gets loaded later if needed
    checkpoint = {
        # Timestamp for latest network weights, so that unchanged models do not need to be copied.
        "weights_timestamp": 0,
        "weights": None,  # Latest network weights
        "optimizer_state": None,  # Saved optimizer state
        "training_step": 0,  # Number of training steps performed so far
        "num_played_games": 0,  # number of all played episodes so far
        "num_played_steps": 0,  # number of all played moves so far
        "terminate": False,
        "best_eval_score": float("-inf"),  # used to asssess currently best model so far
    }

    # Load previous model if specified
    if checkpoint_pth:
        print(f"Loading checkpoint from path {checkpoint_pth}")
        checkpoint_temp = torch.load(checkpoint_pth, weights_only=only_load_model_weights)
        checkpoint.update(checkpoint_temp)
    return checkpoint
