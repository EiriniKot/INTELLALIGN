import ray

from datetime import datetime
from launcher import GAZLauncher
from gaz_singleplayer.config_msa import Config

CUDA_LAUNCH_BLOCKING = 1

if __name__ == "__main__":
    current_dateTime = datetime.now()
    exp_name = f"msa-experiment-{str(current_dateTime)}"

    config = Config(
        save_model=True,
        exp_name=exp_name,
        run_id=0,
        training_games=80000,
        reward_method="SumOfPairs+10*TotalColumn",
        reward_values={"GG": 0, "GL": 0, "LL": 1, "LDL": 0},
    )

    gaz = GAZLauncher(config)
    gaz.setup_workers()

    if config.training_games > 0:
        print(f"Starting Training...")
        gaz.train()
    else:
        if not config.checkpoint_pth:
            print("WARNING: Testing mode, but no checkpoint to load was specified.")
        gaz.test()

    ray.shutdown()
    # mlflow.end_run()
